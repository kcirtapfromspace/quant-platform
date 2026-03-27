"""Vectorized backtesting engine.

Design principles
-----------------
* **No lookahead bias**: signal at bar *t* (using close prices up to and
  including *t*) is applied to the return from close *t* → close *t+1*.
  This is enforced by the Rust kernel — the strategy itself must not peek
  forward either.
* **Realistic fill assumptions**: positions are entered/exited at the next
  close after the signal fires, and a proportional commission is charged on
  every change in position size.
* **Train / test split**: optionally supply ``train_end_date`` to restrict
  signal generation to the training window while measuring performance only
  on the held-out test window.
* **Rust core**: the inner loop (equity curve, trade log, metrics) runs in
  ``quant_rs.backtest.run_backtest`` for maximum throughput.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
from loguru import logger

import quant_rs as _qrs

from quant.backtest import metrics as m
from quant.backtest.report import BacktestReport
from quant.backtest.strategy import Strategy


@dataclass
class BacktestConfig:
    """Configuration for a single backtest run.

    Attributes:
        commission_pct: One-way commission as a fraction of trade value
            (e.g. ``0.001`` for 10 bps).  Applied each time the position
            changes, proportional to the absolute position change.
        initial_capital: Starting portfolio value (default 1.0 — returns are
            expressed as multiples of initial capital).
        train_end_date: If set, strategy signals are generated only on bars up
            to and including this date.  Performance metrics are computed on
            the *test* portion (bars after ``train_end_date``).  When ``None``
            the full date range is used for both signal generation and
            evaluation.
    """

    commission_pct: float = 0.001
    initial_capital: float = 1.0
    train_end_date: date | None = None


class BacktestEngine:
    """Vectorised single-asset backtesting engine backed by Rust kernels.

    Usage::

        engine = BacktestEngine()
        report = engine.run(ohlcv_df, strategy, BacktestConfig())
        print(report.summary())
    """

    def run(
        self,
        ohlcv: pd.DataFrame,
        strategy: Strategy,
        config: BacktestConfig | None = None,
    ) -> BacktestReport:
        """Run a backtest and return a structured report.

        Args:
            ohlcv: DataFrame with at minimum the column ``adj_close`` and a
                date-like index (DatetimeIndex or Index of ``date`` objects).
                Additional columns (``open``, ``high``, ``low``, ``close``,
                ``volume``) are passed through to the strategy as-is.
                Rows must be in **ascending** chronological order.
            strategy: A :class:`~quant.backtest.strategy.Strategy` instance.
            config: Backtest configuration.  Defaults to :class:`BacktestConfig`
                with commission 10 bps and no train/test split.

        Returns:
            A :class:`~quant.backtest.report.BacktestReport` containing scalar
            metrics plus the equity curve and trade log DataFrames.

        Raises:
            ValueError: If *ohlcv* is empty or missing ``adj_close``.
        """
        if config is None:
            config = BacktestConfig()

        ohlcv = self._validate_and_normalise(ohlcv)

        # ── Signal generation ─────────────────────────────────────────────
        if config.train_end_date is not None:
            train_mask = ohlcv.index.date <= config.train_end_date  # type: ignore[union-attr]
            train_df = ohlcv[train_mask]
            logger.debug(
                "Generating signals on training window ({} bars)",
                len(train_df),
            )
            train_signals = strategy.generate_signals(train_df)
            test_mask = ~train_mask
            test_index = ohlcv[test_mask].index
            test_signals = pd.Series(0.0, index=test_index)
            raw_signals = pd.concat([train_signals, test_signals])
        else:
            logger.debug("Generating signals on full window ({} bars)", len(ohlcv))
            raw_signals = strategy.generate_signals(ohlcv)

        raw_signals = raw_signals.reindex(ohlcv.index).fillna(0.0)

        # ── Rust core: equity curve + trade log ───────────────────────────
        result = _qrs.backtest.run_backtest(
            ohlcv["adj_close"].tolist(),
            raw_signals.tolist(),
            config.commission_pct,
            config.initial_capital,
        )

        equity_vals = [pv for pv, _ in result["equity_curve"]]
        dd_vals = [dd for _, dd in result["equity_curve"]]

        equity_curve = pd.DataFrame(
            {
                "date": ohlcv.index,
                "portfolio_value": equity_vals,
                "drawdown": dd_vals,
            }
        )

        # Build trade log with dates from bar indices
        records = [
            {
                "entry_date": ohlcv.index[entry_idx],
                "exit_date": ohlcv.index[exit_idx],
                "direction": direction,
                "return": ret,
            }
            for entry_idx, exit_idx, direction, ret in result["trades"]
        ]
        if records:
            trade_log = pd.DataFrame(records)
        else:
            trade_log = pd.DataFrame(
                columns=["entry_date", "exit_date", "direction", "return"]
            )

        # ── Evaluation window ─────────────────────────────────────────────
        symbol = str(ohlcv.attrs.get("symbol", "UNKNOWN"))
        start_date = ohlcv.index[0].date() if hasattr(ohlcv.index[0], "date") else ohlcv.index[0]  # type: ignore[union-attr]
        end_date = ohlcv.index[-1].date() if hasattr(ohlcv.index[-1], "date") else ohlcv.index[-1]  # type: ignore[union-attr]

        equity_series = pd.Series(equity_vals, index=ohlcv.index)

        if config.train_end_date is not None:
            eval_mask = ohlcv.index.date > config.train_end_date  # type: ignore[union-attr]
            eval_equity = equity_series[eval_mask]
            if not trade_log.empty:
                eval_trade_log = trade_log[
                    pd.to_datetime(trade_log["entry_date"]).dt.date > config.train_end_date
                ]
            else:
                eval_trade_log = trade_log

            if eval_equity.empty:
                logger.warning("Test window is empty — falling back to full window")
                eval_equity = equity_series
                eval_trade_log = trade_log
        else:
            eval_equity = equity_series
            eval_trade_log = trade_log

        first_val = eval_equity.iloc[0] if not eval_equity.empty else 1.0
        eval_equity_rebased = eval_equity / first_val

        # Recompute metrics on eval window from rebased equity
        eval_net_returns = eval_equity_rebased.pct_change().fillna(0.0)
        trade_returns = (
            eval_trade_log["return"] if not eval_trade_log.empty
            else pd.Series([], dtype=float)
        )
        total_ret = float(eval_equity_rebased.iloc[-1] - 1.0) if not eval_equity_rebased.empty else 0.0

        report = BacktestReport(
            strategy_name=strategy.name,
            symbol=symbol,
            start_date=start_date,  # type: ignore[arg-type]
            end_date=end_date,  # type: ignore[arg-type]
            train_end_date=config.train_end_date,
            sharpe_ratio=m.sharpe_ratio(eval_net_returns),
            max_drawdown=m.max_drawdown(eval_equity_rebased),
            cagr=m.cagr(eval_equity_rebased, len(eval_net_returns)),
            win_rate=m.win_rate(trade_returns),
            profit_factor=m.profit_factor(trade_returns),
            total_return=total_ret,
            n_trades=len(eval_trade_log),
            equity_curve=equity_curve,
            trade_log=trade_log,
        )

        logger.info(
            "Backtest complete: {} | {} | Sharpe={:.2f} | MaxDD={:.1%} | CAGR={:.1%}",
            strategy.name,
            symbol,
            report.sharpe_ratio,
            report.max_drawdown,
            report.cagr,
        )
        return report

    # ── Private helpers ───────────────────────────────────────────────────

    @staticmethod
    def _validate_and_normalise(ohlcv: pd.DataFrame) -> pd.DataFrame:
        if ohlcv.empty:
            raise ValueError("ohlcv DataFrame is empty")
        if "adj_close" not in ohlcv.columns:
            raise ValueError("ohlcv must contain an 'adj_close' column")
        df = ohlcv.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df
