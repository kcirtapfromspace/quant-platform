"""Vectorized backtesting engine.

Design principles
-----------------
* **No lookahead bias**: signal at bar *t* (using close prices up to and
  including *t*) is applied to the return from close *t* → close *t+1*.
  This is enforced by ``signals.shift(1)`` inside the engine — the strategy
  itself must not peek forward either.
* **Realistic fill assumptions**: positions are entered/exited at the next
  close after the signal fires, and a proportional commission is charged on
  every change in position size.
* **Train / test split**: optionally supply ``train_end_date`` to restrict
  signal generation to the training window while measuring performance only
  on the held-out test window.
* **Correctness over speed**: vectorised pandas ops — fast enough for years of
  daily data; no sacrifices to correctness for micro-performance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
from loguru import logger

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
    """Vectorised single-asset backtesting engine.

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
            # Extend signals with zeros for the test window (no new signals
            # after train cutoff — position stays flat unless strategy emits
            # a signal on last training bar that carries into the test window
            # via the shift below).
            test_mask = ~train_mask
            test_index = ohlcv[test_mask].index
            test_signals = pd.Series(0.0, index=test_index)
            raw_signals = pd.concat([train_signals, test_signals])
        else:
            logger.debug("Generating signals on full window ({} bars)", len(ohlcv))
            raw_signals = strategy.generate_signals(ohlcv)

        raw_signals = raw_signals.reindex(ohlcv.index).fillna(0.0)

        # ── No-lookahead enforcement: shift signals forward by 1 bar ─────
        # Signal at close of bar t → position held during bar t+1.
        positions = raw_signals.shift(1).fillna(0.0)

        # ── Return computation ────────────────────────────────────────────
        daily_returns = ohlcv["adj_close"].pct_change().fillna(0.0)
        gross_returns = positions * daily_returns

        # Transaction cost: commission_pct * |change in position|
        pos_delta = positions.diff().abs().fillna(0.0)
        costs = config.commission_pct * pos_delta

        net_returns = gross_returns - costs

        # ── Equity curve ──────────────────────────────────────────────────
        equity = config.initial_capital * (1.0 + net_returns).cumprod()
        dd = m.drawdown_series(equity)

        equity_curve = pd.DataFrame(
            {
                "date": ohlcv.index,
                "portfolio_value": equity.values,
                "drawdown": dd.values,
            }
        )

        # ── Evaluation window ─────────────────────────────────────────────
        # When a train/test split is configured, report metrics on test only.
        if config.train_end_date is not None:
            eval_mask = ohlcv.index.date > config.train_end_date  # type: ignore[union-attr]
            eval_returns = net_returns[eval_mask]
            eval_equity = equity[eval_mask]
            if eval_equity.empty:
                logger.warning("Test window is empty — falling back to full window")
                eval_returns = net_returns
                eval_equity = equity
        else:
            eval_returns = net_returns
            eval_equity = equity

        # Re-base equity curve to 1.0 at start of eval window for CAGR/total return
        first_val = eval_equity.iloc[0] if not eval_equity.empty else 1.0
        eval_equity_rebased = eval_equity / first_val

        # ── Trade log ─────────────────────────────────────────────────────
        trade_log = self._build_trade_log(positions, net_returns, ohlcv.index)

        # ── Metrics ───────────────────────────────────────────────────────
        symbol = str(ohlcv.attrs.get("symbol", "UNKNOWN"))
        start_date = ohlcv.index[0].date() if hasattr(ohlcv.index[0], "date") else ohlcv.index[0]  # type: ignore[union-attr]
        end_date = ohlcv.index[-1].date() if hasattr(ohlcv.index[-1], "date") else ohlcv.index[-1]  # type: ignore[union-attr]

        eval_trade_log = trade_log
        if config.train_end_date is not None and not trade_log.empty:
            eval_trade_log = trade_log[
                pd.to_datetime(trade_log["entry_date"]).dt.date > config.train_end_date
            ]

        trade_returns = eval_trade_log["return"] if not eval_trade_log.empty else pd.Series([], dtype=float)

        total_ret = float(eval_equity_rebased.iloc[-1] - 1.0) if not eval_equity_rebased.empty else 0.0

        report = BacktestReport(
            strategy_name=strategy.name,
            symbol=symbol,
            start_date=start_date,  # type: ignore[arg-type]
            end_date=end_date,  # type: ignore[arg-type]
            train_end_date=config.train_end_date,
            sharpe_ratio=m.sharpe_ratio(eval_returns),
            max_drawdown=m.max_drawdown(eval_equity_rebased),
            cagr=m.cagr(eval_equity_rebased, len(eval_returns)),
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
        # Normalise index to DatetimeIndex for consistent .date access
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df

    @staticmethod
    def _build_trade_log(
        positions: pd.Series,
        net_returns: pd.Series,
        index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Identify round-trip trades and compute per-trade returns.

        A trade is defined as a continuous block of non-zero position.  We
        accumulate returns over the block to get the round-trip P&L.
        """
        records: list[dict] = []
        in_trade = False
        entry_date = None
        direction = None
        trade_returns_acc: list[float] = []

        for i, (pos, ret) in enumerate(zip(positions, net_returns)):
            dt = index[i]
            if not in_trade:
                if pos != 0:
                    in_trade = True
                    entry_date = dt
                    direction = "long" if pos > 0 else "short"
                    trade_returns_acc = [ret]
            else:
                # Still in a trade
                if pos != 0:
                    trade_returns_acc.append(ret)
                    if pos > 0:
                        direction = "long"
                    elif pos < 0:
                        direction = "short"
                else:
                    # Position closed
                    trade_ret = float(np.prod([1 + r for r in trade_returns_acc]) - 1)
                    records.append(
                        {
                            "entry_date": entry_date,
                            "exit_date": index[i - 1],
                            "direction": direction,
                            "return": trade_ret,
                        }
                    )
                    in_trade = False
                    entry_date = None
                    direction = None
                    trade_returns_acc = []

        # Close any open trade at end of data
        if in_trade and entry_date is not None and trade_returns_acc:
            trade_ret = float(np.prod([1 + r for r in trade_returns_acc]) - 1)
            records.append(
                {
                    "entry_date": entry_date,
                    "exit_date": index[-1],
                    "direction": direction,
                    "return": trade_ret,
                }
            )

        if records:
            return pd.DataFrame(records)
        return pd.DataFrame(
            columns=["entry_date", "exit_date", "direction", "return"]
        )
