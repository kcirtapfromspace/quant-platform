"""Backtest report dataclass and formatting utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd


@dataclass
class BacktestReport:
    """Structured result of a single backtest run.

    Attributes:
        strategy_name: Name of the strategy under test.
        symbol: Ticker symbol (or comma-separated list for multi-asset).
        start_date: First date of the evaluation window.
        end_date: Last date of the evaluation window.
        train_end_date: Last date of the training split, or None if no split.
        sharpe_ratio: Annualised Sharpe ratio on the evaluation window.
        max_drawdown: Maximum peak-to-trough drawdown as a positive fraction.
        cagr: Compound annual growth rate on the evaluation window.
        win_rate: Fraction of winning trades (by count).
        profit_factor: Gross profit / gross loss.
        total_return: Total return over the evaluation window as a decimal.
        n_trades: Number of completed round-trip trades.
        equity_curve: DataFrame with columns ``date``, ``portfolio_value``,
            ``drawdown``.
        trade_log: DataFrame with columns ``entry_date``, ``exit_date``,
            ``direction`` (``"long"`` or ``"short"``), ``return``.
    """

    strategy_name: str
    symbol: str
    start_date: date
    end_date: date
    train_end_date: date | None

    # Scalar metrics
    sharpe_ratio: float
    max_drawdown: float
    cagr: float
    win_rate: float
    profit_factor: float
    total_return: float
    n_trades: int

    # Time-series outputs
    equity_curve: pd.DataFrame = field(repr=False)
    trade_log: pd.DataFrame = field(repr=False)

    def summary(self) -> str:
        """Return a human-readable multi-line summary of key metrics."""
        pf = (
            f"{self.profit_factor:.2f}"
            if self.profit_factor != float("inf")
            else "inf"
        )
        lines = [
            f"Strategy : {self.strategy_name}",
            f"Symbol   : {self.symbol}",
            f"Period   : {self.start_date} → {self.end_date}",
        ]
        if self.train_end_date:
            lines.append(f"Train end: {self.train_end_date}")
        lines += [
            "─" * 35,
            f"Total return    : {self.total_return:+.2%}",
            f"CAGR            : {self.cagr:+.2%}",
            f"Sharpe ratio    : {self.sharpe_ratio:.2f}",
            f"Max drawdown    : {self.max_drawdown:.2%}",
            f"Win rate        : {self.win_rate:.2%}",
            f"Profit factor   : {pf}",
            f"# trades        : {self.n_trades}",
        ]
        return "\n".join(lines)
