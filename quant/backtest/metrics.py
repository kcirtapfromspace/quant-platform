"""Performance metric calculations for backtesting reports."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualised Sharpe ratio.

    Args:
        returns: Daily net returns (not cumulative).
        risk_free_rate: Annual risk-free rate (default 0).

    Returns:
        Sharpe ratio, or 0.0 if standard deviation is zero.
    """
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = returns - daily_rf
    std = excess.std()
    if std == 0 or math.isnan(std):
        return 0.0
    return float((excess.mean() / std) * math.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a positive fraction.

    Returns:
        A value in [0, 1].  0 means no drawdown occurred.
    """
    rolling_max = equity_curve.cummax()
    dd = (equity_curve - rolling_max) / rolling_max
    return float(-dd.min()) if len(dd) > 0 else 0.0


def cagr(equity_curve: pd.Series, n_trading_days: int) -> float:
    """Compound annual growth rate.

    Args:
        equity_curve: Portfolio value series starting at 1.0.
        n_trading_days: Number of trading days covered by the series.

    Returns:
        CAGR as a decimal (e.g. 0.12 for 12%).
    """
    if n_trading_days <= 0 or equity_curve.empty:
        return 0.0
    total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0])
    years = n_trading_days / TRADING_DAYS_PER_YEAR
    if total_return <= 0 or years <= 0:
        return 0.0
    return float(total_return ** (1.0 / years) - 1.0)


def win_rate(trade_returns: pd.Series) -> float:
    """Fraction of trades with positive return.

    Returns:
        Value in [0, 1], or 0.0 if there are no trades.
    """
    if trade_returns.empty:
        return 0.0
    return float((trade_returns > 0).sum() / len(trade_returns))


def profit_factor(trade_returns: pd.Series) -> float:
    """Gross profit divided by gross loss.

    Returns:
        Profit factor >= 0.  Returns ``float('inf')`` when there are no losing
        trades.  Returns 0.0 when there are no winning trades or no trades.
    """
    if trade_returns.empty:
        return 0.0
    gross_profit = trade_returns[trade_returns > 0].sum()
    gross_loss = (-trade_returns[trade_returns < 0]).sum()
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """Return the rolling drawdown series (negative values, as fraction of peak)."""
    rolling_max = equity_curve.cummax()
    return (equity_curve - rolling_max) / rolling_max
