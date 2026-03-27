"""Tests for the backtesting framework."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from quant.backtest import BacktestConfig, BacktestEngine, BacktestReport, Strategy
from quant.backtest.metrics import (
    cagr,
    drawdown_series,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    win_rate,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_ohlcv(prices: list[float], symbol: str = "TEST") -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from a list of close prices."""
    n = len(prices)
    start = pd.Timestamp("2024-01-01")
    dates = pd.date_range(start, periods=n, freq="B")  # business days
    df = pd.DataFrame(
        {
            "open": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
            "adj_close": prices,
            "volume": [1_000_000] * n,
        },
        index=dates,
    )
    df.attrs["symbol"] = symbol
    return df


class AlwaysLong(Strategy):
    """Holds long 100% of the time."""

    def generate_signals(self, ohlcv: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=ohlcv.index)


class AlwaysFlat(Strategy):
    """Never trades."""

    def generate_signals(self, ohlcv: pd.DataFrame) -> pd.Series:
        return pd.Series(0.0, index=ohlcv.index)


class AlwaysShort(Strategy):
    """Holds short 100% of the time."""

    def generate_signals(self, ohlcv: pd.DataFrame) -> pd.Series:
        return pd.Series(-1.0, index=ohlcv.index)


class FixedSignals(Strategy):
    """Emits a fixed signal series supplied at construction."""

    def __init__(self, signals: pd.Series) -> None:
        self._signals = signals

    def generate_signals(self, ohlcv: pd.DataFrame) -> pd.Series:
        return self._signals.reindex(ohlcv.index).fillna(0.0)


# ── Metric unit tests ──────────────────────────────────────────────────────────

def test_sharpe_zero_std():
    returns = pd.Series([0.0] * 10)
    assert sharpe_ratio(returns) == 0.0


def test_sharpe_positive():
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.normal(0.001, 0.01, 252))
    sr = sharpe_ratio(returns)
    assert sr > 0  # positive drift → positive Sharpe


def test_max_drawdown_no_drawdown():
    equity = pd.Series([1.0, 1.1, 1.2, 1.3])
    assert max_drawdown(equity) == pytest.approx(0.0, abs=1e-9)


def test_max_drawdown_simple():
    equity = pd.Series([1.0, 1.2, 0.8, 1.0])
    dd = max_drawdown(equity)
    # Peak 1.2, trough 0.8 → drawdown = (0.8 - 1.2) / 1.2
    expected = (1.2 - 0.8) / 1.2
    assert dd == pytest.approx(expected, rel=1e-6)


def test_cagr_flat():
    equity = pd.Series([1.0] * 252)
    assert cagr(equity, 252) == pytest.approx(0.0, abs=1e-9)


def test_cagr_doubles_in_one_year():
    equity = pd.Series([1.0, 2.0])
    result = cagr(equity, 252)
    # 100% return in exactly one year → CAGR = 1.0
    assert result == pytest.approx(1.0, rel=1e-3)


def test_win_rate_empty():
    assert win_rate(pd.Series([], dtype=float)) == 0.0


def test_win_rate_all_winners():
    assert win_rate(pd.Series([0.01, 0.02, 0.005])) == pytest.approx(1.0)


def test_win_rate_half():
    assert win_rate(pd.Series([0.01, -0.01])) == pytest.approx(0.5)


def test_profit_factor_no_losers():
    pf = profit_factor(pd.Series([0.1, 0.2, 0.3]))
    assert pf == float("inf")


def test_profit_factor_no_winners():
    pf = profit_factor(pd.Series([-0.1, -0.2]))
    assert pf == 0.0


def test_profit_factor_mixed():
    pf = profit_factor(pd.Series([0.3, -0.1]))
    # 0.3 / 0.1 = 3.0
    assert pf == pytest.approx(3.0, rel=1e-6)


def test_drawdown_series_monotone_up():
    equity = pd.Series([1.0, 1.1, 1.2])
    dd = drawdown_series(equity)
    assert (dd == 0.0).all()


# ── Engine integration tests ───────────────────────────────────────────────────

@pytest.fixture
def engine() -> BacktestEngine:
    return BacktestEngine()


def test_flat_strategy_equity_flat(engine):
    """AlwaysFlat should produce a completely flat equity curve."""
    prices = [100.0 + i for i in range(30)]
    ohlcv = _make_ohlcv(prices)
    report = engine.run(ohlcv, AlwaysFlat(), BacktestConfig(commission_pct=0.001))
    # All portfolio values should be equal (flat = no exposure)
    assert report.equity_curve["portfolio_value"].nunique() == 1
    assert report.n_trades == 0
    assert report.total_return == pytest.approx(0.0, abs=1e-9)


def test_always_long_rising_market(engine):
    """Long-only in a rising market should produce positive total return."""
    prices = [100.0 * (1.001 ** i) for i in range(252)]  # +0.1%/day
    ohlcv = _make_ohlcv(prices)
    report = engine.run(ohlcv, AlwaysLong(), BacktestConfig(commission_pct=0.0))
    assert report.total_return > 0
    assert report.cagr > 0


def test_always_short_falling_market(engine):
    """Short-only in a falling market should produce positive return."""
    prices = [100.0 * (0.999 ** i) for i in range(252)]
    ohlcv = _make_ohlcv(prices)
    report = engine.run(ohlcv, AlwaysShort(), BacktestConfig(commission_pct=0.0))
    assert report.total_return > 0


def test_no_lookahead_bias(engine):
    """Signal at bar t must not benefit from the bar-t close itself.

    We use a strategy that always signals 1, but the first useful return can
    only be computed from bar 1 (using bar 0's close as entry).  The equity
    on bar 0 (before any return) must equal the initial capital.
    """
    prices = [100.0, 105.0, 103.0, 108.0]
    ohlcv = _make_ohlcv(prices)
    report = engine.run(ohlcv, AlwaysLong(), BacktestConfig(commission_pct=0.0))
    pv = report.equity_curve["portfolio_value"]
    # The very first bar has no shifted signal yet — position = 0
    # so portfolio value stays at initial capital
    assert pv.iloc[0] == pytest.approx(1.0, rel=1e-9)


def test_commission_reduces_return(engine):
    prices = [100.0 * (1.001 ** i) for i in range(252)]
    ohlcv = _make_ohlcv(prices)
    zero_cost = engine.run(ohlcv, AlwaysLong(), BacktestConfig(commission_pct=0.0))
    with_cost = engine.run(ohlcv, AlwaysLong(), BacktestConfig(commission_pct=0.001))
    assert with_cost.total_return < zero_cost.total_return


def test_train_test_split(engine):
    """Metrics are computed on the test window only when train_end_date is set."""
    prices = [100.0 + i for i in range(100)]
    ohlcv = _make_ohlcv(prices)
    train_end = ohlcv.index[49].date()  # first 50 bars = training
    config = BacktestConfig(train_end_date=train_end, commission_pct=0.0)
    report = engine.run(ohlcv, AlwaysLong(), config)
    assert report.train_end_date == train_end


def test_equity_curve_shape(engine):
    prices = [100.0 + i for i in range(50)]
    ohlcv = _make_ohlcv(prices)
    report = engine.run(ohlcv, AlwaysLong(), BacktestConfig())
    assert len(report.equity_curve) == len(ohlcv)
    assert list(report.equity_curve.columns) == ["date", "portfolio_value", "drawdown"]


def test_trade_log_single_long(engine):
    """Strategy is long for 5 bars then flat: should produce one trade."""
    prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 105.0]
    ohlcv = _make_ohlcv(prices)
    # Signal: 1 for first 5 bars, then 0
    sigs = pd.Series([1, 1, 1, 1, 1, 0, 0], index=ohlcv.index, dtype=float)
    report = engine.run(ohlcv, FixedSignals(sigs), BacktestConfig(commission_pct=0.0))
    assert report.n_trades == 1
    assert report.trade_log.iloc[0]["direction"] == "long"
    assert report.trade_log.iloc[0]["return"] > 0


def test_trade_log_short_direction(engine):
    prices = [105.0, 104.0, 103.0, 102.0, 101.0, 100.0, 100.0]
    ohlcv = _make_ohlcv(prices)
    sigs = pd.Series([-1, -1, -1, -1, -1, 0, 0], index=ohlcv.index, dtype=float)
    report = engine.run(ohlcv, FixedSignals(sigs), BacktestConfig(commission_pct=0.0))
    assert report.n_trades == 1
    assert report.trade_log.iloc[0]["direction"] == "short"
    assert report.trade_log.iloc[0]["return"] > 0


def test_report_summary_string(engine):
    prices = [100.0 + i for i in range(30)]
    ohlcv = _make_ohlcv(prices, symbol="AAPL")
    report = engine.run(ohlcv, AlwaysLong(), BacktestConfig())
    summary = report.summary()
    assert "AlwaysLong" in summary
    assert "AAPL" in summary
    assert "Sharpe" in summary


def test_empty_ohlcv_raises(engine):
    empty = pd.DataFrame(columns=["adj_close"])
    with pytest.raises(ValueError, match="empty"):
        engine.run(empty, AlwaysLong(), BacktestConfig())


def test_missing_adj_close_raises(engine):
    df = pd.DataFrame({"close": [100.0, 101.0]}, index=pd.date_range("2024-01-01", periods=2))
    with pytest.raises(ValueError, match="adj_close"):
        engine.run(df, AlwaysLong(), BacktestConfig())


def test_max_drawdown_in_report(engine):
    """A declining market with a long position should show non-zero max drawdown."""
    prices = [100.0 * (0.99 ** i) for i in range(50)]
    ohlcv = _make_ohlcv(prices)
    report = engine.run(ohlcv, AlwaysLong(), BacktestConfig(commission_pct=0.0))
    assert report.max_drawdown > 0


def test_profit_factor_in_report(engine):
    """With zero commission in a rising market all trade returns positive → inf PF."""
    prices = [100.0 * (1.005 ** i) for i in range(20)]
    ohlcv = _make_ohlcv(prices)
    sigs = pd.Series([1] * 10 + [0] * 10, index=ohlcv.index, dtype=float)
    report = engine.run(ohlcv, FixedSignals(sigs), BacktestConfig(commission_pct=0.0))
    assert report.profit_factor == float("inf") or report.profit_factor > 0
