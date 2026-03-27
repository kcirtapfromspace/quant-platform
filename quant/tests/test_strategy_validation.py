"""Tests for strategy validation — backtest vs live comparison (QUA-27)."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from quant.backtest.report import BacktestReport
from quant.validation import (
    LiveMetrics,
    Recommendation,
    StrategyValidator,
    ValidationConfig,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_backtest(
    sharpe: float = 1.5,
    max_dd: float = 0.12,
    cagr: float = 0.25,
    win_rate: float = 0.55,
    profit_factor: float = 1.8,
    total_return: float = 0.50,
    n_trades: int = 100,
) -> BacktestReport:
    """Create a BacktestReport with specified metrics."""
    return BacktestReport(
        strategy_name="test_strategy",
        symbol="AAPL",
        start_date=date(2022, 1, 1),
        end_date=date(2024, 1, 1),
        train_end_date=None,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        cagr=cagr,
        win_rate=win_rate,
        profit_factor=profit_factor,
        total_return=total_return,
        n_trades=n_trades,
        equity_curve=pd.DataFrame(
            {"date": [], "portfolio_value": [], "drawdown": []}
        ),
        trade_log=pd.DataFrame(
            columns=["entry_date", "exit_date", "direction", "return"]
        ),
    )


def _make_live(
    sharpe: float = 1.3,
    max_dd: float = 0.14,
    cagr: float = 0.20,
    win_rate: float = 0.50,
    profit_factor: float = 1.5,
    total_return: float = 0.10,
    n_trades: int = 20,
) -> LiveMetrics:
    return LiveMetrics(
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        cagr=cagr,
        win_rate=win_rate,
        profit_factor=profit_factor,
        total_return=total_return,
        n_trades=n_trades,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 6, 1),
    )


def _make_equity_curve(
    n_days: int = 252,
    daily_return: float = 0.001,
    volatility: float = 0.015,
    seed: int = 42,
) -> pd.Series:
    """Generate a synthetic equity curve."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(daily_return, volatility, size=n_days)
    prices = 100.0 * np.cumprod(1 + returns)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    return pd.Series(prices, index=dates)


# ── Validator Tests ──────────────────────────────────────────────────────────


class TestStrategyValidator:
    def test_passing_validation(self):
        """Live metrics within thresholds -> CONTINUE."""
        bt = _make_backtest(sharpe=1.5, max_dd=0.12, cagr=0.25)
        lv = _make_live(sharpe=1.3, max_dd=0.14, cagr=0.20)
        validator = StrategyValidator()
        result = validator.validate(bt, lv)
        assert result.passed
        assert result.recommendation == Recommendation.CONTINUE
        assert result.n_critical == 0

    def test_sharpe_decay_critical(self):
        """Large Sharpe decay -> breach detected."""
        bt = _make_backtest(sharpe=2.0)
        lv = _make_live(sharpe=0.5)  # 75% decay
        validator = StrategyValidator(ValidationConfig(max_sharpe_decay=0.30))
        result = validator.validate(bt, lv)
        assert not result.passed
        assert any(b.metric == "sharpe_ratio" for b in result.breaches)

    def test_sharpe_decay_within_threshold(self):
        """Small Sharpe decay -> no breach."""
        bt = _make_backtest(sharpe=1.5)
        lv = _make_live(sharpe=1.2)  # 20% decay, under 30% threshold
        validator = StrategyValidator(ValidationConfig(max_sharpe_decay=0.30))
        result = validator.validate(bt, lv)
        sharpe_breaches = [b for b in result.breaches if b.metric == "sharpe_ratio"]
        assert len(sharpe_breaches) == 0

    def test_drawdown_growth_critical(self):
        """Drawdown much worse than backtest -> breach."""
        bt = _make_backtest(max_dd=0.10)
        lv = _make_live(max_dd=0.20)  # 100% growth
        validator = StrategyValidator(ValidationConfig(max_drawdown_growth=0.50))
        result = validator.validate(bt, lv)
        assert any(b.metric == "max_drawdown" for b in result.breaches)

    def test_drawdown_within_threshold(self):
        bt = _make_backtest(max_dd=0.10)
        lv = _make_live(max_dd=0.12)  # 20% growth, under 50% threshold
        validator = StrategyValidator(ValidationConfig(max_drawdown_growth=0.50))
        result = validator.validate(bt, lv)
        dd_breaches = [b for b in result.breaches if b.metric == "max_drawdown"]
        assert len(dd_breaches) == 0

    def test_cagr_decay_breach(self):
        bt = _make_backtest(cagr=0.30)
        lv = _make_live(cagr=0.10)  # 67% decay
        validator = StrategyValidator(ValidationConfig(max_cagr_decay=0.40))
        result = validator.validate(bt, lv)
        assert any(b.metric == "cagr" for b in result.breaches)

    def test_win_rate_breach(self):
        bt = _make_backtest(win_rate=0.60)
        lv = _make_live(win_rate=0.30, n_trades=20)  # 50% ratio
        validator = StrategyValidator(ValidationConfig(min_win_rate_ratio=0.70))
        result = validator.validate(bt, lv)
        assert any(b.metric == "win_rate" for b in result.breaches)

    def test_profit_factor_breach(self):
        bt = _make_backtest(profit_factor=2.0)
        lv = _make_live(profit_factor=0.8, n_trades=20)  # 40% ratio
        validator = StrategyValidator(ValidationConfig(min_profit_factor_ratio=0.60))
        result = validator.validate(bt, lv)
        assert any(b.metric == "profit_factor" for b in result.breaches)


class TestRecommendations:
    def test_zero_critical_is_continue(self):
        bt = _make_backtest()
        lv = _make_live(sharpe=1.3, max_dd=0.14, cagr=0.20)
        result = StrategyValidator().validate(bt, lv)
        assert result.recommendation == Recommendation.CONTINUE

    def test_one_critical_is_reduce(self):
        bt = _make_backtest(sharpe=2.0)
        lv = _make_live(sharpe=0.2)  # severe Sharpe decay
        result = StrategyValidator(
            ValidationConfig(max_sharpe_decay=0.30)
        ).validate(bt, lv)
        assert result.recommendation in (
            Recommendation.REDUCE_CAPITAL,
            Recommendation.HALT,
        )

    def test_two_critical_is_halt(self):
        bt = _make_backtest(sharpe=2.0, max_dd=0.05, cagr=0.30)
        lv = _make_live(sharpe=0.2, max_dd=0.25, cagr=0.02)
        result = StrategyValidator(
            ValidationConfig(
                max_sharpe_decay=0.20,
                max_drawdown_growth=0.30,
                max_cagr_decay=0.30,
            )
        ).validate(bt, lv)
        assert result.n_critical >= 2
        assert result.recommendation == Recommendation.HALT

    def test_insufficient_data_downgrades_halt(self):
        """With few trades, HALT -> REDUCE_CAPITAL."""
        bt = _make_backtest(sharpe=2.0, max_dd=0.05, cagr=0.30)
        lv = _make_live(sharpe=0.2, max_dd=0.25, cagr=0.02, n_trades=2)
        result = StrategyValidator(
            ValidationConfig(
                max_sharpe_decay=0.20,
                max_drawdown_growth=0.30,
                max_cagr_decay=0.30,
                min_live_trades=5,
            )
        ).validate(bt, lv)
        assert result.insufficient_data
        assert result.recommendation == Recommendation.REDUCE_CAPITAL


class TestLiveMetrics:
    def test_from_equity_curve_basic(self):
        equity = _make_equity_curve(n_days=252, daily_return=0.001)
        metrics = LiveMetrics.from_equity_curve(equity)
        assert metrics.sharpe_ratio > 0
        assert metrics.max_drawdown > 0
        assert metrics.cagr > 0
        assert metrics.total_return > 0
        assert metrics.start_date is not None
        assert metrics.end_date is not None

    def test_from_equity_curve_with_trades(self):
        equity = _make_equity_curve()
        trade_returns = pd.Series([0.02, -0.01, 0.03, 0.01, -0.02, 0.015])
        metrics = LiveMetrics.from_equity_curve(equity, trade_returns)
        assert metrics.n_trades == 6
        assert metrics.win_rate > 0
        assert metrics.profit_factor > 0

    def test_from_empty_equity(self):
        metrics = LiveMetrics.from_equity_curve(pd.Series([], dtype=float))
        assert metrics.sharpe_ratio == 0.0
        assert metrics.n_trades == 0

    def test_manual_construction(self):
        lv = LiveMetrics(
            sharpe_ratio=1.2,
            max_drawdown=0.15,
            cagr=0.18,
            n_trades=50,
        )
        assert lv.sharpe_ratio == 1.2
        assert lv.n_trades == 50


class TestSignalDrift:
    def test_identical_signals_zero_drift(self):
        signals = pd.Series(np.random.default_rng(42).normal(0, 1, 1000))
        validator = StrategyValidator()
        ks = validator.check_signal_drift(signals, signals)
        assert ks == 0.0

    def test_different_signals_nonzero_drift(self):
        rng = np.random.default_rng(42)
        bt_signals = pd.Series(rng.normal(0, 1, 1000))
        lv_signals = pd.Series(rng.normal(0.5, 1.5, 1000))  # shifted
        validator = StrategyValidator()
        ks = validator.check_signal_drift(bt_signals, lv_signals)
        assert ks > 0.1  # meaningful drift

    def test_empty_signals_zero_drift(self):
        validator = StrategyValidator()
        ks = validator.check_signal_drift(pd.Series([], dtype=float), pd.Series([1.0]))
        assert ks == 0.0


class TestEquityCorrelation:
    def test_identical_curves_high_correlation(self):
        equity = _make_equity_curve(seed=42)
        validator = StrategyValidator()
        corr = validator.check_equity_correlation(equity, equity)
        assert corr > 0.99

    def test_uncorrelated_curves(self):
        eq1 = _make_equity_curve(seed=42)
        eq2 = _make_equity_curve(seed=99)
        validator = StrategyValidator()
        corr = validator.check_equity_correlation(eq1, eq2)
        assert abs(corr) < 0.5

    def test_empty_curve_zero_correlation(self):
        validator = StrategyValidator()
        corr = validator.check_equity_correlation(
            pd.Series([], dtype=float),
            _make_equity_curve(),
        )
        assert corr == 0.0


class TestValidationResult:
    def test_summary_output(self):
        bt = _make_backtest()
        lv = _make_live()
        result = StrategyValidator().validate(bt, lv)
        summary = result.summary()
        assert "Validation" in summary
        assert "Recommendation" in summary
        assert "sharpe_ratio" in summary

    def test_metrics_comparison_populated(self):
        bt = _make_backtest()
        lv = _make_live()
        result = StrategyValidator().validate(bt, lv)
        assert "sharpe_ratio" in result.metrics_comparison
        assert "max_drawdown" in result.metrics_comparison
        assert "cagr" in result.metrics_comparison
        assert "backtest" in result.metrics_comparison["sharpe_ratio"]
        assert "live" in result.metrics_comparison["sharpe_ratio"]
        assert "delta_pct" in result.metrics_comparison["sharpe_ratio"]


class TestEdgeCases:
    def test_zero_backtest_sharpe_skipped(self):
        """Zero or negative backtest Sharpe -> skip check."""
        bt = _make_backtest(sharpe=0.0)
        lv = _make_live(sharpe=-0.5)
        result = StrategyValidator().validate(bt, lv)
        sharpe_breaches = [b for b in result.breaches if b.metric == "sharpe_ratio"]
        assert len(sharpe_breaches) == 0

    def test_zero_backtest_drawdown_skipped(self):
        bt = _make_backtest(max_dd=0.0)
        lv = _make_live(max_dd=0.20)
        result = StrategyValidator().validate(bt, lv)
        dd_breaches = [b for b in result.breaches if b.metric == "max_drawdown"]
        assert len(dd_breaches) == 0

    def test_infinite_profit_factor_skipped(self):
        bt = _make_backtest(profit_factor=float("inf"))
        lv = _make_live(profit_factor=1.5)
        result = StrategyValidator().validate(bt, lv)
        pf_breaches = [b for b in result.breaches if b.metric == "profit_factor"]
        assert len(pf_breaches) == 0

    def test_zero_live_trades_skips_win_rate(self):
        bt = _make_backtest(win_rate=0.60)
        lv = _make_live(win_rate=0.0, n_trades=0)
        result = StrategyValidator().validate(bt, lv)
        wr_breaches = [b for b in result.breaches if b.metric == "win_rate"]
        assert len(wr_breaches) == 0

    def test_default_config(self):
        config = ValidationConfig()
        assert config.max_sharpe_decay == 0.30
        assert config.min_live_trades == 5
