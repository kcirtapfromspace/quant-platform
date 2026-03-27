"""Tests for backtest comparison framework (QUA-48)."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from quant.backtest.comparison import (
    BacktestComparator,
    ComparisonConfig,
    ComparisonResult,
    StrategyRow,
)
from quant.backtest.portfolio_backtest import PortfolioBacktestReport
from quant.portfolio.factor_attribution import FactorAttributionReport

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_report(
    name: str,
    total_return: float = 0.15,
    cagr: float = 0.12,
    sharpe: float = 1.2,
    volatility: float = 0.15,
    max_drawdown: float = 0.08,
    n_days: int = 252,
    seed: int = 42,
    alpha: float | None = None,
) -> PortfolioBacktestReport:
    """Create a synthetic PortfolioBacktestReport for testing."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_days)

    # Generate returns with specified characteristics
    daily_vol = volatility / np.sqrt(252)
    daily_mean = cagr / 252
    returns = pd.Series(
        rng.normal(daily_mean, daily_vol, n_days), index=dates, name=name
    )

    equity = (1 + returns).cumprod() * 1_000_000

    factor_attr = None
    if alpha is not None:
        factor_attr = FactorAttributionReport(
            alpha=alpha,
            alpha_daily=alpha / 252,
            alpha_t_stat=alpha * 10,  # simplified
            r_squared=0.6,
            adjusted_r_squared=0.55,
            total_return=total_return,
            factor_return=total_return - alpha,
            residual_return=alpha,
            n_observations=n_days,
        )

    return PortfolioBacktestReport(
        name=name,
        start_date=date(2023, 1, 1),
        end_date=date(2023, 12, 31),
        initial_capital=1_000_000,
        final_value=1_000_000 * (1 + total_return),
        total_return=total_return,
        cagr=cagr,
        sharpe_ratio=sharpe,
        max_drawdown=max_drawdown,
        volatility=volatility,
        avg_turnover=0.05,
        total_costs=5000.0,
        n_rebalances=12,
        n_trading_days=n_days,
        benchmark_return=0.10,
        benchmark_sharpe=0.8,
        active_return=total_return - 0.10,
        information_ratio=0.5,
        tracking_error=0.05,
        equity_curve=equity,
        returns_series=returns,
        weights_history=pd.DataFrame(),
        rebalances=[],
        factor_attribution=factor_attr,
    )


def _make_three_reports() -> list[PortfolioBacktestReport]:
    """Create three distinct strategy reports."""
    return [
        _make_report("momentum", total_return=0.20, cagr=0.18, sharpe=1.5,
                      max_drawdown=0.10, seed=1, alpha=0.05),
        _make_report("mean_rev", total_return=0.08, cagr=0.06, sharpe=0.7,
                      max_drawdown=0.15, seed=2, alpha=0.02),
        _make_report("low_vol", total_return=0.12, cagr=0.10, sharpe=1.8,
                      max_drawdown=0.04, seed=3, alpha=0.03),
    ]


# ── Tests: Basic comparison ──────────────────────────────────────────────


class TestBasicComparison:
    def test_empty_reports(self):
        comp = BacktestComparator()
        result = comp.compare([])
        assert result.n_strategies == 0

    def test_single_report(self):
        comp = BacktestComparator()
        result = comp.compare([_make_report("a")])
        assert result.n_strategies == 1
        assert result.strategies[0].name == "a"

    def test_three_reports(self):
        comp = BacktestComparator()
        result = comp.compare(_make_three_reports())
        assert result.n_strategies == 3
        assert set(result.names) == {"momentum", "mean_rev", "low_vol"}

    def test_result_type(self):
        comp = BacktestComparator()
        result = comp.compare(_make_three_reports())
        assert isinstance(result, ComparisonResult)


# ── Tests: Strategy rows ─────────────────────────────────────────────────


class TestStrategyRows:
    def test_metrics_populated(self):
        comp = BacktestComparator()
        result = comp.compare([_make_report("a", total_return=0.20)])
        row = result.strategies[0]
        assert isinstance(row, StrategyRow)
        assert row.total_return == 0.20
        assert row.n_rebalances == 12

    def test_calmar_computed(self):
        comp = BacktestComparator()
        result = comp.compare([_make_report("a", cagr=0.12, max_drawdown=0.06)])
        row = result.strategies[0]
        assert abs(row.calmar - 2.0) < 0.01

    def test_factor_attribution_extracted(self):
        comp = BacktestComparator()
        report = _make_report("a", alpha=0.05)
        result = comp.compare([report])
        row = result.strategies[0]
        assert row.alpha == 0.05
        assert row.r_squared == 0.6

    def test_no_factor_attribution(self):
        comp = BacktestComparator()
        report = _make_report("a")
        result = comp.compare([report])
        row = result.strategies[0]
        assert row.alpha is None
        assert row.r_squared is None


# ── Tests: Correlation matrix ────────────────────────────────────────────


class TestCorrelationMatrix:
    def test_correlation_shape(self):
        comp = BacktestComparator()
        result = comp.compare(_make_three_reports())
        assert result.correlation_matrix.shape == (3, 3)

    def test_diagonal_is_one(self):
        comp = BacktestComparator()
        result = comp.compare(_make_three_reports())
        for name in result.names:
            assert abs(result.correlation_matrix.loc[name, name] - 1.0) < 1e-6

    def test_symmetric(self):
        comp = BacktestComparator()
        result = comp.compare(_make_three_reports())
        cm = result.correlation_matrix
        for i in cm.index:
            for j in cm.columns:
                assert abs(cm.loc[i, j] - cm.loc[j, i]) < 1e-6

    def test_single_report_no_correlation(self):
        comp = BacktestComparator()
        result = comp.compare([_make_report("a")])
        assert result.correlation_matrix.empty


# ── Tests: Rolling correlations ──────────────────────────────────────────


class TestRollingCorrelations:
    def test_rolling_corr_computed(self):
        comp = BacktestComparator(ComparisonConfig(correlation_window=30))
        result = comp.compare(_make_three_reports())
        assert not result.rolling_correlations.empty

    def test_rolling_corr_columns(self):
        comp = BacktestComparator()
        reports = _make_three_reports()
        result = comp.compare(reports)
        # 3 strategies → 3 pairs
        assert len(result.rolling_correlations.columns) == 3

    def test_single_report_no_rolling(self):
        comp = BacktestComparator()
        result = comp.compare([_make_report("a")])
        assert result.rolling_correlations.empty


# ── Tests: Drawdown overlap ──────────────────────────────────────────────


class TestDrawdownOverlap:
    def test_overlap_pairs(self):
        comp = BacktestComparator()
        result = comp.compare(_make_three_reports())
        # 3 strategies → 3 pairs
        assert len(result.drawdown_overlap) == 3

    def test_overlap_between_zero_and_one(self):
        comp = BacktestComparator()
        result = comp.compare(_make_three_reports())
        for pct in result.drawdown_overlap.values():
            assert 0.0 <= pct <= 1.0

    def test_single_report_no_overlap(self):
        comp = BacktestComparator()
        result = comp.compare([_make_report("a")])
        assert len(result.drawdown_overlap) == 0


# ── Tests: best_by ───────────────────────────────────────────────────────


class TestBestBy:
    def test_best_sharpe(self):
        comp = BacktestComparator()
        result = comp.compare(_make_three_reports())
        best = result.best_by("sharpe")
        assert best is not None
        assert best.name == "low_vol"  # sharpe=1.8

    def test_best_cagr(self):
        comp = BacktestComparator()
        result = comp.compare(_make_three_reports())
        best = result.best_by("cagr")
        assert best is not None
        assert best.name == "momentum"  # cagr=0.18

    def test_best_max_drawdown(self):
        comp = BacktestComparator()
        result = comp.compare(_make_three_reports())
        best = result.best_by("max_drawdown")
        assert best is not None
        assert best.name == "low_vol"  # dd=0.04 (lowest)

    def test_empty_returns_none(self):
        result = ComparisonResult()
        assert result.best_by("sharpe") is None


# ── Tests: Auto-ranking integration ──────────────────────────────────────


class TestAutoRank:
    def test_auto_rank_produces_result(self):
        comp = BacktestComparator()
        result = comp.compare(_make_three_reports())
        ranking = result.auto_rank()
        assert ranking.n_strategies == 3

    def test_to_strategy_metrics(self):
        comp = BacktestComparator()
        result = comp.compare(_make_three_reports())
        metrics = result.to_strategy_metrics()
        assert len(metrics) == 3
        names = {m.name for m in metrics}
        assert names == {"momentum", "mean_rev", "low_vol"}

    def test_strategy_metrics_fields(self):
        comp = BacktestComparator()
        result = comp.compare(_make_three_reports())
        metrics = result.to_strategy_metrics()
        mom = next(m for m in metrics if m.name == "momentum")
        assert mom.sharpe_ratio == 1.5
        assert mom.alpha == 0.05


# ── Tests: Summary ───────────────────────────────────────────────────────


class TestSummary:
    def test_summary_with_data(self):
        comp = BacktestComparator()
        result = comp.compare(_make_three_reports())
        summary = result.summary()
        assert "Backtest Comparison" in summary
        assert "momentum" in summary
        assert "Correlations" in summary
        assert "Highlights" in summary

    def test_empty_summary(self):
        result = ComparisonResult()
        assert "no strategies" in result.summary()


# ── Tests: Aligned returns ───────────────────────────────────────────────


class TestAlignedReturns:
    def test_aligned_returns_shape(self):
        comp = BacktestComparator()
        result = comp.compare(_make_three_reports())
        assert not result.aligned_returns.empty
        assert set(result.aligned_returns.columns) == {"momentum", "mean_rev", "low_vol"}

    def test_aligned_dates_match(self):
        comp = BacktestComparator(ComparisonConfig(align_dates=True))
        result = comp.compare(_make_three_reports())
        # All columns should have same index (no NaN after dropna)
        assert result.aligned_returns.isna().sum().sum() == 0
