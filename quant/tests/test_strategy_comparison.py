"""Tests for multi-strategy comparison (QUA-95)."""
from __future__ import annotations

import pytest

from quant.backtest.strategy_comparison import (
    ComparisonConfig,
    ComparisonResult,
    PairwiseComparison,
    RankMetric,
    StrategyComparator,
    StrategyRanking,
    StrategySummary,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _momentum() -> StrategySummary:
    return StrategySummary(
        name="Momentum", sharpe=1.8, cagr=0.15, max_drawdown=-0.20,
        annualised_vol=0.12, calmar=0.75, avg_turnover=0.40,
        total_return=0.60, win_rate=0.55, profit_factor=1.5, n_trades=200,
        capacity_aum=500_000_000,
    )


def _mean_rev() -> StrategySummary:
    return StrategySummary(
        name="MeanRev", sharpe=1.2, cagr=0.08, max_drawdown=-0.15,
        annualised_vol=0.08, calmar=0.53, avg_turnover=0.60,
        total_return=0.35, win_rate=0.60, profit_factor=1.3, n_trades=350,
        capacity_aum=200_000_000,
    )


def _stat_arb() -> StrategySummary:
    return StrategySummary(
        name="StatArb", sharpe=2.1, cagr=0.18, max_drawdown=-0.10,
        annualised_vol=0.10, calmar=1.80, avg_turnover=0.80,
        total_return=0.80, win_rate=0.58, profit_factor=1.8, n_trades=500,
        capacity_aum=100_000_000,
    )


def _bad_strategy() -> StrategySummary:
    return StrategySummary(
        name="BadStrat", sharpe=-0.3, cagr=-0.05, max_drawdown=-0.50,
        annualised_vol=0.25, calmar=-0.10, avg_turnover=1.5,
        total_return=-0.20, win_rate=0.35, profit_factor=0.7, n_trades=100,
    )


def _sample_strategies() -> list[StrategySummary]:
    return [_momentum(), _mean_rev(), _stat_arb()]


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestBasic:
    def test_returns_result(self):
        result = StrategyComparator().compare(_sample_strategies())
        assert isinstance(result, ComparisonResult)

    def test_n_strategies(self):
        result = StrategyComparator().compare(_sample_strategies())
        assert result.n_strategies == 3

    def test_rankings_populated(self):
        result = StrategyComparator().compare(_sample_strategies())
        assert len(result.rankings) == 3

    def test_ranking_types(self):
        result = StrategyComparator().compare(_sample_strategies())
        for r in result.rankings:
            assert isinstance(r, StrategyRanking)

    def test_metric_table_populated(self):
        result = StrategyComparator().compare(_sample_strategies())
        assert len(result.metric_table) == 3

    def test_best_strategy_identified(self):
        result = StrategyComparator().compare(_sample_strategies())
        assert result.best_strategy != ""


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


class TestRanking:
    def test_ranked_by_sharpe_default(self):
        result = StrategyComparator().compare(_sample_strategies())
        assert result.rankings[0].name == "StatArb"  # Sharpe 2.1
        assert result.rankings[1].name == "Momentum"  # Sharpe 1.8
        assert result.rankings[2].name == "MeanRev"   # Sharpe 1.2

    def test_ranked_by_calmar(self):
        cfg = ComparisonConfig(rank_by=RankMetric.CALMAR)
        result = StrategyComparator(cfg).compare(_sample_strategies())
        assert result.rankings[0].name == "StatArb"  # Calmar 1.80

    def test_ranked_by_cagr(self):
        cfg = ComparisonConfig(rank_by=RankMetric.CAGR)
        result = StrategyComparator(cfg).compare(_sample_strategies())
        assert result.rankings[0].name == "StatArb"  # CAGR 0.18

    def test_rank_numbers_sequential(self):
        result = StrategyComparator().compare(_sample_strategies())
        ranks = [r.rank for r in result.rankings]
        assert ranks == [1, 2, 3]


# ---------------------------------------------------------------------------
# Viability
# ---------------------------------------------------------------------------


class TestViability:
    def test_good_strategies_viable(self):
        result = StrategyComparator().compare(_sample_strategies())
        assert result.n_viable == 3

    def test_bad_strategy_not_viable(self):
        strategies = _sample_strategies() + [_bad_strategy()]
        result = StrategyComparator(ComparisonConfig(
            min_sharpe=0.0, max_drawdown_limit=-0.30,
        )).compare(strategies)
        bad = next(r for r in result.rankings if r.name == "BadStrat")
        assert not bad.is_viable  # HIGH_DRAWDOWN flag

    def test_min_sharpe_filter(self):
        cfg = ComparisonConfig(min_sharpe=1.5)
        result = StrategyComparator(cfg).compare(_sample_strategies())
        mean_rev = next(r for r in result.rankings if r.name == "MeanRev")
        assert not mean_rev.is_viable  # Sharpe 1.2 < 1.5

    def test_n_viable_counted(self):
        strategies = _sample_strategies() + [_bad_strategy()]
        result = StrategyComparator(ComparisonConfig(
            max_drawdown_limit=-0.30,
        )).compare(strategies)
        assert result.n_viable == 3  # 3 good, 1 bad


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------


class TestFlags:
    def test_high_drawdown_flag(self):
        strategies = [_bad_strategy()]
        result = StrategyComparator(ComparisonConfig(
            max_drawdown_limit=-0.30,
        )).compare(strategies)
        assert "HIGH_DRAWDOWN" in result.rankings[0].flags

    def test_high_turnover_flag(self):
        strategies = [_bad_strategy()]
        result = StrategyComparator(ComparisonConfig(
            max_turnover_limit=1.0,
        )).compare(strategies)
        assert "HIGH_TURNOVER" in result.rankings[0].flags

    def test_low_sharpe_flag(self):
        strategies = [_bad_strategy()]
        result = StrategyComparator(ComparisonConfig(
            min_sharpe=0.5,
        )).compare(strategies)
        assert "LOW_SHARPE" in result.rankings[0].flags

    def test_good_strategy_no_flags(self):
        result = StrategyComparator().compare([_stat_arb()])
        assert len(result.rankings[0].flags) == 0


# ---------------------------------------------------------------------------
# Pairwise comparison
# ---------------------------------------------------------------------------


class TestPairwise:
    def test_pairwise_populated(self):
        result = StrategyComparator().compare(_sample_strategies())
        # 3 choose 2 = 3
        assert len(result.pairwise) == 3

    def test_pairwise_types(self):
        result = StrategyComparator().compare(_sample_strategies())
        for p in result.pairwise:
            assert isinstance(p, PairwiseComparison)

    def test_winner_identified(self):
        result = StrategyComparator().compare(_sample_strategies())
        for p in result.pairwise:
            assert p.winner in [p.strategy_a, p.strategy_b]

    def test_sharpe_diff_sign(self):
        result = StrategyComparator().compare([_stat_arb(), _mean_rev()])
        pw = result.pairwise[0]
        # StatArb Sharpe 2.1 > MeanRev 1.2 => positive diff
        assert pw.sharpe_diff > 0

    def test_no_pairwise_for_single(self):
        result = StrategyComparator().compare([_momentum()])
        assert len(result.pairwise) == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_strategy(self):
        result = StrategyComparator().compare([_momentum()])
        assert result.n_strategies == 1
        assert result.rankings[0].rank == 1
        assert result.best_strategy == "Momentum"

    def test_empty_strategies_raises(self):
        with pytest.raises(ValueError, match="at least 1"):
            StrategyComparator().compare([])

    def test_identical_strategies(self):
        s1 = _momentum()
        s2 = StrategySummary(name="MomentumCopy", sharpe=1.8, cagr=0.15,
                             max_drawdown=-0.20, annualised_vol=0.12,
                             calmar=0.75, avg_turnover=0.40)
        result = StrategyComparator().compare([s1, s2])
        assert result.n_strategies == 2

    def test_many_strategies(self):
        strategies = [
            StrategySummary(name=f"S{i}", sharpe=float(i))
            for i in range(10)
        ]
        result = StrategyComparator().compare(strategies)
        assert result.rankings[0].name == "S9"  # Highest Sharpe


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_readable(self):
        result = StrategyComparator().compare(_sample_strategies())
        summary = result.summary()
        assert "Strategy Comparison" in summary
        assert "Best strategy" in summary
        assert "Momentum" in summary
        assert "StatArb" in summary
        assert "Sharpe" in summary
        assert "Viable" in summary

    def test_summary_with_flags(self):
        strategies = _sample_strategies() + [_bad_strategy()]
        result = StrategyComparator(ComparisonConfig(
            max_drawdown_limit=-0.30,
        )).compare(strategies)
        summary = result.summary()
        assert "HIGH_DRAWDOWN" in summary
