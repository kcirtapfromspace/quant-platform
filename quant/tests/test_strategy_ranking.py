"""Tests for strategy ranking and comparison (QUA-46)."""
from __future__ import annotations

from quant.portfolio.strategy_ranking import (
    RankingConfig,
    RankingResult,
    StrategyMetrics,
    StrategyRanker,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_good_strategy(name: str = "good") -> StrategyMetrics:
    return StrategyMetrics(
        name=name,
        sharpe_ratio=2.0,
        sortino_ratio=2.5,
        total_return=0.25,
        annualised_return=0.20,
        max_drawdown=0.05,
        calmar_ratio=4.0,
        alpha=0.08,
        alpha_t_stat=3.0,
        residual_vol=0.05,
        execution_quality=0.95,
        positive_months_pct=0.75,
        n_months=36,
    )


def _make_bad_strategy(name: str = "bad") -> StrategyMetrics:
    return StrategyMetrics(
        name=name,
        sharpe_ratio=-0.5,
        sortino_ratio=-0.3,
        total_return=-0.10,
        annualised_return=-0.08,
        max_drawdown=0.30,
        calmar_ratio=-0.27,
        alpha=-0.02,
        alpha_t_stat=-1.0,
        residual_vol=0.15,
        execution_quality=0.30,
        positive_months_pct=0.35,
        n_months=36,
    )


def _make_mediocre_strategy(name: str = "mediocre") -> StrategyMetrics:
    return StrategyMetrics(
        name=name,
        sharpe_ratio=0.5,
        sortino_ratio=0.7,
        total_return=0.05,
        annualised_return=0.04,
        max_drawdown=0.12,
        calmar_ratio=0.33,
        alpha=0.01,
        alpha_t_stat=0.8,
        residual_vol=0.08,
        execution_quality=0.70,
        positive_months_pct=0.55,
        n_months=24,
    )


# ── Tests: Basic ranking ─────────────────────────────────────────────────


class TestBasicRanking:
    def test_empty_strategies(self):
        ranker = StrategyRanker()
        result = ranker.rank([])
        assert result.n_strategies == 0

    def test_single_strategy(self):
        ranker = StrategyRanker()
        result = ranker.rank([_make_good_strategy()])
        assert result.n_strategies == 1
        assert result.rankings[0].rank == 1

    def test_good_beats_bad(self):
        ranker = StrategyRanker()
        result = ranker.rank([_make_bad_strategy(), _make_good_strategy()])
        assert result.rankings[0].name == "good"
        assert result.rankings[1].name == "bad"

    def test_ordering_three_strategies(self):
        ranker = StrategyRanker()
        result = ranker.rank([
            _make_bad_strategy(),
            _make_mediocre_strategy(),
            _make_good_strategy(),
        ])
        assert [r.name for r in result.rankings] == ["good", "mediocre", "bad"]

    def test_ranks_are_1_indexed(self):
        ranker = StrategyRanker()
        result = ranker.rank([
            _make_good_strategy(),
            _make_bad_strategy(),
        ])
        assert result.rankings[0].rank == 1
        assert result.rankings[1].rank == 2


# ── Tests: Composite score ───────────────────────────────────────────────


class TestCompositeScore:
    def test_score_between_zero_and_one(self):
        ranker = StrategyRanker()
        result = ranker.rank([
            _make_good_strategy(),
            _make_bad_strategy(),
            _make_mediocre_strategy(),
        ])
        for r in result.rankings:
            assert 0.0 <= r.composite_score <= 1.0

    def test_good_score_above_bad(self):
        ranker = StrategyRanker()
        result = ranker.rank([_make_good_strategy(), _make_bad_strategy()])
        good = result.by_name("good")
        bad = result.by_name("bad")
        assert good is not None and bad is not None
        assert good.composite_score > bad.composite_score

    def test_dimension_scores_present(self):
        ranker = StrategyRanker()
        result = ranker.rank([_make_good_strategy()])
        dims = result.rankings[0].dimension_scores
        assert "sharpe" in dims
        assert "alpha" in dims
        assert "execution" in dims
        assert "consistency" in dims


# ── Tests: Missing data handling ─────────────────────────────────────────


class TestMissingData:
    def test_all_none_strategy(self):
        """Strategy with all None metrics should get neutral 0.5 scores."""
        ranker = StrategyRanker()
        result = ranker.rank([StrategyMetrics(name="empty")])
        r = result.rankings[0]
        assert abs(r.composite_score - 0.5) < 0.01

    def test_partial_data(self):
        ranker = StrategyRanker()
        partial = StrategyMetrics(name="partial", sharpe_ratio=1.5, alpha=0.05)
        result = ranker.rank([partial])
        r = result.rankings[0]
        # Should score above 0.5 with positive sharpe and alpha
        assert r.composite_score > 0.5

    def test_none_vs_filled_ranking(self):
        ranker = StrategyRanker()
        result = ranker.rank([
            StrategyMetrics(name="empty"),
            _make_good_strategy(),
        ])
        assert result.rankings[0].name == "good"


# ── Tests: Custom weights ────────────────────────────────────────────────


class TestCustomWeights:
    def test_sharpe_only(self):
        """When only Sharpe matters, strategy with best Sharpe wins."""
        config = RankingConfig(
            w_sharpe=1.0, w_sortino=0.0, w_alpha=0.0,
            w_alpha_persistence=0.0, w_max_drawdown=0.0,
            w_calmar=0.0, w_execution=0.0, w_consistency=0.0,
        )
        ranker = StrategyRanker(config)

        # Bad has better Sharpe than mediocre-sharpe-override
        high_sharpe = StrategyMetrics(name="high_sharpe", sharpe_ratio=3.0)
        low_sharpe = StrategyMetrics(name="low_sharpe", sharpe_ratio=0.2)

        result = ranker.rank([low_sharpe, high_sharpe])
        assert result.rankings[0].name == "high_sharpe"

    def test_execution_only(self):
        config = RankingConfig(
            w_sharpe=0.0, w_sortino=0.0, w_alpha=0.0,
            w_alpha_persistence=0.0, w_max_drawdown=0.0,
            w_calmar=0.0, w_execution=1.0, w_consistency=0.0,
        )
        ranker = StrategyRanker(config)

        good_exec = StrategyMetrics(name="good_exec", execution_quality=0.95)
        bad_exec = StrategyMetrics(name="bad_exec", execution_quality=0.20)

        result = ranker.rank([bad_exec, good_exec])
        assert result.rankings[0].name == "good_exec"


# ── Tests: Lookup helpers ────────────────────────────────────────────────


class TestLookupHelpers:
    def test_by_name_found(self):
        ranker = StrategyRanker()
        result = ranker.rank([_make_good_strategy(), _make_bad_strategy()])
        assert result.by_name("good") is not None
        assert result.by_name("good").rank == 1

    def test_by_name_not_found(self):
        ranker = StrategyRanker()
        result = ranker.rank([_make_good_strategy()])
        assert result.by_name("nonexistent") is None

    def test_top(self):
        ranker = StrategyRanker()
        strategies = [
            _make_good_strategy("a"),
            _make_mediocre_strategy("b"),
            _make_bad_strategy("c"),
        ]
        result = ranker.rank(strategies)
        top2 = result.top(2)
        assert len(top2) == 2
        assert top2[0].rank == 1

    def test_top_exceeds_count(self):
        ranker = StrategyRanker()
        result = ranker.rank([_make_good_strategy()])
        top5 = result.top(5)
        assert len(top5) == 1


# ── Tests: Summary ───────────────────────────────────────────────────────


class TestSummary:
    def test_summary_with_data(self):
        ranker = StrategyRanker()
        result = ranker.rank([
            _make_good_strategy(),
            _make_bad_strategy(),
        ])
        summary = result.summary()
        assert "Strategy Ranking" in summary
        assert "good" in summary
        assert "bad" in summary

    def test_empty_summary(self):
        result = RankingResult()
        assert "no strategies" in result.summary()


# ── Tests: Normalisation monotonicity ────────────────────────────────────


class TestNormalisationMonotonicity:
    def test_higher_sharpe_higher_score(self):
        ranker = StrategyRanker()
        low = StrategyMetrics(name="low", sharpe_ratio=0.5)
        high = StrategyMetrics(name="high", sharpe_ratio=2.5)
        result = ranker.rank([low, high])
        low_r = result.by_name("low")
        high_r = result.by_name("high")
        assert high_r.dimension_scores["sharpe"] > low_r.dimension_scores["sharpe"]

    def test_higher_alpha_higher_score(self):
        ranker = StrategyRanker()
        low = StrategyMetrics(name="low", alpha=0.01)
        high = StrategyMetrics(name="high", alpha=0.10)
        result = ranker.rank([low, high])
        assert result.by_name("high").dimension_scores["alpha"] > result.by_name("low").dimension_scores["alpha"]

    def test_lower_drawdown_higher_score(self):
        ranker = StrategyRanker()
        low_dd = StrategyMetrics(name="low_dd", max_drawdown=0.03)
        high_dd = StrategyMetrics(name="high_dd", max_drawdown=0.40)
        result = ranker.rank([low_dd, high_dd])
        assert result.by_name("low_dd").dimension_scores["max_drawdown"] > result.by_name("high_dd").dimension_scores["max_drawdown"]

    def test_higher_execution_higher_score(self):
        ranker = StrategyRanker()
        low = StrategyMetrics(name="low", execution_quality=0.20)
        high = StrategyMetrics(name="high", execution_quality=0.95)
        result = ranker.rank([low, high])
        assert result.by_name("high").dimension_scores["execution"] > result.by_name("low").dimension_scores["execution"]
