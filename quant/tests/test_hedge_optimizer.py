"""Tests for portfolio hedge optimiser (QUA-115)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.risk.hedge_optimizer import (
    HedgeCandidate,
    HedgeConfig,
    HedgeOptimizer,
    RiskProfile,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_DAYS = 500


def _portfolio_and_hedges(
    seed: int = 42,
    n_hedges: int = 3,
    betas: list[float] | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Generate portfolio returns and correlated hedge instrument returns.

    Hedge_i = beta_i * portfolio + noise, so higher beta → more effective.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-01", periods=N_DAYS)
    port = rng.normal(0.0003, 0.015, N_DAYS)

    if betas is None:
        betas = [0.8, 0.5, 0.1]

    cols = {}
    for i, beta in enumerate(betas):
        noise = rng.normal(0.0, 0.01, N_DAYS)
        cols[f"H{i}"] = beta * port + noise

    port_series = pd.Series(port, index=dates, name="portfolio")
    hedge_df = pd.DataFrame(cols, index=dates)
    return port_series, hedge_df


def _negatively_correlated(seed: int = 42) -> tuple[pd.Series, pd.Series]:
    """Portfolio + hedge with negative correlation (ideal hedge)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-01", periods=N_DAYS)
    port = rng.normal(0.0003, 0.015, N_DAYS)
    hedge = -0.9 * port + rng.normal(0.0, 0.003, N_DAYS)
    return (
        pd.Series(port, index=dates, name="portfolio"),
        pd.Series(hedge, index=dates, name="put_overlay"),
    )


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestBasic:
    def test_returns_candidate_list(self):
        port, hedges = _portfolio_and_hedges()
        result = HedgeOptimizer().evaluate_candidates(port, hedges)
        assert isinstance(result, list)
        assert all(isinstance(c, HedgeCandidate) for c in result)

    def test_default_config(self):
        opt = HedgeOptimizer()
        assert opt.config.var_confidence == 0.95
        assert opt.config.min_observations == 60

    def test_custom_config(self):
        cfg = HedgeConfig(var_confidence=0.99)
        assert HedgeOptimizer(cfg).config.var_confidence == 0.99

    def test_candidate_count(self):
        port, hedges = _portfolio_and_hedges(n_hedges=3)
        result = HedgeOptimizer().evaluate_candidates(port, hedges)
        assert len(result) == 3

    def test_n_observations(self):
        port, hedges = _portfolio_and_hedges()
        result = HedgeOptimizer().evaluate_candidates(port, hedges)
        assert result[0].n_observations == N_DAYS


# ---------------------------------------------------------------------------
# Hedge effectiveness ranking
# ---------------------------------------------------------------------------


class TestRanking:
    def test_sorted_by_effectiveness(self):
        """Candidates sorted by R² descending."""
        port, hedges = _portfolio_and_hedges(betas=[0.8, 0.5, 0.1])
        result = HedgeOptimizer().evaluate_candidates(port, hedges)
        r2s = [c.effectiveness for c in result]
        assert r2s == sorted(r2s, reverse=True)

    def test_higher_beta_more_effective(self):
        """Instrument with higher beta should have higher R²."""
        port, hedges = _portfolio_and_hedges(betas=[0.9, 0.2])
        result = HedgeOptimizer().evaluate_candidates(port, hedges)
        # H0 (beta=0.9) should rank first
        assert result[0].symbol == "H0"

    def test_effectiveness_bounded(self):
        port, hedges = _portfolio_and_hedges()
        result = HedgeOptimizer().evaluate_candidates(port, hedges)
        for c in result:
            assert 0 <= c.effectiveness <= 1.0

    def test_correlation_sign(self):
        """Positively correlated hedges should have positive correlation."""
        port, hedges = _portfolio_and_hedges(betas=[0.8])
        result = HedgeOptimizer().evaluate_candidates(port, hedges)
        assert result[0].correlation > 0


# ---------------------------------------------------------------------------
# Hedge ratio
# ---------------------------------------------------------------------------


class TestHedgeRatio:
    def test_hedge_ratio_positive_for_positive_beta(self):
        port, hedges = _portfolio_and_hedges(betas=[0.8])
        result = HedgeOptimizer().evaluate_candidates(port, hedges)
        assert result[0].hedge_ratio > 0

    def test_hedge_ratio_near_true_beta(self):
        """OLS should recover approximately the true beta."""
        port, hedges = _portfolio_and_hedges(betas=[1.0])
        result = HedgeOptimizer().evaluate_candidates(port, hedges)
        # With noise, won't be exact, but should be in range
        assert 0.5 < result[0].hedge_ratio < 2.0

    def test_negative_hedge_ratio_for_negative_correlation(self):
        port, hedge = _negatively_correlated()
        hedges = pd.DataFrame({"put": hedge.values}, index=hedge.index)
        result = HedgeOptimizer().evaluate_candidates(port, hedges)
        assert result[0].hedge_ratio < 0


# ---------------------------------------------------------------------------
# Vol and VaR reduction
# ---------------------------------------------------------------------------


class TestRiskReduction:
    def test_vol_reduction_positive(self):
        """A correlated hedge should reduce portfolio vol."""
        port, hedges = _portfolio_and_hedges(betas=[0.8])
        result = HedgeOptimizer().evaluate_candidates(port, hedges)
        assert result[0].vol_reduction_pct > 0

    def test_var_reduction_positive(self):
        port, hedges = _portfolio_and_hedges(betas=[0.8])
        result = HedgeOptimizer().evaluate_candidates(port, hedges)
        assert result[0].var_reduction_pct > 0

    def test_weak_hedge_small_reduction(self):
        """Very weak hedge should give small vol reduction."""
        port, hedges = _portfolio_and_hedges(betas=[0.01])
        result = HedgeOptimizer().evaluate_candidates(port, hedges)
        assert result[0].vol_reduction_pct < 0.1

    def test_negative_corr_hedge_reduction(self):
        """Negatively correlated hedge should give large vol reduction."""
        port, hedge = _negatively_correlated()
        hedges = pd.DataFrame({"put": hedge.values}, index=hedge.index)
        result = HedgeOptimizer().evaluate_candidates(port, hedges)
        assert result[0].vol_reduction_pct > 0.3


# ---------------------------------------------------------------------------
# Risk profile
# ---------------------------------------------------------------------------


class TestRiskProfile:
    def test_returns_profile(self):
        port, hedges = _portfolio_and_hedges(betas=[0.8])
        profile = HedgeOptimizer().risk_profile(port, hedges["H0"])
        assert isinstance(profile, RiskProfile)

    def test_profile_point_count(self):
        cfg = HedgeConfig(n_profile_points=10)
        port, hedges = _portfolio_and_hedges(betas=[0.8])
        profile = HedgeOptimizer(cfg).risk_profile(port, hedges["H0"])
        assert len(profile.points) == 11  # 0 to 10 inclusive

    def test_profile_starts_at_zero(self):
        port, hedges = _portfolio_and_hedges(betas=[0.8])
        profile = HedgeOptimizer().risk_profile(port, hedges["H0"])
        assert profile.points[0].hedge_fraction == pytest.approx(0.0)

    def test_profile_ends_at_one(self):
        port, hedges = _portfolio_and_hedges(betas=[0.8])
        profile = HedgeOptimizer().risk_profile(port, hedges["H0"])
        assert profile.points[-1].hedge_fraction == pytest.approx(1.0)

    def test_vol_decreases_with_good_hedge(self):
        """Vol should generally decrease as hedge fraction increases."""
        port, hedges = _portfolio_and_hedges(betas=[0.8])
        profile = HedgeOptimizer().risk_profile(port, hedges["H0"])
        # First point (unhedged) should have higher vol than last
        assert profile.points[0].portfolio_vol > profile.points[-1].portfolio_vol

    def test_insufficient_data_raises(self):
        cfg = HedgeConfig(min_observations=100)
        port = pd.Series(
            np.random.default_rng(42).normal(0, 0.01, 50),
            index=pd.bdate_range("2022-01-01", periods=50),
        )
        hedge = pd.Series(
            np.random.default_rng(43).normal(0, 0.01, 50),
            index=pd.bdate_range("2022-01-01", periods=50),
        )
        with pytest.raises(ValueError, match="at least 100"):
            HedgeOptimizer(cfg).risk_profile(port, hedge)


# ---------------------------------------------------------------------------
# Optimal hedge size
# ---------------------------------------------------------------------------


class TestOptimalSize:
    def test_returns_fraction(self):
        port, hedges = _portfolio_and_hedges(betas=[0.8])
        frac = HedgeOptimizer().optimal_hedge_size(port, hedges["H0"], 0.10)
        assert 0.0 <= frac <= 1.0

    def test_small_target_small_fraction(self):
        port, hedges = _portfolio_and_hedges(betas=[0.8])
        small = HedgeOptimizer().optimal_hedge_size(port, hedges["H0"], 0.05)
        large = HedgeOptimizer().optimal_hedge_size(port, hedges["H0"], 0.30)
        assert small <= large

    def test_unreachable_target_returns_one(self):
        """If target reduction exceeds what's achievable, return 1.0."""
        port, hedges = _portfolio_and_hedges(betas=[0.1])
        frac = HedgeOptimizer().optimal_hedge_size(port, hedges["H0"], 0.99)
        assert frac == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Carry costs
# ---------------------------------------------------------------------------


class TestCarryCost:
    def test_zero_carry_infinite_ratio(self):
        port, hedges = _portfolio_and_hedges(betas=[0.8])
        result = HedgeOptimizer().evaluate_candidates(port, hedges)
        assert result[0].risk_per_cost == float("inf")

    def test_positive_carry_finite_ratio(self):
        port, hedges = _portfolio_and_hedges(betas=[0.8])
        carry = {"H0": 50.0}  # 50 bps annual
        result = HedgeOptimizer().evaluate_candidates(port, hedges, carry)
        h0 = next(c for c in result if c.symbol == "H0")
        assert h0.risk_per_cost < float("inf")
        assert h0.risk_per_cost > 0

    def test_carry_cost_stored(self):
        port, hedges = _portfolio_and_hedges(betas=[0.8])
        carry = {"H0": 100.0}
        result = HedgeOptimizer().evaluate_candidates(port, hedges, carry)
        h0 = next(c for c in result if c.symbol == "H0")
        assert h0.carry_cost == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_hedge(self):
        port, hedges = _portfolio_and_hedges(n_hedges=1, betas=[0.5])
        result = HedgeOptimizer().evaluate_candidates(port, hedges)
        assert len(result) == 1

    def test_skips_insufficient_data(self):
        """Hedges with too few overlapping periods are skipped."""
        cfg = HedgeConfig(min_observations=600)
        port, hedges = _portfolio_and_hedges()  # 500 days
        result = HedgeOptimizer(cfg).evaluate_candidates(port, hedges)
        assert len(result) == 0

    def test_partial_overlap(self):
        """Different date ranges should use overlapping portion."""
        rng = np.random.default_rng(42)
        dates_a = pd.bdate_range("2022-01-01", periods=300)
        dates_b = pd.bdate_range("2022-06-01", periods=300)
        port = pd.Series(rng.normal(0, 0.01, 300), index=dates_a)
        hedge = pd.DataFrame(
            {"H0": rng.normal(0, 0.01, 300)}, index=dates_b,
        )
        cfg = HedgeConfig(min_observations=10)
        result = HedgeOptimizer(cfg).evaluate_candidates(port, hedge)
        if result:
            assert result[0].n_observations < 300


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_candidate_summary(self):
        port, hedges = _portfolio_and_hedges(betas=[0.8])
        result = HedgeOptimizer().evaluate_candidates(port, hedges)
        summary = result[0].summary()
        assert "Hedge" in summary
        assert "ratio" in summary.lower()

    def test_profile_summary(self):
        port, hedges = _portfolio_and_hedges(betas=[0.8])
        profile = HedgeOptimizer().risk_profile(port, hedges["H0"])
        summary = profile.summary()
        assert "Risk Profile" in summary
        assert "Vol" in summary
