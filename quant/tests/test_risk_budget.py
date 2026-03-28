"""Tests for risk budget allocation (QUA-89)."""
from __future__ import annotations

import pytest

from quant.risk.risk_budget import (
    AllocationMethod,
    RiskBudgetAllocator,
    RiskBudgetConfig,
    RiskBudgetResult,
    SleeveRiskBudget,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VOLS = {"momentum": 0.15, "mean_rev": 0.08, "stat_arb": 0.12}
SHARPES = {"momentum": 1.2, "mean_rev": 0.8, "stat_arb": 1.5}
CORR = {
    ("momentum", "mean_rev"): -0.10,
    ("momentum", "stat_arb"): 0.20,
    ("mean_rev", "stat_arb"): 0.05,
}


def _allocator(**overrides) -> RiskBudgetAllocator:
    return RiskBudgetAllocator(RiskBudgetConfig(**overrides))


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestBasic:
    def test_returns_result(self):
        result = _allocator().allocate(VOLS, SHARPES, CORR)
        assert isinstance(result, RiskBudgetResult)

    def test_n_sleeves(self):
        result = _allocator().allocate(VOLS, SHARPES, CORR)
        assert result.n_sleeves == 3

    def test_sleeve_types(self):
        result = _allocator().allocate(VOLS, SHARPES, CORR)
        for s in result.sleeves:
            assert isinstance(s, SleeveRiskBudget)

    def test_capital_weights_populated(self):
        result = _allocator().allocate(VOLS, SHARPES, CORR)
        assert len(result.capital_weights) == 3

    def test_portfolio_vol_positive(self):
        result = _allocator().allocate(VOLS, SHARPES, CORR)
        assert result.expected_portfolio_vol > 0


# ---------------------------------------------------------------------------
# Risk shares
# ---------------------------------------------------------------------------


class TestRiskShares:
    def test_shares_sum_to_one(self):
        result = _allocator().allocate(VOLS, SHARPES, CORR)
        total = sum(s.risk_share for s in result.sleeves)
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_shares_non_negative(self):
        result = _allocator().allocate(VOLS, SHARPES, CORR)
        for s in result.sleeves:
            assert s.risk_share >= 0

    def test_shares_within_bounds(self):
        result = _allocator(
            min_risk_share=0.10, max_risk_share=0.60,
        ).allocate(VOLS, SHARPES, CORR)
        for s in result.sleeves:
            assert s.risk_share >= 0.10 - 1e-6
            assert s.risk_share <= 0.60 + 1e-6


# ---------------------------------------------------------------------------
# Equal risk
# ---------------------------------------------------------------------------


class TestEqualRisk:
    def test_equal_shares(self):
        result = _allocator(method=AllocationMethod.EQUAL_RISK).allocate(VOLS, SHARPES, CORR)
        for s in result.sleeves:
            assert s.risk_share == pytest.approx(1.0 / 3, abs=1e-6)

    def test_allocated_vol_equal(self):
        result = _allocator(
            method=AllocationMethod.EQUAL_RISK,
            total_vol_target=0.12,
        ).allocate(VOLS, SHARPES, CORR)
        for s in result.sleeves:
            assert s.allocated_vol == pytest.approx(0.04, abs=1e-6)


# ---------------------------------------------------------------------------
# Sharpe-weighted
# ---------------------------------------------------------------------------


class TestSharpeWeighted:
    def test_higher_sharpe_more_budget(self):
        result = _allocator(
            method=AllocationMethod.SHARPE_WEIGHTED,
        ).allocate(VOLS, SHARPES, CORR)
        stat_arb = next(s for s in result.sleeves if s.name == "stat_arb")
        mean_rev = next(s for s in result.sleeves if s.name == "mean_rev")
        # stat_arb Sharpe 1.5 > mean_rev 0.8
        assert stat_arb.risk_share > mean_rev.risk_share

    def test_proportional_to_sharpe(self):
        result = _allocator(
            method=AllocationMethod.SHARPE_WEIGHTED,
            min_risk_share=0.0,
            max_risk_share=1.0,
        ).allocate(VOLS, SHARPES, CORR)
        total_sharpe = sum(SHARPES.values())
        for s in result.sleeves:
            expected = SHARPES[s.name] / total_sharpe
            assert s.risk_share == pytest.approx(expected, abs=1e-4)

    def test_zero_sharpes_fall_back_to_equal(self):
        zero_sharpes = dict.fromkeys(VOLS, 0.0)
        result = _allocator(
            method=AllocationMethod.SHARPE_WEIGHTED,
        ).allocate(VOLS, zero_sharpes, CORR)
        for s in result.sleeves:
            assert s.risk_share == pytest.approx(1.0 / 3, abs=0.05)


# ---------------------------------------------------------------------------
# Optimized
# ---------------------------------------------------------------------------


class TestOptimized:
    def test_optimized_runs(self):
        result = _allocator(
            method=AllocationMethod.OPTIMIZED,
        ).allocate(VOLS, SHARPES, CORR)
        assert result.n_sleeves == 3

    def test_optimized_favours_high_sharpe_low_vol(self):
        """mean_rev: low vol (0.08) + decent Sharpe (0.8) should get good allocation."""
        result = _allocator(
            method=AllocationMethod.OPTIMIZED,
            min_risk_share=0.0,
            max_risk_share=1.0,
        ).allocate(VOLS, SHARPES, CORR)
        mean_rev = next(s for s in result.sleeves if s.name == "mean_rev")
        # mean_rev has lowest vol so inv_vol * sharpe is competitive
        assert mean_rev.risk_share > 0.15

    def test_optimized_shares_sum_to_one(self):
        result = _allocator(
            method=AllocationMethod.OPTIMIZED,
        ).allocate(VOLS, SHARPES, CORR)
        total = sum(s.risk_share for s in result.sleeves)
        assert total == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Capital weights
# ---------------------------------------------------------------------------


class TestCapitalWeights:
    def test_capital_weights_positive(self):
        result = _allocator().allocate(VOLS, SHARPES, CORR)
        for w in result.capital_weights.values():
            assert w >= 0

    def test_capital_weight_from_vol(self):
        """cap_weight = allocated_vol / sleeve_vol."""
        result = _allocator().allocate(VOLS, SHARPES, CORR)
        for s in result.sleeves:
            expected = s.allocated_vol / s.sleeve_vol if s.sleeve_vol > 0 else 0
            assert s.capital_weight == pytest.approx(expected, rel=1e-6)

    def test_capital_weights_match_result(self):
        result = _allocator().allocate(VOLS, SHARPES, CORR)
        for s in result.sleeves:
            assert result.capital_weights[s.name] == pytest.approx(
                s.capital_weight, abs=1e-8,
            )


# ---------------------------------------------------------------------------
# VaR conversion
# ---------------------------------------------------------------------------


class TestVaR:
    def test_var_positive(self):
        result = _allocator().allocate(VOLS, SHARPES, CORR)
        for s in result.sleeves:
            assert s.allocated_var > 0

    def test_higher_confidence_higher_var(self):
        r95 = _allocator(var_confidence=0.95).allocate(VOLS, SHARPES, CORR)
        r99 = _allocator(var_confidence=0.99).allocate(VOLS, SHARPES, CORR)
        var95 = sum(s.allocated_var for s in r95.sleeves)
        var99 = sum(s.allocated_var for s in r99.sleeves)
        assert var99 > var95

    def test_longer_horizon_higher_var(self):
        r1 = _allocator(var_horizon_days=1).allocate(VOLS, SHARPES, CORR)
        r10 = _allocator(var_horizon_days=10).allocate(VOLS, SHARPES, CORR)
        var1 = sum(s.allocated_var for s in r1.sleeves)
        var10 = sum(s.allocated_var for s in r10.sleeves)
        assert var10 > var1


# ---------------------------------------------------------------------------
# Marginal risk
# ---------------------------------------------------------------------------


class TestMarginalRisk:
    def test_marginal_risk_sums_near_portfolio_vol(self):
        """Sum of marginal contributions should approximately equal portfolio vol."""
        result = _allocator().allocate(VOLS, SHARPES, CORR)
        total_marginal = sum(s.marginal_risk for s in result.sleeves)
        assert total_marginal == pytest.approx(
            result.expected_portfolio_vol, rel=0.01,
        )

    def test_marginal_risk_non_negative(self):
        result = _allocator().allocate(VOLS, SHARPES, CORR)
        for s in result.sleeves:
            assert s.marginal_risk >= -1e-6


# ---------------------------------------------------------------------------
# Correlation effects
# ---------------------------------------------------------------------------


class TestCorrelation:
    def test_diversification_reduces_vol(self):
        """With negative correlations, portfolio vol should be below sum of weighted vols."""
        result = _allocator().allocate(VOLS, SHARPES, CORR)
        # Weighted sum of vols (undiversified)
        undiversified = sum(
            s.capital_weight * s.sleeve_vol for s in result.sleeves
        )
        assert result.expected_portfolio_vol < undiversified + 1e-6

    def test_no_correlation_matrix_defaults_to_zero(self):
        """Without correlation, off-diagonal = 0."""
        result = _allocator().allocate(VOLS, SHARPES)
        assert result.expected_portfolio_vol > 0

    def test_higher_correlation_higher_vol(self):
        high_corr = {
            ("momentum", "mean_rev"): 0.80,
            ("momentum", "stat_arb"): 0.80,
            ("mean_rev", "stat_arb"): 0.80,
        }
        low_result = _allocator().allocate(VOLS, SHARPES, CORR)
        high_result = _allocator().allocate(VOLS, SHARPES, high_corr)
        assert high_result.expected_portfolio_vol > low_result.expected_portfolio_vol


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_sleeve(self):
        result = _allocator().allocate(
            {"alpha": 0.15}, {"alpha": 1.5},
        )
        assert result.n_sleeves == 1
        assert result.sleeves[0].risk_share == pytest.approx(1.0)

    def test_zero_sleeve_raises(self):
        with pytest.raises(ValueError, match="at least 1"):
            _allocator().allocate({}, {})

    def test_negative_sharpe_clipped(self):
        """Negative Sharpe should be treated as 0 for allocation."""
        neg_sharpes = {"a": -0.5, "b": 1.0}
        result = _allocator(
            method=AllocationMethod.SHARPE_WEIGHTED,
        ).allocate(
            {"a": 0.15, "b": 0.10}, neg_sharpes,
        )
        b = next(s for s in result.sleeves if s.name == "b")
        a = next(s for s in result.sleeves if s.name == "a")
        # b has positive Sharpe, should get more budget
        assert b.risk_share > a.risk_share

    def test_two_sleeves(self):
        result = _allocator().allocate(
            {"a": 0.10, "b": 0.20},
            {"a": 1.0, "b": 1.0},
        )
        assert result.n_sleeves == 2
        total = sum(s.risk_share for s in result.sleeves)
        assert total == pytest.approx(1.0)

    def test_default_sharpes(self):
        """When no sharpes provided, default to 1.0."""
        result = _allocator(
            method=AllocationMethod.SHARPE_WEIGHTED,
        ).allocate(VOLS)
        for s in result.sleeves:
            assert s.risk_share == pytest.approx(1.0 / 3, abs=0.05)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_readable(self):
        result = _allocator().allocate(VOLS, SHARPES, CORR)
        summary = result.summary()
        assert "Risk Budget Allocation" in summary
        assert "Target portfolio vol" in summary
        assert "Expected portfolio Sharpe" in summary
        assert "momentum" in summary
        assert "stat_arb" in summary
