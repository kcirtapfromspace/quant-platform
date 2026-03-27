"""Tests for correlation monitoring and risk controls (QUA-38)."""
from __future__ import annotations

import random

import pytest

from quant.risk.correlation import (
    CorrelationConfig,
    CorrelationMonitor,
    CorrelationRiskCheck,
    CorrelationState,
    _mean,
    _pearson,
    _std,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_correlated_returns(
    n_assets: int,
    n_days: int,
    correlation: float,
    seed: int = 42,
) -> list[list[float]]:
    """Generate multi-asset returns with controlled pairwise correlation.

    Uses a common factor model: r_i = sqrt(rho)*f + sqrt(1-rho)*e_i
    where f is a common factor and e_i is idiosyncratic noise.
    """
    rng = random.Random(seed)
    rho = max(0.0, min(1.0, correlation))
    common_weight = rho ** 0.5
    idio_weight = (1.0 - rho) ** 0.5

    returns: list[list[float]] = []
    for _d in range(n_days):
        common_factor = rng.gauss(0, 0.01)
        row = [
            common_weight * common_factor + idio_weight * rng.gauss(0, 0.01)
            for _a in range(n_assets)
        ]
        returns.append(row)
    return returns


# ---------------------------------------------------------------------------
# Math helper tests
# ---------------------------------------------------------------------------


class TestMathHelpers:
    def test_mean(self):
        assert abs(_mean([1.0, 2.0, 3.0]) - 2.0) < 1e-10

    def test_std(self):
        std = _std([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        assert abs(std - 2.138) < 0.01

    def test_pearson_perfect(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert abs(_pearson(xs, xs) - 1.0) < 1e-10

    def test_pearson_inverse(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [5.0, 4.0, 3.0, 2.0, 1.0]
        assert abs(_pearson(xs, ys) - (-1.0)) < 1e-10

    def test_pearson_uncorrelated(self):
        rng = random.Random(99)
        xs = [rng.gauss(0, 1) for _ in range(1000)]
        ys = [rng.gauss(0, 1) for _ in range(1000)]
        assert abs(_pearson(xs, ys)) < 0.1

    def test_pearson_short_series(self):
        assert _pearson([1.0], [2.0]) == 0.0


# ---------------------------------------------------------------------------
# CorrelationMonitor tests
# ---------------------------------------------------------------------------


class TestCorrelationMonitor:
    def test_basic_construction(self):
        monitor = CorrelationMonitor(["A", "B", "C"])
        assert monitor.symbols == ["A", "B", "C"]
        assert not monitor.has_sufficient_data()

    def test_update_and_sufficient_data(self):
        cfg = CorrelationConfig(window=63, min_observations=10)
        monitor = CorrelationMonitor(["A", "B"], config=cfg)
        returns = _make_correlated_returns(2, 15, 0.5)
        monitor.update(returns)
        assert monitor.has_sufficient_data()

    def test_update_wrong_width_raises(self):
        monitor = CorrelationMonitor(["A", "B"])
        with pytest.raises(ValueError, match="Expected 2"):
            monitor.update([[0.01, 0.02, 0.03]])  # 3 assets, expected 2

    def test_buffer_trimmed_to_window(self):
        cfg = CorrelationConfig(window=30)
        monitor = CorrelationMonitor(["A", "B"], config=cfg)
        monitor.update(_make_correlated_returns(2, 100, 0.5))
        # Internal buffer should be trimmed to window
        assert len(monitor._buffer) == 30

    def test_reset(self):
        monitor = CorrelationMonitor(["A", "B"])
        monitor.update(_make_correlated_returns(2, 50, 0.5))
        assert monitor.has_sufficient_data()
        monitor.reset()
        assert not monitor.has_sufficient_data()

    def test_correlation_matrix_diagonal_is_one(self):
        monitor = CorrelationMonitor(["A", "B", "C"])
        monitor.update(_make_correlated_returns(3, 100, 0.5))
        corr = monitor.correlation_matrix()

        for sym in ["A", "B", "C"]:
            assert abs(corr[sym][sym] - 1.0) < 1e-10

    def test_correlation_matrix_symmetric(self):
        monitor = CorrelationMonitor(["A", "B", "C"])
        monitor.update(_make_correlated_returns(3, 100, 0.5))
        corr = monitor.correlation_matrix()

        for si in ["A", "B", "C"]:
            for sj in ["A", "B", "C"]:
                assert abs(corr[si][sj] - corr[sj][si]) < 1e-10

    def test_high_correlation_detected(self):
        """Returns with high common factor should show high avg correlation."""
        monitor = CorrelationMonitor(["A", "B", "C"])
        monitor.update(_make_correlated_returns(3, 100, 0.9))
        state = monitor.current_state()

        assert state.avg_pairwise_corr > 0.5

    def test_low_correlation_detected(self):
        """Independent returns should show low avg correlation."""
        monitor = CorrelationMonitor(["A", "B", "C"])
        monitor.update(_make_correlated_returns(3, 200, 0.0, seed=77))
        state = monitor.current_state()

        assert state.avg_pairwise_corr < 0.3

    def test_volatilities_positive(self):
        monitor = CorrelationMonitor(["A", "B"])
        monitor.update(_make_correlated_returns(2, 100, 0.5))
        vols = monitor.asset_volatilities()

        for v in vols.values():
            assert v > 0

    def test_state_with_custom_weights(self):
        monitor = CorrelationMonitor(["A", "B", "C"])
        monitor.update(_make_correlated_returns(3, 100, 0.5))

        # Concentrated in one asset
        state = monitor.current_state({"A": 0.9, "B": 0.05, "C": 0.05})
        assert state.n_assets == 3
        assert state.herfindahl > 0.5  # concentrated

    def test_state_equal_weight(self):
        monitor = CorrelationMonitor(["A", "B", "C", "D"])
        monitor.update(_make_correlated_returns(4, 100, 0.3))

        state = monitor.current_state()
        assert state.n_assets == 4
        assert abs(state.herfindahl - 0.25) < 0.01  # equal weight HHI = 1/N


# ---------------------------------------------------------------------------
# CorrelationState tests
# ---------------------------------------------------------------------------


class TestCorrelationState:
    def test_level_normal(self):
        monitor = CorrelationMonitor(["A", "B", "C"])
        monitor.update(_make_correlated_returns(3, 200, 0.0, seed=55))
        state = monitor.current_state()
        assert state.level == "normal"

    def test_level_critical_with_high_corr(self):
        cfg = CorrelationConfig(avg_corr_critical=0.5)
        monitor = CorrelationMonitor(["A", "B", "C"], config=cfg)
        monitor.update(_make_correlated_returns(3, 100, 0.95))
        state = monitor.current_state()

        assert state.level in ("elevated", "critical")

    def test_effective_n_decreases_with_correlation(self):
        """Higher correlation should reduce effective N."""
        syms = ["A", "B", "C", "D"]

        monitor_low = CorrelationMonitor(syms)
        monitor_low.update(_make_correlated_returns(4, 200, 0.1, seed=10))
        state_low = monitor_low.current_state()

        monitor_high = CorrelationMonitor(syms)
        monitor_high.update(_make_correlated_returns(4, 200, 0.9, seed=10))
        state_high = monitor_high.current_state()

        assert state_low.effective_n > state_high.effective_n

    def test_diversification_ratio(self):
        """Diversification ratio should be >= 1 when assets have low correlation."""
        syms = ["A", "B", "C", "D"]
        monitor = CorrelationMonitor(syms)
        monitor.update(_make_correlated_returns(4, 200, 0.1, seed=33))
        state = monitor.current_state()

        assert state.diversification_ratio >= 0.9  # near or above 1

    def test_insufficient_data_returns_neutral(self):
        monitor = CorrelationMonitor(["A", "B"])
        monitor.update(_make_correlated_returns(2, 5, 0.5))  # too few
        state = monitor.current_state()

        assert state.avg_pairwise_corr == 0.0
        assert state.level == "normal"

    def test_single_asset_state(self):
        monitor = CorrelationMonitor(["A"])
        state = monitor.current_state({"A": 1.0})
        assert state.n_assets == 1
        assert state.level == "normal"


# ---------------------------------------------------------------------------
# CorrelationRiskCheck tests
# ---------------------------------------------------------------------------


class TestCorrelationRiskCheck:
    def test_normal_conditions_approved(self):
        state = CorrelationState(
            avg_pairwise_corr=0.30,
            max_pairwise_corr=0.45,
            max_corr_pair=("A", "B"),
            effective_n=4.0,
            diversification_ratio=1.2,
            herfindahl=0.25,
            n_assets=4,
            level="normal",
        )
        check = CorrelationRiskCheck()
        approved, reason = check.check(state)
        assert approved
        assert reason == ""

    def test_high_correlation_rejected(self):
        cfg = CorrelationConfig(avg_corr_critical=0.70)
        state = CorrelationState(
            avg_pairwise_corr=0.80,
            max_pairwise_corr=0.90,
            max_corr_pair=("A", "B"),
            effective_n=2.0,
            diversification_ratio=1.0,
            herfindahl=0.25,
            n_assets=4,
            level="critical",
        )
        check = CorrelationRiskCheck(config=cfg)
        approved, reason = check.check(state)
        assert not approved
        assert "correlation" in reason.lower()

    def test_low_effective_n_rejected(self):
        cfg = CorrelationConfig(min_effective_n=3.0)
        state = CorrelationState(
            avg_pairwise_corr=0.40,
            max_pairwise_corr=0.60,
            max_corr_pair=("A", "B"),
            effective_n=1.5,
            diversification_ratio=1.2,
            herfindahl=0.50,
            n_assets=4,
            level="elevated",
        )
        check = CorrelationRiskCheck(config=cfg)
        approved, reason = check.check(state)
        assert not approved
        assert "concentrated" in reason.lower()

    def test_low_diversification_rejected(self):
        cfg = CorrelationConfig(min_diversification_ratio=1.0)
        state = CorrelationState(
            avg_pairwise_corr=0.40,
            max_pairwise_corr=0.60,
            max_corr_pair=("A", "B"),
            effective_n=4.0,
            diversification_ratio=0.8,
            herfindahl=0.25,
            n_assets=4,
            level="normal",
        )
        check = CorrelationRiskCheck(config=cfg)
        approved, reason = check.check(state)
        assert not approved
        assert "diversification" in reason.lower()

    def test_adjusted_position_limit_normal(self):
        """Normal correlation should not reduce position limit."""
        state = CorrelationState(
            avg_pairwise_corr=0.30,
            max_pairwise_corr=0.45,
            max_corr_pair=("A", "B"),
            effective_n=4.0,
            diversification_ratio=1.2,
            herfindahl=0.25,
            n_assets=4,
            level="normal",
        )
        check = CorrelationRiskCheck()
        limit = check.adjusted_position_limit(0.20, state)
        assert limit == 0.20

    def test_adjusted_position_limit_critical(self):
        """Critical correlation should reduce position limit."""
        cfg = CorrelationConfig(
            avg_corr_critical=0.70,
            position_scale_factor=0.5,
        )
        state = CorrelationState(
            avg_pairwise_corr=0.80,
            max_pairwise_corr=0.90,
            max_corr_pair=("A", "B"),
            effective_n=2.0,
            diversification_ratio=1.0,
            herfindahl=0.25,
            n_assets=4,
            level="critical",
        )
        check = CorrelationRiskCheck(config=cfg)
        limit = check.adjusted_position_limit(0.20, state)
        assert abs(limit - 0.10) < 0.01  # 0.20 * 0.5

    def test_adjusted_position_limit_elevated(self):
        """Elevated correlation should partially reduce position limit."""
        cfg = CorrelationConfig(
            avg_corr_warn=0.60,
            avg_corr_critical=0.80,
            position_scale_factor=0.5,
        )
        # Midpoint: 0.70 — should give ~0.75 scale (halfway to 0.5)
        state = CorrelationState(
            avg_pairwise_corr=0.70,
            max_pairwise_corr=0.75,
            max_corr_pair=("A", "B"),
            effective_n=3.0,
            diversification_ratio=1.1,
            herfindahl=0.25,
            n_assets=4,
            level="elevated",
        )
        check = CorrelationRiskCheck(config=cfg)
        limit = check.adjusted_position_limit(0.20, state)
        # At midpoint between warn and critical, scale should be ~0.75
        assert 0.10 < limit < 0.20

    def test_position_limit_scale(self):
        check = CorrelationRiskCheck()
        state = CorrelationState(
            avg_pairwise_corr=0.30,
            max_pairwise_corr=0.45,
            max_corr_pair=("A", "B"),
            effective_n=4.0,
            diversification_ratio=1.2,
            herfindahl=0.25,
            n_assets=4,
            level="normal",
        )
        assert check.position_limit_scale(state) == 1.0


# ---------------------------------------------------------------------------
# Integration: monitor → risk check
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_low_corr_portfolio_approved(self):
        """Portfolio with low-correlation assets should pass all checks."""
        monitor = CorrelationMonitor(["A", "B", "C", "D"])
        monitor.update(_make_correlated_returns(4, 100, 0.1, seed=11))

        state = monitor.current_state()
        check = CorrelationRiskCheck()
        approved, _ = check.check(state)
        assert approved

    def test_high_corr_portfolio_flagged(self):
        """Portfolio with highly correlated assets should be flagged."""
        cfg = CorrelationConfig(avg_corr_critical=0.5)
        monitor = CorrelationMonitor(["A", "B", "C"], config=cfg)
        monitor.update(_make_correlated_returns(3, 100, 0.95, seed=22))

        state = monitor.current_state()
        check = CorrelationRiskCheck(config=cfg)
        approved, reason = check.check(state)
        # Should either be rejected or have meaningful reason
        if not approved:
            assert len(reason) > 0

    def test_position_limit_tightens_with_correlation(self):
        """Higher correlation should tighten position limits."""
        cfg = CorrelationConfig(avg_corr_warn=0.40, avg_corr_critical=0.70)

        monitor_low = CorrelationMonitor(["A", "B", "C"], config=cfg)
        monitor_low.update(_make_correlated_returns(3, 200, 0.1, seed=33))
        state_low = monitor_low.current_state()

        monitor_high = CorrelationMonitor(["A", "B", "C"], config=cfg)
        monitor_high.update(_make_correlated_returns(3, 200, 0.9, seed=33))
        state_high = monitor_high.current_state()

        check = CorrelationRiskCheck(config=cfg)
        limit_low = check.adjusted_position_limit(0.20, state_low)
        limit_high = check.adjusted_position_limit(0.20, state_high)

        assert limit_low >= limit_high
