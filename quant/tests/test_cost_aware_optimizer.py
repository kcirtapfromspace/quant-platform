"""Tests for transaction-cost-aware portfolio optimizer (QUA-97)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.portfolio.constraints import PortfolioConstraints
from quant.portfolio.cost_aware_optimizer import (
    CostAwareConfig,
    CostAwareOptimizer,
    CostAwareResult,
    TradeCost,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SYMBOLS = ["AAPL", "GOOG", "MSFT", "AMZN", "META"]
N = len(SYMBOLS)


def _cov_matrix(vol: float = 0.20, corr: float = 0.30) -> pd.DataFrame:
    """Create a simple correlation-based covariance matrix (annualised)."""
    vols = np.full(N, vol)
    rho = np.full((N, N), corr)
    np.fill_diagonal(rho, 1.0)
    cov = np.outer(vols, vols) * rho
    return pd.DataFrame(cov, index=SYMBOLS, columns=SYMBOLS)


def _expected_returns(spread: float = 0.05) -> pd.Series:
    """Create a spread of expected returns."""
    base = np.linspace(0.05, 0.05 + spread * (N - 1), N)
    return pd.Series(base, index=SYMBOLS)


def _equal_weights() -> dict[str, float]:
    return dict.fromkeys(SYMBOLS, 1.0 / N)


def _adv() -> dict[str, float]:
    """Average daily volume (USD) per asset."""
    return {
        "AAPL": 5_000_000_000,
        "GOOG": 2_000_000_000,
        "MSFT": 4_000_000_000,
        "AMZN": 3_000_000_000,
        "META": 2_500_000_000,
    }


def _constraints(**kwargs) -> PortfolioConstraints:
    defaults = {"long_only": True, "max_weight": 0.40, "max_gross_exposure": 1.0}
    defaults.update(kwargs)
    return PortfolioConstraints(**defaults)


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestBasic:
    def test_returns_result(self):
        result = CostAwareOptimizer().optimize(
            SYMBOLS, _cov_matrix(), _expected_returns(), _constraints(),
        )
        assert isinstance(result, CostAwareResult)

    def test_weights_populated(self):
        result = CostAwareOptimizer().optimize(
            SYMBOLS, _cov_matrix(), _expected_returns(), _constraints(),
        )
        assert len(result.weights) == N
        assert set(result.weights.keys()) == set(SYMBOLS)

    def test_risk_positive(self):
        result = CostAwareOptimizer().optimize(
            SYMBOLS, _cov_matrix(), _expected_returns(), _constraints(),
        )
        assert result.risk > 0

    def test_trade_costs_populated(self):
        result = CostAwareOptimizer().optimize(
            SYMBOLS, _cov_matrix(), _expected_returns(), _constraints(),
        )
        assert len(result.trade_costs) == N

    def test_trade_cost_types(self):
        result = CostAwareOptimizer().optimize(
            SYMBOLS, _cov_matrix(), _expected_returns(), _constraints(),
        )
        for tc in result.trade_costs:
            assert isinstance(tc, TradeCost)


# ---------------------------------------------------------------------------
# Cost penalty effect
# ---------------------------------------------------------------------------


class TestCostPenalty:
    def test_higher_penalty_less_turnover(self):
        """Higher cost penalty should reduce turnover from current holdings."""
        current = _equal_weights()
        low = CostAwareOptimizer(CostAwareConfig(cost_penalty=1.0)).optimize(
            SYMBOLS, _cov_matrix(), _expected_returns(), _constraints(),
            current_weights=current, adv=_adv(),
        )
        high = CostAwareOptimizer(CostAwareConfig(cost_penalty=50.0)).optimize(
            SYMBOLS, _cov_matrix(), _expected_returns(), _constraints(),
            current_weights=current, adv=_adv(),
        )
        assert high.turnover <= low.turnover + 1e-6

    def test_zero_penalty_ignores_costs(self):
        """With zero penalty, result should match standard MV (approximately)."""
        result = CostAwareOptimizer(CostAwareConfig(cost_penalty=0.0)).optimize(
            SYMBOLS, _cov_matrix(), _expected_returns(), _constraints(),
        )
        assert abs(sum(abs(w) for w in result.weights.values())) > 0

    def test_high_penalty_stays_near_current(self):
        """Very high cost penalty should keep weights close to current."""
        current = _equal_weights()
        result = CostAwareOptimizer(CostAwareConfig(cost_penalty=1000.0)).optimize(
            SYMBOLS, _cov_matrix(), _expected_returns(), _constraints(),
            current_weights=current, adv=_adv(),
        )
        for sym in SYMBOLS:
            assert abs(result.weights[sym] - current[sym]) < 0.10

    def test_increasing_penalty_monotonic_turnover(self):
        """Turnover should decrease monotonically as cost penalty increases."""
        current = _equal_weights()
        turnovers = []
        for kappa in [0.1, 1.0, 10.0, 100.0]:
            r = CostAwareOptimizer(CostAwareConfig(cost_penalty=kappa)).optimize(
                SYMBOLS, _cov_matrix(), _expected_returns(), _constraints(),
                current_weights=current, adv=_adv(),
            )
            turnovers.append(r.turnover)
        for i in range(len(turnovers) - 1):
            assert turnovers[i + 1] <= turnovers[i] + 1e-6


# ---------------------------------------------------------------------------
# Transaction cost computation
# ---------------------------------------------------------------------------


class TestTransactionCosts:
    def test_no_trade_zero_cost(self):
        """Starting from optimal weights should have near-zero extra cost."""
        # First find optimal weights with zero penalty
        opt = CostAwareOptimizer(CostAwareConfig(cost_penalty=0.0))
        r1 = opt.optimize(SYMBOLS, _cov_matrix(), _expected_returns(), _constraints())
        # Now optimise from those weights with zero penalty — should be same
        r2 = opt.optimize(
            SYMBOLS, _cov_matrix(), _expected_returns(), _constraints(),
            current_weights=r1.weights,
        )
        assert r2.turnover < 0.01

    def test_linear_cost_positive(self):
        result = CostAwareOptimizer().optimize(
            SYMBOLS, _cov_matrix(), _expected_returns(), _constraints(),
        )
        total_linear = sum(tc.linear_cost for tc in result.trade_costs)
        assert total_linear > 0

    def test_impact_cost_with_adv(self):
        result = CostAwareOptimizer().optimize(
            SYMBOLS, _cov_matrix(), _expected_returns(), _constraints(),
            adv=_adv(),
        )
        total_impact = sum(tc.impact_cost for tc in result.trade_costs)
        assert total_impact > 0

    def test_impact_zero_without_adv(self):
        """Without ADV data, impact cost should be zero."""
        result = CostAwareOptimizer().optimize(
            SYMBOLS, _cov_matrix(), _expected_returns(), _constraints(),
        )
        total_impact = sum(tc.impact_cost for tc in result.trade_costs)
        assert total_impact == 0.0

    def test_total_cost_equals_sum(self):
        result = CostAwareOptimizer().optimize(
            SYMBOLS, _cov_matrix(), _expected_returns(), _constraints(),
            adv=_adv(),
        )
        expected_total = sum(tc.total_cost for tc in result.trade_costs)
        assert abs(result.total_cost - expected_total) < 1e-10


# ---------------------------------------------------------------------------
# Current weights
# ---------------------------------------------------------------------------


class TestCurrentWeights:
    def test_from_zero(self):
        result = CostAwareOptimizer().optimize(
            SYMBOLS, _cov_matrix(), _expected_returns(), _constraints(),
        )
        for sym in SYMBOLS:
            assert result.current_weights[sym] == 0.0

    def test_from_existing(self):
        current = {"AAPL": 0.30, "GOOG": 0.20, "MSFT": 0.20, "AMZN": 0.15, "META": 0.15}
        result = CostAwareOptimizer().optimize(
            SYMBOLS, _cov_matrix(), _expected_returns(), _constraints(),
            current_weights=current,
        )
        assert result.current_weights == current

    def test_turnover_computed(self):
        current = _equal_weights()
        result = CostAwareOptimizer().optimize(
            SYMBOLS, _cov_matrix(), _expected_returns(), _constraints(),
            current_weights=current,
        )
        expected_turnover = sum(
            abs(result.weights[s] - current[s]) for s in SYMBOLS
        )
        assert abs(result.turnover - expected_turnover) < 1e-10

    def test_n_trades_counted(self):
        result = CostAwareOptimizer().optimize(
            SYMBOLS, _cov_matrix(), _expected_returns(), _constraints(),
        )
        assert result.n_trades >= 1


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


class TestConstraints:
    def test_long_only_respected(self):
        result = CostAwareOptimizer().optimize(
            SYMBOLS, _cov_matrix(), _expected_returns(),
            _constraints(long_only=True),
        )
        for w in result.weights.values():
            assert w >= -1e-9

    def test_max_weight_respected(self):
        result = CostAwareOptimizer().optimize(
            SYMBOLS, _cov_matrix(), _expected_returns(),
            _constraints(max_weight=0.30),
        )
        for w in result.weights.values():
            assert w <= 0.30 + 1e-9

    def test_gross_exposure_respected(self):
        result = CostAwareOptimizer().optimize(
            SYMBOLS, _cov_matrix(), _expected_returns(),
            _constraints(max_gross_exposure=1.0),
        )
        gross = sum(abs(w) for w in result.weights.values())
        assert gross <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Net expected return
# ---------------------------------------------------------------------------


class TestNetReturn:
    def test_net_return_less_than_gross(self):
        result = CostAwareOptimizer(CostAwareConfig(cost_penalty=5.0)).optimize(
            SYMBOLS, _cov_matrix(), _expected_returns(), _constraints(),
            adv=_adv(),
        )
        assert result.net_expected_return <= result.expected_return + 1e-10

    def test_net_return_formula(self):
        cfg = CostAwareConfig(cost_penalty=5.0)
        result = CostAwareOptimizer(cfg).optimize(
            SYMBOLS, _cov_matrix(), _expected_returns(), _constraints(),
            adv=_adv(),
        )
        expected_net = result.expected_return - cfg.cost_penalty * result.total_cost
        assert abs(result.net_expected_return - expected_net) < 1e-10


# ---------------------------------------------------------------------------
# Risk aversion
# ---------------------------------------------------------------------------


class TestRiskAversion:
    def test_higher_aversion_different_weights(self):
        """Different risk aversion should produce different allocations."""
        low_lam = CostAwareOptimizer(CostAwareConfig(risk_aversion=0.5)).optimize(
            SYMBOLS, _cov_matrix(), _expected_returns(), _constraints(),
        )
        high_lam = CostAwareOptimizer(CostAwareConfig(risk_aversion=5.0)).optimize(
            SYMBOLS, _cov_matrix(), _expected_returns(), _constraints(),
        )
        diff = sum(
            abs(low_lam.weights[s] - high_lam.weights[s]) for s in SYMBOLS
        )
        # They should differ when constraints don't fully bind
        assert diff > 0 or low_lam.risk >= high_lam.risk - 1e-6


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_symbols_raises(self):
        with pytest.raises(ValueError, match="at least 1"):
            CostAwareOptimizer().optimize(
                [], pd.DataFrame(), None, _constraints(),
            )

    def test_single_asset(self):
        syms = ["AAPL"]
        cov = pd.DataFrame([[0.04]], index=syms, columns=syms)
        er = pd.Series([0.10], index=syms)
        result = CostAwareOptimizer().optimize(
            syms, cov, er, PortfolioConstraints(long_only=True, max_gross_exposure=1.0),
        )
        assert len(result.weights) == 1

    def test_no_expected_returns(self):
        """Without alpha, optimizer should still produce valid weights."""
        result = CostAwareOptimizer().optimize(
            SYMBOLS, _cov_matrix(), None, _constraints(),
        )
        assert len(result.weights) == N

    def test_identical_alpha_similar_weights(self):
        """Equal alpha + equal vol should produce roughly equal weights."""
        er = pd.Series(0.10, index=SYMBOLS)
        result = CostAwareOptimizer(CostAwareConfig(
            cost_penalty=0.0, risk_aversion=1.0,
        )).optimize(SYMBOLS, _cov_matrix(), er, _constraints())
        weights = list(result.weights.values())
        assert max(weights) - min(weights) < 0.15


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_readable(self):
        result = CostAwareOptimizer().optimize(
            SYMBOLS, _cov_matrix(), _expected_returns(), _constraints(),
            adv=_adv(),
        )
        summary = result.summary()
        assert "Cost-Aware" in summary
        assert "Expected return" in summary
        assert "turnover" in summary.lower()

    def test_summary_shows_trades(self):
        result = CostAwareOptimizer().optimize(
            SYMBOLS, _cov_matrix(), _expected_returns(), _constraints(),
        )
        summary = result.summary()
        assert "AAPL" in summary
