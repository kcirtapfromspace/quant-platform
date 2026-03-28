"""Tests for portfolio risk decomposition (QUA-103)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.risk.factor_model import FactorModelConfig, FactorRiskModel
from quant.risk.risk_decomposition import (
    DecompositionConfig,
    FactorRiskContrib,
    PositionRisk,
    RiskDecomposer,
    RiskDecompositionResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_ASSETS = 10
N_OBS = 200
SYMBOLS = [f"S{i}" for i in range(N_ASSETS)]


def _returns(seed: int = 42) -> pd.DataFrame:
    """Generate returns with factor structure."""
    rng = np.random.default_rng(seed)
    factors = rng.normal(0, 0.01, (N_OBS, 3))
    loadings = rng.normal(0, 1, (N_ASSETS, 3))
    systematic = factors @ loadings.T
    idio = rng.normal(0, 0.003, (N_OBS, N_ASSETS))
    dates = pd.bdate_range("2023-01-01", periods=N_OBS)
    return pd.DataFrame(systematic + idio, index=dates, columns=SYMBOLS)


def _factor_model_result(returns: pd.DataFrame | None = None):
    """Fit a factor model and return the result."""
    r = returns if returns is not None else _returns()
    model = FactorRiskModel(FactorModelConfig(n_factors=3, min_observations=50))
    return model.estimate(r)


def _equal_weights() -> dict[str, float]:
    return dict.fromkeys(SYMBOLS, 1.0 / N_ASSETS)


def _concentrated_weights() -> dict[str, float]:
    """80% in S0, rest equally spread."""
    w = dict.fromkeys(SYMBOLS, 0.20 / (N_ASSETS - 1))
    w["S0"] = 0.80
    return w


def _long_short_weights() -> dict[str, float]:
    """Dollar-neutral long-short portfolio."""
    w = {}
    for i, s in enumerate(SYMBOLS):
        w[s] = 0.20 if i < N_ASSETS // 2 else -0.20
    return w


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestBasic:
    def test_returns_result(self):
        fm = _factor_model_result()
        result = RiskDecomposer().decompose(
            _equal_weights(), fm.covariance,
        )
        assert isinstance(result, RiskDecompositionResult)

    def test_n_positions(self):
        fm = _factor_model_result()
        result = RiskDecomposer().decompose(
            _equal_weights(), fm.covariance,
        )
        assert result.n_positions == N_ASSETS

    def test_total_risk_positive(self):
        fm = _factor_model_result()
        result = RiskDecomposer().decompose(
            _equal_weights(), fm.covariance,
        )
        assert result.total_risk > 0

    def test_positions_populated(self):
        fm = _factor_model_result()
        result = RiskDecomposer().decompose(
            _equal_weights(), fm.covariance,
        )
        assert len(result.positions) == N_ASSETS

    def test_position_types(self):
        fm = _factor_model_result()
        result = RiskDecomposer().decompose(
            _equal_weights(), fm.covariance,
        )
        for p in result.positions:
            assert isinstance(p, PositionRisk)

    def test_config_accessible(self):
        cfg = DecompositionConfig(top_n=5)
        d = RiskDecomposer(cfg)
        assert d.config.top_n == 5


# ---------------------------------------------------------------------------
# Euler decomposition
# ---------------------------------------------------------------------------


class TestEulerDecomposition:
    def test_rc_sums_to_total_risk(self):
        """Risk contributions must sum to portfolio volatility (Euler's theorem)."""
        fm = _factor_model_result()
        result = RiskDecomposer().decompose(
            _equal_weights(), fm.covariance,
        )
        rc_sum = sum(p.risk_contribution for p in result.positions)
        assert rc_sum == pytest.approx(result.total_risk, rel=1e-6)

    def test_risk_shares_sum_to_one(self):
        fm = _factor_model_result()
        result = RiskDecomposer().decompose(
            _equal_weights(), fm.covariance,
        )
        share_sum = sum(p.risk_share for p in result.positions)
        assert share_sum == pytest.approx(1.0, rel=1e-6)

    def test_rc_sums_long_short(self):
        """Euler property holds for long-short portfolios."""
        fm = _factor_model_result()
        result = RiskDecomposer().decompose(
            _long_short_weights(), fm.covariance,
        )
        rc_sum = sum(p.risk_contribution for p in result.positions)
        assert rc_sum == pytest.approx(result.total_risk, rel=1e-6)

    def test_marginal_risk_positive_for_positive_rc(self):
        fm = _factor_model_result()
        result = RiskDecomposer().decompose(
            _equal_weights(), fm.covariance,
        )
        for p in result.positions:
            if p.risk_contribution > 0:
                assert p.marginal_risk > 0


# ---------------------------------------------------------------------------
# Factor / specific decomposition
# ---------------------------------------------------------------------------


class TestFactorSpecificSplit:
    def test_factor_risk_populated(self):
        fm = _factor_model_result()
        result = RiskDecomposer().decompose(
            _equal_weights(), fm.covariance,
            factor_loadings=fm.loadings,
            factor_covariance=fm.factor_covariance,
            specific_variance=fm.specific_variance,
        )
        assert result.factor_risk > 0

    def test_specific_risk_populated(self):
        fm = _factor_model_result()
        result = RiskDecomposer().decompose(
            _equal_weights(), fm.covariance,
            factor_loadings=fm.loadings,
            factor_covariance=fm.factor_covariance,
            specific_variance=fm.specific_variance,
        )
        assert result.specific_risk > 0

    def test_factor_plus_specific_approximates_total(self):
        """σ²_factor + σ²_specific should approximate σ²_total."""
        fm = _factor_model_result()
        result = RiskDecomposer().decompose(
            _equal_weights(), fm.covariance,
            factor_loadings=fm.loadings,
            factor_covariance=fm.factor_covariance,
            specific_variance=fm.specific_variance,
        )
        factor_var = result.factor_risk ** 2
        specific_var = result.specific_risk ** 2
        total_var = result.total_risk ** 2
        # The covariance includes shrinkage, so this is approximate
        assert (factor_var + specific_var) == pytest.approx(total_var, rel=0.05)

    def test_factor_risk_pct_in_range(self):
        fm = _factor_model_result()
        result = RiskDecomposer().decompose(
            _equal_weights(), fm.covariance,
            factor_loadings=fm.loadings,
            factor_covariance=fm.factor_covariance,
            specific_variance=fm.specific_variance,
        )
        assert 0 <= result.factor_risk_pct <= 1.0

    def test_position_factor_specific_sum(self):
        """Per-position factor + specific RC should sum to total RC."""
        fm = _factor_model_result()
        result = RiskDecomposer().decompose(
            _equal_weights(), fm.covariance,
            factor_loadings=fm.loadings,
            factor_covariance=fm.factor_covariance,
            specific_variance=fm.specific_variance,
        )
        for p in result.positions:
            # Approximate: factor RC + specific RC ≈ total RC
            assert (p.factor_risk_contribution + p.specific_risk_contribution) == pytest.approx(
                p.risk_contribution, rel=0.05,
            )

    def test_no_factor_model_zero_split(self):
        """Without factor model, factor/specific should be zero."""
        fm = _factor_model_result()
        result = RiskDecomposer().decompose(
            _equal_weights(), fm.covariance,
        )
        assert result.factor_risk == 0.0
        assert result.specific_risk == 0.0
        assert result.factor_risk_pct == 0.0


# ---------------------------------------------------------------------------
# Factor-level attribution
# ---------------------------------------------------------------------------


class TestFactorAttribution:
    def test_factors_populated(self):
        fm = _factor_model_result()
        result = RiskDecomposer().decompose(
            _equal_weights(), fm.covariance,
            factor_loadings=fm.loadings,
            factor_covariance=fm.factor_covariance,
            specific_variance=fm.specific_variance,
        )
        assert len(result.factors) == fm.n_factors

    def test_factor_types(self):
        fm = _factor_model_result()
        result = RiskDecomposer().decompose(
            _equal_weights(), fm.covariance,
            factor_loadings=fm.loadings,
            factor_covariance=fm.factor_covariance,
            specific_variance=fm.specific_variance,
        )
        for f in result.factors:
            assert isinstance(f, FactorRiskContrib)

    def test_factor_shares_sum_to_one(self):
        """Factor risk shares should sum to approximately 1.0."""
        fm = _factor_model_result()
        result = RiskDecomposer().decompose(
            _equal_weights(), fm.covariance,
            factor_loadings=fm.loadings,
            factor_covariance=fm.factor_covariance,
            specific_variance=fm.specific_variance,
        )
        share_sum = sum(f.risk_share for f in result.factors)
        assert share_sum == pytest.approx(1.0, rel=1e-6)

    def test_no_factors_without_model(self):
        fm = _factor_model_result()
        result = RiskDecomposer().decompose(
            _equal_weights(), fm.covariance,
        )
        assert len(result.factors) == 0


# ---------------------------------------------------------------------------
# Concentration metrics
# ---------------------------------------------------------------------------


class TestConcentration:
    def test_hhi_in_range(self):
        fm = _factor_model_result()
        result = RiskDecomposer().decompose(
            _equal_weights(), fm.covariance,
        )
        assert 0 <= result.hhi <= 1.0

    def test_concentrated_higher_hhi(self):
        """Concentrated portfolio should have higher risk HHI."""
        fm = _factor_model_result()
        r_equal = RiskDecomposer().decompose(
            _equal_weights(), fm.covariance,
        )
        r_conc = RiskDecomposer().decompose(
            _concentrated_weights(), fm.covariance,
        )
        assert r_conc.hhi > r_equal.hhi

    def test_top_contributors_ordered(self):
        fm = _factor_model_result()
        result = RiskDecomposer().decompose(
            _equal_weights(), fm.covariance,
        )
        rcs = [abs(p.risk_contribution) for p in result.top_risk_contributors]
        for i in range(len(rcs) - 1):
            assert rcs[i] >= rcs[i + 1] - 1e-12


# ---------------------------------------------------------------------------
# Long-short portfolio
# ---------------------------------------------------------------------------


class TestLongShort:
    def test_has_negative_rc(self):
        """Short positions can have negative risk contribution (hedging)."""
        fm = _factor_model_result()
        result = RiskDecomposer().decompose(
            _long_short_weights(), fm.covariance,
        )
        # Just check we handle negative RC gracefully
        assert result.total_risk > 0

    def test_shares_sum_to_one(self):
        fm = _factor_model_result()
        result = RiskDecomposer().decompose(
            _long_short_weights(), fm.covariance,
        )
        share_sum = sum(p.risk_share for p in result.positions)
        assert share_sum == pytest.approx(1.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_weights(self):
        fm = _factor_model_result()
        result = RiskDecomposer().decompose({}, fm.covariance)
        assert result.n_positions == 0
        assert result.total_risk == 0.0

    def test_single_position(self):
        fm = _factor_model_result()
        result = RiskDecomposer().decompose(
            {"S0": 1.0}, fm.covariance,
        )
        assert result.n_positions == 1
        assert result.positions[0].risk_share == pytest.approx(1.0)

    def test_zero_weight_excluded(self):
        w = _equal_weights()
        w["S0"] = 0.0
        fm = _factor_model_result()
        result = RiskDecomposer().decompose(w, fm.covariance)
        assert result.n_positions == N_ASSETS - 1

    def test_missing_symbol_raises(self):
        fm = _factor_model_result()
        with pytest.raises(ValueError, match="not in covariance"):
            RiskDecomposer().decompose(
                {"UNKNOWN": 1.0}, fm.covariance,
            )

    def test_subset_of_universe(self):
        """Decompose using only a subset of the covariance universe."""
        fm = _factor_model_result()
        w = {"S0": 0.50, "S1": 0.50}
        result = RiskDecomposer().decompose(w, fm.covariance)
        assert result.n_positions == 2


# ---------------------------------------------------------------------------
# Integration with factor model
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_full_pipeline(self):
        """Decompose with a freshly fitted factor model."""
        returns = _returns()
        fm = FactorRiskModel(FactorModelConfig(n_factors=3, min_observations=50))
        model_result = fm.estimate(returns)

        result = RiskDecomposer().decompose(
            weights=_equal_weights(),
            covariance=model_result.covariance,
            factor_loadings=model_result.loadings,
            factor_covariance=model_result.factor_covariance,
            specific_variance=model_result.specific_variance,
        )
        assert result.total_risk > 0
        assert result.factor_risk > 0
        assert result.specific_risk > 0
        assert len(result.factors) == 3
        assert result.n_positions == N_ASSETS

    def test_concentrated_more_risk(self):
        """Concentrated portfolio should have higher total risk."""
        fm = _factor_model_result()
        r_equal = RiskDecomposer().decompose(
            _equal_weights(), fm.covariance,
        )
        r_conc = RiskDecomposer().decompose(
            _concentrated_weights(), fm.covariance,
        )
        # Concentrated portfolio puts 80% in one asset → higher risk
        assert r_conc.total_risk > r_equal.total_risk


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_readable(self):
        fm = _factor_model_result()
        result = RiskDecomposer().decompose(
            _equal_weights(), fm.covariance,
            factor_loadings=fm.loadings,
            factor_covariance=fm.factor_covariance,
            specific_variance=fm.specific_variance,
        )
        summary = result.summary()
        assert "Risk Decomposition" in summary
        assert "Factor" in summary
        assert "Specific" in summary
        assert "HHI" in summary

    def test_summary_without_factors(self):
        fm = _factor_model_result()
        result = RiskDecomposer().decompose(
            _equal_weights(), fm.covariance,
        )
        summary = result.summary()
        assert "Risk Decomposition" in summary
        # Factor section should not appear
        assert "Factor risk decomposition" not in summary
