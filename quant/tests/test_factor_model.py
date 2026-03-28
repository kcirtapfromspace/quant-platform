"""Tests for statistical factor risk model (QUA-96)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.risk.factor_model import (
    FactorInfo,
    FactorModelConfig,
    FactorModelResult,
    FactorRiskModel,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_OBS = 300
N_ASSETS = 10
DATES = pd.bdate_range("2023-01-01", periods=N_OBS)
SYMBOLS = [f"A{i}" for i in range(N_ASSETS)]


def _make_factor_returns(
    n_factors: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Create asset returns driven by a known number of latent factors.

    Returns have a clear factor structure making PCA recovery straightforward.
    """
    rng = np.random.default_rng(seed)

    # Latent factors (N_OBS x n_factors)
    factors = rng.normal(0, 0.01, (N_OBS, n_factors))

    # Random loadings (N_ASSETS x n_factors)
    loadings = rng.normal(0, 1, (N_ASSETS, n_factors))

    # Asset returns = loadings @ factors' + idiosyncratic noise
    systematic = factors @ loadings.T
    idio = rng.normal(0, 0.003, (N_OBS, N_ASSETS))

    returns = systematic + idio
    return pd.DataFrame(returns, index=DATES, columns=SYMBOLS)


def _make_independent_returns(seed: int = 42) -> pd.DataFrame:
    """Returns with no factor structure (all idiosyncratic)."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.01, (N_OBS, N_ASSETS))
    return pd.DataFrame(returns, index=DATES, columns=SYMBOLS)


def _make_single_factor_returns(seed: int = 42) -> pd.DataFrame:
    """All assets driven by a single common factor (high correlation)."""
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0005, 0.01, N_OBS)
    returns = np.column_stack([
        market * (0.8 + 0.4 * rng.random()) + rng.normal(0, 0.002, N_OBS)
        for _ in range(N_ASSETS)
    ])
    return pd.DataFrame(returns, index=DATES, columns=SYMBOLS)


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestBasic:
    def test_returns_result(self):
        result = FactorRiskModel().estimate(_make_factor_returns())
        assert isinstance(result, FactorModelResult)

    def test_n_assets(self):
        result = FactorRiskModel().estimate(_make_factor_returns())
        assert result.n_assets == N_ASSETS

    def test_n_observations(self):
        result = FactorRiskModel().estimate(_make_factor_returns())
        assert result.n_observations == N_OBS

    def test_covariance_shape(self):
        result = FactorRiskModel().estimate(_make_factor_returns())
        assert result.covariance.shape == (N_ASSETS, N_ASSETS)

    def test_covariance_is_dataframe(self):
        result = FactorRiskModel().estimate(_make_factor_returns())
        assert isinstance(result.covariance, pd.DataFrame)

    def test_covariance_symmetric(self):
        result = FactorRiskModel().estimate(_make_factor_returns())
        cov = result.covariance.values
        np.testing.assert_allclose(cov, cov.T, atol=1e-10)

    def test_covariance_positive_diagonal(self):
        result = FactorRiskModel().estimate(_make_factor_returns())
        assert all(result.covariance.values[i, i] > 0 for i in range(N_ASSETS))

    def test_covariance_positive_semidefinite(self):
        result = FactorRiskModel().estimate(_make_factor_returns())
        eigenvalues = np.linalg.eigvalsh(result.covariance.values)
        assert all(ev >= -1e-10 for ev in eigenvalues)

    def test_symbols_preserved(self):
        result = FactorRiskModel().estimate(_make_factor_returns())
        assert list(result.covariance.index) == SYMBOLS
        assert list(result.covariance.columns) == SYMBOLS


# ---------------------------------------------------------------------------
# Factor extraction
# ---------------------------------------------------------------------------


class TestFactorExtraction:
    def test_factors_populated(self):
        result = FactorRiskModel().estimate(_make_factor_returns())
        assert len(result.factors) > 0

    def test_factor_types(self):
        result = FactorRiskModel().estimate(_make_factor_returns())
        for f in result.factors:
            assert isinstance(f, FactorInfo)

    def test_eigenvalues_descending(self):
        result = FactorRiskModel().estimate(_make_factor_returns())
        evs = [f.eigenvalue for f in result.factors]
        for i in range(len(evs) - 1):
            assert evs[i] >= evs[i + 1] - 1e-10

    def test_cumulative_variance_monotonic(self):
        result = FactorRiskModel().estimate(_make_factor_returns())
        cum = [f.cumulative_variance_ratio for f in result.factors]
        for i in range(len(cum) - 1):
            assert cum[i + 1] >= cum[i] - 1e-10

    def test_variance_explained_positive(self):
        result = FactorRiskModel().estimate(_make_factor_returns())
        assert result.total_variance_explained > 0.0

    def test_factor_structure_detected(self):
        """Data with 3 latent factors should have high explained variance."""
        cfg = FactorModelConfig(n_factors=3)
        result = FactorRiskModel(cfg).estimate(_make_factor_returns(n_factors=3))
        assert result.total_variance_explained > 0.50

    def test_more_factors_more_variance(self):
        r = _make_factor_returns(n_factors=3)
        r3 = FactorRiskModel(FactorModelConfig(n_factors=3)).estimate(r)
        r5 = FactorRiskModel(FactorModelConfig(n_factors=5)).estimate(r)
        assert r5.total_variance_explained >= r3.total_variance_explained - 1e-10


# ---------------------------------------------------------------------------
# Auto factor selection
# ---------------------------------------------------------------------------


class TestAutoSelection:
    def test_auto_selects_factors(self):
        """With no n_factors, model should auto-select."""
        cfg = FactorModelConfig(n_factors=None, variance_threshold=0.80)
        result = FactorRiskModel(cfg).estimate(_make_factor_returns(n_factors=3))
        assert result.n_factors >= 1

    def test_higher_threshold_more_factors(self):
        r = _make_factor_returns(n_factors=5)
        low = FactorRiskModel(FactorModelConfig(
            n_factors=None, variance_threshold=0.50,
        )).estimate(r)
        high = FactorRiskModel(FactorModelConfig(
            n_factors=None, variance_threshold=0.95,
        )).estimate(r)
        assert high.n_factors >= low.n_factors

    def test_max_factors_respected(self):
        cfg = FactorModelConfig(n_factors=None, variance_threshold=0.99, max_factors=2)
        result = FactorRiskModel(cfg).estimate(_make_factor_returns())
        assert result.n_factors <= 2

    def test_explicit_n_factors_used(self):
        cfg = FactorModelConfig(n_factors=4)
        result = FactorRiskModel(cfg).estimate(_make_factor_returns())
        assert result.n_factors == 4


# ---------------------------------------------------------------------------
# Loadings
# ---------------------------------------------------------------------------


class TestLoadings:
    def test_loadings_shape(self):
        cfg = FactorModelConfig(n_factors=3)
        result = FactorRiskModel(cfg).estimate(_make_factor_returns())
        assert result.loadings.shape == (N_ASSETS, 3)

    def test_loadings_is_dataframe(self):
        result = FactorRiskModel().estimate(_make_factor_returns())
        assert isinstance(result.loadings, pd.DataFrame)

    def test_loadings_index_matches(self):
        result = FactorRiskModel().estimate(_make_factor_returns())
        assert list(result.loadings.index) == SYMBOLS


# ---------------------------------------------------------------------------
# Specific (idiosyncratic) variance
# ---------------------------------------------------------------------------


class TestSpecificVariance:
    def test_specific_variance_positive(self):
        result = FactorRiskModel().estimate(_make_factor_returns())
        assert (result.specific_variance >= 0).all()

    def test_specific_variance_series(self):
        result = FactorRiskModel().estimate(_make_factor_returns())
        assert isinstance(result.specific_variance, pd.Series)
        assert list(result.specific_variance.index) == SYMBOLS

    def test_factor_driven_low_specific(self):
        """Assets driven by factors should have relatively low idiosyncratic risk."""
        cfg = FactorModelConfig(n_factors=3)
        result = FactorRiskModel(cfg).estimate(_make_factor_returns(n_factors=3))
        # Systematic risk should dominate for factor-driven data
        avg_sys = result.systematic_risk_pct.mean()
        assert avg_sys > 0.30


# ---------------------------------------------------------------------------
# Systematic risk decomposition
# ---------------------------------------------------------------------------


class TestSystematicRisk:
    def test_systematic_pct_in_range(self):
        result = FactorRiskModel().estimate(_make_factor_returns())
        assert (result.systematic_risk_pct >= 0).all()
        assert (result.systematic_risk_pct <= 1.0 + 1e-10).all()

    def test_single_factor_high_systematic(self):
        """Single-factor data should show high systematic risk."""
        cfg = FactorModelConfig(n_factors=1)
        result = FactorRiskModel(cfg).estimate(_make_single_factor_returns())
        avg_sys = result.systematic_risk_pct.mean()
        assert avg_sys > 0.30

    def test_independent_low_systematic(self):
        """Independent returns should have low systematic risk per factor."""
        cfg = FactorModelConfig(n_factors=1)
        result = FactorRiskModel(cfg).estimate(_make_independent_returns())
        avg_sys = result.systematic_risk_pct.mean()
        assert avg_sys < 0.50


# ---------------------------------------------------------------------------
# Shrinkage
# ---------------------------------------------------------------------------


class TestShrinkage:
    def test_shrinkage_in_range(self):
        result = FactorRiskModel().estimate(_make_factor_returns())
        assert 0.0 <= result.shrinkage_intensity <= 1.0

    def test_explicit_shrinkage(self):
        cfg = FactorModelConfig(shrinkage_intensity=0.5)
        result = FactorRiskModel(cfg).estimate(_make_factor_returns())
        assert result.shrinkage_intensity == 0.5

    def test_zero_shrinkage_closer_to_sample(self):
        """With shrinkage=0, result should be the pure sample covariance."""
        r = _make_factor_returns()
        cfg = FactorModelConfig(shrinkage_intensity=0.0, n_factors=3, annualise=False)
        result = FactorRiskModel(cfg).estimate(r)
        sample_cov = r.cov().values
        np.testing.assert_allclose(
            result.covariance.values, sample_cov, atol=1e-10,
        )

    def test_full_shrinkage_uses_factor_model(self):
        """With shrinkage=1, result should equal the factor model covariance."""
        r = _make_factor_returns()
        cfg0 = FactorModelConfig(shrinkage_intensity=0.0, n_factors=3, annualise=False)
        cfg1 = FactorModelConfig(shrinkage_intensity=1.0, n_factors=3, annualise=False)
        r0 = FactorRiskModel(cfg0).estimate(r)
        r1 = FactorRiskModel(cfg1).estimate(r)
        # They should differ (unless sample == factor model, which is unlikely)
        diff = np.abs(r0.covariance.values - r1.covariance.values).max()
        assert diff > 1e-6  # Not identical


# ---------------------------------------------------------------------------
# Annualisation
# ---------------------------------------------------------------------------


class TestAnnualisation:
    def test_annualised_larger(self):
        r = _make_factor_returns()
        ann = FactorRiskModel(FactorModelConfig(annualise=True, n_factors=3)).estimate(r)
        raw = FactorRiskModel(FactorModelConfig(annualise=False, n_factors=3)).estimate(r)
        # Annualised covariance should be ~252x larger
        ratio = ann.covariance.values[0, 0] / raw.covariance.values[0, 0]
        assert 200 < ratio < 300

    def test_not_annualised(self):
        r = _make_factor_returns()
        cfg = FactorModelConfig(annualise=False, n_factors=3, shrinkage_intensity=0.0)
        result = FactorRiskModel(cfg).estimate(r)
        # Should be close to sample daily covariance
        sample_var = float(r.iloc[:, 0].var())
        model_var = float(result.covariance.values[0, 0])
        assert abs(model_var - sample_var) / sample_var < 0.01


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_too_few_observations_raises(self):
        dates = pd.bdate_range("2024-01-01", periods=10)
        r = pd.DataFrame(
            np.random.default_rng(42).normal(0, 0.01, (10, 5)),
            index=dates, columns=[f"X{i}" for i in range(5)],
        )
        with pytest.raises(ValueError, match="at least"):
            FactorRiskModel(FactorModelConfig(min_observations=60)).estimate(r)

    def test_single_asset_raises(self):
        r = pd.DataFrame(
            np.random.default_rng(42).normal(0, 0.01, (100, 1)),
            index=pd.bdate_range("2024-01-01", periods=100),
            columns=["X0"],
        )
        with pytest.raises(ValueError, match="at least 2"):
            FactorRiskModel().estimate(r)

    def test_two_assets(self):
        rng = np.random.default_rng(42)
        r = pd.DataFrame(
            rng.normal(0, 0.01, (200, 2)),
            index=pd.bdate_range("2024-01-01", periods=200),
            columns=["X0", "X1"],
        )
        result = FactorRiskModel(FactorModelConfig(n_factors=1)).estimate(r)
        assert result.n_assets == 2
        assert result.n_factors == 1

    def test_n_factors_capped_at_n_assets_minus_one(self):
        rng = np.random.default_rng(42)
        r = pd.DataFrame(
            rng.normal(0, 0.01, (200, 3)),
            index=pd.bdate_range("2024-01-01", periods=200),
            columns=["X0", "X1", "X2"],
        )
        cfg = FactorModelConfig(n_factors=10)  # More than N-1=2
        result = FactorRiskModel(cfg).estimate(r)
        assert result.n_factors <= 2

    def test_many_assets(self):
        rng = np.random.default_rng(42)
        n_a = 50
        r = pd.DataFrame(
            rng.normal(0, 0.01, (200, n_a)),
            index=pd.bdate_range("2024-01-01", periods=200),
            columns=[f"X{i}" for i in range(n_a)],
        )
        cfg = FactorModelConfig(n_factors=5)
        result = FactorRiskModel(cfg).estimate(r)
        assert result.covariance.shape == (n_a, n_a)
        assert result.n_factors == 5

    def test_nan_handling(self):
        r = _make_factor_returns()
        # Sprinkle some NaNs
        r.iloc[0, 0] = np.nan
        r.iloc[5, 3] = np.nan
        result = FactorRiskModel().estimate(r)
        assert result.n_assets == N_ASSETS


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_readable(self):
        result = FactorRiskModel().estimate(_make_factor_returns())
        summary = result.summary()
        assert "Factor Risk Model" in summary
        assert "variance explained" in summary.lower()
        assert "Shrinkage" in summary
        assert "Eigenvalue" in summary

    def test_summary_shows_systematic(self):
        result = FactorRiskModel().estimate(_make_factor_returns())
        summary = result.summary()
        assert "Systematic" in summary or "systematic" in summary.lower()
