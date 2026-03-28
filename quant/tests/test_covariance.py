"""Tests for robust covariance estimation (QUA-107)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.risk.covariance import (
    CovarianceConfig,
    CovarianceEstimator,
    CovarianceResult,
    EstimationMethod,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_ASSETS = 10
N_OBS = 250


def _returns(seed: int = 42) -> pd.DataFrame:
    """Generate correlated returns with a known factor structure."""
    rng = np.random.default_rng(seed)
    n_factors = 3
    loadings = rng.normal(0, 0.3, (N_ASSETS, n_factors))
    factors = rng.normal(0, 0.01, (N_OBS, n_factors))
    specific = rng.normal(0, 0.005, (N_OBS, N_ASSETS))
    r = factors @ loadings.T + specific
    symbols = [f"ASSET_{i}" for i in range(N_ASSETS)]
    dates = pd.bdate_range("2023-01-01", periods=N_OBS)
    return pd.DataFrame(r, index=dates, columns=symbols)


def _high_dim_returns(seed: int = 42) -> pd.DataFrame:
    """Returns with p close to T (stress test for shrinkage)."""
    rng = np.random.default_rng(seed)
    n_obs, n_cols = 50, 40
    x = rng.normal(0, 0.01, (n_obs, n_cols))
    symbols = [f"A{i}" for i in range(n_cols)]
    return pd.DataFrame(x, columns=symbols)


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestBasic:
    def test_returns_result(self):
        result = CovarianceEstimator().estimate(_returns())
        assert isinstance(result, CovarianceResult)

    def test_correct_shape(self):
        result = CovarianceEstimator().estimate(_returns())
        assert result.covariance.shape == (N_ASSETS, N_ASSETS)
        assert result.correlation.shape == (N_ASSETS, N_ASSETS)
        assert len(result.volatilities) == N_ASSETS

    def test_symmetric(self):
        result = CovarianceEstimator().estimate(_returns())
        cov = result.covariance.values
        np.testing.assert_array_almost_equal(cov, cov.T)

    def test_positive_semidefinite(self):
        result = CovarianceEstimator().estimate(_returns())
        eigvals = np.linalg.eigvalsh(result.covariance.values)
        assert eigvals.min() > -1e-10

    def test_n_observations(self):
        result = CovarianceEstimator().estimate(_returns())
        assert result.n_observations == N_OBS

    def test_n_assets(self):
        result = CovarianceEstimator().estimate(_returns())
        assert result.n_assets == N_ASSETS

    def test_too_few_observations_raises(self):
        r = _returns().iloc[:5]
        with pytest.raises(ValueError, match="at least 30"):
            CovarianceEstimator().estimate(r)

    def test_single_asset_raises(self):
        r = _returns()[["ASSET_0"]]
        with pytest.raises(ValueError, match="at least 2"):
            CovarianceEstimator().estimate(r)

    def test_default_method_is_ledoit_wolf(self):
        result = CovarianceEstimator().estimate(_returns())
        assert result.method == EstimationMethod.LEDOIT_WOLF


# ---------------------------------------------------------------------------
# Sample covariance
# ---------------------------------------------------------------------------


class TestSampleCovariance:
    def test_matches_numpy(self):
        r = _returns()
        cfg = CovarianceConfig(method=EstimationMethod.SAMPLE, annualise=False)
        result = CovarianceEstimator(cfg).estimate(r)
        expected = np.cov(r.values, rowvar=False, ddof=1)
        np.testing.assert_array_almost_equal(result.covariance.values, expected)

    def test_no_shrinkage(self):
        cfg = CovarianceConfig(method=EstimationMethod.SAMPLE)
        result = CovarianceEstimator(cfg).estimate(_returns())
        assert result.shrinkage_intensity == 0.0

    def test_diagonal_positive(self):
        cfg = CovarianceConfig(method=EstimationMethod.SAMPLE, annualise=False)
        result = CovarianceEstimator(cfg).estimate(_returns())
        assert all(result.covariance.values.diagonal() > 0)


# ---------------------------------------------------------------------------
# Ledoit-Wolf
# ---------------------------------------------------------------------------


class TestLedoitWolf:
    def test_positive_shrinkage(self):
        cfg = CovarianceConfig(method=EstimationMethod.LEDOIT_WOLF)
        result = CovarianceEstimator(cfg).estimate(_returns())
        assert result.shrinkage_intensity > 0

    def test_shrinkage_bounded(self):
        cfg = CovarianceConfig(method=EstimationMethod.LEDOIT_WOLF)
        result = CovarianceEstimator(cfg).estimate(_returns())
        assert 0 <= result.shrinkage_intensity <= 1

    def test_better_conditioned_than_sample(self):
        r = _returns()
        lw = CovarianceEstimator(
            CovarianceConfig(method=EstimationMethod.LEDOIT_WOLF, annualise=False),
        ).estimate(r)
        sample = CovarianceEstimator(
            CovarianceConfig(method=EstimationMethod.SAMPLE, annualise=False),
        ).estimate(r)
        assert lw.condition_number <= sample.condition_number

    def test_positive_definite(self):
        cfg = CovarianceConfig(method=EstimationMethod.LEDOIT_WOLF, annualise=False)
        result = CovarianceEstimator(cfg).estimate(_returns())
        eigvals = np.linalg.eigvalsh(result.covariance.values)
        assert eigvals.min() > 0

    def test_converges_to_sample_with_many_obs(self):
        """With T >> p and non-trivial correlation, shrinkage approaches 0."""
        rng = np.random.default_rng(42)
        n_obs, n_cols = 5000, 5
        loadings = rng.normal(0, 0.5, (n_cols, 2))
        factors = rng.normal(0, 0.01, (n_obs, 2))
        specific = rng.normal(0, 0.005, (n_obs, n_cols))
        x = factors @ loadings.T + specific
        r = pd.DataFrame(x, columns=[f"A{i}" for i in range(n_cols)])
        cfg = CovarianceConfig(
            method=EstimationMethod.LEDOIT_WOLF,
            min_observations=10,
            annualise=False,
        )
        result = CovarianceEstimator(cfg).estimate(r)
        assert result.shrinkage_intensity < 0.05

    def test_high_dimension_high_shrinkage(self):
        """With p close to T, shrinkage should be substantial."""
        cfg = CovarianceConfig(
            method=EstimationMethod.LEDOIT_WOLF,
            min_observations=10,
            annualise=False,
        )
        result = CovarianceEstimator(cfg).estimate(_high_dim_returns())
        assert result.shrinkage_intensity > 0.3


# ---------------------------------------------------------------------------
# OAS
# ---------------------------------------------------------------------------


class TestOAS:
    def test_positive_shrinkage(self):
        cfg = CovarianceConfig(method=EstimationMethod.OAS)
        result = CovarianceEstimator(cfg).estimate(_returns())
        assert result.shrinkage_intensity > 0

    def test_shrinkage_bounded(self):
        cfg = CovarianceConfig(method=EstimationMethod.OAS)
        result = CovarianceEstimator(cfg).estimate(_returns())
        assert 0 <= result.shrinkage_intensity <= 1

    def test_positive_definite(self):
        cfg = CovarianceConfig(method=EstimationMethod.OAS, annualise=False)
        result = CovarianceEstimator(cfg).estimate(_returns())
        eigvals = np.linalg.eigvalsh(result.covariance.values)
        assert eigvals.min() > 0

    def test_better_conditioned_than_sample(self):
        r = _returns()
        oas = CovarianceEstimator(
            CovarianceConfig(method=EstimationMethod.OAS, annualise=False),
        ).estimate(r)
        sample = CovarianceEstimator(
            CovarianceConfig(method=EstimationMethod.SAMPLE, annualise=False),
        ).estimate(r)
        assert oas.condition_number <= sample.condition_number


# ---------------------------------------------------------------------------
# Exponential weighting
# ---------------------------------------------------------------------------


class TestExponential:
    def test_no_shrinkage_reported(self):
        cfg = CovarianceConfig(method=EstimationMethod.EXPONENTIAL)
        result = CovarianceEstimator(cfg).estimate(_returns())
        assert result.shrinkage_intensity == 0.0

    def test_positive_definite(self):
        cfg = CovarianceConfig(
            method=EstimationMethod.EXPONENTIAL, annualise=False,
        )
        result = CovarianceEstimator(cfg).estimate(_returns())
        eigvals = np.linalg.eigvalsh(result.covariance.values)
        assert eigvals.min() > 0

    def test_different_halflife_different_result(self):
        """Different half-lives should produce different covariances."""
        r = _returns()
        long_cov = CovarianceEstimator(
            CovarianceConfig(
                method=EstimationMethod.EXPONENTIAL, halflife=126, annualise=False,
            ),
        ).estimate(r)
        short_cov = CovarianceEstimator(
            CovarianceConfig(
                method=EstimationMethod.EXPONENTIAL, halflife=21, annualise=False,
            ),
        ).estimate(r)
        assert not np.allclose(
            long_cov.covariance.values, short_cov.covariance.values,
        )

    def test_symmetric(self):
        cfg = CovarianceConfig(
            method=EstimationMethod.EXPONENTIAL, annualise=False,
        )
        result = CovarianceEstimator(cfg).estimate(_returns())
        cov = result.covariance.values
        np.testing.assert_array_almost_equal(cov, cov.T)


# ---------------------------------------------------------------------------
# Annualisation
# ---------------------------------------------------------------------------


class TestAnnualisation:
    def test_annualised_equals_raw_times_252(self):
        r = _returns()
        ann = CovarianceEstimator(
            CovarianceConfig(method=EstimationMethod.SAMPLE, annualise=True),
        ).estimate(r)
        raw = CovarianceEstimator(
            CovarianceConfig(method=EstimationMethod.SAMPLE, annualise=False),
        ).estimate(r)
        np.testing.assert_array_almost_equal(
            ann.covariance.values, raw.covariance.values * 252,
        )

    def test_annualised_vol_matches_daily_scaled(self):
        r = _returns()
        cfg = CovarianceConfig(method=EstimationMethod.SAMPLE, annualise=True)
        result = CovarianceEstimator(cfg).estimate(r)
        for sym in r.columns[:3]:
            daily_std = r[sym].std()
            expected = daily_std * np.sqrt(252)
            assert result.volatilities[sym] == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------


class TestCorrelation:
    def test_diagonal_ones(self):
        result = CovarianceEstimator().estimate(_returns())
        diag = result.correlation.values.diagonal()
        np.testing.assert_array_almost_equal(diag, np.ones(N_ASSETS))

    def test_bounded(self):
        result = CovarianceEstimator().estimate(_returns())
        assert result.correlation.values.min() >= -1.0 - 1e-10
        assert result.correlation.values.max() <= 1.0 + 1e-10

    def test_symmetric(self):
        result = CovarianceEstimator().estimate(_returns())
        corr = result.correlation.values
        np.testing.assert_array_almost_equal(corr, corr.T)


# ---------------------------------------------------------------------------
# Condition number
# ---------------------------------------------------------------------------


class TestConditionNumber:
    def test_positive(self):
        result = CovarianceEstimator().estimate(_returns())
        assert result.condition_number > 0

    def test_shrinkage_reduces_condition(self):
        r = _returns()
        sample = CovarianceEstimator(
            CovarianceConfig(method=EstimationMethod.SAMPLE, annualise=False),
        ).estimate(r)
        lw = CovarianceEstimator(
            CovarianceConfig(method=EstimationMethod.LEDOIT_WOLF, annualise=False),
        ).estimate(r)
        assert lw.condition_number <= sample.condition_number


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_two_assets(self):
        r = _returns()[["ASSET_0", "ASSET_1"]]
        result = CovarianceEstimator().estimate(r)
        assert result.n_assets == 2

    def test_exact_min_observations(self):
        r = _returns().iloc[:30]
        cfg = CovarianceConfig(min_observations=30)
        result = CovarianceEstimator(cfg).estimate(r)
        assert result.n_observations == 30

    def test_nan_rows_dropped(self):
        r = _returns().copy()
        r.iloc[5, 2] = np.nan
        r.iloc[10, 3] = np.nan
        result = CovarianceEstimator().estimate(r)
        assert result.n_observations == N_OBS - 2

    def test_all_methods_produce_psd(self):
        """Every estimation method should yield a PSD matrix."""
        r = _returns()
        for method in EstimationMethod:
            cfg = CovarianceConfig(method=method, annualise=False)
            result = CovarianceEstimator(cfg).estimate(r)
            eigvals = np.linalg.eigvalsh(result.covariance.values)
            assert eigvals.min() > -1e-10, f"{method.name} not PSD"


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_contains_key_info(self):
        result = CovarianceEstimator().estimate(_returns())
        summary = result.summary()
        assert "Covariance Estimation" in summary
        assert "Condition" in summary
        assert "Shrinkage" in summary
        assert "Volatilities" in summary
        assert "Correlation" in summary
