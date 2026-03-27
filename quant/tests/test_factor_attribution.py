"""Tests for factor-based return attribution (QUA-40)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant.portfolio.factor_attribution import (
    FactorAttributionReport,
    FactorAttributor,
    construct_factors,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_factor_data(
    n_days: int = 300,
    seed: int = 42,
) -> tuple[pd.Series, pd.DataFrame]:
    """Create portfolio returns that are a known linear combination of factors.

    Returns (portfolio_returns, factor_returns) where:
        portfolio = 0.0002 + 1.0*market + 0.5*momentum + noise
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_days)

    # Factor returns
    market = rng.normal(0.0005, 0.01, n_days)
    momentum = rng.normal(0.0002, 0.005, n_days)
    low_vol = rng.normal(0.0001, 0.003, n_days)

    factor_df = pd.DataFrame(
        {"market": market, "momentum": momentum, "low_vol": low_vol},
        index=dates,
    )

    # Portfolio = alpha + beta_market*market + beta_mom*momentum + noise
    alpha_daily = 0.0002
    noise = rng.normal(0, 0.002, n_days)
    port_ret = alpha_daily + 1.0 * market + 0.5 * momentum + noise

    return pd.Series(port_ret, index=dates), factor_df


def _make_asset_returns(
    n_assets: int = 10,
    n_days: int = 300,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate multi-asset return data with a common factor structure."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    symbols = [f"SYM_{i:02d}" for i in range(n_assets)]

    # Common factor + idiosyncratic
    factor = rng.normal(0.0005, 0.01, n_days)
    betas = rng.uniform(0.5, 1.5, n_assets)
    idio = rng.normal(0, 0.015, (n_days, n_assets))

    returns = factor[:, None] * betas[None, :] + idio
    return pd.DataFrame(returns, index=dates, columns=symbols)


# ── Tests: FactorAttributor with explicit factors ─────────────────────────


class TestFactorAttributorExplicit:
    """Test attribution with user-supplied factor returns."""

    def test_basic_attribution(self):
        port_ret, factor_df = _make_factor_data()
        attributor = FactorAttributor()
        report = attributor.attribute(port_ret, factor_returns=factor_df)

        assert isinstance(report, FactorAttributionReport)
        assert report.n_observations == len(port_ret)
        assert report.r_squared > 0.5  # should explain most variance

    def test_market_beta_close_to_one(self):
        port_ret, factor_df = _make_factor_data()
        attributor = FactorAttributor()
        report = attributor.attribute(port_ret, factor_returns=factor_df)

        market_fc = next(fc for fc in report.factor_contributions if fc.factor_name == "market")
        assert abs(market_fc.beta - 1.0) < 0.15  # true beta is 1.0

    def test_momentum_beta_close_to_half(self):
        port_ret, factor_df = _make_factor_data()
        attributor = FactorAttributor()
        report = attributor.attribute(port_ret, factor_returns=factor_df)

        mom_fc = next(fc for fc in report.factor_contributions if fc.factor_name == "momentum")
        assert abs(mom_fc.beta - 0.5) < 0.15  # true beta is 0.5

    def test_alpha_positive(self):
        port_ret, factor_df = _make_factor_data()
        attributor = FactorAttributor()
        report = attributor.attribute(port_ret, factor_returns=factor_df)

        # True daily alpha is 0.0002, annualised ~ 0.05
        assert report.alpha_daily > 0
        assert report.alpha > 0

    def test_r_squared_range(self):
        port_ret, factor_df = _make_factor_data()
        attributor = FactorAttributor()
        report = attributor.attribute(port_ret, factor_returns=factor_df)

        assert 0.0 <= report.r_squared <= 1.0
        assert 0.0 <= report.adjusted_r_squared <= 1.0

    def test_factor_contributions_count(self):
        port_ret, factor_df = _make_factor_data()
        attributor = FactorAttributor()
        report = attributor.attribute(port_ret, factor_returns=factor_df)

        assert len(report.factor_contributions) == 3
        names = {fc.factor_name for fc in report.factor_contributions}
        assert names == {"market", "momentum", "low_vol"}

    def test_factor_exposures_dict(self):
        port_ret, factor_df = _make_factor_data()
        attributor = FactorAttributor()
        report = attributor.attribute(port_ret, factor_returns=factor_df)

        assert "market" in report.factor_exposures
        assert abs(report.factor_exposures["market"] - 1.0) < 0.15

    def test_total_return_computed(self):
        port_ret, factor_df = _make_factor_data()
        attributor = FactorAttributor()
        report = attributor.attribute(port_ret, factor_returns=factor_df)

        expected = float((1 + port_ret).prod() - 1)
        assert abs(report.total_return - expected) < 1e-6

    def test_residual_vol_positive(self):
        port_ret, factor_df = _make_factor_data()
        attributor = FactorAttributor()
        report = attributor.attribute(port_ret, factor_returns=factor_df)

        assert report.residual_vol > 0
        assert report.factor_vol > 0

    def test_t_statistics_present(self):
        port_ret, factor_df = _make_factor_data()
        attributor = FactorAttributor()
        report = attributor.attribute(port_ret, factor_returns=factor_df)

        # Market beta should be statistically significant
        market_fc = next(fc for fc in report.factor_contributions if fc.factor_name == "market")
        assert abs(market_fc.t_stat) > 2.0  # significant at 95%


class TestFactorAttributorEdgeCases:
    """Test edge cases and error handling."""

    def test_insufficient_data(self):
        dates = pd.bdate_range("2023-01-01", periods=10)
        port_ret = pd.Series(np.random.default_rng(42).normal(0, 0.01, 10), index=dates)
        factor_df = pd.DataFrame(
            {"market": np.random.default_rng(42).normal(0, 0.01, 10)},
            index=dates,
        )
        attributor = FactorAttributor(min_observations=30)
        report = attributor.attribute(port_ret, factor_returns=factor_df)

        # Should return empty report
        assert report.r_squared == 0.0
        assert len(report.factor_contributions) == 0

    def test_no_factors_or_assets_raises(self):
        import pytest

        dates = pd.bdate_range("2023-01-01", periods=100)
        port_ret = pd.Series(np.random.default_rng(42).normal(0, 0.01, 100), index=dates)
        attributor = FactorAttributor()

        with pytest.raises(ValueError, match="Must provide"):
            attributor.attribute(port_ret)

    def test_single_factor(self):
        rng = np.random.default_rng(42)
        n = 200
        dates = pd.bdate_range("2023-01-01", periods=n)
        market = rng.normal(0.0005, 0.01, n)
        port_ret = pd.Series(0.8 * market + rng.normal(0, 0.002, n), index=dates)
        factor_df = pd.DataFrame({"market": market}, index=dates)

        attributor = FactorAttributor()
        report = attributor.attribute(port_ret, factor_returns=factor_df)

        assert len(report.factor_contributions) == 1
        assert abs(report.factor_contributions[0].beta - 0.8) < 0.1

    def test_zero_returns(self):
        n = 100
        dates = pd.bdate_range("2023-01-01", periods=n)
        port_ret = pd.Series(0.0, index=dates)
        factor_df = pd.DataFrame(
            {"market": np.random.default_rng(42).normal(0, 0.01, n)},
            index=dates,
        )

        attributor = FactorAttributor()
        report = attributor.attribute(port_ret, factor_returns=factor_df)

        assert isinstance(report, FactorAttributionReport)
        assert report.total_return == 0.0


class TestFactorRiskDecomposition:
    """Test risk attribution across factors."""

    def test_risk_contributions_bounded(self):
        port_ret, factor_df = _make_factor_data()
        attributor = FactorAttributor()
        report = attributor.attribute(port_ret, factor_returns=factor_df)

        for fc in report.factor_contributions:
            # Marginal risk contributions can be slightly negative for
            # near-zero-beta factors due to covariance cross-terms
            assert fc.risk_contribution >= -0.01

    def test_risk_contributions_sum_to_factor_share(self):
        port_ret, factor_df = _make_factor_data()
        attributor = FactorAttributor()
        report = attributor.attribute(port_ret, factor_returns=factor_df)

        total_risk = sum(fc.risk_contribution for fc in report.factor_contributions)
        # Factor risk contributions should sum to <= 1.0
        assert total_risk <= 1.0 + 0.01

    def test_market_dominates_risk(self):
        port_ret, factor_df = _make_factor_data()
        attributor = FactorAttributor()
        report = attributor.attribute(port_ret, factor_returns=factor_df)

        market_fc = next(fc for fc in report.factor_contributions if fc.factor_name == "market")
        # Market has the largest beta and highest vol → should dominate risk
        assert market_fc.risk_contribution > 0.2


# ── Tests: Built-in factor construction ───────────────────────────────────


class TestConstructFactors:
    """Test automatic factor construction from asset returns."""

    def test_constructs_expected_factors(self):
        asset_ret = _make_asset_returns()
        factors = construct_factors(asset_ret)

        assert "market" in factors.columns
        assert "momentum" in factors.columns
        assert "low_vol" in factors.columns
        assert "mean_reversion" in factors.columns

    def test_market_factor_is_average(self):
        asset_ret = _make_asset_returns()
        factors = construct_factors(asset_ret)

        expected_market = asset_ret.mean(axis=1)
        pd.testing.assert_series_equal(
            factors["market"], expected_market, check_names=False
        )

    def test_factor_length_matches_input(self):
        asset_ret = _make_asset_returns(n_days=200)
        factors = construct_factors(asset_ret)

        assert len(factors) == 200

    def test_too_few_assets_returns_empty(self):
        dates = pd.bdate_range("2023-01-01", periods=100)
        asset_ret = pd.DataFrame(
            np.random.default_rng(42).normal(0, 0.01, (100, 1)),
            index=dates,
            columns=["SYM_00"],
        )
        factors = construct_factors(asset_ret)
        assert factors.empty

    def test_too_few_days_returns_empty(self):
        dates = pd.bdate_range("2023-01-01", periods=5)
        asset_ret = pd.DataFrame(
            np.random.default_rng(42).normal(0, 0.01, (5, 3)),
            index=dates,
            columns=["A", "B", "C"],
        )
        factors = construct_factors(asset_ret)
        assert factors.empty


# ── Tests: Attribution with auto-constructed factors ──────────────────────


class TestFactorAttributorAutoConstruct:
    """Test attribution using automatic factor construction from assets."""

    def test_auto_construct_succeeds(self):
        asset_ret = _make_asset_returns()
        # Portfolio = equal-weight
        port_ret = asset_ret.mean(axis=1)

        attributor = FactorAttributor()
        report = attributor.attribute(port_ret, asset_returns=asset_ret)

        assert isinstance(report, FactorAttributionReport)
        assert len(report.factor_contributions) > 0

    def test_market_exposure_high_for_ew_portfolio(self):
        asset_ret = _make_asset_returns()
        port_ret = asset_ret.mean(axis=1)

        attributor = FactorAttributor()
        report = attributor.attribute(port_ret, asset_returns=asset_ret)

        # EW portfolio should have strong market exposure
        assert "market" in report.factor_exposures
        assert report.factor_exposures["market"] > 0.5

    def test_r_squared_reasonable(self):
        asset_ret = _make_asset_returns()
        port_ret = asset_ret.mean(axis=1)

        attributor = FactorAttributor()
        report = attributor.attribute(port_ret, asset_returns=asset_ret)

        # Market factor alone should explain a lot of EW portfolio
        assert report.r_squared > 0.3


# ── Tests: Integration with PerformanceAttributor ─────────────────────────


class TestPerformanceAttributorFactorIntegration:
    """Test that PerformanceAttributor now populates factor_exposures."""

    def test_factor_exposures_populated(self):
        from quant.portfolio.attribution import PerformanceAttributor

        asset_ret = _make_asset_returns()
        port_ret = asset_ret.mean(axis=1)

        attributor = PerformanceAttributor()
        report = attributor.attribute(
            portfolio_returns=port_ret,
            asset_returns=asset_ret,
        )

        # factor_exposures should now be populated (was always empty before QUA-40)
        assert len(report.factor_exposures) > 0
        assert "market" in report.factor_exposures

    def test_factor_exposures_empty_without_asset_returns(self):
        from quant.portfolio.attribution import PerformanceAttributor

        dates = pd.bdate_range("2023-01-01", periods=100)
        port_ret = pd.Series(
            np.random.default_rng(42).normal(0.0005, 0.01, 100),
            index=dates,
        )

        attributor = PerformanceAttributor()
        report = attributor.attribute(portfolio_returns=port_ret)

        assert report.factor_exposures == {}
