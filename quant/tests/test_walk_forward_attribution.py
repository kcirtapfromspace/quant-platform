"""Tests for walk-forward factor attribution (QUA-45)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.portfolio.walk_forward_attribution import (
    WalkForwardAttributionConfig,
    WalkForwardAttributionResult,
    WalkForwardAttributor,
    WindowSnapshot,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_data(
    n_days: int = 500,
    n_assets: int = 5,
    seed: int = 42,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Generate synthetic portfolio returns, factor returns, and asset returns.

    Portfolio = 0.0002 + 1.0*market + 0.5*momentum + noise.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-01", periods=n_days)

    # Factor returns
    market = rng.normal(0.0003, 0.01, n_days)
    momentum = rng.normal(0.0001, 0.005, n_days)
    factors = pd.DataFrame(
        {"market": market, "momentum": momentum}, index=dates
    )

    # Portfolio returns: alpha + factor exposure + noise
    noise = rng.normal(0, 0.003, n_days)
    port_ret = 0.0002 + 1.0 * market + 0.5 * momentum + noise
    portfolio = pd.Series(port_ret, index=dates, name="portfolio")

    # Asset returns (for auto-construction path)
    common = rng.normal(0.0003, 0.01, n_days)
    assets = pd.DataFrame(
        {
            f"A{i}": common * rng.uniform(0.5, 1.5) + rng.normal(0, 0.015, n_days)
            for i in range(n_assets)
        },
        index=dates,
    )

    return portfolio, factors, assets


# ── Tests: Basic run ─────────────────────────────────────────────────────


class TestBasicRun:
    def test_run_with_factor_returns(self):
        portfolio, factors, _ = _make_data()
        wfa = WalkForwardAttributor()
        result = wfa.run(portfolio, factor_returns=factors)
        assert isinstance(result, WalkForwardAttributionResult)
        assert result.n_windows > 0

    def test_run_with_asset_returns(self):
        portfolio, _, assets = _make_data()
        wfa = WalkForwardAttributor()
        result = wfa.run(portfolio, asset_returns=assets)
        assert result.n_windows > 0

    def test_run_with_both(self):
        portfolio, factors, assets = _make_data()
        wfa = WalkForwardAttributor()
        # factor_returns takes precedence
        result = wfa.run(portfolio, factor_returns=factors, asset_returns=assets)
        assert result.n_windows > 0

    def test_no_data_raises(self):
        portfolio, _, _ = _make_data()
        wfa = WalkForwardAttributor()
        with pytest.raises(ValueError, match="factor_returns or asset_returns"):
            wfa.run(portfolio)

    def test_too_short_raises(self):
        portfolio, factors, _ = _make_data(n_days=50)
        wfa = WalkForwardAttributor()
        config = WalkForwardAttributionConfig(window=126)
        with pytest.raises(ValueError, match="at least 126"):
            wfa.run(portfolio, factor_returns=factors, config=config)


# ── Tests: Window generation ─────────────────────────────────────────────


class TestWindowGeneration:
    def test_window_count(self):
        portfolio, factors, _ = _make_data(n_days=500)
        config = WalkForwardAttributionConfig(window=126, step=21)
        wfa = WalkForwardAttributor()
        result = wfa.run(portfolio, factor_returns=factors, config=config)
        # (500 - 126) / 21 + 1 = 17.8 -> 18 windows
        expected = (500 - 126) // 21 + 1
        assert result.n_windows == expected

    def test_non_overlapping_windows(self):
        portfolio, factors, _ = _make_data(n_days=500)
        config = WalkForwardAttributionConfig(window=126, step=126)
        wfa = WalkForwardAttributor()
        result = wfa.run(portfolio, factor_returns=factors, config=config)
        # Non-overlapping: 500 / 126 = 3 full windows
        assert result.n_windows == 3

    def test_snapshots_ordered_by_time(self):
        portfolio, factors, _ = _make_data()
        wfa = WalkForwardAttributor()
        result = wfa.run(portfolio, factor_returns=factors)
        dates = [s.window_end for s in result.snapshots]
        assert dates == sorted(dates)

    def test_snapshot_fields(self):
        portfolio, factors, _ = _make_data()
        wfa = WalkForwardAttributor()
        result = wfa.run(portfolio, factor_returns=factors)
        snap = result.snapshots[0]
        assert isinstance(snap, WindowSnapshot)
        assert snap.window_start < snap.window_end
        assert snap.report.n_observations > 0


# ── Tests: Beta paths ────────────────────────────────────────────────────


class TestBetaPaths:
    def test_beta_paths_shape(self):
        portfolio, factors, _ = _make_data()
        wfa = WalkForwardAttributor()
        result = wfa.run(portfolio, factor_returns=factors)
        df = result.beta_paths()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == result.n_windows
        assert set(df.columns) == {"market", "momentum"}

    def test_market_beta_near_one(self):
        portfolio, factors, _ = _make_data()
        wfa = WalkForwardAttributor()
        result = wfa.run(portfolio, factor_returns=factors)
        betas = result.beta_paths()
        # Portfolio was constructed with market beta = 1.0
        mean_market = betas["market"].mean()
        assert 0.7 < mean_market < 1.3

    def test_momentum_beta_near_half(self):
        portfolio, factors, _ = _make_data()
        wfa = WalkForwardAttributor()
        result = wfa.run(portfolio, factor_returns=factors)
        betas = result.beta_paths()
        mean_mom = betas["momentum"].mean()
        assert 0.2 < mean_mom < 0.8

    def test_empty_result_returns_empty_df(self):
        result = WalkForwardAttributionResult(
            config=WalkForwardAttributionConfig()
        )
        assert result.beta_paths().empty


# ── Tests: Rolling alpha ─────────────────────────────────────────────────


class TestRollingAlpha:
    def test_rolling_alpha_shape(self):
        portfolio, factors, _ = _make_data()
        wfa = WalkForwardAttributor()
        result = wfa.run(portfolio, factor_returns=factors)
        alpha = result.rolling_alpha()
        assert isinstance(alpha, pd.Series)
        assert len(alpha) == result.n_windows

    def test_alpha_positive_on_average(self):
        portfolio, factors, _ = _make_data()
        wfa = WalkForwardAttributor()
        result = wfa.run(portfolio, factor_returns=factors)
        alpha = result.rolling_alpha()
        # Portfolio was constructed with daily alpha of 0.0002
        assert alpha.mean() > 0.0

    def test_empty_result_returns_empty_series(self):
        result = WalkForwardAttributionResult(
            config=WalkForwardAttributionConfig()
        )
        assert result.rolling_alpha().empty


# ── Tests: Rolling R-squared ─────────────────────────────────────────────


class TestRollingRSquared:
    def test_r_squared_range(self):
        portfolio, factors, _ = _make_data()
        wfa = WalkForwardAttributor()
        result = wfa.run(portfolio, factor_returns=factors)
        r2 = result.rolling_r_squared()
        assert all(0.0 <= v <= 1.0 for v in r2)

    def test_r_squared_reasonably_high(self):
        portfolio, factors, _ = _make_data()
        wfa = WalkForwardAttributor()
        result = wfa.run(portfolio, factor_returns=factors)
        r2 = result.rolling_r_squared()
        # Factor model should explain a good chunk of variance
        assert r2.mean() > 0.3


# ── Tests: Risk contribution paths ───────────────────────────────────────


class TestRiskContributionPaths:
    def test_risk_paths_shape(self):
        portfolio, factors, _ = _make_data()
        wfa = WalkForwardAttributor()
        result = wfa.run(portfolio, factor_returns=factors)
        rc = result.risk_contribution_paths()
        assert len(rc) == result.n_windows
        assert set(rc.columns) == {"market", "momentum"}

    def test_risk_contributions_non_negative_on_average(self):
        portfolio, factors, _ = _make_data()
        wfa = WalkForwardAttributor()
        result = wfa.run(portfolio, factor_returns=factors)
        rc = result.risk_contribution_paths()
        # Market should dominate risk
        assert rc["market"].mean() > 0.0


# ── Tests: Residual vol ─────────────────────────────────────────────────


class TestResidualVol:
    def test_residual_vol_positive(self):
        portfolio, factors, _ = _make_data()
        wfa = WalkForwardAttributor()
        result = wfa.run(portfolio, factor_returns=factors)
        rv = result.rolling_residual_vol()
        assert all(v >= 0.0 for v in rv)

    def test_residual_vol_shape(self):
        portfolio, factors, _ = _make_data()
        wfa = WalkForwardAttributor()
        result = wfa.run(portfolio, factor_returns=factors)
        rv = result.rolling_residual_vol()
        assert len(rv) == result.n_windows


# ── Tests: Summary ───────────────────────────────────────────────────────


class TestSummary:
    def test_summary_with_data(self):
        portfolio, factors, _ = _make_data()
        wfa = WalkForwardAttributor()
        result = wfa.run(portfolio, factor_returns=factors)
        summary = result.summary()
        assert "Walk-Forward" in summary
        assert "alpha" in summary.lower()
        assert "R-squared" in summary
        assert "market" in summary
        assert "momentum" in summary

    def test_empty_summary(self):
        result = WalkForwardAttributionResult(
            config=WalkForwardAttributionConfig()
        )
        assert "no windows" in result.summary()


# ── Tests: Factor names ─────────────────────────────────────────────────


class TestFactorNames:
    def test_factor_names_populated(self):
        portfolio, factors, _ = _make_data()
        wfa = WalkForwardAttributor()
        result = wfa.run(portfolio, factor_returns=factors)
        assert "market" in result.factor_names
        assert "momentum" in result.factor_names

    def test_factor_names_sorted(self):
        portfolio, factors, _ = _make_data()
        wfa = WalkForwardAttributor()
        result = wfa.run(portfolio, factor_returns=factors)
        assert result.factor_names == sorted(result.factor_names)


# ── Tests: Custom config ────────────────────────────────────────────────


class TestCustomConfig:
    def test_small_window(self):
        portfolio, factors, _ = _make_data(n_days=200)
        config = WalkForwardAttributionConfig(window=60, step=10, min_observations=20)
        wfa = WalkForwardAttributor()
        result = wfa.run(portfolio, factor_returns=factors, config=config)
        assert result.n_windows > 10

    def test_min_observations_override(self):
        portfolio, factors, _ = _make_data(n_days=200)
        config = WalkForwardAttributionConfig(window=40, step=10, min_observations=20)
        wfa = WalkForwardAttributor(min_observations=15)
        result = wfa.run(portfolio, factor_returns=factors, config=config)
        assert result.n_windows > 0
        # Constructor min_observations takes precedence
        for snap in result.snapshots:
            assert snap.report.n_observations >= 15 or snap.report.r_squared == 0.0
