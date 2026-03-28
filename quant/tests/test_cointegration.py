"""Tests for cointegration tester (QUA-110)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.research.cointegration import (
    CointegrationConfig,
    CointegrationTester,
    HedgeMethod,
    PairResult,
    ScreenResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_OBS = 500


def _cointegrated_pair(
    seed: int = 42,
    beta: float = 1.5,
    alpha: float = 2.0,
    noise_std: float = 0.3,
    theta: float = 0.1,
) -> tuple[pd.Series, pd.Series]:
    """Generate a cointegrated pair:  y = α + β·x + stationary_noise."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-01", periods=N_OBS)

    # x follows a random walk
    x = np.cumsum(rng.normal(0, 1, N_OBS)) + 100.0

    # y = α + β·x + mean-reverting noise (OU process)
    ou_noise = np.zeros(N_OBS)
    for t in range(1, N_OBS):
        ou_noise[t] = (1 - theta) * ou_noise[t - 1] + rng.normal(0, noise_std)

    y = alpha + beta * x + ou_noise

    return (
        pd.Series(y, index=dates, name="Y"),
        pd.Series(x, index=dates, name="X"),
    )


def _independent_pair(seed: int = 42) -> tuple[pd.Series, pd.Series]:
    """Generate two independent random walks (not cointegrated)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-01", periods=N_OBS)

    x = np.cumsum(rng.normal(0, 1, N_OBS)) + 100.0
    y = np.cumsum(rng.normal(0, 1, N_OBS)) + 100.0

    return (
        pd.Series(y, index=dates, name="Y"),
        pd.Series(x, index=dates, name="X"),
    )


def _universe_prices(
    seed: int = 42,
    n_assets: int = 6,
) -> pd.DataFrame:
    """Generate prices with some cointegrated and some independent pairs."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-01", periods=N_OBS)

    # Two random walks as base
    rw1 = np.cumsum(rng.normal(0, 1, N_OBS)) + 100.0
    rw2 = np.cumsum(rng.normal(0, 1, N_OBS)) + 100.0

    prices = {}
    # A, B are cointegrated (both follow rw1 + noise)
    prices["A"] = rw1 + rng.normal(0, 0.3, N_OBS)
    prices["B"] = 1.2 * rw1 + 5.0 + rng.normal(0, 0.3, N_OBS)
    # C, D are cointegrated (both follow rw2 + noise)
    prices["C"] = rw2 + rng.normal(0, 0.3, N_OBS)
    prices["D"] = 0.8 * rw2 - 3.0 + rng.normal(0, 0.3, N_OBS)
    # E, F are independent random walks
    for sym in ["E", "F"]:
        prices[sym] = np.cumsum(rng.normal(0, 1, N_OBS)) + 100.0

    return pd.DataFrame(prices, index=dates)


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestBasic:
    def test_returns_pair_result(self):
        y, x = _cointegrated_pair()
        result = CointegrationTester().test_pair(y, x)
        assert isinstance(result, PairResult)

    def test_asset_names(self):
        y, x = _cointegrated_pair()
        result = CointegrationTester().test_pair(y, x)
        assert result.asset_y == "Y"
        assert result.asset_x == "X"

    def test_custom_asset_names(self):
        y, x = _cointegrated_pair()
        result = CointegrationTester().test_pair(
            y, x, asset_y="AAPL", asset_x="MSFT",
        )
        assert result.asset_y == "AAPL"
        assert result.asset_x == "MSFT"

    def test_n_observations(self):
        y, x = _cointegrated_pair()
        result = CointegrationTester().test_pair(y, x)
        assert result.n_observations == N_OBS

    def test_too_few_observations_raises(self):
        y, x = _cointegrated_pair()
        cfg = CointegrationConfig(min_observations=9999)
        with pytest.raises(ValueError, match="at least 9999"):
            CointegrationTester(cfg).test_pair(y, x)

    def test_config_accessible(self):
        cfg = CointegrationConfig(significance=0.01)
        tester = CointegrationTester(cfg)
        assert tester.config.significance == 0.01


# ---------------------------------------------------------------------------
# Cointegration detection
# ---------------------------------------------------------------------------


class TestCointegrationDetection:
    def test_detects_cointegrated_pair(self):
        y, x = _cointegrated_pair()
        result = CointegrationTester().test_pair(y, x)
        assert result.is_cointegrated is True

    def test_rejects_independent_pair(self):
        y, x = _independent_pair()
        result = CointegrationTester().test_pair(y, x)
        assert result.is_cointegrated is False

    def test_adf_statistic_negative_for_coint(self):
        y, x = _cointegrated_pair()
        result = CointegrationTester().test_pair(y, x)
        assert result.adf_statistic < result.adf_critical

    def test_adf_statistic_less_negative_for_non_coint(self):
        y, x = _independent_pair()
        result = CointegrationTester().test_pair(y, x)
        assert result.adf_statistic > result.adf_critical

    def test_stricter_significance(self):
        """1% significance should accept fewer pairs than 5%."""
        y, x = _cointegrated_pair()
        r_05 = CointegrationTester(CointegrationConfig(significance=0.05)).test_pair(y, x)
        r_01 = CointegrationTester(CointegrationConfig(significance=0.01)).test_pair(y, x)
        assert r_01.adf_critical < r_05.adf_critical


# ---------------------------------------------------------------------------
# Hedge ratio
# ---------------------------------------------------------------------------


class TestHedgeRatio:
    def test_ols_recovers_true_beta(self):
        y, x = _cointegrated_pair(beta=1.5, noise_std=0.2)
        result = CointegrationTester().test_pair(y, x)
        assert result.hedge_ratio == pytest.approx(1.5, abs=0.1)

    def test_ols_recovers_true_intercept(self):
        y, x = _cointegrated_pair(beta=1.5, alpha=2.0, noise_std=0.2)
        result = CointegrationTester().test_pair(y, x)
        assert result.intercept == pytest.approx(2.0, abs=5.0)

    def test_tls_returns_result(self):
        y, x = _cointegrated_pair()
        cfg = CointegrationConfig(hedge_method=HedgeMethod.TLS)
        result = CointegrationTester(cfg).test_pair(y, x)
        assert isinstance(result.hedge_ratio, float)

    def test_tls_close_to_ols(self):
        """TLS and OLS should be close for well-conditioned data."""
        y, x = _cointegrated_pair(noise_std=0.2)
        ols = CointegrationTester().test_pair(y, x)
        tls = CointegrationTester(
            CointegrationConfig(hedge_method=HedgeMethod.TLS),
        ).test_pair(y, x)
        assert tls.hedge_ratio == pytest.approx(ols.hedge_ratio, abs=0.3)


# ---------------------------------------------------------------------------
# Half-life
# ---------------------------------------------------------------------------


class TestHalfLife:
    def test_finite_for_cointegrated(self):
        y, x = _cointegrated_pair()
        result = CointegrationTester().test_pair(y, x)
        assert result.half_life > 0
        assert result.half_life < float("inf")

    def test_faster_reversion_shorter_halflife(self):
        """Higher OU theta → shorter half-life."""
        y_fast, x_fast = _cointegrated_pair(theta=0.30, seed=42)
        y_slow, x_slow = _cointegrated_pair(theta=0.03, seed=42)
        r_fast = CointegrationTester().test_pair(y_fast, x_fast)
        r_slow = CointegrationTester().test_pair(y_slow, x_slow)
        assert r_fast.half_life < r_slow.half_life

    def test_reasonable_range(self):
        y, x = _cointegrated_pair()
        result = CointegrationTester().test_pair(y, x)
        assert 1 < result.half_life < 200


# ---------------------------------------------------------------------------
# Spread and z-score
# ---------------------------------------------------------------------------


class TestSpread:
    def test_spread_series_length(self):
        y, x = _cointegrated_pair()
        result = CointegrationTester().test_pair(y, x)
        assert len(result.spread_series) == N_OBS

    def test_spread_mean_near_zero(self):
        """OLS ensures residuals have ~zero mean."""
        y, x = _cointegrated_pair()
        result = CointegrationTester().test_pair(y, x)
        assert abs(result.spread_mean) < 1.0

    def test_spread_std_positive(self):
        y, x = _cointegrated_pair()
        result = CointegrationTester().test_pair(y, x)
        assert result.spread_std > 0

    def test_z_score_series_length(self):
        y, x = _cointegrated_pair()
        result = CointegrationTester().test_pair(y, x)
        assert len(result.z_score_series) == N_OBS

    def test_z_score_bounded(self):
        """Z-scores should be reasonable for mean-reverting spread."""
        y, x = _cointegrated_pair()
        result = CointegrationTester().test_pair(y, x)
        # Not a hard bound but z-scores > 10 would be unusual
        assert result.z_score_series.abs().max() < 10


# ---------------------------------------------------------------------------
# Universe screen
# ---------------------------------------------------------------------------


class TestUniverseScreen:
    def test_returns_screen_result(self):
        prices = _universe_prices()
        result = CointegrationTester().screen_universe(prices)
        assert isinstance(result, ScreenResult)

    def test_n_tested(self):
        prices = _universe_prices(n_assets=6)
        result = CointegrationTester().screen_universe(prices)
        # C(6,2) = 15 pairs
        assert result.n_tested == 15

    def test_finds_cointegrated_pairs(self):
        prices = _universe_prices()
        result = CointegrationTester().screen_universe(prices)
        assert result.n_cointegrated > 0

    def test_sorted_by_adf(self):
        prices = _universe_prices()
        result = CointegrationTester().screen_universe(prices)
        stats = [p.adf_statistic for p in result.pairs]
        assert stats == sorted(stats)

    def test_cointegrated_pairs_at_top(self):
        """Cointegrated pairs (A~B, C~D) should rank near the top."""
        prices = _universe_prices()
        result = CointegrationTester().screen_universe(prices)
        top_syms = {
            (p.asset_y, p.asset_x) for p in result.pairs[:4]
        }
        # At least one of the known cointegrated pairs should be in top 4
        known = [
            frozenset(("A", "B")),
            frozenset(("C", "D")),
        ]
        found = sum(
            1 for k in known
            if any(frozenset(p) == k for p in top_syms)
        )
        assert found >= 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_identical_series(self):
        """Identical series should be trivially cointegrated."""
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2022-01-01", periods=200)
        x = pd.Series(np.cumsum(rng.normal(0, 1, 200)) + 100, index=dates)
        result = CointegrationTester(
            CointegrationConfig(min_observations=30),
        ).test_pair(x, x.copy())
        assert result.hedge_ratio == pytest.approx(1.0, abs=0.01)

    def test_nan_handling(self):
        """NaN rows should be dropped, not crash."""
        y, x = _cointegrated_pair()
        y.iloc[5] = np.nan
        x.iloc[10] = np.nan
        result = CointegrationTester().test_pair(y, x)
        assert result.n_observations == N_OBS - 2

    def test_misaligned_index(self):
        """Series with partially overlapping indices."""
        dates_a = pd.bdate_range("2022-01-01", periods=N_OBS)
        dates_b = pd.bdate_range("2022-02-01", periods=N_OBS)
        rng = np.random.default_rng(42)
        x = pd.Series(
            np.cumsum(rng.normal(0, 1, N_OBS)) + 100, index=dates_a, name="X",
        )
        y = pd.Series(
            np.cumsum(rng.normal(0, 1, N_OBS)) + 100, index=dates_b, name="Y",
        )
        # Should work on the overlap
        result = CointegrationTester(
            CointegrationConfig(min_observations=30),
        ).test_pair(y, x)
        assert result.n_observations > 0


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_pair_summary(self):
        y, x = _cointegrated_pair()
        result = CointegrationTester().test_pair(y, x)
        summary = result.summary()
        assert "Cointegration" in summary
        assert "ADF statistic" in summary
        assert "Hedge ratio" in summary
        assert "Half-life" in summary

    def test_screen_summary(self):
        prices = _universe_prices()
        result = CointegrationTester().screen_universe(prices)
        summary = result.summary()
        assert "Screen" in summary
        assert "Cointegrated" in summary
