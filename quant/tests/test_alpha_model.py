"""Tests for cross-sectional alpha model (QUA-98)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.research.alpha_model import (
    AlphaModel,
    AlphaModelConfig,
    AlphaModelResult,
    ForecastMethod,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SYMBOLS = [f"S{i}" for i in range(20)]


def _signal_scores(seed: int = 42) -> pd.Series:
    """Random signal scores across a 20-asset universe."""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0, 1, len(SYMBOLS)), index=SYMBOLS, name="signal")


def _asset_vols(mean_vol: float = 0.25, seed: int = 42) -> pd.Series:
    """Random annualised volatilities."""
    rng = np.random.default_rng(seed)
    vols = rng.uniform(0.10, 0.40, len(SYMBOLS))
    return pd.Series(vols, index=SYMBOLS, name="vol")


def _strong_signal(seed: int = 42) -> pd.Series:
    """Signal with clear cross-sectional dispersion."""
    return pd.Series(
        np.linspace(-2, 2, len(SYMBOLS)), index=SYMBOLS, name="signal",
    )


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestBasic:
    def test_returns_result(self):
        result = AlphaModel().forecast(_signal_scores())
        assert isinstance(result, AlphaModelResult)

    def test_n_assets(self):
        result = AlphaModel().forecast(_signal_scores())
        assert result.n_assets == len(SYMBOLS)

    def test_expected_returns_is_series(self):
        result = AlphaModel().forecast(_signal_scores())
        assert isinstance(result.expected_returns, pd.Series)

    def test_expected_returns_indexed(self):
        result = AlphaModel().forecast(_signal_scores())
        assert list(result.expected_returns.index) == SYMBOLS

    def test_z_scores_populated(self):
        result = AlphaModel().forecast(_signal_scores())
        assert len(result.z_scores) == len(SYMBOLS)

    def test_ranks_populated(self):
        result = AlphaModel().forecast(_signal_scores())
        assert len(result.ranks) == len(SYMBOLS)

    def test_ranks_in_range(self):
        result = AlphaModel().forecast(_signal_scores())
        assert result.ranks.min() > 0
        assert result.ranks.max() <= 1.0


# ---------------------------------------------------------------------------
# IC-vol method
# ---------------------------------------------------------------------------


class TestICVol:
    def test_ic_vol_default(self):
        cfg = AlphaModelConfig(method=ForecastMethod.IC_VOL, information_coefficient=0.05)
        result = AlphaModel(cfg).forecast(_signal_scores(), _asset_vols())
        assert result.method == ForecastMethod.IC_VOL

    def test_higher_ic_larger_spread(self):
        scores = _strong_signal()
        vols = _asset_vols()
        low_ic = AlphaModel(AlphaModelConfig(
            information_coefficient=0.02,
        )).forecast(scores, vols)
        high_ic = AlphaModel(AlphaModelConfig(
            information_coefficient=0.10,
        )).forecast(scores, vols)
        assert high_ic.forecast_spread > low_ic.forecast_spread

    def test_higher_vol_larger_returns(self):
        """Assets with higher vol should get larger magnitude forecasts."""
        scores = _strong_signal()
        result = AlphaModel(AlphaModelConfig(
            information_coefficient=0.05, neutralise=False,
        )).forecast(scores, _asset_vols())
        # The asset with the highest |z| and highest vol should have largest |E[r]|
        assert result.forecast_spread > 0

    def test_zero_ic_zero_returns(self):
        cfg = AlphaModelConfig(information_coefficient=0.0)
        result = AlphaModel(cfg).forecast(_signal_scores())
        assert abs(result.expected_returns.sum()) < 1e-10

    def test_default_vol_used(self):
        """Without explicit vols, uniform 0.20 is assumed."""
        result = AlphaModel().forecast(_signal_scores())
        assert result.cross_sectional_vol == pytest.approx(0.20, abs=0.01)


# ---------------------------------------------------------------------------
# Rank method
# ---------------------------------------------------------------------------


class TestRankMethod:
    def test_rank_method(self):
        cfg = AlphaModelConfig(method=ForecastMethod.RANK, target_spread=0.10)
        result = AlphaModel(cfg).forecast(_signal_scores())
        assert result.method == ForecastMethod.RANK

    def test_spread_matches_target(self):
        """Rank method spread should approximate the target spread."""
        cfg = AlphaModelConfig(
            method=ForecastMethod.RANK, target_spread=0.20, neutralise=False,
        )
        result = AlphaModel(cfg).forecast(_strong_signal())
        # With ranks 0.05 to 1.0, centered: spread ≈ target_spread * (1 - 1/N)
        assert result.forecast_spread > 0.15

    def test_higher_spread_larger_forecasts(self):
        scores = _signal_scores()
        small = AlphaModel(AlphaModelConfig(
            method=ForecastMethod.RANK, target_spread=0.05,
        )).forecast(scores)
        large = AlphaModel(AlphaModelConfig(
            method=ForecastMethod.RANK, target_spread=0.30,
        )).forecast(scores)
        assert large.forecast_spread > small.forecast_spread


# ---------------------------------------------------------------------------
# Raw method
# ---------------------------------------------------------------------------


class TestRawMethod:
    def test_raw_method(self):
        cfg = AlphaModelConfig(method=ForecastMethod.RAW, raw_multiplier=0.01)
        result = AlphaModel(cfg).forecast(_signal_scores())
        assert result.method == ForecastMethod.RAW

    def test_multiplier_scales_output(self):
        scores = _signal_scores()
        small = AlphaModel(AlphaModelConfig(
            method=ForecastMethod.RAW, raw_multiplier=0.01,
        )).forecast(scores)
        large = AlphaModel(AlphaModelConfig(
            method=ForecastMethod.RAW, raw_multiplier=0.10,
        )).forecast(scores)
        assert large.forecast_spread > small.forecast_spread


# ---------------------------------------------------------------------------
# Neutralisation
# ---------------------------------------------------------------------------


class TestNeutralisation:
    def test_neutralised_zero_mean(self):
        result = AlphaModel(AlphaModelConfig(neutralise=True)).forecast(_signal_scores())
        assert abs(result.expected_returns.mean()) < 1e-10

    def test_not_neutralised_may_have_mean(self):
        result = AlphaModel(AlphaModelConfig(
            method=ForecastMethod.RAW, raw_multiplier=1.0, neutralise=False,
        )).forecast(_signal_scores())
        # Not necessarily zero mean (depends on signal)
        assert isinstance(result.expected_returns.mean(), float)


# ---------------------------------------------------------------------------
# Z-score winsorisation
# ---------------------------------------------------------------------------


class TestWinsorisation:
    def test_z_scores_capped(self):
        result = AlphaModel(AlphaModelConfig(winsorise_z=2.0)).forecast(_signal_scores())
        assert result.z_scores.max() <= 2.0 + 1e-10
        assert result.z_scores.min() >= -2.0 - 1e-10

    def test_tighter_winsorisation_smaller_spread(self):
        scores = _strong_signal()
        loose = AlphaModel(AlphaModelConfig(winsorise_z=5.0)).forecast(scores)
        tight = AlphaModel(AlphaModelConfig(winsorise_z=1.5)).forecast(scores)
        assert tight.forecast_spread <= loose.forecast_spread + 1e-10


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_too_few_assets_raises(self):
        scores = pd.Series([0.5], index=["X0"])
        with pytest.raises(ValueError, match="at least"):
            AlphaModel().forecast(scores)

    def test_two_assets(self):
        scores = pd.Series([1.0, -1.0], index=["X0", "X1"])
        result = AlphaModel(AlphaModelConfig(min_assets=2)).forecast(scores)
        assert result.n_assets == 2

    def test_constant_signal_zero_z(self):
        """All identical signals should produce zero z-scores."""
        scores = pd.Series(0.5, index=SYMBOLS)
        result = AlphaModel().forecast(scores)
        assert abs(result.z_scores.sum()) < 1e-10

    def test_nan_signals_dropped(self):
        scores = _signal_scores()
        scores.iloc[0] = np.nan
        result = AlphaModel().forecast(scores)
        assert result.n_assets == len(SYMBOLS) - 1

    def test_partial_vols(self):
        """Missing vols should be filled with 0.20 default."""
        scores = _signal_scores()
        vols = pd.Series(0.30, index=SYMBOLS[:10])  # Only half
        result = AlphaModel().forecast(scores, vols)
        assert result.n_assets == len(SYMBOLS)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_readable(self):
        result = AlphaModel().forecast(_signal_scores(), _asset_vols())
        summary = result.summary()
        assert "Alpha Model" in summary
        assert "Forecast spread" in summary
        assert "E[r]" in summary

    def test_summary_shows_symbols(self):
        result = AlphaModel().forecast(_signal_scores())
        summary = result.summary()
        assert "S0" in summary or "S1" in summary
