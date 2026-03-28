"""Tests for regime-conditional signal blending (QUA-91)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.research.signal_blender import (
    BlenderConfig,
    BlendMethod,
    BlendResult,
    RegimeWeights,
    SignalBlender,
    SignalWeight,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_DATES = 100
SYMBOLS = ["A", "B", "C", "D", "E"]
DATES = pd.bdate_range("2023-01-01", periods=N_DATES)


def _make_signals(seed: int = 42) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    return {
        "momentum": pd.DataFrame(
            rng.normal(0, 1, (N_DATES, len(SYMBOLS))),
            index=DATES, columns=SYMBOLS,
        ),
        "reversal": pd.DataFrame(
            rng.normal(0, 1, (N_DATES, len(SYMBOLS))),
            index=DATES, columns=SYMBOLS,
        ),
        "quality": pd.DataFrame(
            rng.normal(0, 1, (N_DATES, len(SYMBOLS))),
            index=DATES, columns=SYMBOLS,
        ),
    }


def _make_returns(seed: int = 99) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(0.0005, 0.02, (N_DATES, len(SYMBOLS))),
        index=DATES, columns=SYMBOLS,
    )


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestBasic:
    def test_returns_result(self):
        blender = SignalBlender()
        result = blender.blend(_make_signals())
        assert isinstance(result, BlendResult)

    def test_composite_shape(self):
        blender = SignalBlender()
        result = blender.blend(_make_signals())
        assert result.composite_scores.shape == (N_DATES, len(SYMBOLS))

    def test_n_signals(self):
        blender = SignalBlender()
        result = blender.blend(_make_signals())
        assert result.n_signals == 3

    def test_n_dates(self):
        blender = SignalBlender()
        result = blender.blend(_make_signals())
        assert result.n_dates == N_DATES

    def test_signal_weights_populated(self):
        blender = SignalBlender()
        result = blender.blend(_make_signals())
        assert len(result.signal_weights) == 3
        for sw in result.signal_weights:
            assert isinstance(sw, SignalWeight)


# ---------------------------------------------------------------------------
# Equal weight
# ---------------------------------------------------------------------------


class TestEqualWeight:
    def test_equal_weights(self):
        blender = SignalBlender(BlenderConfig(method=BlendMethod.EQUAL_WEIGHT))
        result = blender.blend(_make_signals())
        for sw in result.signal_weights:
            assert sw.weight == pytest.approx(1.0 / 3)

    def test_weights_sum_to_one(self):
        blender = SignalBlender(BlenderConfig(method=BlendMethod.EQUAL_WEIGHT))
        result = blender.blend(_make_signals())
        total = sum(sw.weight for sw in result.signal_weights)
        assert total == pytest.approx(1.0)

    def test_composite_is_average(self):
        cfg = BlenderConfig(method=BlendMethod.EQUAL_WEIGHT, z_score_signals=False)
        blender = SignalBlender(cfg)
        signals = _make_signals()
        result = blender.blend(signals)
        # Manual average
        expected = (signals["momentum"] + signals["reversal"] + signals["quality"]) / 3
        pd.testing.assert_frame_equal(result.composite_scores, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# IC-weighted
# ---------------------------------------------------------------------------


class TestICWeighted:
    def test_runs_with_returns(self):
        blender = SignalBlender(BlenderConfig(
            method=BlendMethod.IC_WEIGHTED,
            ic_lookback=50,
            ic_min_periods=10,
        ))
        result = blender.blend(_make_signals(), forward_returns=_make_returns())
        assert result.n_signals == 3

    def test_weights_sum_to_one(self):
        blender = SignalBlender(BlenderConfig(
            method=BlendMethod.IC_WEIGHTED,
            ic_lookback=50,
            ic_min_periods=10,
        ))
        result = blender.blend(_make_signals(), forward_returns=_make_returns())
        total = sum(sw.weight for sw in result.signal_weights)
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_requires_forward_returns(self):
        blender = SignalBlender(BlenderConfig(method=BlendMethod.IC_WEIGHTED))
        with pytest.raises(ValueError, match="forward_returns"):
            blender.blend(_make_signals())

    def test_weights_non_negative(self):
        blender = SignalBlender(BlenderConfig(
            method=BlendMethod.IC_WEIGHTED,
            ic_lookback=50,
            ic_min_periods=10,
        ))
        result = blender.blend(_make_signals(), forward_returns=_make_returns())
        for sw in result.signal_weights:
            assert sw.weight >= 0


# ---------------------------------------------------------------------------
# Regime-conditional
# ---------------------------------------------------------------------------


class TestRegimeConditional:
    def _regime_config(self) -> BlenderConfig:
        return BlenderConfig(
            method=BlendMethod.REGIME_CONDITIONAL,
            regime_weights={
                "high_vol": RegimeWeights(
                    {"momentum": 0.2, "reversal": 0.5, "quality": 0.3}
                ),
                "low_vol": RegimeWeights(
                    {"momentum": 0.5, "reversal": 0.2, "quality": 0.3}
                ),
            },
            default_regime="low_vol",
        )

    def test_high_vol_weights(self):
        blender = SignalBlender(self._regime_config())
        result = blender.blend(_make_signals(), regime="high_vol")
        reversal = next(sw for sw in result.signal_weights if sw.signal_name == "reversal")
        momentum = next(sw for sw in result.signal_weights if sw.signal_name == "momentum")
        assert reversal.weight > momentum.weight

    def test_low_vol_weights(self):
        blender = SignalBlender(self._regime_config())
        result = blender.blend(_make_signals(), regime="low_vol")
        momentum = next(sw for sw in result.signal_weights if sw.signal_name == "momentum")
        reversal = next(sw for sw in result.signal_weights if sw.signal_name == "reversal")
        assert momentum.weight > reversal.weight

    def test_regime_recorded(self):
        blender = SignalBlender(self._regime_config())
        result = blender.blend(_make_signals(), regime="high_vol")
        assert result.regime == "high_vol"

    def test_default_regime_fallback(self):
        blender = SignalBlender(self._regime_config())
        result = blender.blend(_make_signals(), regime="unknown_regime")
        assert result.regime == "low_vol"  # Falls back to default

    def test_no_regime_uses_default(self):
        blender = SignalBlender(self._regime_config())
        result = blender.blend(_make_signals())
        assert result.regime == "low_vol"

    def test_weights_sum_to_one(self):
        blender = SignalBlender(self._regime_config())
        result = blender.blend(_make_signals(), regime="high_vol")
        total = sum(sw.weight for sw in result.signal_weights)
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_different_regimes_different_output(self):
        blender = SignalBlender(self._regime_config())
        signals = _make_signals()
        high = blender.blend(signals, regime="high_vol")
        low = blender.blend(signals, regime="low_vol")
        # Different weights should produce different composites
        assert not high.composite_scores.equals(low.composite_scores)


# ---------------------------------------------------------------------------
# Z-scoring
# ---------------------------------------------------------------------------


class TestZScoring:
    def test_z_scored_mean_near_zero(self):
        blender = SignalBlender(BlenderConfig(z_score_signals=True))
        result = blender.blend(_make_signals())
        # Cross-sectional mean should be near zero
        row_means = result.composite_scores.mean(axis=1)
        assert abs(row_means.mean()) < 0.5

    def test_no_z_score_preserves_raw(self):
        cfg = BlenderConfig(method=BlendMethod.EQUAL_WEIGHT, z_score_signals=False)
        blender = SignalBlender(cfg)
        signals = _make_signals()
        result = blender.blend(signals)
        expected = (signals["momentum"] + signals["quality"] + signals["reversal"]) / 3
        pd.testing.assert_frame_equal(result.composite_scores, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_signal(self):
        blender = SignalBlender()
        signals = {"alpha": _make_signals()["momentum"]}
        result = blender.blend(signals)
        assert result.n_signals == 1
        assert result.signal_weights[0].weight == pytest.approx(1.0)

    def test_empty_signals_raises(self):
        blender = SignalBlender()
        with pytest.raises(ValueError, match="at least 1"):
            blender.blend({})

    def test_two_signals(self):
        blender = SignalBlender()
        signals = dict(list(_make_signals().items())[:2])
        result = blender.blend(signals)
        assert result.n_signals == 2

    def test_regime_weights_normalised(self):
        rw = RegimeWeights({"a": 3.0, "b": 7.0})
        norm = rw.normalised()
        assert norm["a"] == pytest.approx(0.3)
        assert norm["b"] == pytest.approx(0.7)

    def test_zero_regime_weights_fallback(self):
        rw = RegimeWeights({"a": 0.0, "b": 0.0})
        norm = rw.normalised()
        assert norm["a"] == pytest.approx(0.5)
        assert norm["b"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_readable(self):
        blender = SignalBlender()
        result = blender.blend(_make_signals())
        summary = result.summary()
        assert "Signal Blend" in summary
        assert "momentum" in summary
        assert "Weight" in summary
