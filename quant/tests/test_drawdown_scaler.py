"""Tests for drawdown-aware position scaler (QUA-105)."""
from __future__ import annotations

import pytest

from quant.risk.drawdown_scaler import (
    DrawdownScaler,
    DrawdownScalerConfig,
    ScaledWeights,
    ScalerState,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _weights() -> dict[str, float]:
    return {"AAPL": 0.30, "GOOG": 0.30, "MSFT": 0.40}


def _sim_drawdown(scaler: DrawdownScaler, peak: float, trough: float) -> None:
    """Simulate a peak then drawdown."""
    scaler.update(peak)
    scaler.update(trough)


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestBasic:
    def test_returns_state(self):
        scaler = DrawdownScaler()
        state = scaler.update(100.0)
        assert isinstance(state, ScalerState)

    def test_full_exposure_at_peak(self):
        scaler = DrawdownScaler()
        scaler.update(100.0)
        assert scaler.state.scale_factor == pytest.approx(1.0)

    def test_drawdown_computed(self):
        scaler = DrawdownScaler()
        scaler.update(100.0)
        scaler.update(90.0)
        assert scaler.state.drawdown == pytest.approx(0.10)

    def test_scale_reduces_on_drawdown(self):
        scaler = DrawdownScaler(DrawdownScalerConfig(max_drawdown=0.10))
        _sim_drawdown(scaler, 100.0, 95.0)  # 5% drawdown
        assert scaler.state.scale_factor < 1.0

    def test_peak_tracked(self):
        scaler = DrawdownScaler()
        scaler.update(100.0)
        scaler.update(110.0)
        scaler.update(105.0)
        assert scaler.state.peak == 110.0

    def test_config_accessible(self):
        cfg = DrawdownScalerConfig(max_drawdown=0.20)
        scaler = DrawdownScaler(cfg)
        assert scaler.config.max_drawdown == 0.20


# ---------------------------------------------------------------------------
# Scaling function
# ---------------------------------------------------------------------------


class TestScalingFunction:
    def test_linear_at_half_drawdown(self):
        """Linear scaling at 50% of max drawdown should give ~0.50 scale."""
        cfg = DrawdownScalerConfig(max_drawdown=0.10, floor=0.0, exponent=1.0)
        scaler = DrawdownScaler(cfg)
        _sim_drawdown(scaler, 100.0, 95.0)  # 5% = 50% of 10% max
        assert scaler.state.scale_factor == pytest.approx(0.50, abs=0.01)

    def test_at_max_drawdown_hits_floor(self):
        cfg = DrawdownScalerConfig(max_drawdown=0.10, floor=0.25)
        scaler = DrawdownScaler(cfg)
        _sim_drawdown(scaler, 100.0, 90.0)  # 10% = 100% of max
        assert scaler.state.scale_factor == pytest.approx(0.25, abs=0.01)

    def test_beyond_max_drawdown_at_floor(self):
        cfg = DrawdownScalerConfig(max_drawdown=0.10, floor=0.25)
        scaler = DrawdownScaler(cfg)
        _sim_drawdown(scaler, 100.0, 85.0)  # 15% > 10% max
        assert scaler.state.scale_factor == pytest.approx(0.25, abs=0.01)

    def test_zero_floor(self):
        cfg = DrawdownScalerConfig(max_drawdown=0.10, floor=0.0)
        scaler = DrawdownScaler(cfg)
        _sim_drawdown(scaler, 100.0, 90.0)
        assert scaler.state.scale_factor == pytest.approx(0.0, abs=0.01)

    def test_convex_exponent(self):
        """Exponent > 1 should scale less aggressively at small drawdowns."""
        cfg_lin = DrawdownScalerConfig(max_drawdown=0.10, floor=0.0, exponent=1.0)
        cfg_conv = DrawdownScalerConfig(max_drawdown=0.10, floor=0.0, exponent=2.0)

        scaler_lin = DrawdownScaler(cfg_lin)
        scaler_conv = DrawdownScaler(cfg_conv)

        _sim_drawdown(scaler_lin, 100.0, 97.0)   # 3% drawdown
        _sim_drawdown(scaler_conv, 100.0, 97.0)

        # Convex should have HIGHER scale (less reduction) at small drawdowns
        assert scaler_conv.state.scale_factor > scaler_lin.state.scale_factor

    def test_concave_exponent(self):
        """Exponent < 1 should scale more aggressively at small drawdowns."""
        cfg_lin = DrawdownScalerConfig(max_drawdown=0.10, floor=0.0, exponent=1.0)
        cfg_conc = DrawdownScalerConfig(max_drawdown=0.10, floor=0.0, exponent=0.5)

        scaler_lin = DrawdownScaler(cfg_lin)
        scaler_conc = DrawdownScaler(cfg_conc)

        _sim_drawdown(scaler_lin, 100.0, 97.0)
        _sim_drawdown(scaler_conc, 100.0, 97.0)

        # Concave should have LOWER scale (more reduction) at small drawdowns
        assert scaler_conc.state.scale_factor < scaler_lin.state.scale_factor

    def test_monotonic_with_depth(self):
        """Deeper drawdown should always give lower scale."""
        cfg = DrawdownScalerConfig(max_drawdown=0.20, floor=0.10, exponent=1.5)
        scales = []
        for dd_pct in [0, 2, 5, 10, 15, 20]:
            scaler = DrawdownScaler(cfg)
            _sim_drawdown(scaler, 100.0, 100.0 - dd_pct)
            scales.append(scaler.state.scale_factor)

        for i in range(1, len(scales)):
            assert scales[i] <= scales[i - 1] + 1e-9


# ---------------------------------------------------------------------------
# Weight scaling
# ---------------------------------------------------------------------------


class TestWeightScaling:
    def test_returns_scaled_weights(self):
        scaler = DrawdownScaler()
        scaler.update(100.0)
        result = scaler.scale_weights(_weights())
        assert isinstance(result, ScaledWeights)

    def test_full_exposure_weights_unchanged(self):
        scaler = DrawdownScaler()
        scaler.update(100.0)
        result = scaler.scale_weights(_weights())
        for sym, w in _weights().items():
            assert result.weights[sym] == pytest.approx(w)

    def test_scaled_weights_smaller(self):
        cfg = DrawdownScalerConfig(max_drawdown=0.10, floor=0.0)
        scaler = DrawdownScaler(cfg)
        _sim_drawdown(scaler, 100.0, 95.0)
        result = scaler.scale_weights(_weights())
        for sym, w in _weights().items():
            assert abs(result.weights[sym]) < abs(w)

    def test_cash_weight_positive_on_drawdown(self):
        cfg = DrawdownScalerConfig(max_drawdown=0.10, floor=0.0)
        scaler = DrawdownScaler(cfg)
        _sim_drawdown(scaler, 100.0, 95.0)
        result = scaler.scale_weights(_weights())
        assert result.cash_weight > 0

    def test_cash_weight_zero_at_peak(self):
        scaler = DrawdownScaler()
        scaler.update(100.0)
        result = scaler.scale_weights(_weights())
        assert result.cash_weight == pytest.approx(0.0, abs=1e-9)

    def test_weights_proportional(self):
        """Relative weights should be preserved after scaling."""
        cfg = DrawdownScalerConfig(max_drawdown=0.10, floor=0.0)
        scaler = DrawdownScaler(cfg)
        _sim_drawdown(scaler, 100.0, 95.0)
        result = scaler.scale_weights(_weights())
        # AAPL/GOOG ratio should be preserved (0.30/0.30 = 1.0)
        if result.weights["GOOG"] > 0:
            ratio = result.weights["AAPL"] / result.weights["GOOG"]
            assert ratio == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# Peak recovery
# ---------------------------------------------------------------------------


class TestPeakRecovery:
    def test_new_peak_restores_exposure(self):
        cfg = DrawdownScalerConfig(max_drawdown=0.10, reset_on_peak=True)
        scaler = DrawdownScaler(cfg)
        scaler.update(100.0)
        scaler.update(92.0)  # 8% drawdown
        assert scaler.state.scale_factor < 1.0

        scaler.update(105.0)  # New peak
        assert scaler.state.scale_factor == pytest.approx(1.0)

    def test_no_reset_on_peak_when_disabled(self):
        """With reset_on_peak=True and recovery, drawdown becomes 0."""
        cfg = DrawdownScalerConfig(max_drawdown=0.10, reset_on_peak=True)
        scaler = DrawdownScaler(cfg)
        _sim_drawdown(scaler, 100.0, 92.0)
        scaler.update(105.0)
        assert scaler.state.drawdown == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------


class TestWarmup:
    def test_warmup_delays_scaling(self):
        cfg = DrawdownScalerConfig(max_drawdown=0.10, warmup_periods=5)
        scaler = DrawdownScaler(cfg)
        scaler.update(100.0)
        scaler.update(90.0)  # Only 2 updates, within warmup
        assert scaler.state.scale_factor == 1.0  # Not active yet
        assert not scaler.state.is_active

    def test_warmup_completed(self):
        cfg = DrawdownScalerConfig(max_drawdown=0.10, warmup_periods=3)
        scaler = DrawdownScaler(cfg)
        scaler.update(100.0)
        scaler.update(100.0)
        scaler.update(90.0)  # 3rd update, warmup complete
        assert scaler.state.is_active
        assert scaler.state.scale_factor < 1.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_zero_value_ignored(self):
        scaler = DrawdownScaler()
        scaler.update(100.0)
        scaler.update(0.0)  # Should be ignored
        assert scaler.state.peak == 100.0
        assert scaler.state.current_value == 100.0

    def test_negative_value_ignored(self):
        scaler = DrawdownScaler()
        scaler.update(100.0)
        scaler.update(-50.0)
        assert scaler.state.peak == 100.0

    def test_empty_weights(self):
        scaler = DrawdownScaler()
        scaler.update(100.0)
        result = scaler.scale_weights({})
        assert len(result.weights) == 0
        assert result.cash_weight == pytest.approx(0.0)

    def test_reset(self):
        scaler = DrawdownScaler()
        scaler.update(100.0)
        scaler.update(90.0)
        scaler.reset()
        assert scaler.state.peak == 0.0
        assert scaler.state.n_updates == 0

    def test_initial_state(self):
        scaler = DrawdownScaler()
        state = scaler.state
        assert state.peak == 0.0
        assert state.drawdown == 0.0
        assert state.scale_factor == 1.0


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_readable(self):
        cfg = DrawdownScalerConfig(max_drawdown=0.10)
        scaler = DrawdownScaler(cfg)
        _sim_drawdown(scaler, 100.0, 95.0)
        result = scaler.scale_weights(_weights())
        summary = result.summary()
        assert "Drawdown Scaling" in summary
        assert "Scale factor" in summary
        assert "Cash weight" in summary
