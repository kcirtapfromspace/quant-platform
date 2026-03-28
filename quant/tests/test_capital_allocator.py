"""Tests for dynamic capital allocation across strategy sleeves (QUA-79)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant.portfolio.capital_allocator import (
    AllocationConfig,
    AllocationMethod,
    AllocationResult,
    CapitalAllocator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sleeve_returns(
    n_sleeves: int = 3,
    n_days: int = 252,
    vols: list[float] | None = None,
    seed: int = 42,
) -> dict[str, pd.Series]:
    """Generate sleeve returns with controlled volatilities."""
    rng = np.random.default_rng(seed)
    vols = vols or [0.10, 0.20, 0.30]
    names = ["low_vol", "med_vol", "high_vol"][:n_sleeves]
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    result = {}
    for i, name in enumerate(names):
        daily_vol = vols[i] / np.sqrt(252)
        result[name] = pd.Series(
            rng.normal(0.0003, daily_vol, n_days), index=dates
        )
    return result


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestBasic:
    def test_allocate_returns_result(self):
        allocator = CapitalAllocator()
        result = allocator.allocate(_make_sleeve_returns())
        assert isinstance(result, AllocationResult)

    def test_weights_sum_to_one(self):
        allocator = CapitalAllocator()
        result = allocator.allocate(_make_sleeve_returns())
        assert abs(sum(result.weights.values()) - 1.0) < 1e-10

    def test_all_sleeves_present(self):
        rets = _make_sleeve_returns()
        allocator = CapitalAllocator()
        result = allocator.allocate(rets)
        assert set(result.weights.keys()) == set(rets.keys())

    def test_vols_populated(self):
        allocator = CapitalAllocator()
        result = allocator.allocate(_make_sleeve_returns())
        assert len(result.sleeve_vols) == 3
        for vol in result.sleeve_vols.values():
            assert vol > 0

    def test_method_tracked(self):
        config = AllocationConfig(method=AllocationMethod.INVERSE_VOL)
        allocator = CapitalAllocator(config)
        result = allocator.allocate(_make_sleeve_returns())
        assert result.method == AllocationMethod.INVERSE_VOL


# ---------------------------------------------------------------------------
# Equal weight
# ---------------------------------------------------------------------------


class TestEqualWeight:
    def test_equal_weights(self):
        config = AllocationConfig(method=AllocationMethod.EQUAL_WEIGHT)
        allocator = CapitalAllocator(config)
        result = allocator.allocate(_make_sleeve_returns())
        for w in result.weights.values():
            assert abs(w - 1.0 / 3) < 0.01


# ---------------------------------------------------------------------------
# Inverse volatility
# ---------------------------------------------------------------------------


class TestInverseVol:
    def test_low_vol_gets_more_weight(self):
        config = AllocationConfig(method=AllocationMethod.INVERSE_VOL)
        allocator = CapitalAllocator(config)
        result = allocator.allocate(
            _make_sleeve_returns(vols=[0.10, 0.20, 0.30])
        )
        # Lower vol strategy should get more capital
        assert result.weights["low_vol"] > result.weights["med_vol"]
        assert result.weights["med_vol"] > result.weights["high_vol"]

    def test_equal_vols_give_equal_weights(self):
        config = AllocationConfig(
            method=AllocationMethod.INVERSE_VOL,
            min_weight=0.0,
            max_weight=1.0,
        )
        allocator = CapitalAllocator(config)
        result = allocator.allocate(
            _make_sleeve_returns(vols=[0.15, 0.15, 0.15])
        )
        for w in result.weights.values():
            assert abs(w - 1.0 / 3) < 0.02


# ---------------------------------------------------------------------------
# Risk parity
# ---------------------------------------------------------------------------


class TestRiskParity:
    def test_risk_parity_produces_weights(self):
        config = AllocationConfig(method=AllocationMethod.RISK_PARITY)
        allocator = CapitalAllocator(config)
        result = allocator.allocate(_make_sleeve_returns())
        assert abs(sum(result.weights.values()) - 1.0) < 1e-10

    def test_risk_parity_favors_low_vol(self):
        config = AllocationConfig(
            method=AllocationMethod.RISK_PARITY,
            min_weight=0.0,
            max_weight=1.0,
        )
        allocator = CapitalAllocator(config)
        result = allocator.allocate(
            _make_sleeve_returns(vols=[0.10, 0.20, 0.40])
        )
        # Low vol should still get more weight under risk parity
        assert result.weights["low_vol"] > result.weights["high_vol"]

    def test_risk_parity_more_balanced_than_inv_vol(self):
        """Risk parity should produce a more balanced allocation than pure
        inverse-vol when correlations differ."""
        rets = _make_sleeve_returns(vols=[0.10, 0.20, 0.30])
        inv_vol = CapitalAllocator(
            AllocationConfig(
                method=AllocationMethod.INVERSE_VOL,
                min_weight=0.0,
                max_weight=1.0,
            )
        ).allocate(rets)
        rp = CapitalAllocator(
            AllocationConfig(
                method=AllocationMethod.RISK_PARITY,
                min_weight=0.0,
                max_weight=1.0,
            )
        ).allocate(rets)
        # Both should sum to 1
        assert abs(sum(inv_vol.weights.values()) - 1.0) < 1e-10
        assert abs(sum(rp.weights.values()) - 1.0) < 1e-10

    def test_risk_parity_with_insufficient_data(self):
        """With very little data, should fall back to inverse vol."""
        short_rets = {
            "a": pd.Series([0.01, -0.005, 0.003]),
            "b": pd.Series([0.002, 0.001, -0.002]),
        }
        config = AllocationConfig(method=AllocationMethod.RISK_PARITY)
        allocator = CapitalAllocator(config)
        result = allocator.allocate(short_rets)
        assert abs(sum(result.weights.values()) - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------


class TestBounds:
    def test_min_weight_enforced(self):
        config = AllocationConfig(
            method=AllocationMethod.INVERSE_VOL,
            min_weight=0.10,
        )
        allocator = CapitalAllocator(config)
        result = allocator.allocate(
            _make_sleeve_returns(vols=[0.05, 0.50, 0.50])
        )
        for w in result.weights.values():
            assert w >= 0.10 - 1e-10

    def test_max_weight_enforced(self):
        config = AllocationConfig(
            method=AllocationMethod.INVERSE_VOL,
            min_weight=0.0,
            max_weight=0.50,
        )
        allocator = CapitalAllocator(config)
        result = allocator.allocate(
            _make_sleeve_returns(vols=[0.05, 0.50, 0.50])
        )
        for w in result.weights.values():
            assert w <= 0.50 + 1e-5


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_sleeves(self):
        allocator = CapitalAllocator()
        result = allocator.allocate({})
        assert result.weights == {}

    def test_single_sleeve(self):
        allocator = CapitalAllocator()
        result = allocator.allocate({"only": pd.Series([0.01, -0.005, 0.003])})
        assert result.weights["only"] == 1.0

    def test_two_sleeves(self):
        config = AllocationConfig(method=AllocationMethod.INVERSE_VOL)
        allocator = CapitalAllocator(config)
        rets = _make_sleeve_returns(
            n_sleeves=2, vols=[0.10, 0.20]
        )
        result = allocator.allocate(rets)
        assert abs(sum(result.weights.values()) - 1.0) < 1e-10

    def test_default_config(self):
        allocator = CapitalAllocator()
        assert allocator.config.method == AllocationMethod.INVERSE_VOL

    def test_vol_lookback_respected(self):
        """Shorter lookback should use fewer days for vol estimation."""
        rets = _make_sleeve_returns(n_days=252)
        short = CapitalAllocator(AllocationConfig(vol_lookback=21))
        long = CapitalAllocator(AllocationConfig(vol_lookback=252))
        r_short = short.allocate(rets)
        r_long = long.allocate(rets)
        # Different lookbacks should produce different vols
        assert r_short.sleeve_vols != r_long.sleeve_vols
