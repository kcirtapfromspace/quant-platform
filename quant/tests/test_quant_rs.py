"""Integration tests validating Rust kernels via PyO3 (quant_rs).

These tests import the compiled Rust extension and verify numerical
parity with the Python implementations.  They are skipped automatically
if quant_rs is not installed (maturin develop has not been run).

To run locally:
    cd quant-rs && maturin develop --release && cd ..
    pytest quant/tests/test_quant_rs.py -v
"""
from __future__ import annotations

import math
import pytest

try:
    import quant_rs
    HAS_QUANT_RS = True
except ImportError:
    HAS_QUANT_RS = False

pytestmark = pytest.mark.skipif(
    not HAS_QUANT_RS,
    reason="quant_rs not installed — run 'cd quant-rs && maturin develop'",
)

# ─── Fixtures ────────────────────────────────────────────────────────────────

PRICES_FLAT = [100.0] * 50
PRICES_UP = [float(i + 1) for i in range(50)]       # 1, 2, ..., 50
PRICES_DOWN = [float(50 - i) for i in range(50)]     # 50, 49, ..., 1


def _nan_or_approx(a: float, b: float, rel: float = 1e-9) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isnan(a) or math.isnan(b):
        return False
    return abs(a - b) <= rel * max(abs(b), 1.0)


# ─── quant_rs.features ───────────────────────────────────────────────────────

class TestFeatures:
    def test_returns_length(self):
        r = quant_rs.features.returns(PRICES_UP)
        assert len(r) == len(PRICES_UP)

    def test_returns_first_is_nan(self):
        r = quant_rs.features.returns(PRICES_UP)
        assert math.isnan(r[0])

    def test_returns_value(self):
        r = quant_rs.features.returns([100.0, 110.0, 99.0])
        assert _nan_or_approx(r[1], 0.1)
        assert _nan_or_approx(r[2], (99.0 - 110.0) / 110.0)

    def test_log_returns_value(self):
        r = quant_rs.features.log_returns([100.0, 110.0])
        assert math.isnan(r[0])
        assert _nan_or_approx(r[1], math.log(110.0 / 100.0))

    def test_rolling_mean(self):
        r = quant_rs.features.rolling_mean(PRICES_UP, 3)
        assert math.isnan(r[0])
        assert math.isnan(r[1])
        assert _nan_or_approx(r[2], 2.0)  # mean(1,2,3) = 2

    def test_rolling_std_warm_up(self):
        r = quant_rs.features.rolling_std(PRICES_UP, 20)
        assert all(math.isnan(x) for x in r[:19])
        assert not math.isnan(r[19])

    def test_ema_span1_identity(self):
        e = quant_rs.features.ema(PRICES_UP, 1)
        for a, b in zip(e, PRICES_UP):
            assert _nan_or_approx(a, b)

    def test_rsi_all_gains_is_100(self):
        r = quant_rs.features.rsi(PRICES_UP, 14)
        assert all(math.isnan(x) for x in r[:14])
        assert _nan_or_approx(r[14], 100.0)

    def test_rsi_all_losses_is_0(self):
        r = quant_rs.features.rsi(PRICES_DOWN, 14)
        assert _nan_or_approx(r[14], 0.0)

    def test_macd_length(self):
        m = quant_rs.features.macd(PRICES_UP, 12, 26)
        assert len(m) == len(PRICES_UP)

    def test_bb_mid_equals_rolling_mean(self):
        mid = quant_rs.features.bb_mid(PRICES_UP, 20)
        rm = quant_rs.features.rolling_mean(PRICES_UP, 20)
        for a, b in zip(mid, rm):
            assert _nan_or_approx(a, b)

    def test_bb_upper_gte_mid(self):
        upper = quant_rs.features.bb_upper(PRICES_UP, 20, 2.0)
        mid = quant_rs.features.bb_mid(PRICES_UP, 20)
        for u, m in zip(upper, mid):
            if not math.isnan(u) and not math.isnan(m):
                assert u >= m

    def test_volume_ratio_warm_up(self):
        v = [float(i + 100) for i in range(30)]
        r = quant_rs.features.volume_ratio(v, 5)
        assert all(math.isnan(x) for x in r[:4])
        assert not math.isnan(r[4])

    def test_version_attribute(self):
        assert quant_rs.__version__ == "0.1.0"


# ─── quant_rs.risk ───────────────────────────────────────────────────────────

class TestRisk:
    def test_fixed_fraction_sizing(self):
        qty = quant_rs.risk.position_size_fixed_fraction(100_000.0, 50.0, 0.02)
        assert abs(qty - 40.0) < 1e-9

    def test_fixed_fraction_zero_price_returns_zero(self):
        qty = quant_rs.risk.position_size_fixed_fraction(100_000.0, 0.0, 0.02)
        assert qty == 0.0

    def test_kelly_fraction(self):
        f = quant_rs.risk.kelly_fraction(0.6, 2.0)
        # 0.6 - 0.4/2 = 0.4
        assert abs(f - 0.4) < 1e-9

    def test_kelly_negative_edge_returns_zero(self):
        f = quant_rs.risk.kelly_fraction(0.3, 1.0)
        assert f == 0.0

    def test_vol_target_sizing(self):
        qty = quant_rs.risk.position_size_vol_target(1_000_000.0, 100.0, 0.2, 0.1)
        # (0.1 * 1M) / (0.2 * 100) = 5000
        assert abs(qty - 5_000.0) < 1e-9

    def test_exposure_check_approved(self):
        result = quant_rs.risk.check_exposure(
            capital=1_000_000.0,
            current_gross=500_000.0,
            current_net=100_000.0,
            order_value=50_000.0,
        )
        assert result is None

    def test_exposure_check_gross_breach(self):
        result = quant_rs.risk.check_exposure(
            capital=1_000_000.0,
            current_gross=1_000_000.0,
            current_net=0.0,
            order_value=600_000.0,
        )
        assert result is not None
        assert "gross" in result.lower()

    def test_circuit_breaker_not_tripped(self):
        assert not quant_rs.risk.is_circuit_tripped(1_000_000.0, 950_000.0, 0.10)

    def test_circuit_breaker_tripped(self):
        assert quant_rs.risk.is_circuit_tripped(1_000_000.0, 890_000.0, 0.10)

    def test_drawdown_calculation(self):
        dd = quant_rs.risk.drawdown(1_000_000.0, 900_000.0)
        assert abs(dd - 0.10) < 1e-9


# ─── Numerical parity with Python implementations ────────────────────────────

class TestNumericalParity:
    """Cross-validate Rust output against Python reference implementations."""

    @pytest.fixture
    def spy_closes(self):
        """Synthetic close price series (100 bars, random-walk-like)."""
        import random
        random.seed(42)
        prices = [150.0]
        for _ in range(99):
            prices.append(prices[-1] * (1.0 + random.gauss(0, 0.01)))
        return prices

    def test_returns_parity(self, spy_closes):
        import numpy as np
        closes = spy_closes
        rust = quant_rs.features.returns(closes)
        py = [float("nan")] + [
            (closes[i] - closes[i - 1]) / closes[i - 1]
            for i in range(1, len(closes))
        ]
        for r, p in zip(rust[1:], py[1:]):
            assert abs(r - p) < 1e-10

    def test_rolling_mean_parity(self, spy_closes):
        import numpy as np
        closes = np.array(spy_closes)
        rust = quant_rs.features.rolling_mean(spy_closes, 20)
        # pandas-style rolling mean with min_periods=20
        py_vals = []
        for i in range(len(closes)):
            if i < 19:
                py_vals.append(float("nan"))
            else:
                py_vals.append(float(closes[i - 19 : i + 1].mean()))
        for r, p in zip(rust[19:], py_vals[19:]):
            assert abs(r - p) < 1e-9
