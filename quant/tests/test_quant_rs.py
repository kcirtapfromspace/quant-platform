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
import random
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


def _spy_closes(n: int = 100, seed: int = 42) -> list[float]:
    """Synthetic close price series (random-walk-like)."""
    random.seed(seed)
    prices = [150.0]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1.0 + random.gauss(0, 0.01)))
    return prices


def _spy_volumes(n: int = 100, seed: int = 7) -> list[float]:
    random.seed(seed)
    return [float(1_000_000 + random.randint(-200_000, 200_000)) for _ in range(n)]


def _nan_or_close(a: float, b: float, tol: float = 1e-9) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isnan(a) or math.isnan(b):
        return False
    return abs(a - b) <= tol * max(abs(b), 1.0)


def _assert_parity(rust: list[float], py: list[float], tol: float = 1e-9, label: str = "") -> None:
    assert len(rust) == len(py), f"{label}: length mismatch {len(rust)} vs {len(py)}"
    for i, (r, p) in enumerate(zip(rust, py)):
        assert _nan_or_close(r, p, tol), (
            f"{label}[{i}]: Rust={r!r} vs Python={p!r} (diff={abs(r-p) if not (math.isnan(r) or math.isnan(p)) else 'nan-mismatch'})"
        )


# ─── Pure-Python reference implementations ───────────────────────────────────
# These mirror the Rust kernel algorithms exactly so parity tests are valid.

def _py_returns(prices: list[float]) -> list[float]:
    out = [math.nan]
    for i in range(1, len(prices)):
        prev = prices[i - 1]
        out.append((prices[i] - prev) / prev if prev != 0.0 else math.nan)
    return out


def _py_log_returns(prices: list[float]) -> list[float]:
    out = [math.nan]
    for i in range(1, len(prices)):
        prev, curr = prices[i - 1], prices[i]
        out.append(math.log(curr / prev) if prev > 0.0 and curr > 0.0 else math.nan)
    return out


def _py_rolling_mean(prices: list[float], period: int) -> list[float]:
    n = len(prices)
    out = [math.nan] * n
    for i in range(period - 1, n):
        out[i] = sum(prices[i - period + 1 : i + 1]) / period
    return out


def _py_rolling_std(prices: list[float], period: int) -> list[float]:
    n = len(prices)
    out = [math.nan] * n
    for i in range(period - 1, n):
        window = prices[i - period + 1 : i + 1]
        mean = sum(window) / period
        var = sum((x - mean) ** 2 for x in window) / (period - 1)
        out[i] = math.sqrt(var)
    return out


def _py_ema(prices: list[float], span: int) -> list[float]:
    alpha = 2.0 / (span + 1.0)
    out = [prices[0]]
    for p in prices[1:]:
        out.append(alpha * p + (1.0 - alpha) * out[-1])
    return out


def _py_macd(prices: list[float], fast: int, slow: int) -> list[float]:
    ef = _py_ema(prices, fast)
    es = _py_ema(prices, slow)
    return [f - s for f, s in zip(ef, es)]


def _py_macd_signal(prices: list[float], fast: int, slow: int, signal: int) -> list[float]:
    return _py_ema(_py_macd(prices, fast, slow), signal)


def _py_macd_histogram(prices: list[float], fast: int, slow: int, signal: int) -> list[float]:
    macd = _py_macd(prices, fast, slow)
    sig = _py_ema(macd, signal)
    return [m - s for m, s in zip(macd, sig)]


def _py_rsi(prices: list[float], period: int) -> list[float]:
    """RSI matching the Rust EWM-from-first-diff implementation."""
    n = len(prices)
    out = [math.nan] * n
    alpha = 1.0 / period
    avg_gain = avg_loss = 0.0
    count = 0
    for i in range(1, n):
        d = prices[i] - prices[i - 1]
        gain = d if d > 0.0 else 0.0
        loss = -d if d < 0.0 else 0.0
        count += 1
        if count == 1:
            avg_gain, avg_loss = gain, loss
        else:
            avg_gain = alpha * gain + (1.0 - alpha) * avg_gain
            avg_loss = alpha * loss + (1.0 - alpha) * avg_loss
        if count >= period:
            if avg_gain == 0.0 and avg_loss == 0.0:
                out[i] = 50.0
            elif avg_loss == 0.0:
                out[i] = 100.0
            elif avg_gain == 0.0:
                out[i] = 0.0
            else:
                out[i] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return out


def _py_bb_mid(prices: list[float], period: int) -> list[float]:
    return _py_rolling_mean(prices, period)


def _py_bb_upper(prices: list[float], period: int, num_std: float) -> list[float]:
    mid = _py_rolling_mean(prices, period)
    std = _py_rolling_std(prices, period)
    return [m + num_std * s if not (math.isnan(m) or math.isnan(s)) else math.nan
            for m, s in zip(mid, std)]


def _py_bb_lower(prices: list[float], period: int, num_std: float) -> list[float]:
    mid = _py_rolling_mean(prices, period)
    std = _py_rolling_std(prices, period)
    return [m - num_std * s if not (math.isnan(m) or math.isnan(s)) else math.nan
            for m, s in zip(mid, std)]


def _py_bb_bandwidth(prices: list[float], period: int, num_std: float) -> list[float]:
    mid = _py_rolling_mean(prices, period)
    std = _py_rolling_std(prices, period)
    out = []
    for m, s in zip(mid, std):
        if math.isnan(m) or math.isnan(s) or m == 0.0:
            out.append(math.nan)
        else:
            out.append(2.0 * num_std * s / m)
    return out


def _py_volume_sma(volume: list[float], period: int) -> list[float]:
    return _py_rolling_mean(volume, period)


def _py_volume_ratio(volume: list[float], period: int) -> list[float]:
    sma = _py_rolling_mean(volume, period)
    out = []
    for v, s in zip(volume, sma):
        if math.isnan(s) or s == 0.0:
            out.append(math.nan)
        else:
            out.append(v / s)
    return out


# ─── quant_rs.features — basic correctness ───────────────────────────────────

class TestFeatures:
    def test_returns_length(self):
        r = quant_rs.features.returns(PRICES_UP)
        assert len(r) == len(PRICES_UP)

    def test_returns_first_is_nan(self):
        r = quant_rs.features.returns(PRICES_UP)
        assert math.isnan(r[0])

    def test_returns_value(self):
        r = quant_rs.features.returns([100.0, 110.0, 99.0])
        assert _nan_or_close(r[1], 0.1)
        assert _nan_or_close(r[2], (99.0 - 110.0) / 110.0)

    def test_log_returns_value(self):
        r = quant_rs.features.log_returns([100.0, 110.0])
        assert math.isnan(r[0])
        assert _nan_or_close(r[1], math.log(110.0 / 100.0))

    def test_rolling_mean(self):
        r = quant_rs.features.rolling_mean(PRICES_UP, 3)
        assert math.isnan(r[0])
        assert math.isnan(r[1])
        assert _nan_or_close(r[2], 2.0)  # mean(1,2,3) = 2

    def test_rolling_std_warm_up(self):
        r = quant_rs.features.rolling_std(PRICES_UP, 20)
        assert all(math.isnan(x) for x in r[:19])
        assert not math.isnan(r[19])

    def test_ema_span1_identity(self):
        e = quant_rs.features.ema(PRICES_UP, 1)
        for a, b in zip(e, PRICES_UP):
            assert _nan_or_close(a, b)

    def test_rsi_all_gains_is_100(self):
        r = quant_rs.features.rsi(PRICES_UP, 14)
        assert all(math.isnan(x) for x in r[:14])
        assert _nan_or_close(r[14], 100.0)

    def test_rsi_all_losses_is_0(self):
        r = quant_rs.features.rsi(PRICES_DOWN, 14)
        assert _nan_or_close(r[14], 0.0)

    def test_macd_length(self):
        m = quant_rs.features.macd(PRICES_UP, 12, 26)
        assert len(m) == len(PRICES_UP)

    def test_bb_mid_equals_rolling_mean(self):
        mid = quant_rs.features.bb_mid(PRICES_UP, 20)
        rm = quant_rs.features.rolling_mean(PRICES_UP, 20)
        for a, b in zip(mid, rm):
            assert _nan_or_close(a, b)

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


# ─── Numerical parity — all 15 built-in features ─────────────────────────────

class TestNumericalParity:
    """Cross-validate Rust output against pure-Python reference implementations.

    Each test compares quant_rs kernel output against a hand-coded Python
    function that mirrors the Rust algorithm.  Tolerance is 1e-9 (relative).
    """

    @pytest.fixture
    def closes(self):
        return _spy_closes(100, seed=42)

    @pytest.fixture
    def volumes(self):
        return _spy_volumes(100, seed=7)

    # 1. returns
    def test_returns_parity(self, closes):
        _assert_parity(quant_rs.features.returns(closes), _py_returns(closes), label="returns")

    # 2. log_returns
    def test_log_returns_parity(self, closes):
        _assert_parity(
            quant_rs.features.log_returns(closes), _py_log_returns(closes), label="log_returns"
        )

    # 3. rolling_mean (period=20)
    def test_rolling_mean_parity(self, closes):
        _assert_parity(
            quant_rs.features.rolling_mean(closes, 20),
            _py_rolling_mean(closes, 20),
            label="rolling_mean_20",
        )

    # 4. rolling_std (period=20)
    def test_rolling_std_parity(self, closes):
        _assert_parity(
            quant_rs.features.rolling_std(closes, 20),
            _py_rolling_std(closes, 20),
            label="rolling_std_20",
        )

    # 5. ema (span=12)
    def test_ema_parity(self, closes):
        _assert_parity(
            quant_rs.features.ema(closes, 12),
            _py_ema(closes, 12),
            label="ema_12",
        )

    # 6. rsi (period=14)
    def test_rsi_parity(self, closes):
        _assert_parity(
            quant_rs.features.rsi(closes, 14),
            _py_rsi(closes, 14),
            label="rsi_14",
        )

    # 7. macd (fast=12, slow=26)
    def test_macd_parity(self, closes):
        _assert_parity(
            quant_rs.features.macd(closes, 12, 26),
            _py_macd(closes, 12, 26),
            label="macd_12_26",
        )

    # 8. macd_signal (fast=12, slow=26, signal=9)
    def test_macd_signal_parity(self, closes):
        _assert_parity(
            quant_rs.features.macd_signal(closes, 12, 26, 9),
            _py_macd_signal(closes, 12, 26, 9),
            label="macd_signal_12_26_9",
        )

    # 9. macd_histogram (fast=12, slow=26, signal=9)
    def test_macd_histogram_parity(self, closes):
        _assert_parity(
            quant_rs.features.macd_histogram(closes, 12, 26, 9),
            _py_macd_histogram(closes, 12, 26, 9),
            label="macd_hist_12_26_9",
        )

    # 10. bb_mid (period=20)
    def test_bb_mid_parity(self, closes):
        _assert_parity(
            quant_rs.features.bb_mid(closes, 20),
            _py_bb_mid(closes, 20),
            label="bb_mid_20",
        )

    # 11. bb_upper (period=20, num_std=2.0)
    def test_bb_upper_parity(self, closes):
        _assert_parity(
            quant_rs.features.bb_upper(closes, 20, 2.0),
            _py_bb_upper(closes, 20, 2.0),
            label="bb_upper_20",
        )

    # 12. bb_lower (period=20, num_std=2.0)
    def test_bb_lower_parity(self, closes):
        _assert_parity(
            quant_rs.features.bb_lower(closes, 20, 2.0),
            _py_bb_lower(closes, 20, 2.0),
            label="bb_lower_20",
        )

    # 13. bb_bandwidth (period=20, num_std=2.0)
    def test_bb_bandwidth_parity(self, closes):
        _assert_parity(
            quant_rs.features.bb_bandwidth(closes, 20, 2.0),
            _py_bb_bandwidth(closes, 20, 2.0),
            label="bb_bandwidth_20",
        )

    # 14. volume_sma (period=20)
    def test_volume_sma_parity(self, volumes):
        _assert_parity(
            quant_rs.features.volume_sma(volumes, 20),
            _py_volume_sma(volumes, 20),
            label="volume_sma_20",
        )

    # 15. volume_ratio (period=20)
    def test_volume_ratio_parity(self, volumes):
        _assert_parity(
            quant_rs.features.volume_ratio(volumes, 20),
            _py_volume_ratio(volumes, 20),
            label="volume_ratio_20",
        )
