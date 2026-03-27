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


# ─── Pure-Python reference implementations for signal kernels ─────────────────
# Mirror the Rust algorithms exactly for parity validation.


def _last_valid_py(values: list[float]) -> float | None:
    for v in reversed(values):
        if not math.isnan(v) and not math.isinf(v):
            return v
    return None


def _py_momentum_signal(
    rsi_values: list[float],
    returns: list[float],
    lookback: int = 5,
    return_scale: float = 0.05,
) -> tuple[float, float, float]:
    rsi_val = _last_valid_py(rsi_values)
    if rsi_val is None:
        return (0.0, 0.0, 0.0)

    score = max(-1.0, min(1.0, (rsi_val - 50.0) / 20.0))

    valid_rets = [r for r in returns if not math.isnan(r) and not math.isinf(r)]
    if len(valid_rets) >= lookback:
        recent_abs = sum(abs(r) for r in valid_rets[-lookback:]) / lookback
        confidence = max(0.0, min(1.0, recent_abs / return_scale))
    else:
        confidence = 0.5

    target = max(-1.0, min(1.0, score * confidence))
    return (score, confidence, target)


def _py_mean_reversion_signal(
    bb_mid: list[float],
    bb_upper: list[float],
    bb_lower: list[float],
    returns: list[float],
    num_std: float = 2.0,
) -> tuple[float, float, float]:
    mid = _last_valid_py(bb_mid)
    upper = _last_valid_py(bb_upper)
    lower = _last_valid_py(bb_lower)
    if mid is None or upper is None or lower is None:
        return (0.0, 0.0, 0.0)

    band_width = upper - lower
    if band_width < 1e-12:
        return (0.0, 0.0, 0.0)

    last_ret = _last_valid_py(returns)
    if last_ret is None:
        last_ret = 0.0

    price_approx = mid * (1.0 + last_ret)
    half_band = band_width / 2.0
    z = (price_approx - mid) / half_band if half_band > 0.0 else 0.0

    score = max(-1.0, min(1.0, -z / num_std))
    confidence = max(0.0, min(1.0, abs(z) / num_std))
    target = max(-1.0, min(1.0, score * confidence))
    return (score, confidence, target)


def _py_trend_following_signal(
    macd_hist: list[float],
    fast_ma: list[float],
    slow_ma: list[float],
) -> tuple[float, float, float]:
    hist_valid = [h for h in macd_hist if not math.isnan(h) and not math.isinf(h)]
    fast_val = _last_valid_py(fast_ma)
    slow_val = _last_valid_py(slow_ma)

    if not hist_valid or fast_val is None or slow_val is None:
        return (0.0, 0.0, 0.0)

    last_hist = hist_valid[-1]

    if len(hist_valid) >= 10:
        window = min(20, len(hist_valid))
        w_slice = hist_valid[-window:]
        mean = sum(w_slice) / window
        var = sum((x - mean) ** 2 for x in w_slice) / (window - 1)
        hist_std = math.sqrt(var)
    else:
        mean_abs = sum(abs(h) for h in hist_valid) / len(hist_valid)
        hist_std = mean_abs if mean_abs != 0.0 else 1.0

    if hist_std < 1e-12:
        hist_std = 1.0

    score = max(-1.0, min(1.0, last_hist / hist_std))
    sma_bullish = fast_val > slow_val
    hist_bullish = last_hist > 0.0
    aligned = sma_bullish == hist_bullish
    base_confidence = abs(score)
    confidence = max(0.0, min(1.0, base_confidence * (1.2 if aligned else 0.6)))
    target = max(-1.0, min(1.0, score * confidence))
    return (score, confidence, target)


# ─── quant_rs.signals — parity tests ─────────────────────────────────────────


class TestSignals:
    """Validate Rust signal kernels against pure-Python reference implementations.

    Tolerance: 1e-9 (relative), matching the feature/risk parity tests.
    """

    @pytest.fixture
    def closes(self):
        return _spy_closes(100, seed=42)

    def _assert_signal_parity(
        self,
        rust: tuple[float, float, float],
        py: tuple[float, float, float],
        label: str = "",
        tol: float = 1e-9,
    ) -> None:
        labels = ["score", "confidence", "target_position"]
        for r, p, name in zip(rust, py, labels):
            assert abs(r - p) <= tol * max(abs(p), 1.0), (
                f"{label}.{name}: Rust={r!r} vs Python={p!r} (diff={abs(r - p)!r})"
            )

    # ── momentum_signal ───────────────────────────────────────────────────

    def test_momentum_signal_parity(self, closes):
        rsi = quant_rs.features.rsi(closes, 14)
        rets = quant_rs.features.returns(closes)
        rust = quant_rs.signals.momentum_signal(rsi, rets)
        py = _py_momentum_signal(rsi, rets)
        self._assert_signal_parity(rust, py, "momentum_signal")

    def test_momentum_signal_no_rsi_returns_zero(self):
        rsi = [math.nan] * 20
        rets = [0.01] * 20
        rust = quant_rs.signals.momentum_signal(rsi, rets)
        assert rust == (0.0, 0.0, 0.0)

    def test_momentum_signal_few_returns_half_confidence(self):
        rsi = [60.0]
        rets = [0.01, 0.02]  # < lookback=5
        _, confidence, _ = quant_rs.signals.momentum_signal(rsi, rets)
        assert abs(confidence - 0.5) < 1e-12

    def test_momentum_signal_output_in_range(self, closes):
        rsi = quant_rs.features.rsi(closes, 14)
        rets = quant_rs.features.returns(closes)
        score, confidence, target = quant_rs.signals.momentum_signal(rsi, rets)
        assert -1.0 <= score <= 1.0
        assert 0.0 <= confidence <= 1.0
        assert -1.0 <= target <= 1.0

    def test_momentum_signal_custom_params_parity(self, closes):
        rsi = quant_rs.features.rsi(closes, 14)
        rets = quant_rs.features.returns(closes)
        rust = quant_rs.signals.momentum_signal(rsi, rets, lookback=10, return_scale=0.02)
        py = _py_momentum_signal(rsi, rets, lookback=10, return_scale=0.02)
        self._assert_signal_parity(rust, py, "momentum_signal_custom")

    # ── mean_reversion_signal ─────────────────────────────────────────────

    def test_mean_reversion_signal_parity(self, closes):
        bb_mid = quant_rs.features.bb_mid(closes, 20)
        bb_upper = quant_rs.features.bb_upper(closes, 20, 2.0)
        bb_lower = quant_rs.features.bb_lower(closes, 20, 2.0)
        rets = quant_rs.features.returns(closes)
        rust = quant_rs.signals.mean_reversion_signal(bb_mid, bb_upper, bb_lower, rets)
        py = _py_mean_reversion_signal(bb_mid, bb_upper, bb_lower, rets)
        self._assert_signal_parity(rust, py, "mean_reversion_signal")

    def test_mean_reversion_no_bands_returns_zero(self):
        nans = [math.nan]
        rust = quant_rs.signals.mean_reversion_signal(nans, nans, nans, [0.0])
        assert rust == (0.0, 0.0, 0.0)

    def test_mean_reversion_zero_bandwidth_returns_zero(self):
        rust = quant_rs.signals.mean_reversion_signal([100.0], [100.0], [100.0], [0.0])
        assert rust == (0.0, 0.0, 0.0)

    def test_mean_reversion_at_mid_returns_neutral(self):
        # last_ret=0 → price_approx=mid → z=0 → score=0
        score, confidence, _ = quant_rs.signals.mean_reversion_signal(
            [100.0], [102.0], [98.0], [0.0]
        )
        assert abs(score) < 1e-12
        assert abs(confidence) < 1e-12

    def test_mean_reversion_output_in_range(self, closes):
        bb_mid = quant_rs.features.bb_mid(closes, 20)
        bb_upper = quant_rs.features.bb_upper(closes, 20, 2.0)
        bb_lower = quant_rs.features.bb_lower(closes, 20, 2.0)
        rets = quant_rs.features.returns(closes)
        score, confidence, target = quant_rs.signals.mean_reversion_signal(
            bb_mid, bb_upper, bb_lower, rets
        )
        assert -1.0 <= score <= 1.0
        assert 0.0 <= confidence <= 1.0
        assert -1.0 <= target <= 1.0

    # ── trend_following_signal ────────────────────────────────────────────

    def test_trend_following_signal_parity(self, closes):
        hist = quant_rs.features.macd_histogram(closes, 12, 26, 9)
        fast_ma = quant_rs.features.rolling_mean(closes, 20)
        slow_ma = quant_rs.features.rolling_mean(closes, 50)
        rust = quant_rs.signals.trend_following_signal(hist, fast_ma, slow_ma)
        py = _py_trend_following_signal(hist, fast_ma, slow_ma)
        self._assert_signal_parity(rust, py, "trend_following_signal")

    def test_trend_following_no_hist_returns_zero(self):
        rust = quant_rs.signals.trend_following_signal(
            [math.nan] * 10, [100.0], [99.0]
        )
        assert rust == (0.0, 0.0, 0.0)

    def test_trend_following_positive_hist_bullish(self):
        hist = [float(i + 1) * 0.1 for i in range(20)]
        score, _, _ = quant_rs.signals.trend_following_signal(hist, [105.0], [100.0])
        assert score > 0.0

    def test_trend_following_output_in_range(self, closes):
        hist = quant_rs.features.macd_histogram(closes, 12, 26, 9)
        fast_ma = quant_rs.features.rolling_mean(closes, 20)
        slow_ma = quant_rs.features.rolling_mean(closes, 50)
        score, confidence, target = quant_rs.signals.trend_following_signal(
            hist, fast_ma, slow_ma
        )
        assert -1.0 <= score <= 1.0
        assert 0.0 <= confidence <= 1.0
        assert -1.0 <= target <= 1.0


# ─── Pure-Python backtest reference ──────────────────────────────────────────
# Mirrors quant.backtest.engine.BacktestEngine exactly for parity validation.

import math as _math


def _py_run_backtest(
    adj_close: list[float],
    signals: list[float],
    commission_pct: float = 0.001,
    initial_capital: float = 1.0,
) -> dict:
    """Pure-Python reference implementation mirroring the BacktestEngine."""
    n = len(adj_close)
    assert len(signals) == n

    # Daily returns (pct_change; first bar = 0)
    daily_returns = [0.0]
    for i in range(1, n):
        prev = adj_close[i - 1]
        daily_returns.append((adj_close[i] - prev) / prev if prev != 0.0 else 0.0)

    # Positions = signals shifted by 1 bar
    positions = [0.0] + signals[:-1]

    # Net returns = gross - commission
    net_returns = []
    for i in range(n):
        gross = positions[i] * daily_returns[i]
        delta = 0.0 if i == 0 else abs(positions[i] - positions[i - 1])
        net_returns.append(gross - commission_pct * delta)

    # Equity curve
    equity = [initial_capital * (1.0 + net_returns[0])]
    for i in range(1, n):
        equity.append(equity[-1] * (1.0 + net_returns[i]))

    # Drawdown series
    running_max = equity[0]
    drawdown = []
    for e in equity:
        running_max = max(running_max, e)
        drawdown.append((e - running_max) / running_max if running_max > 0 else 0.0)

    # Trade log
    trades = []
    in_trade = False
    entry_idx = 0
    direction = ""
    trade_acc: list[float] = []
    for i in range(n):
        pos = positions[i]
        ret = net_returns[i]
        if not in_trade:
            if pos != 0.0:
                in_trade = True
                entry_idx = i
                direction = "long" if pos > 0 else "short"
                trade_acc = [ret]
        else:
            if pos != 0.0:
                trade_acc.append(ret)
                direction = "long" if pos > 0 else "short"
            else:
                compound = 1.0
                for r in trade_acc:
                    compound *= (1.0 + r)
                trades.append((entry_idx, i - 1, direction, compound - 1.0))
                in_trade = False
                trade_acc = []
    if in_trade and trade_acc:
        compound = 1.0
        for r in trade_acc:
            compound *= (1.0 + r)
        trades.append((entry_idx, n - 1, direction, compound - 1.0))

    # Metrics
    mean_r = sum(net_returns) / n
    var_r = sum((r - mean_r) ** 2 for r in net_returns) / (n - 1) if n > 1 else 0.0
    std_r = var_r ** 0.5
    sharpe = (mean_r / std_r) * (252 ** 0.5) if std_r > 0 else 0.0

    max_dd = -min(drawdown) if drawdown else 0.0

    total_ret = equity[-1] / initial_capital - 1.0

    years = n / 252.0
    total_ratio = equity[-1] / initial_capital
    cagr_val = (total_ratio ** (1.0 / years) - 1.0) if total_ratio > 0 and years > 0 else 0.0

    trade_rets = [t[3] for t in trades]
    wr = sum(1 for r in trade_rets if r > 0) / len(trade_rets) if trade_rets else 0.0
    gross_profit = sum(r for r in trade_rets if r > 0)
    gross_loss = sum(-r for r in trade_rets if r < 0)
    pf = (float("inf") if gross_profit > 0 else 0.0) if gross_loss == 0 else gross_profit / gross_loss

    return {
        "equity_curve": list(zip(equity, drawdown)),
        "trades": trades,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "cagr": cagr_val,
        "win_rate": wr,
        "profit_factor": pf,
        "total_return": total_ret,
        "n_trades": len(trades),
    }


# ─── quant_rs.backtest — parity tests ────────────────────────────────────────


class TestBacktest:
    """Validate Rust backtest engine against the pure-Python reference.

    All numeric comparisons use 1e-9 relative tolerance (matching feature/risk tests).
    profit_factor == inf is handled as an exact equality check.
    """

    TOL = 1e-9

    def _assert_close(self, rust: float, py: float, label: str = "") -> None:
        if _math.isinf(py) and _math.isinf(rust):
            return
        assert abs(rust - py) <= self.TOL * max(abs(py), 1.0), (
            f"{label}: Rust={rust!r} vs Python={py!r} (diff={abs(rust - py)!r})"
        )

    @pytest.fixture
    def prices(self):
        return _spy_closes(252, seed=99)

    # ── Basic correctness ─────────────────────────────────────────────────

    def test_flat_strategy_equity_flat(self):
        prices = [100.0 + float(i) for i in range(50)]
        signals = [0.0] * 50
        r = quant_rs.backtest.run_backtest(prices, signals, 0.001, 1.0)
        for pv, _ in r["equity_curve"]:
            assert abs(pv - 1.0) < 1e-12, f"equity should be flat, got {pv}"
        assert r["n_trades"] == 0
        assert abs(r["total_return"]) < 1e-12

    def test_no_lookahead_bias(self):
        prices = [100.0, 105.0, 103.0, 108.0]
        signals = [1.0, 1.0, 1.0, 1.0]
        r = quant_rs.backtest.run_backtest(prices, signals, 0.0, 1.0)
        first_pv = r["equity_curve"][0][0]
        assert abs(first_pv - 1.0) < 1e-12, f"equity[0] should equal initial_capital, got {first_pv}"

    def test_always_long_rising_market(self, prices):
        signals = [1.0] * len(prices)
        r = quant_rs.backtest.run_backtest(prices, signals, 0.0, 1.0)
        assert r["total_return"] != 0.0  # some movement in 252 bars

    def test_commission_reduces_return(self, prices):
        signals = [1.0] * len(prices)
        zero = quant_rs.backtest.run_backtest(prices, signals, 0.0, 1.0)
        cost = quant_rs.backtest.run_backtest(prices, signals, 0.001, 1.0)
        assert cost["total_return"] < zero["total_return"]

    def test_single_long_trade(self):
        prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 105.0]
        signals = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
        r = quant_rs.backtest.run_backtest(prices, signals, 0.0, 1.0)
        assert r["n_trades"] == 1
        assert r["trades"][0][2] == "long"
        assert r["trades"][0][3] > 0.0

    def test_single_short_trade(self):
        prices = [105.0, 104.0, 103.0, 102.0, 101.0, 100.0, 100.0]
        signals = [-1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0]
        r = quant_rs.backtest.run_backtest(prices, signals, 0.0, 1.0)
        assert r["n_trades"] == 1
        assert r["trades"][0][2] == "short"
        assert r["trades"][0][3] > 0.0

    def test_equity_curve_length(self, prices):
        signals = [1.0] * len(prices)
        r = quant_rs.backtest.run_backtest(prices, signals, 0.001, 1.0)
        assert len(r["equity_curve"]) == len(prices)

    # ── Numerical parity vs pure-Python reference ─────────────────────────

    @pytest.fixture
    def _parity_pair(self, prices):
        signals = [1.0 if i % 5 < 3 else -1.0 for i in range(len(prices))]
        signals[-10:] = [0.0] * 10  # ensure last trade closes
        rust = quant_rs.backtest.run_backtest(prices, signals, 0.001, 10_000.0)
        py = _py_run_backtest(prices, signals, 0.001, 10_000.0)
        return rust, py

    def test_parity_total_return(self, _parity_pair):
        rust, py = _parity_pair
        self._assert_close(rust["total_return"], py["total_return"], "total_return")

    def test_parity_sharpe_ratio(self, _parity_pair):
        rust, py = _parity_pair
        self._assert_close(rust["sharpe_ratio"], py["sharpe_ratio"], "sharpe_ratio")

    def test_parity_max_drawdown(self, _parity_pair):
        rust, py = _parity_pair
        self._assert_close(rust["max_drawdown"], py["max_drawdown"], "max_drawdown")

    def test_parity_cagr(self, _parity_pair):
        rust, py = _parity_pair
        self._assert_close(rust["cagr"], py["cagr"], "cagr")

    def test_parity_win_rate(self, _parity_pair):
        rust, py = _parity_pair
        self._assert_close(rust["win_rate"], py["win_rate"], "win_rate")

    def test_parity_profit_factor(self, _parity_pair):
        rust, py = _parity_pair
        self._assert_close(rust["profit_factor"], py["profit_factor"], "profit_factor")

    def test_parity_n_trades(self, _parity_pair):
        rust, py = _parity_pair
        assert rust["n_trades"] == py["n_trades"], (
            f"n_trades mismatch: Rust={rust['n_trades']} vs Python={py['n_trades']}"
        )

    def test_parity_equity_curve_all_bars(self, _parity_pair):
        rust, py = _parity_pair
        assert len(rust["equity_curve"]) == len(py["equity_curve"])
        for i, ((rpv, rdd), (ppv, pdd)) in enumerate(
            zip(rust["equity_curve"], py["equity_curve"])
        ):
            self._assert_close(rpv, ppv, f"equity_curve[{i}].portfolio_value")
            self._assert_close(rdd, pdd, f"equity_curve[{i}].drawdown")

    def test_parity_trade_returns(self, _parity_pair):
        rust, py = _parity_pair
        assert len(rust["trades"]) == len(py["trades"]), "trade count mismatch"
        for i, (rt, pt) in enumerate(zip(rust["trades"], py["trades"])):
            # (entry_idx, exit_idx, direction, return)
            assert rt[0] == pt[0], f"trade[{i}] entry_idx mismatch: {rt[0]} vs {pt[0]}"
            assert rt[1] == pt[1], f"trade[{i}] exit_idx mismatch: {rt[1]} vs {pt[1]}"
            assert rt[2] == pt[2], f"trade[{i}] direction mismatch: {rt[2]} vs {pt[2]}"
            self._assert_close(rt[3], pt[3], f"trade[{i}].return")

    # ── Profit-factor edge cases ──────────────────────────────────────────

    def test_profit_factor_all_winners(self):
        prices = [100.0 * (1.01 ** i) for i in range(20)]
        signals = [1.0] * 10 + [0.0] * 10
        r = quant_rs.backtest.run_backtest(prices, signals, 0.0, 1.0)
        assert r["profit_factor"] == float("inf") or r["profit_factor"] > 0

    def test_profit_factor_no_trades(self):
        prices = [100.0] * 20
        signals = [0.0] * 20
        r = quant_rs.backtest.run_backtest(prices, signals, 0.001, 1.0)
        assert r["profit_factor"] == 0.0
