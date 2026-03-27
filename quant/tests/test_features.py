"""Unit tests for the feature engine (QUA-5)."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quant.data.ingest.base import OHLCVRecord
from quant.data.storage.duckdb import MarketDataStore
from quant.features.built_in import (
    BollingerBandwidth,
    BollingerLower,
    BollingerMid,
    BollingerUpper,
    LogReturns,
    MACD,
    MACDHistogram,
    MACDSignal,
    Returns,
    RollingMean,
    RollingStd,
    RSI,
    VolumeSMA,
    VolumeRatio,
    DEFAULT_REGISTRY,
)
from quant.features.cache import InMemoryFeatureCache
from quant.features.engine import FeatureEngine
from quant.features.registry import FeatureRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(closes: list[float], volumes: list[float] | None = None) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame for testing."""
    n = len(closes)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    if volumes is None:
        volumes = [1_000_000.0] * n
    return pd.DataFrame(
        {
            "symbol": ["AAPL"] * n,
            "date": dates,
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "volume": volumes,
            "adj_close": closes,
        }
    )


def _make_store_with_data() -> MarketDataStore:
    closes = [100.0 + i * 0.5 for i in range(60)]
    store = MarketDataStore(":memory:")
    records = [
        OHLCVRecord(
            symbol="AAPL",
            date=date(2024, 1, 1) + pd.Timedelta(days=i),
            open=closes[i],
            high=closes[i] * 1.01,
            low=closes[i] * 0.99,
            close=closes[i],
            volume=1_000_000.0,
            adj_close=closes[i],
        )
        for i in range(60)
    ]
    store.upsert(records)
    return store


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------

class TestReturns:
    def test_name(self):
        assert Returns().name == "returns"

    def test_first_value_is_nan(self):
        df = _make_df([100.0, 105.0, 110.0])
        result = Returns().compute(df)
        assert pd.isna(result.iloc[0])

    def test_simple_return(self):
        df = _make_df([100.0, 110.0])
        result = Returns().compute(df)
        assert pytest.approx(result.iloc[1], rel=1e-6) == 0.1

    def test_length_matches_input(self):
        closes = [100.0 + i for i in range(20)]
        df = _make_df(closes)
        result = Returns().compute(df)
        assert len(result) == len(closes)

    def test_log_returns_name(self):
        assert LogReturns().name == "log_returns"

    def test_log_returns_value(self):
        df = _make_df([100.0, np.e * 100.0])
        result = LogReturns().compute(df)
        assert pytest.approx(result.iloc[1], rel=1e-4) == 1.0


# ---------------------------------------------------------------------------
# Rolling statistics
# ---------------------------------------------------------------------------

class TestRollingMean:
    def test_name(self):
        assert RollingMean(20).name == "rolling_mean_20"

    def test_warmup_nans(self):
        df = _make_df([float(i) for i in range(1, 21)])
        result = RollingMean(5).compute(df)
        assert result.iloc[:4].isna().all()
        assert not pd.isna(result.iloc[4])

    def test_correct_value(self):
        closes = [1.0, 2.0, 3.0, 4.0, 5.0]
        df = _make_df(closes)
        result = RollingMean(5).compute(df)
        assert pytest.approx(result.iloc[4]) == 3.0

    def test_invalid_period(self):
        with pytest.raises(ValueError):
            RollingMean(0)


class TestRollingStd:
    def test_name(self):
        assert RollingStd(20).name == "rolling_std_20"

    def test_warmup_nans(self):
        df = _make_df([float(i) for i in range(1, 21)])
        result = RollingStd(5).compute(df)
        assert result.iloc[:4].isna().all()

    def test_constant_series_std_zero(self):
        df = _make_df([10.0] * 10)
        result = RollingStd(5).compute(df)
        assert pytest.approx(result.iloc[4]) == 0.0

    def test_invalid_period(self):
        with pytest.raises(ValueError):
            RollingStd(1)


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

class TestRSI:
    def test_name(self):
        assert RSI(14).name == "rsi_14"

    def test_warmup_nans(self):
        closes = [float(i) for i in range(1, 20)]
        df = _make_df(closes)
        result = RSI(14).compute(df)
        # First 14 values should be NaN (EWM with min_periods=14)
        assert result.iloc[:13].isna().all()

    def test_all_gains_gives_high_rsi(self):
        closes = [100.0 + i for i in range(30)]
        df = _make_df(closes)
        result = RSI(14).compute(df)
        valid = result.dropna()
        assert (valid > 90).all(), f"Expected RSI > 90 for all-up series, got {valid.values}"

    def test_all_losses_gives_low_rsi(self):
        closes = [200.0 - i for i in range(30)]
        df = _make_df(closes)
        result = RSI(14).compute(df)
        valid = result.dropna()
        assert (valid < 10).all(), f"Expected RSI < 10 for all-down series, got {valid.values}"

    def test_rsi_in_range(self):
        rng = np.random.default_rng(42)
        closes = list(100.0 + rng.standard_normal(100).cumsum())
        df = _make_df(closes)
        result = RSI(14).compute(df)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_invalid_period(self):
        with pytest.raises(ValueError):
            RSI(1)


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

class TestMACD:
    def test_name(self):
        assert MACD(12, 26).name == "macd_12_26"

    def test_macd_signal_name(self):
        assert MACDSignal(12, 26, 9).name == "macd_signal_12_26_9"

    def test_macd_hist_name(self):
        assert MACDHistogram(12, 26, 9).name == "macd_hist_12_26_9"

    def test_histogram_equals_macd_minus_signal(self):
        rng = np.random.default_rng(7)
        closes = list(100.0 + rng.standard_normal(60).cumsum())
        df = _make_df(closes)
        macd = MACD(12, 26).compute(df)
        signal = MACDSignal(12, 26, 9).compute(df)
        hist = MACDHistogram(12, 26, 9).compute(df)
        pd.testing.assert_series_equal(
            hist.reset_index(drop=True),
            (macd - signal).reset_index(drop=True),
            check_names=False,
            atol=1e-10,
        )

    def test_invalid_fast_slow(self):
        with pytest.raises(ValueError):
            MACD(26, 12)


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

class TestBollingerBands:
    def test_names(self):
        assert BollingerMid(20).name == "bb_mid_20"
        assert BollingerUpper(20).name == "bb_upper_20"
        assert BollingerLower(20).name == "bb_lower_20"
        assert BollingerBandwidth(20).name == "bb_bandwidth_20"

    def test_upper_above_mid_above_lower(self):
        rng = np.random.default_rng(99)
        closes = list(100.0 + rng.standard_normal(40).cumsum())
        df = _make_df(closes)
        upper = BollingerUpper(20).compute(df).dropna()
        mid = BollingerMid(20).compute(df).dropna()
        lower = BollingerLower(20).compute(df).dropna()
        assert (upper >= mid).all()
        assert (mid >= lower).all()

    def test_bandwidth_positive(self):
        rng = np.random.default_rng(5)
        closes = list(100.0 + rng.standard_normal(40).cumsum())
        df = _make_df(closes)
        bw = BollingerBandwidth(20).compute(df).dropna()
        assert (bw >= 0).all()

    def test_constant_prices_zero_bandwidth(self):
        df = _make_df([100.0] * 30)
        bw = BollingerBandwidth(20).compute(df).dropna()
        assert pytest.approx(bw.values, abs=1e-9) == [0.0] * len(bw)


# ---------------------------------------------------------------------------
# Volume metrics
# ---------------------------------------------------------------------------

class TestVolumeMetrics:
    def test_names(self):
        assert VolumeSMA(20).name == "volume_sma_20"
        assert VolumeRatio(20).name == "volume_ratio_20"

    def test_volume_sma_warmup(self):
        df = _make_df([100.0] * 25, volumes=[1e6] * 25)
        result = VolumeSMA(5).compute(df)
        assert result.iloc[:4].isna().all()
        assert pytest.approx(result.iloc[4]) == 1e6

    def test_volume_ratio_double_volume(self):
        vols = [1e6] * 25
        vols[-1] = 2e6
        df = _make_df([100.0] * 25, volumes=vols)
        result = VolumeRatio(20).compute(df)
        # Last bar volume is roughly 2x the average (window includes the spike itself)
        assert result.iloc[-1] > 1.0


# ---------------------------------------------------------------------------
# FeatureRegistry
# ---------------------------------------------------------------------------

class TestFeatureRegistry:
    def test_register_and_get(self):
        reg = FeatureRegistry()
        feat = Returns()
        reg.register(feat)
        assert reg.get("returns") is feat

    def test_duplicate_register_raises(self):
        reg = FeatureRegistry()
        reg.register(Returns())
        with pytest.raises(ValueError):
            reg.register(Returns())

    def test_get_missing_raises(self):
        reg = FeatureRegistry()
        with pytest.raises(KeyError):
            reg.get("nonexistent")

    def test_contains(self):
        reg = FeatureRegistry()
        reg.register(Returns())
        assert "returns" in reg
        assert "rsi_14" not in reg

    def test_default_registry_has_expected_features(self):
        names = DEFAULT_REGISTRY.names()
        for expected in ["returns", "rsi_14", "macd_12_26", "bb_upper_20", "volume_sma_20"]:
            assert expected in names


# ---------------------------------------------------------------------------
# FeatureEngine (integration with MarketDataStore)
# ---------------------------------------------------------------------------

class TestFeatureEngine:
    def setup_method(self):
        self.store = _make_store_with_data()
        self.engine = FeatureEngine(self.store, DEFAULT_REGISTRY)

    def test_compute_returns_for_symbol(self):
        result = self.engine.compute(
            symbols=["AAPL"],
            features=["returns"],
            start=date(2024, 1, 1),
            end=date(2024, 2, 29),
        )
        assert "AAPL" in result
        assert "returns" in result["AAPL"]
        assert isinstance(result["AAPL"]["returns"], pd.Series)

    def test_missing_symbol_omitted(self):
        result = self.engine.compute(
            symbols=["ZZZZ"],
            features=["returns"],
            start=date(2024, 1, 1),
            end=date(2024, 2, 29),
        )
        assert "ZZZZ" not in result

    def test_unknown_feature_raises(self):
        with pytest.raises(KeyError):
            self.engine.compute(
                symbols=["AAPL"],
                features=["nonexistent_feature"],
                start=date(2024, 1, 1),
                end=date(2024, 2, 29),
            )

    def test_cache_hit_avoids_recomputation(self):
        cache = InMemoryFeatureCache()
        engine = FeatureEngine(self.store, DEFAULT_REGISTRY, cache=cache)

        start, end = date(2024, 1, 1), date(2024, 2, 29)
        engine.compute(["AAPL"], ["returns"], start, end)
        assert len(cache) == 1

        # Second call should hit cache (len stays at 1)
        engine.compute(["AAPL"], ["returns"], start, end)
        assert len(cache) == 1

    def test_compute_dataframe(self):
        df = self.engine.compute_dataframe(
            symbol="AAPL",
            features=["returns", "rsi_14"],
            start=date(2024, 1, 1),
            end=date(2024, 2, 29),
        )
        assert "date" in df.columns
        assert "returns" in df.columns
        assert "rsi_14" in df.columns

    def test_multiple_features_computed(self):
        features = ["returns", "rsi_14", "macd_12_26", "bb_upper_20"]
        result = self.engine.compute(
            symbols=["AAPL"],
            features=features,
            start=date(2024, 1, 1),
            end=date(2024, 2, 29),
        )
        for feat in features:
            assert feat in result["AAPL"]


# ---------------------------------------------------------------------------
# InMemoryFeatureCache
# ---------------------------------------------------------------------------

class TestInMemoryFeatureCache:
    def test_set_and_get(self):
        cache = InMemoryFeatureCache()
        series = pd.Series([1.0, 2.0, 3.0])
        cache.set("test_key", series)
        result = cache.get("test_key")
        assert result is not None
        pd.testing.assert_series_equal(result, series)

    def test_miss_returns_none(self):
        cache = InMemoryFeatureCache()
        assert cache.get("missing") is None

    def test_delete(self):
        cache = InMemoryFeatureCache()
        cache.set("key", pd.Series([1.0]))
        cache.delete("key")
        assert cache.get("key") is None

    def test_clear(self):
        cache = InMemoryFeatureCache()
        cache.set("a", pd.Series([1.0]))
        cache.set("b", pd.Series([2.0]))
        cache.clear()
        assert len(cache) == 0

    def test_make_key(self):
        cache = InMemoryFeatureCache()
        key = cache.make_key("AAPL", "rsi_14", date(2024, 1, 1), date(2024, 12, 31))
        assert key == "features:AAPL:rsi_14:2024-01-01:2024-12-31"
