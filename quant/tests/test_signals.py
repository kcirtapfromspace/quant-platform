"""Unit tests for the signal registry and strategy framework (QUA-6)."""
from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from quant.data.ingest.base import OHLCVRecord
from quant.data.storage.duckdb import MarketDataStore
from quant.features.built_in import DEFAULT_REGISTRY
from quant.features.engine import FeatureEngine
from quant.signals.base import SignalOutput
from quant.signals.registry import SignalRegistry
from quant.signals.strategies import (
    MeanReversionSignal,
    MomentumSignal,
    TrendFollowingSignal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2024, 3, 1, tzinfo=timezone.utc)


def _make_features(closes: list[float], volumes: list[float] | None = None) -> dict[str, pd.Series]:
    """Compute all default features from synthetic close prices."""
    n = len(closes)
    dates = [date(2024, 1, 1) + pd.Timedelta(days=i) for i in range(n)]
    if volumes is None:
        volumes = [1_000_000.0] * n

    store = MarketDataStore(":memory:")
    records = [
        OHLCVRecord(
            symbol="TEST",
            date=dates[i],
            open=closes[i],
            high=closes[i] * 1.01,
            low=closes[i] * 0.99,
            close=closes[i],
            volume=volumes[i],
            adj_close=closes[i],
        )
        for i in range(n)
    ]
    store.upsert(records)
    engine = FeatureEngine(store, DEFAULT_REGISTRY)

    all_features = DEFAULT_REGISTRY.names()
    result = engine.compute(["TEST"], all_features, dates[0], dates[-1])
    return result.get("TEST", {})


def _trending_up(n: int = 60) -> list[float]:
    return [100.0 + i for i in range(n)]


def _trending_down(n: int = 60) -> list[float]:
    return [200.0 - i for i in range(n)]


def _flat(n: int = 60) -> list[float]:
    rng = np.random.default_rng(42)
    return list(100.0 + rng.standard_normal(n) * 0.5)


# ---------------------------------------------------------------------------
# SignalOutput
# ---------------------------------------------------------------------------

class TestSignalOutput:
    def test_valid_construction(self):
        out = SignalOutput(
            symbol="AAPL",
            timestamp=_NOW,
            score=0.5,
            confidence=0.8,
            target_position=0.4,
        )
        assert out.score == 0.5

    def test_invalid_score_raises(self):
        with pytest.raises(ValueError):
            SignalOutput("A", _NOW, score=1.5, confidence=0.5, target_position=0.5)

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError):
            SignalOutput("A", _NOW, score=0.5, confidence=1.1, target_position=0.5)

    def test_invalid_target_position_raises(self):
        with pytest.raises(ValueError):
            SignalOutput("A", _NOW, score=0.5, confidence=0.5, target_position=-1.5)

    def test_metadata_defaults_to_empty_dict(self):
        out = SignalOutput("A", _NOW, 0.0, 0.0, 0.0)
        assert out.metadata == {}


# ---------------------------------------------------------------------------
# SignalRegistry
# ---------------------------------------------------------------------------

class TestSignalRegistry:
    def test_register_and_get(self):
        reg = SignalRegistry()
        sig = MomentumSignal()
        reg.register(sig)
        assert reg.get("momentum") is sig

    def test_duplicate_raises(self):
        reg = SignalRegistry()
        reg.register(MomentumSignal())
        with pytest.raises(ValueError):
            reg.register(MomentumSignal())

    def test_get_missing_raises(self):
        reg = SignalRegistry()
        with pytest.raises(KeyError):
            reg.get("nonexistent")

    def test_contains(self):
        reg = SignalRegistry()
        reg.register(MomentumSignal())
        assert "momentum" in reg
        assert "trend_following" not in reg

    def test_names_sorted(self):
        reg = SignalRegistry()
        reg.register(TrendFollowingSignal())
        reg.register(MomentumSignal())
        reg.register(MeanReversionSignal())
        assert reg.names() == sorted(["momentum", "mean_reversion", "trend_following"])

    def test_unregister(self):
        reg = SignalRegistry()
        reg.register(MomentumSignal())
        reg.unregister("momentum")
        assert "momentum" not in reg

    def test_unregister_missing_raises(self):
        reg = SignalRegistry()
        with pytest.raises(KeyError):
            reg.unregister("nonexistent")

    def test_all_returns_signals(self):
        reg = SignalRegistry()
        reg.register(MomentumSignal())
        reg.register(TrendFollowingSignal())
        assert len(reg.all()) == 2


# ---------------------------------------------------------------------------
# MomentumSignal
# ---------------------------------------------------------------------------

class TestMomentumSignal:
    def test_name(self):
        assert MomentumSignal().name == "momentum"

    def test_required_features(self):
        assert "rsi_14" in MomentumSignal().required_features

    def test_bullish_on_strong_uptrend(self):
        features = _make_features(_trending_up(60))
        out = MomentumSignal().compute("TEST", features, _NOW)
        assert isinstance(out, SignalOutput)
        assert out.score > 0, f"Expected bullish score for uptrend, got {out.score}"

    def test_bearish_on_strong_downtrend(self):
        features = _make_features(_trending_down(60))
        out = MomentumSignal().compute("TEST", features, _NOW)
        assert out.score < 0, f"Expected bearish score for downtrend, got {out.score}"

    def test_score_in_valid_range(self):
        features = _make_features(_flat(60))
        out = MomentumSignal().compute("TEST", features, _NOW)
        assert -1.0 <= out.score <= 1.0
        assert 0.0 <= out.confidence <= 1.0
        assert -1.0 <= out.target_position <= 1.0

    def test_insufficient_data_returns_zero_signal(self):
        # Only 3 bars — not enough for RSI warmup
        features = _make_features([100.0, 101.0, 102.0])
        out = MomentumSignal().compute("TEST", features, _NOW)
        assert out.score == 0.0
        assert out.confidence == 0.0

    def test_symbol_preserved_in_output(self):
        features = _make_features(_trending_up(60))
        out = MomentumSignal().compute("AAPL", features, _NOW)
        assert out.symbol == "AAPL"

    def test_timestamp_preserved(self):
        features = _make_features(_trending_up(60))
        out = MomentumSignal().compute("TEST", features, _NOW)
        assert out.timestamp == _NOW


# ---------------------------------------------------------------------------
# MeanReversionSignal
# ---------------------------------------------------------------------------

class TestMeanReversionSignal:
    def test_name(self):
        assert MeanReversionSignal().name == "mean_reversion"

    def test_required_features_includes_bollinger(self):
        req = MeanReversionSignal().required_features
        assert "bb_mid_20" in req
        assert "bb_upper_20" in req
        assert "bb_lower_20" in req

    def test_output_valid_range(self):
        features = _make_features(_flat(60))
        out = MeanReversionSignal().compute("TEST", features, _NOW)
        assert -1.0 <= out.score <= 1.0
        assert 0.0 <= out.confidence <= 1.0
        assert -1.0 <= out.target_position <= 1.0

    def test_insufficient_data_returns_zero(self):
        features = _make_features([100.0, 101.0])
        out = MeanReversionSignal().compute("TEST", features, _NOW)
        assert out.score == 0.0

    def test_metadata_contains_z_score(self):
        features = _make_features(_flat(60))
        out = MeanReversionSignal().compute("TEST", features, _NOW)
        if out.confidence > 0:
            assert "z_score" in out.metadata


# ---------------------------------------------------------------------------
# TrendFollowingSignal
# ---------------------------------------------------------------------------

class TestTrendFollowingSignal:
    def test_name(self):
        assert TrendFollowingSignal().name == "trend_following"

    def test_required_features(self):
        req = TrendFollowingSignal().required_features
        assert "macd_hist_12_26_9" in req
        assert "rolling_mean_20" in req
        assert "rolling_mean_50" in req

    def test_bullish_on_uptrend(self):
        features = _make_features(_trending_up(80))
        out = TrendFollowingSignal().compute("TEST", features, _NOW)
        # In a strong uptrend, MACD histogram should be positive
        assert out.score >= 0.0, f"Expected non-negative score for uptrend, got {out.score}"

    def test_bearish_on_downtrend(self):
        features = _make_features(_trending_down(80))
        out = TrendFollowingSignal().compute("TEST", features, _NOW)
        assert out.score <= 0.0, f"Expected non-positive score for downtrend, got {out.score}"

    def test_output_valid_range(self):
        features = _make_features(_flat(80))
        out = TrendFollowingSignal().compute("TEST", features, _NOW)
        assert -1.0 <= out.score <= 1.0
        assert 0.0 <= out.confidence <= 1.0
        assert -1.0 <= out.target_position <= 1.0

    def test_insufficient_data_returns_zero(self):
        features = _make_features([100.0 + i for i in range(10)])
        out = TrendFollowingSignal().compute("TEST", features, _NOW)
        assert out.score == 0.0

    def test_invalid_ma_params(self):
        with pytest.raises(ValueError):
            TrendFollowingSignal(fast_ma=50, slow_ma=20)

    def test_metadata_contains_alignment(self):
        features = _make_features(_trending_up(80))
        out = TrendFollowingSignal().compute("TEST", features, _NOW)
        if out.confidence > 0:
            assert "sma_aligned" in out.metadata


# ---------------------------------------------------------------------------
# Integration: SignalRegistry + FeatureEngine
# ---------------------------------------------------------------------------

class TestSignalRegistryIntegration:
    def test_run_all_signals_on_uptrend(self):
        features = _make_features(_trending_up(80))
        reg = SignalRegistry()
        reg.register(MomentumSignal())
        reg.register(MeanReversionSignal())
        reg.register(TrendFollowingSignal())

        outputs = []
        for sig in reg.all():
            out = sig.compute("TEST", features, _NOW)
            outputs.append(out)
            assert isinstance(out, SignalOutput)
            assert -1.0 <= out.score <= 1.0

        assert len(outputs) == 3
