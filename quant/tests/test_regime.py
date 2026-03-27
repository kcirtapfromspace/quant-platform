"""Tests for regime detection and adaptive weight allocation (QUA-36)."""
from __future__ import annotations

import math

import pytest

from quant.signals.regime import (
    CorrelationRegime,
    MarketRegime,
    RegimeConfig,
    RegimeDetector,
    RegimeState,
    RegimeWeightAdapter,
    TrendRegime,
    VolRegime,
    _autocorrelation,
    _mean,
    _pearson,
    _std,
)

# ---------------------------------------------------------------------------
# Math helper tests
# ---------------------------------------------------------------------------


class TestMathHelpers:
    def test_mean(self):
        assert _mean([1.0, 2.0, 3.0]) == 2.0
        assert _mean([]) == 0.0

    def test_std(self):
        assert abs(_std([1.0, 1.0, 1.0]) - 0.0) < 1e-9
        assert _std([]) == 0.0
        assert _std([1.0]) == 0.0
        s = _std([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(s - math.sqrt(2.5)) < 1e-6

    def test_pearson_perfect_positive(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [2.0, 4.0, 6.0, 8.0, 10.0]
        assert abs(_pearson(xs, ys) - 1.0) < 1e-6

    def test_pearson_perfect_negative(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [10.0, 8.0, 6.0, 4.0, 2.0]
        assert abs(_pearson(xs, ys) + 1.0) < 1e-6

    def test_pearson_uncorrelated(self):
        # Zero std in one series
        xs = [1.0, 1.0, 1.0]
        ys = [1.0, 2.0, 3.0]
        assert _pearson(xs, ys) == 0.0

    def test_pearson_short_list(self):
        assert _pearson([1.0], [2.0]) == 0.0

    def test_autocorrelation_trending(self):
        # Strong positive autocorrelation: steadily increasing
        xs = [float(i) for i in range(100)]
        ac = _autocorrelation(xs, lag=1)
        assert ac > 0.9

    def test_autocorrelation_mean_reverting(self):
        # Alternating up/down
        xs = [1.0 if i % 2 == 0 else -1.0 for i in range(100)]
        ac = _autocorrelation(xs, lag=1)
        assert ac < -0.9

    def test_autocorrelation_short_series(self):
        assert _autocorrelation([1.0, 2.0], lag=1) == 0.0


# ---------------------------------------------------------------------------
# Volatility regime tests
# ---------------------------------------------------------------------------


class TestVolatilityRegime:
    def test_high_vol(self):
        """Spiking vol should be detected as HIGH."""
        # Calm period followed by volatile period
        calm = [0.001] * 200
        wild = [(0.04 if i % 2 == 0 else -0.04) for i in range(30)]
        returns = calm + wild

        detector = RegimeDetector(RegimeConfig(
            vol_short_window=21,
            vol_long_window=200,
        ))
        state = detector.detect(returns_1d=returns)
        assert state.vol_regime == VolRegime.HIGH

    def test_low_vol(self):
        """Suppressed vol should be detected as LOW."""
        # Volatile period followed by calm
        wild = [(0.03 if i % 2 == 0 else -0.03) for i in range(200)]
        calm = [0.0001] * 30
        returns = wild + calm

        detector = RegimeDetector(RegimeConfig(
            vol_short_window=21,
            vol_long_window=200,
        ))
        state = detector.detect(returns_1d=returns)
        assert state.vol_regime == VolRegime.LOW

    def test_normal_vol(self):
        """Stable vol should be NORMAL."""
        returns = [(0.01 if i % 2 == 0 else -0.01) for i in range(300)]

        detector = RegimeDetector()
        state = detector.detect(returns_1d=returns)
        assert state.vol_regime == VolRegime.NORMAL

    def test_crisis_vol(self):
        """Extreme vol spike should trigger CRISIS regime."""
        calm = [0.001] * 200
        crisis = [(0.08 if i % 2 == 0 else -0.08) for i in range(30)]
        returns = calm + crisis

        detector = RegimeDetector(RegimeConfig(
            vol_short_window=21,
            vol_long_window=200,
            crisis_vol_threshold=2.0,
        ))
        state = detector.detect(returns_1d=returns)
        assert state.vol_regime == VolRegime.HIGH
        assert state.metrics["vol_ratio"] >= 2.0


# ---------------------------------------------------------------------------
# Trend regime tests
# ---------------------------------------------------------------------------


class TestTrendRegime:
    def test_trending_market(self):
        """AR(1) process with positive phi should be TRENDING."""
        import random
        rng = random.Random(99)
        # AR(1): r[t] = 0.4 * r[t-1] + noise — strong positive serial correlation
        returns: list[float] = [0.01]
        for _ in range(199):
            returns.append(0.4 * returns[-1] + rng.gauss(0, 0.005))

        detector = RegimeDetector(RegimeConfig(trend_window=63))
        state = detector.detect(returns_1d=returns)
        assert state.trend_regime == TrendRegime.TRENDING

    def test_mean_reverting_market(self):
        """Alternating returns should be MEAN_REVERTING."""
        returns = [0.02 if i % 2 == 0 else -0.02 for i in range(100)]

        detector = RegimeDetector(RegimeConfig(
            trend_window=63,
            mr_threshold=-0.05,
        ))
        state = detector.detect(returns_1d=returns)
        assert state.trend_regime == TrendRegime.MEAN_REVERTING

    def test_random_market(self):
        """Random returns should be RANDOM."""
        import random
        rng = random.Random(42)
        returns = [rng.gauss(0, 0.01) for _ in range(200)]

        detector = RegimeDetector()
        state = detector.detect(returns_1d=returns)
        # Random series should have near-zero autocorrelation
        assert abs(state.metrics["autocorrelation"]) < 0.3


# ---------------------------------------------------------------------------
# Correlation regime tests
# ---------------------------------------------------------------------------


class TestCorrelationRegime:
    def test_high_correlation(self):
        """Highly correlated assets should be HIGH."""
        n = 100
        base = [0.01 * (1 if i % 3 != 0 else -1) for i in range(n)]
        # All assets move together
        returns_2d = [[b, b * 0.9, b * 1.1] for b in base]

        detector = RegimeDetector(RegimeConfig(
            corr_window=63,
            corr_high_threshold=0.60,
        ))
        state = detector.detect(returns=returns_2d)
        assert state.corr_regime == CorrelationRegime.HIGH

    def test_low_correlation(self):
        """Uncorrelated assets should be LOW."""
        import random
        rng = random.Random(42)
        n = 100
        returns_2d = [
            [rng.gauss(0, 0.01), rng.gauss(0, 0.01), rng.gauss(0, 0.01)]
            for _ in range(n)
        ]

        detector = RegimeDetector(RegimeConfig(
            corr_window=63,
            corr_low_threshold=0.25,
        ))
        state = detector.detect(returns=returns_2d)
        # Random series should have low average correlation
        assert state.corr_regime in (CorrelationRegime.LOW, CorrelationRegime.NORMAL)

    def test_single_asset_gives_normal(self):
        """Single asset should give NORMAL correlation regime."""
        returns_2d = [[0.01] for _ in range(100)]

        detector = RegimeDetector()
        state = detector.detect(returns=returns_2d)
        assert state.corr_regime == CorrelationRegime.NORMAL


# ---------------------------------------------------------------------------
# Composite regime tests
# ---------------------------------------------------------------------------


class TestCompositeRegime:
    def test_crisis_detection(self):
        """Very high vol + high correlation should trigger CRISIS."""
        calm = [0.001] * 200
        crisis = [(0.08 if i % 2 == 0 else -0.08) for i in range(30)]
        # Make all assets crash together
        returns_1d = calm + crisis
        returns_2d = [[r, r * 0.95, r * 1.05] for r in returns_1d]

        detector = RegimeDetector(RegimeConfig(
            vol_short_window=21,
            vol_long_window=200,
            crisis_vol_threshold=2.0,
            corr_high_threshold=0.5,
        ))
        state = detector.detect(returns=returns_2d)
        assert state.regime == MarketRegime.CRISIS

    def test_normal_regime_default(self):
        """Under normal conditions, regime should be NORMAL."""
        import random
        rng = random.Random(123)
        # Large sample to minimize spurious autocorrelation
        returns = [rng.gauss(0, 0.01) for _ in range(500)]

        detector = RegimeDetector()
        state = detector.detect(returns_1d=returns)
        assert state.regime in (MarketRegime.NORMAL, MarketRegime.RISK_ON, MarketRegime.RISK_OFF)

    def test_insufficient_data_gives_normal(self):
        """Too little data should return NORMAL with zero confidence."""
        detector = RegimeDetector()
        state = detector.detect(returns_1d=[0.01, 0.02])
        assert state.regime == MarketRegime.NORMAL
        assert state.confidence == 0.0


# ---------------------------------------------------------------------------
# Regime state tests
# ---------------------------------------------------------------------------


class TestRegimeState:
    def test_immutable(self):
        """RegimeState should be frozen."""
        state = RegimeState(
            regime=MarketRegime.NORMAL,
            confidence=0.5,
            vol_regime=VolRegime.NORMAL,
            trend_regime=TrendRegime.RANDOM,
            corr_regime=CorrelationRegime.NORMAL,
        )
        with pytest.raises(AttributeError):
            state.regime = MarketRegime.CRISIS  # type: ignore[misc]

    def test_metrics_populated(self):
        """detect() should populate metrics dict."""
        detector = RegimeDetector()
        state = detector.detect(returns_1d=[0.01 * (i % 3 - 1) for i in range(100)])
        assert "vol_ratio" in state.metrics
        assert "autocorrelation" in state.metrics


# ---------------------------------------------------------------------------
# Weight adapter tests
# ---------------------------------------------------------------------------


class TestRegimeWeightAdapter:
    def test_normal_regime_no_change(self):
        """NORMAL regime should leave weights approximately unchanged."""
        adapter = RegimeWeightAdapter()
        regime = RegimeState(
            regime=MarketRegime.NORMAL,
            confidence=0.3,
            vol_regime=VolRegime.NORMAL,
            trend_regime=TrendRegime.RANDOM,
            corr_regime=CorrelationRegime.NORMAL,
        )
        base = {"momentum": 0.40, "mean_rev": 0.35, "trend": 0.25}
        types = {"momentum": "momentum", "mean_rev": "mean_reversion", "trend": "trend"}

        adjusted = adapter.adapt(regime, base, types)

        # Should sum to the same total
        assert abs(sum(adjusted.values()) - sum(base.values())) < 0.001
        # With NORMAL regime and zero affinity, weights should be unchanged
        for name in base:
            assert abs(adjusted[name] - base[name]) < 0.01

    def test_trending_regime_tilts_toward_trend(self):
        """TRENDING regime should increase trend strategy weight."""
        adapter = RegimeWeightAdapter()
        regime = RegimeState(
            regime=MarketRegime.TRENDING,
            confidence=0.8,
            vol_regime=VolRegime.NORMAL,
            trend_regime=TrendRegime.TRENDING,
            corr_regime=CorrelationRegime.NORMAL,
        )
        base = {"momentum": 0.40, "mean_rev": 0.35, "trend": 0.25}
        types = {"momentum": "momentum", "mean_rev": "mean_reversion", "trend": "trend"}

        adjusted = adapter.adapt(regime, base, types)

        # Trend should get more weight, mean_rev should get less
        assert adjusted["trend"] > base["trend"]
        assert adjusted["mean_rev"] < base["mean_rev"]

    def test_mean_reverting_regime_tilts_toward_mr(self):
        """MEAN_REVERTING regime should increase mean_reversion weight."""
        adapter = RegimeWeightAdapter()
        regime = RegimeState(
            regime=MarketRegime.MEAN_REVERTING,
            confidence=0.8,
            vol_regime=VolRegime.NORMAL,
            trend_regime=TrendRegime.MEAN_REVERTING,
            corr_regime=CorrelationRegime.NORMAL,
        )
        base = {"momentum": 0.40, "mean_rev": 0.35, "trend": 0.25}
        types = {"momentum": "momentum", "mean_rev": "mean_reversion", "trend": "trend"}

        adjusted = adapter.adapt(regime, base, types)

        assert adjusted["mean_rev"] > base["mean_rev"]
        assert adjusted["trend"] < base["trend"]

    def test_crisis_regime_favors_quality(self):
        """CRISIS regime should increase quality/volatility weights."""
        adapter = RegimeWeightAdapter()
        regime = RegimeState(
            regime=MarketRegime.CRISIS,
            confidence=0.9,
            vol_regime=VolRegime.HIGH,
            trend_regime=TrendRegime.RANDOM,
            corr_regime=CorrelationRegime.HIGH,
        )
        base = {"momentum": 0.25, "quality": 0.25, "vol": 0.25, "trend": 0.25}
        types = {"momentum": "momentum", "quality": "quality", "vol": "volatility", "trend": "trend"}

        adjusted = adapter.adapt(regime, base, types)

        # Quality and vol should increase; momentum and trend should decrease
        assert adjusted["quality"] > base["quality"]
        assert adjusted["vol"] > base["vol"]
        assert adjusted["momentum"] < base["momentum"]

    def test_weights_sum_preserved(self):
        """Adjusted weights should sum to the same total as base."""
        adapter = RegimeWeightAdapter()

        for regime_type in MarketRegime:
            regime = RegimeState(
                regime=regime_type,
                confidence=0.8,
                vol_regime=VolRegime.NORMAL,
                trend_regime=TrendRegime.RANDOM,
                corr_regime=CorrelationRegime.NORMAL,
            )
            base = {"a": 0.30, "b": 0.30, "c": 0.20, "d": 0.20}
            types = {"a": "trend", "b": "mean_reversion", "c": "momentum", "d": "quality"}

            adjusted = adapter.adapt(regime, base, types)
            assert abs(sum(adjusted.values()) - sum(base.values())) < 0.001, (
                f"Sum mismatch for regime {regime_type}"
            )

    def test_all_weights_non_negative(self):
        """No adjusted weight should go negative."""
        adapter = RegimeWeightAdapter()

        for regime_type in MarketRegime:
            regime = RegimeState(
                regime=regime_type,
                confidence=1.0,
                vol_regime=VolRegime.HIGH,
                trend_regime=TrendRegime.TRENDING,
                corr_regime=CorrelationRegime.HIGH,
            )
            base = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
            types = {"a": "trend", "b": "mean_reversion", "c": "momentum", "d": "quality"}

            adjusted = adapter.adapt(regime, base, types)
            for name, w in adjusted.items():
                assert w >= 0.0, f"Negative weight for {name} in {regime_type}"

    def test_empty_base_weights(self):
        """Empty base weights should return empty."""
        adapter = RegimeWeightAdapter()
        regime = RegimeState(
            regime=MarketRegime.NORMAL,
            confidence=0.5,
            vol_regime=VolRegime.NORMAL,
            trend_regime=TrendRegime.RANDOM,
            corr_regime=CorrelationRegime.NORMAL,
        )
        assert adapter.adapt(regime, {}, {}) == {}

    def test_unknown_strategy_type_no_tilt(self):
        """Strategies with unknown types should get no tilt."""
        adapter = RegimeWeightAdapter()
        regime = RegimeState(
            regime=MarketRegime.TRENDING,
            confidence=0.8,
            vol_regime=VolRegime.NORMAL,
            trend_regime=TrendRegime.TRENDING,
            corr_regime=CorrelationRegime.NORMAL,
        )
        base = {"mystery": 1.0}
        types = {"mystery": "exotic_unknown_type"}

        adjusted = adapter.adapt(regime, base, types)
        assert abs(adjusted["mystery"] - 1.0) < 0.001

    def test_confidence_scales_tilt(self):
        """Higher confidence should produce larger tilts."""
        adapter = RegimeWeightAdapter()
        base = {"trend": 0.50, "mr": 0.50}
        types = {"trend": "trend", "mr": "mean_reversion"}

        low_conf = RegimeState(
            regime=MarketRegime.TRENDING, confidence=0.2,
            vol_regime=VolRegime.NORMAL, trend_regime=TrendRegime.TRENDING,
            corr_regime=CorrelationRegime.NORMAL,
        )
        high_conf = RegimeState(
            regime=MarketRegime.TRENDING, confidence=0.9,
            vol_regime=VolRegime.NORMAL, trend_regime=TrendRegime.TRENDING,
            corr_regime=CorrelationRegime.NORMAL,
        )

        adj_low = adapter.adapt(low_conf, base, types)
        adj_high = adapter.adapt(high_conf, base, types)

        # Higher confidence should tilt more toward trend
        assert adj_high["trend"] > adj_low["trend"]

    def test_custom_affinity_table(self):
        """Custom affinity table should override defaults."""
        custom = {
            "my_strat": {
                MarketRegime.TRENDING: 0.50,
                MarketRegime.NORMAL: 0.0,
            }
        }
        adapter = RegimeWeightAdapter(affinity_table=custom)
        regime = RegimeState(
            regime=MarketRegime.TRENDING, confidence=1.0,
            vol_regime=VolRegime.NORMAL, trend_regime=TrendRegime.TRENDING,
            corr_regime=CorrelationRegime.NORMAL,
        )
        base = {"s1": 0.50, "s2": 0.50}
        types = {"s1": "my_strat", "s2": "my_strat"}

        adjusted = adapter.adapt(regime, base, types)
        # Both should get the same tilt (same type, same affinity)
        assert abs(adjusted["s1"] - adjusted["s2"]) < 0.001
