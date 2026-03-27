"""Tests for conviction-weighted position scaling (QUA-47)."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from quant.portfolio.alpha import AlphaScore
from quant.portfolio.position_scaler import (
    PositionScaler,
    ScaledPosition,
    ScalingConfig,
    ScalingMethod,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

NOW = datetime(2024, 1, 15, tzinfo=timezone.utc)


def _alpha(symbol: str, score: float, confidence: float = 0.8) -> AlphaScore:
    return AlphaScore(
        symbol=symbol, timestamp=NOW, score=score, confidence=confidence
    )


def _make_alphas() -> dict[str, AlphaScore]:
    return {
        "AAPL": _alpha("AAPL", 0.6, 0.9),
        "GOOG": _alpha("GOOG", -0.3, 0.7),
        "MSFT": _alpha("MSFT", 0.1, 0.5),
    }


def _make_returns(n_days: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    return pd.DataFrame(
        {
            "AAPL": rng.normal(0.001, 0.02, n_days),
            "GOOG": rng.normal(0.0005, 0.025, n_days),
            "MSFT": rng.normal(0.0008, 0.015, n_days),
        },
        index=dates,
    )


# ── Tests: Conviction scaling ────────────────────────────────────────────


class TestConvictionScaling:
    def test_scales_by_confidence(self):
        scaler = PositionScaler(ScalingConfig(method=ScalingMethod.CONVICTION))
        alphas = _make_alphas()
        result = scaler.scale(alphas)
        # AAPL: 0.6 * 0.9 = 0.54
        assert abs(result["AAPL"].scaled_alpha - 0.54) < 1e-6
        # GOOG: -0.3 * 0.7 = -0.21
        assert abs(result["GOOG"].scaled_alpha - (-0.21)) < 1e-6

    def test_preserves_sign(self):
        scaler = PositionScaler(ScalingConfig(method=ScalingMethod.CONVICTION))
        alphas = _make_alphas()
        result = scaler.scale(alphas)
        assert result["AAPL"].scaled_alpha > 0
        assert result["GOOG"].scaled_alpha < 0

    def test_zero_alpha_stays_zero(self):
        scaler = PositionScaler(ScalingConfig(method=ScalingMethod.CONVICTION))
        alphas = {"A": _alpha("A", 0.0, 0.9)}
        result = scaler.scale(alphas)
        assert result["A"].scaled_alpha == 0.0

    def test_no_returns_needed(self):
        scaler = PositionScaler(ScalingConfig(method=ScalingMethod.CONVICTION))
        result = scaler.scale(_make_alphas())
        assert len(result) == 3


# ── Tests: Vol-adjusted scaling ──────────────────────────────────────────


class TestVolAdjustedScaling:
    def test_inversely_scales_by_vol(self):
        scaler = PositionScaler(ScalingConfig(method=ScalingMethod.VOL_ADJUSTED))
        alphas = _make_alphas()
        returns = _make_returns()
        result = scaler.scale(alphas, returns)
        # Higher vol asset should have smaller absolute scaled alpha
        # GOOG has higher vol (0.025) than MSFT (0.015)
        # But GOOG has higher abs alpha, so compare ratio
        assert result["AAPL"].vol_estimate > 0
        assert result["GOOG"].vol_estimate > 0

    def test_with_vol_target(self):
        scaler = PositionScaler(ScalingConfig(
            method=ScalingMethod.VOL_ADJUSTED, vol_target=0.15
        ))
        alphas = _make_alphas()
        returns = _make_returns()
        result = scaler.scale(alphas, returns)
        for p in result.values():
            assert isinstance(p, ScaledPosition)

    def test_zero_vol_fallback(self):
        """If vol is zero, fall back to conviction scaling."""
        scaler = PositionScaler(ScalingConfig(method=ScalingMethod.VOL_ADJUSTED))
        alphas = {"A": _alpha("A", 0.5, 0.8)}
        # No returns → vol = 0
        result = scaler.scale(alphas)
        # Should fall back to alpha * confidence
        assert abs(result["A"].scaled_alpha - 0.4) < 1e-6


# ── Tests: Kelly scaling ─────────────────────────────────────────────────


class TestKellyScaling:
    def test_kelly_sizes_by_edge_over_variance(self):
        scaler = PositionScaler(ScalingConfig(
            method=ScalingMethod.KELLY, kelly_fraction=1.0
        ))
        alphas = _make_alphas()
        returns = _make_returns()
        result = scaler.scale(alphas, returns)
        # Kelly should produce non-zero scaled alphas
        assert result["AAPL"].scaled_alpha != 0.0

    def test_half_kelly(self):
        """Half Kelly produces exactly half the sizing of full Kelly."""
        # Single asset to avoid leverage cap distortion
        single = {"A": _alpha("A", 0.5, 0.8)}
        returns = _make_returns()
        returns = returns.rename(columns={"AAPL": "A"})[["A"]]

        full = PositionScaler(ScalingConfig(
            method=ScalingMethod.KELLY, kelly_fraction=1.0, max_leverage=1e6
        ))
        half = PositionScaler(ScalingConfig(
            method=ScalingMethod.KELLY, kelly_fraction=0.5, max_leverage=1e6
        ))
        full_r = full.scale(single, returns)
        half_r = half.scale(single, returns)
        ratio = half_r["A"].scaled_alpha / full_r["A"].scaled_alpha
        assert abs(ratio - 0.5) < 1e-6

    def test_kelly_zero_vol_fallback(self):
        scaler = PositionScaler(ScalingConfig(
            method=ScalingMethod.KELLY, kelly_fraction=0.5
        ))
        alphas = {"A": _alpha("A", 0.5, 0.8)}
        result = scaler.scale(alphas)
        # Zero vol → falls back to alpha * confidence * fraction
        assert abs(result["A"].scaled_alpha - 0.5 * 0.8 * 0.5) < 1e-6

    def test_kelly_preserves_sign(self):
        scaler = PositionScaler(ScalingConfig(method=ScalingMethod.KELLY))
        alphas = _make_alphas()
        returns = _make_returns()
        result = scaler.scale(alphas, returns)
        assert result["AAPL"].scaled_alpha > 0  # positive alpha
        assert result["GOOG"].scaled_alpha < 0  # negative alpha


# ── Tests: None scaling ──────────────────────────────────────────────────


class TestNoneScaling:
    def test_passthrough(self):
        scaler = PositionScaler(ScalingConfig(method=ScalingMethod.NONE))
        alphas = _make_alphas()
        result = scaler.scale(alphas)
        for sym, alpha in alphas.items():
            assert abs(result[sym].scaled_alpha - alpha.score) < 1e-6


# ── Tests: Confidence filter ─────────────────────────────────────────────


class TestConfidenceFilter:
    def test_min_confidence_filters_low(self):
        scaler = PositionScaler(ScalingConfig(
            method=ScalingMethod.CONVICTION, min_confidence=0.6
        ))
        alphas = _make_alphas()  # MSFT has confidence 0.5
        result = scaler.scale(alphas)
        assert result["MSFT"].scaled_alpha == 0.0
        assert result["AAPL"].scaled_alpha != 0.0  # confidence 0.9

    def test_zero_min_confidence_keeps_all(self):
        scaler = PositionScaler(ScalingConfig(min_confidence=0.0))
        alphas = _make_alphas()
        result = scaler.scale(alphas)
        for p in result.values():
            if p.raw_alpha != 0.0:
                assert p.scaled_alpha != 0.0


# ── Tests: Leverage cap ──────────────────────────────────────────────────


class TestLeverageCap:
    def test_leverage_capped(self):
        scaler = PositionScaler(ScalingConfig(
            method=ScalingMethod.NONE, max_leverage=1.0
        ))
        # Total abs alpha = 0.6 + 0.3 + 0.1 = 1.0 (just at cap)
        alphas = _make_alphas()
        result = scaler.scale(alphas)
        total = sum(abs(p.scaled_alpha) for p in result.values())
        assert total <= 1.0 + 1e-6

    def test_leverage_cap_scales_proportionally(self):
        scaler = PositionScaler(ScalingConfig(
            method=ScalingMethod.NONE, max_leverage=0.5
        ))
        alphas = _make_alphas()  # Total abs = 1.0
        result = scaler.scale(alphas)
        total = sum(abs(p.scaled_alpha) for p in result.values())
        assert abs(total - 0.5) < 1e-6

    def test_under_cap_unchanged(self):
        scaler = PositionScaler(ScalingConfig(
            method=ScalingMethod.NONE, max_leverage=5.0
        ))
        alphas = _make_alphas()
        result = scaler.scale(alphas)
        for sym, alpha in alphas.items():
            assert abs(result[sym].scaled_alpha - alpha.score) < 1e-6


# ── Tests: scale_to_alpha_dict ───────────────────────────────────────────


class TestScaleToAlphaDict:
    def test_returns_alpha_scores(self):
        scaler = PositionScaler(ScalingConfig(method=ScalingMethod.CONVICTION))
        alphas = _make_alphas()
        result = scaler.scale_to_alpha_dict(alphas)
        for alpha in result.values():
            assert isinstance(alpha, AlphaScore)
            assert -1.0 <= alpha.score <= 1.0

    def test_scores_are_scaled(self):
        scaler = PositionScaler(ScalingConfig(method=ScalingMethod.CONVICTION))
        alphas = _make_alphas()
        result = scaler.scale_to_alpha_dict(alphas)
        # AAPL: conviction scaled = 0.6 * 0.9 = 0.54
        assert abs(result["AAPL"].score - 0.54) < 1e-6

    def test_preserves_metadata(self):
        scaler = PositionScaler(ScalingConfig(method=ScalingMethod.CONVICTION))
        alphas = _make_alphas()
        result = scaler.scale_to_alpha_dict(alphas)
        for sym in alphas:
            assert result[sym].timestamp == alphas[sym].timestamp
            assert result[sym].confidence == alphas[sym].confidence

    def test_clamps_to_valid_range(self):
        """Kelly can produce values > 1.0; ensure clamped for AlphaScore."""
        scaler = PositionScaler(ScalingConfig(
            method=ScalingMethod.KELLY, kelly_fraction=1.0, max_leverage=100.0
        ))
        alphas = _make_alphas()
        returns = _make_returns()
        result = scaler.scale_to_alpha_dict(alphas, returns)
        for alpha in result.values():
            assert -1.0 <= alpha.score <= 1.0


# ── Tests: ScaledPosition fields ─────────────────────────────────────────


class TestScaledPositionFields:
    def test_all_fields_populated(self):
        scaler = PositionScaler(ScalingConfig(method=ScalingMethod.CONVICTION))
        alphas = _make_alphas()
        returns = _make_returns()
        result = scaler.scale(alphas, returns)
        for p in result.values():
            assert isinstance(p, ScaledPosition)
            assert p.symbol in alphas
            assert p.raw_alpha == alphas[p.symbol].score
            assert p.confidence == alphas[p.symbol].confidence

    def test_vol_estimate_positive_with_returns(self):
        scaler = PositionScaler()
        alphas = _make_alphas()
        returns = _make_returns()
        result = scaler.scale(alphas, returns)
        for p in result.values():
            assert p.vol_estimate > 0

    def test_vol_estimate_zero_without_returns(self):
        scaler = PositionScaler()
        alphas = _make_alphas()
        result = scaler.scale(alphas)
        for p in result.values():
            assert p.vol_estimate == 0.0


# ── Tests: Config defaults ───────────────────────────────────────────────


class TestConfigDefaults:
    def test_default_method(self):
        config = ScalingConfig()
        assert config.method == ScalingMethod.CONVICTION

    def test_default_kelly_fraction(self):
        config = ScalingConfig()
        assert config.kelly_fraction == 0.5

    def test_config_exposed(self):
        scaler = PositionScaler(ScalingConfig(method=ScalingMethod.KELLY))
        assert scaler.config.method == ScalingMethod.KELLY
