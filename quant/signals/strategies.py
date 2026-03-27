"""Example strategy implementations: momentum, mean-reversion, trend-following.

Each strategy consumes pre-computed features from the FeatureEngine and outputs
a SignalOutput with score in [-1, 1], confidence in [0, 1], and target_position.

Scoring convention:
  +1.0 = maximum long
  -1.0 = maximum short
   0.0 = flat / no view
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from quant.signals.base import BaseSignal, SignalOutput

if TYPE_CHECKING:
    pass  # pd.Series imported at runtime below


def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _last_valid(series: pd.Series) -> float | None:
    """Return the last non-NaN value or None."""
    valid = series.dropna()
    if valid.empty:
        return None
    return float(valid.iloc[-1])


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------

class MomentumSignal(BaseSignal):
    """Momentum signal based on RSI and short-term returns.

    Logic:
    - RSI > 70  → bullish (score approaches +1)
    - RSI < 30  → bearish (score approaches -1)
    - RSI 30–70 → linear interpolation through 0

    Confidence is derived from the absolute return over *lookback* bars:
    higher absolute return → higher confidence, capped at 1.

    Args:
        rsi_period:       RSI period. Default 14.
        lookback:         Return lookback for confidence estimation. Default 5.
        return_scale:     Absolute return magnitude mapped to confidence=1. Default 0.05 (5%).
    """

    def __init__(
        self,
        rsi_period: int = 14,
        lookback: int = 5,
        return_scale: float = 0.05,
    ) -> None:
        self._rsi_period = rsi_period
        self._lookback = lookback
        self._return_scale = return_scale

    @property
    def name(self) -> str:
        return "momentum"

    @property
    def required_features(self) -> list[str]:
        return [f"rsi_{self._rsi_period}", "returns"]

    def compute(
        self,
        symbol: str,
        features: dict[str, pd.Series],
        timestamp: datetime,
    ) -> SignalOutput:
        rsi_val = _last_valid(features[f"rsi_{self._rsi_period}"])
        ret_series = features["returns"].dropna()

        if rsi_val is None:
            return SignalOutput(
                symbol=symbol,
                timestamp=timestamp,
                score=0.0,
                confidence=0.0,
                target_position=0.0,
                metadata={"reason": "insufficient data"},
            )

        # Score: linearly map RSI → [-1, 1] with midpoint at 50
        # RSI=30 → -1, RSI=50 → 0, RSI=70 → +1
        score = _clamp((rsi_val - 50.0) / 20.0)

        # Confidence: magnitude of recent returns
        if len(ret_series) >= self._lookback:
            recent_abs_ret = float(ret_series.iloc[-self._lookback:].abs().mean())
            confidence = _clamp(recent_abs_ret / self._return_scale, 0.0, 1.0)
        else:
            confidence = 0.5

        target_position = _clamp(score * confidence)

        return SignalOutput(
            symbol=symbol,
            timestamp=timestamp,
            score=score,
            confidence=confidence,
            target_position=target_position,
            metadata={"rsi": rsi_val},
        )


# ---------------------------------------------------------------------------
# Mean-reversion
# ---------------------------------------------------------------------------

class MeanReversionSignal(BaseSignal):
    """Mean-reversion signal based on Bollinger Band z-score.

    Logic:
    - Price at upper band (z = +2σ) → bearish (score → -1)
    - Price at lower band (z = -2σ) → bullish (score → +1)
    - Price at mid band             → neutral (score = 0)

    Confidence is proportional to how far price is from the mid band
    (normalised by bandwidth), capped at 1.

    Args:
        bb_period:  Bollinger Band period. Default 20.
        num_std:    Band multiplier. Default 2.0.
    """

    def __init__(self, bb_period: int = 20, num_std: float = 2.0) -> None:
        self._bb_period = bb_period
        self._num_std = num_std

    @property
    def name(self) -> str:
        return "mean_reversion"

    @property
    def required_features(self) -> list[str]:
        return [
            f"bb_mid_{self._bb_period}",
            f"bb_upper_{self._bb_period}",
            f"bb_lower_{self._bb_period}",
            f"bb_bandwidth_{self._bb_period}",
            "returns",
        ]

    def compute(
        self,
        symbol: str,
        features: dict[str, pd.Series],
        timestamp: datetime,
    ) -> SignalOutput:
        mid = _last_valid(features[f"bb_mid_{self._bb_period}"])
        upper = _last_valid(features[f"bb_upper_{self._bb_period}"])
        lower = _last_valid(features[f"bb_lower_{self._bb_period}"])
        bw = _last_valid(features[f"bb_bandwidth_{self._bb_period}"])

        # Need actual close price from returns series index
        ret_series = features["returns"]

        if mid is None or upper is None or lower is None:
            return SignalOutput(
                symbol=symbol,
                timestamp=timestamp,
                score=0.0,
                confidence=0.0,
                target_position=0.0,
                metadata={"reason": "insufficient data"},
            )

        band_width = upper - lower
        if band_width < 1e-12:
            return SignalOutput(
                symbol=symbol,
                timestamp=timestamp,
                score=0.0,
                confidence=0.0,
                target_position=0.0,
                metadata={"reason": "zero bandwidth"},
            )

        # Estimate current price from mid and returns
        # (We can't directly access close here without the raw df, so we use
        # mid as a reasonable proxy for normalisation)
        # score: +1 when at lower band, -1 when at upper band
        # Normalised deviation: (price - mid) / (band_width / 2)
        # We approximate "price ≈ mid * (1 + last_return)" for the deviation sign
        last_ret = _last_valid(ret_series)
        if last_ret is None:
            last_ret = 0.0

        # Approximate z-score: how many half-bandwidths from the mid
        price_approx = mid * (1.0 + last_ret)
        half_band = band_width / 2.0
        z = (price_approx - mid) / half_band if half_band > 0 else 0.0

        # Mean-reversion: negative of z (high price → sell, low price → buy)
        score = _clamp(-z / self._num_std)

        # Confidence: normalised bandwidth (thin bands → less reliable)
        confidence = _clamp(abs(z) / self._num_std, 0.0, 1.0)

        target_position = _clamp(score * confidence)

        return SignalOutput(
            symbol=symbol,
            timestamp=timestamp,
            score=score,
            confidence=confidence,
            target_position=target_position,
            metadata={"z_score": z, "band_width": band_width, "mid": mid},
        )


# ---------------------------------------------------------------------------
# Trend-following
# ---------------------------------------------------------------------------

class TrendFollowingSignal(BaseSignal):
    """Trend-following signal based on MACD histogram direction and moving-average crossover.

    Logic:
    - MACD histogram > 0 and rising → bullish
    - MACD histogram < 0 and falling → bearish
    - Score magnitude proportional to histogram size relative to recent range.

    Confidence is derived from alignment between MACD histogram sign and
    short-vs-long SMA relationship.

    Args:
        fast_ma:   Fast MA period for SMA crossover confirmation. Default 20.
        slow_ma:   Slow MA period. Default 50.
    """

    def __init__(self, fast_ma: int = 20, slow_ma: int = 50) -> None:
        if fast_ma >= slow_ma:
            raise ValueError("fast_ma must be < slow_ma")
        self._fast_ma = fast_ma
        self._slow_ma = slow_ma

    @property
    def name(self) -> str:
        return "trend_following"

    @property
    def required_features(self) -> list[str]:
        return [
            "macd_hist_12_26_9",
            f"rolling_mean_{self._fast_ma}",
            f"rolling_mean_{self._slow_ma}",
        ]

    def compute(
        self,
        symbol: str,
        features: dict[str, pd.Series],
        timestamp: datetime,
    ) -> SignalOutput:
        hist_series = features["macd_hist_12_26_9"].dropna()
        fast_ma = _last_valid(features[f"rolling_mean_{self._fast_ma}"])
        slow_ma = _last_valid(features[f"rolling_mean_{self._slow_ma}"])

        if hist_series.empty or fast_ma is None or slow_ma is None:
            return SignalOutput(
                symbol=symbol,
                timestamp=timestamp,
                score=0.0,
                confidence=0.0,
                target_position=0.0,
                metadata={"reason": "insufficient data"},
            )

        last_hist = float(hist_series.iloc[-1])

        # Normalise histogram by its rolling std to get a standardised score
        if len(hist_series) >= 10:
            hist_std = float(hist_series.rolling(min(20, len(hist_series))).std().iloc[-1])
        else:
            hist_std = float(hist_series.abs().mean()) or 1.0

        if hist_std < 1e-12:
            hist_std = 1.0

        score = _clamp(last_hist / hist_std)

        # Confidence: SMA crossover alignment with MACD direction
        sma_bullish = fast_ma > slow_ma
        hist_bullish = last_hist > 0.0
        aligned = sma_bullish == hist_bullish

        base_confidence = abs(score)
        confidence = _clamp(base_confidence * (1.2 if aligned else 0.6), 0.0, 1.0)

        target_position = _clamp(score * confidence)

        return SignalOutput(
            symbol=symbol,
            timestamp=timestamp,
            score=score,
            confidence=confidence,
            target_position=target_position,
            metadata={
                "macd_hist": last_hist,
                "sma_fast": fast_ma,
                "sma_slow": slow_ma,
                "sma_aligned": aligned,
            },
        )
