"""Example strategy implementations: momentum, mean-reversion, trend-following.

Each strategy consumes pre-computed features from the FeatureEngine and outputs
a SignalOutput with score in [-1, 1], confidence in [0, 1], and target_position.

Scoring convention:
  +1.0 = maximum long
  -1.0 = maximum short
   0.0 = flat / no view

Signal computation delegates to ``quant_rs.signals`` Rust kernels.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

import quant_rs as _qrs

from quant.signals.base import BaseSignal, SignalOutput


def _to_list(series: pd.Series) -> list[float]:
    """Convert a pd.Series to a list of floats, replacing NaN with float('nan')."""
    return [float(x) for x in series]


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
        rsi_values = _to_list(features[f"rsi_{self._rsi_period}"])
        returns = _to_list(features["returns"])

        score, confidence, target_position = _qrs.signals.momentum_signal(
            rsi_values, returns, self._lookback, self._return_scale
        )

        rsi_last = next(
            (v for v in reversed(rsi_values) if v == v),  # last non-NaN
            None,
        )
        return SignalOutput(
            symbol=symbol,
            timestamp=timestamp,
            score=score,
            confidence=confidence,
            target_position=target_position,
            metadata={"rsi": rsi_last},
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
        bb_mid = _to_list(features[f"bb_mid_{self._bb_period}"])
        bb_upper = _to_list(features[f"bb_upper_{self._bb_period}"])
        bb_lower = _to_list(features[f"bb_lower_{self._bb_period}"])
        returns = _to_list(features["returns"])

        score, confidence, target_position = _qrs.signals.mean_reversion_signal(
            bb_mid, bb_upper, bb_lower, returns, self._num_std
        )

        mid_last = next((v for v in reversed(bb_mid) if v == v), None)
        upper_last = next((v for v in reversed(bb_upper) if v == v), None)
        lower_last = next((v for v in reversed(bb_lower) if v == v), None)
        band_width = (upper_last - lower_last) if upper_last is not None and lower_last is not None else None

        return SignalOutput(
            symbol=symbol,
            timestamp=timestamp,
            score=score,
            confidence=confidence,
            target_position=target_position,
            metadata={"mid": mid_last, "band_width": band_width},
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
        macd_hist = _to_list(features["macd_hist_12_26_9"])
        fast_ma = _to_list(features[f"rolling_mean_{self._fast_ma}"])
        slow_ma = _to_list(features[f"rolling_mean_{self._slow_ma}"])

        score, confidence, target_position = _qrs.signals.trend_following_signal(
            macd_hist, fast_ma, slow_ma
        )

        fast_last = next((v for v in reversed(fast_ma) if v == v), None)
        slow_last = next((v for v in reversed(slow_ma) if v == v), None)
        hist_last = next((v for v in reversed(macd_hist) if v == v), None)

        return SignalOutput(
            symbol=symbol,
            timestamp=timestamp,
            score=score,
            confidence=confidence,
            target_position=target_position,
            metadata={
                "macd_hist": hist_last,
                "sma_fast": fast_last,
                "sma_slow": slow_last,
            },
        )
