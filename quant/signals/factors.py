"""Factor-based trading signals — volatility, quality, and breakout.

These signals implement classic quant factors using only per-symbol
time-series features (no Rust kernels required).  They complement the
existing technical signals in ``strategies.py`` by capturing different
sources of alpha:

  * **VolatilitySignal**: Low-volatility anomaly — low-vol stocks receive
    positive scores (long bias), high-vol stocks receive negative scores.
  * **ReturnQualitySignal**: Favors assets with high risk-adjusted returns
    (rolling Sharpe) and penalises assets with poor consistency.
  * **BreakoutSignal**: Detects price breakouts above/below N-day
    high/low channels.  Breakout above → bullish, below → bearish.

All signals produce scores in [-1, 1] following the platform convention.
"""
from __future__ import annotations

import math
from datetime import datetime

import pandas as pd

from quant.signals.base import BaseSignal, SignalOutput


def _last_valid(series: pd.Series) -> float | None:
    """Return the last non-NaN value in a series, or None."""
    vals = series.dropna()
    if len(vals) == 0:
        return None
    return float(vals.iloc[-1])


# ── Volatility factor ────────────────────────────────────────────────────────


class VolatilitySignal(BaseSignal):
    """Low-volatility factor signal.

    Scores based on realised volatility relative to configurable thresholds:
      * vol < ``low_vol`` → score approaches +1 (long bias)
      * vol > ``high_vol`` → score approaches -1 (short / avoid)
      * vol between thresholds → linear interpolation through 0

    Confidence scales with the number of valid return observations
    available (more data → higher confidence).

    Args:
        period: Rolling window for volatility estimation (default 20).
        annualise: If True, multiply daily vol by sqrt(252). Default True.
        low_vol: Annualised vol threshold below which score is maximally positive (default 0.12).
        high_vol: Annualised vol threshold above which score is maximally negative (default 0.40).
    """

    def __init__(
        self,
        period: int = 20,
        annualise: bool = True,
        low_vol: float = 0.12,
        high_vol: float = 0.40,
    ) -> None:
        self._period = period
        self._annualise = annualise
        self._low_vol = low_vol
        self._high_vol = high_vol

    @property
    def name(self) -> str:
        return "volatility"

    @property
    def required_features(self) -> list[str]:
        return ["returns"]

    def compute(
        self,
        symbol: str,
        features: dict[str, pd.Series],
        timestamp: datetime,
    ) -> SignalOutput:
        returns = features["returns"].dropna()

        if len(returns) < self._period:
            return SignalOutput(
                symbol=symbol, timestamp=timestamp,
                score=0.0, confidence=0.0, target_position=0.0,
                metadata={"vol": None, "period": self._period},
            )

        tail = returns.iloc[-self._period :]
        daily_vol = float(tail.std())
        vol = daily_vol * math.sqrt(252) if self._annualise else daily_vol

        # Linear mapping: low_vol → +1, high_vol → -1
        mid = (self._low_vol + self._high_vol) / 2
        half_range = (self._high_vol - self._low_vol) / 2
        raw_score = 0.0 if half_range == 0 else -(vol - mid) / half_range
        score = max(-1.0, min(1.0, raw_score))

        # Confidence from data availability (saturates at 2× period)
        confidence = min(1.0, len(returns) / (2 * self._period))

        target_position = score * confidence

        return SignalOutput(
            symbol=symbol,
            timestamp=timestamp,
            score=score,
            confidence=confidence,
            target_position=max(-1.0, min(1.0, target_position)),
            metadata={"vol": round(vol, 6), "period": self._period},
        )


# ── Return quality factor ────────────────────────────────────────────────────


class ReturnQualitySignal(BaseSignal):
    """Return quality (rolling Sharpe) signal.

    Scores based on a rolling risk-adjusted return metric:
      * High rolling Sharpe → positive score (long)
      * Negative Sharpe → negative score (avoid)

    The Sharpe ratio is computed as ``mean(returns) / std(returns)``
    over a rolling window, then annualised and mapped to [-1, 1].

    Args:
        period: Rolling window for Sharpe estimation (default 60 trading days).
        sharpe_cap: Annualised Sharpe value mapped to score = ±1 (default 3.0).
    """

    def __init__(self, period: int = 60, sharpe_cap: float = 3.0) -> None:
        self._period = period
        self._sharpe_cap = sharpe_cap

    @property
    def name(self) -> str:
        return "return_quality"

    @property
    def required_features(self) -> list[str]:
        return ["returns"]

    def compute(
        self,
        symbol: str,
        features: dict[str, pd.Series],
        timestamp: datetime,
    ) -> SignalOutput:
        returns = features["returns"].dropna()

        if len(returns) < self._period:
            return SignalOutput(
                symbol=symbol, timestamp=timestamp,
                score=0.0, confidence=0.0, target_position=0.0,
                metadata={"sharpe": None, "period": self._period},
            )

        tail = returns.iloc[-self._period :]
        mu = float(tail.mean())
        sigma = float(tail.std())

        sharpe = 0.0 if sigma == 0 or math.isnan(sigma) else (mu / sigma) * math.sqrt(252)

        # Map to [-1, 1] with cap
        score = max(-1.0, min(1.0, sharpe / self._sharpe_cap))

        confidence = min(1.0, len(returns) / (2 * self._period))
        target_position = max(-1.0, min(1.0, score * confidence))

        return SignalOutput(
            symbol=symbol,
            timestamp=timestamp,
            score=score,
            confidence=confidence,
            target_position=target_position,
            metadata={"sharpe": round(sharpe, 4), "period": self._period},
        )


# ── Breakout signal ──────────────────────────────────────────────────────────


class BreakoutSignal(BaseSignal):
    """Donchian channel breakout signal.

    Detects when price breaks above the N-day high or below the N-day low:
      * Close > N-day high → bullish breakout (score → +1)
      * Close < N-day low → bearish breakout (score → -1)
      * Close within channel → score proportional to position in channel

    Confidence is derived from the relative width of the channel — wider
    channels (more decisive breakouts) produce higher confidence.

    Args:
        period: Lookback period for high/low channel (default 20).
        bandwidth_scale: Channel width (as fraction of mid) mapped to
            confidence = 1.0 (default 0.10 = 10%).
    """

    def __init__(self, period: int = 20, bandwidth_scale: float = 0.10) -> None:
        self._period = period
        self._bandwidth_scale = bandwidth_scale

    @property
    def name(self) -> str:
        return "breakout"

    @property
    def required_features(self) -> list[str]:
        return ["returns"]  # we derive close from cumulative returns

    def compute(
        self,
        symbol: str,
        features: dict[str, pd.Series],
        timestamp: datetime,
    ) -> SignalOutput:
        returns = features["returns"].dropna()

        if len(returns) < self._period + 1:
            return SignalOutput(
                symbol=symbol, timestamp=timestamp,
                score=0.0, confidence=0.0, target_position=0.0,
                metadata={"channel_pos": None, "period": self._period},
            )

        # Reconstruct synthetic close from cumulative returns (normalised to 100)
        cum = (1 + returns).cumprod() * 100
        tail = cum.iloc[-(self._period + 1) :]
        channel = tail.iloc[:-1]  # lookback window (excluding current bar)
        current = float(tail.iloc[-1])

        high = float(channel.max())
        low = float(channel.min())

        if high == low:
            score = 0.0
            channel_pos = 0.5
        else:
            # Position in channel: 0 = at low, 1 = at high
            channel_pos = (current - low) / (high - low)
            # Map to [-1, 1]: 0.5 → 0, 0 → -1, 1 → +1
            score = max(-1.0, min(1.0, 2.0 * channel_pos - 1.0))

        # Confidence from channel width
        mid = (high + low) / 2
        if mid > 0:
            bandwidth = (high - low) / mid
            confidence = min(1.0, bandwidth / self._bandwidth_scale)
        else:
            confidence = 0.0

        target_position = max(-1.0, min(1.0, score * confidence))

        return SignalOutput(
            symbol=symbol,
            timestamp=timestamp,
            score=score,
            confidence=confidence,
            target_position=target_position,
            metadata={
                "channel_pos": round(channel_pos, 4) if channel_pos is not None else None,
                "channel_high": round(high, 4),
                "channel_low": round(low, 4),
                "period": self._period,
            },
        )
