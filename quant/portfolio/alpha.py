"""Alpha combination and ensemble framework.

Combines multiple signal scores into a single composite alpha per symbol,
supporting static weights, inverse-volatility weighting, and rank-based
combination.

Scoring convention (inherited from signals):
  +1.0 = maximum long conviction
  -1.0 = maximum short conviction
   0.0 = flat / no view
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime

from quant.signals.base import SignalOutput


@dataclass(frozen=True, slots=True)
class AlphaScore:
    """Composite alpha for a single symbol at a point in time.

    Attributes:
        symbol:     Ticker symbol.
        timestamp:  Bar timestamp.
        score:      Combined alpha in [-1, 1].
        confidence: Aggregate confidence in [0, 1].
        signal_contributions: Per-signal weighted contribution to the final score.
    """

    symbol: str
    timestamp: datetime
    score: float
    confidence: float
    signal_contributions: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (-1.0 <= self.score <= 1.0):
            raise ValueError(f"score must be in [-1, 1], got {self.score}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


class CombinationMethod(enum.Enum):
    STATIC_WEIGHT = "static_weight"
    INVERSE_VOLATILITY = "inverse_volatility"
    RANK_WEIGHTED = "rank_weighted"
    EQUAL_WEIGHT = "equal_weight"


def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


class AlphaCombiner:
    """Combines multiple signal outputs into a single composite alpha.

    Usage::

        combiner = AlphaCombiner(
            method=CombinationMethod.STATIC_WEIGHT,
            weights={"momentum": 0.4, "mean_reversion": 0.3, "trend_following": 0.3},
        )
        alpha = combiner.combine("AAPL", timestamp, signal_outputs)
    """

    def __init__(
        self,
        method: CombinationMethod = CombinationMethod.EQUAL_WEIGHT,
        weights: dict[str, float] | None = None,
    ) -> None:
        self._method = method
        self._weights = weights or {}
        if method == CombinationMethod.STATIC_WEIGHT and not self._weights:
            raise ValueError("weights required for STATIC_WEIGHT method")

    @property
    def method(self) -> CombinationMethod:
        return self._method

    def combine(
        self,
        symbol: str,
        timestamp: datetime,
        signals: list[SignalOutput],
    ) -> AlphaScore:
        """Combine signal outputs into a composite alpha score.

        Args:
            symbol:    Ticker symbol.
            timestamp: Current bar timestamp.
            signals:   List of SignalOutput from different strategies.

        Returns:
            AlphaScore with combined score and per-signal contributions.
        """
        if not signals:
            return AlphaScore(
                symbol=symbol,
                timestamp=timestamp,
                score=0.0,
                confidence=0.0,
            )

        if self._method == CombinationMethod.STATIC_WEIGHT:
            return self._static_weight(symbol, timestamp, signals)
        if self._method == CombinationMethod.INVERSE_VOLATILITY:
            return self._inverse_volatility(symbol, timestamp, signals)
        if self._method == CombinationMethod.RANK_WEIGHTED:
            return self._rank_weighted(symbol, timestamp, signals)
        return self._equal_weight(symbol, timestamp, signals)

    def combine_universe(
        self,
        timestamp: datetime,
        universe_signals: dict[str, list[SignalOutput]],
    ) -> dict[str, AlphaScore]:
        """Combine signals for every symbol in the universe.

        Args:
            timestamp:        Current bar timestamp.
            universe_signals: Dict of {symbol: [SignalOutput, ...]}.

        Returns:
            Dict of {symbol: AlphaScore}.
        """
        return {
            sym: self.combine(sym, timestamp, sigs)
            for sym, sigs in universe_signals.items()
        }

    def _static_weight(
        self, symbol: str, timestamp: datetime, signals: list[SignalOutput]
    ) -> AlphaScore:
        contributions: dict[str, float] = {}
        total_weight = 0.0
        weighted_score = 0.0
        weighted_conf = 0.0

        for sig in signals:
            # Look up weight by signal name from metadata, or default to 0
            sig_name = sig.metadata.get("signal_name", "")
            w = self._weights.get(sig_name, 0.0)
            contributions[sig_name] = sig.score * w
            weighted_score += sig.score * w
            weighted_conf += sig.confidence * w
            total_weight += w

        if total_weight > 0:
            weighted_conf /= total_weight

        return AlphaScore(
            symbol=symbol,
            timestamp=timestamp,
            score=_clamp(weighted_score),
            confidence=_clamp(weighted_conf, 0.0, 1.0),
            signal_contributions=contributions,
        )

    def _equal_weight(
        self, symbol: str, timestamp: datetime, signals: list[SignalOutput]
    ) -> AlphaScore:
        n = len(signals)
        contributions: dict[str, float] = {}
        total_score = 0.0
        total_conf = 0.0

        for sig in signals:
            sig_name = sig.metadata.get("signal_name", f"signal_{id(sig)}")
            contribution = sig.score / n
            contributions[sig_name] = contribution
            total_score += contribution
            total_conf += sig.confidence

        return AlphaScore(
            symbol=symbol,
            timestamp=timestamp,
            score=_clamp(total_score),
            confidence=_clamp(total_conf / n, 0.0, 1.0),
            signal_contributions=contributions,
        )

    def _inverse_volatility(
        self, symbol: str, timestamp: datetime, signals: list[SignalOutput]
    ) -> AlphaScore:
        """Weight signals inversely to their confidence uncertainty (1 - confidence).

        Higher-confidence signals receive proportionally more weight.
        """
        contributions: dict[str, float] = {}
        # Use confidence as a proxy for signal quality
        raw_weights: list[float] = []
        for sig in signals:
            # inv_vol weight: higher confidence → lower "vol" → higher weight
            raw_weights.append(sig.confidence + 1e-8)

        total_w = sum(raw_weights)
        weights = [w / total_w for w in raw_weights]

        weighted_score = 0.0
        weighted_conf = 0.0
        for sig, w in zip(signals, weights, strict=True):
            sig_name = sig.metadata.get("signal_name", f"signal_{id(sig)}")
            contributions[sig_name] = sig.score * w
            weighted_score += sig.score * w
            weighted_conf += sig.confidence * w

        return AlphaScore(
            symbol=symbol,
            timestamp=timestamp,
            score=_clamp(weighted_score),
            confidence=_clamp(weighted_conf, 0.0, 1.0),
            signal_contributions=contributions,
        )

    def _rank_weighted(
        self, symbol: str, timestamp: datetime, signals: list[SignalOutput]
    ) -> AlphaScore:
        """Rank signals by absolute score magnitude; higher-ranked get more weight."""
        contributions: dict[str, float] = {}
        # Sort by |score| descending
        sorted_sigs = sorted(signals, key=lambda s: abs(s.score), reverse=True)
        n = len(sorted_sigs)

        # Rank weights: top signal gets n, second gets n-1, etc.
        raw_weights = [float(n - i) for i in range(n)]
        total_w = sum(raw_weights)
        weights = [w / total_w for w in raw_weights]

        weighted_score = 0.0
        weighted_conf = 0.0
        for sig, w in zip(sorted_sigs, weights, strict=True):
            sig_name = sig.metadata.get("signal_name", f"signal_{id(sig)}")
            contributions[sig_name] = sig.score * w
            weighted_score += sig.score * w
            weighted_conf += sig.confidence * w

        return AlphaScore(
            symbol=symbol,
            timestamp=timestamp,
            score=_clamp(weighted_score),
            confidence=_clamp(weighted_conf, 0.0, 1.0),
            signal_contributions=contributions,
        )
