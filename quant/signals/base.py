"""Abstract base class and output schema for trading signals."""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class SignalOutput:
    """Canonical output of a signal computation for one symbol at one point in time.

    Attributes:
        symbol:     Ticker symbol (upper-case).
        timestamp:  UTC datetime corresponding to the bar used.
        score:      Signal strength in [-1, 1].
                    Negative = bearish, positive = bullish, 0 = neutral.
        confidence: Confidence in the score, in [0, 1].
                    1.0 = fully confident, 0.0 = no information.
        target_position: Suggested normalised position size derived from
                    score * confidence, in [-1, 1].
        metadata:   Optional dict of diagnostics (indicator values, etc.).
    """

    symbol: str
    timestamp: datetime
    score: float
    confidence: float
    target_position: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (-1.0 <= self.score <= 1.0):
            raise ValueError(f"score must be in [-1, 1], got {self.score}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if not (-1.0 <= self.target_position <= 1.0):
            raise ValueError(
                f"target_position must be in [-1, 1], got {self.target_position}"
            )


class BaseSignal(abc.ABC):
    """Abstract trading signal.

    Subclasses declare the symbol and timeframe they operate on, and implement
    ``compute()`` which receives feature output and returns a ``SignalOutput``.

    The convention for ``compute`` is:
    - Receive a dict mapping feature name → pd.Series (indexed by date).
    - Return a single ``SignalOutput`` using the latest available bar.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique signal identifier."""

    @property
    @abc.abstractmethod
    def required_features(self) -> list[str]:
        """Feature names this signal depends on (must exist in FeatureRegistry)."""

    @abc.abstractmethod
    def compute(
        self,
        symbol: str,
        features: dict[str, "pd.Series"],  # noqa: F821  (avoid import at base level)
        timestamp: datetime,
    ) -> SignalOutput:
        """Compute the signal for *symbol* from the provided feature series.

        Args:
            symbol:    Ticker symbol.
            features:  Dict of {feature_name: pd.Series} pre-computed by FeatureEngine.
            timestamp: Timestamp of the latest bar.

        Returns:
            SignalOutput with score, confidence, and target_position.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
