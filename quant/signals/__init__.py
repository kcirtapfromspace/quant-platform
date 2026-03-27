"""Signal registry and strategy framework.

Consumes features from the feature engine and produces signal scores
and target positions in the range [-1, 1].
"""
from quant.signals.base import BaseSignal, SignalOutput
from quant.signals.registry import SignalRegistry
from quant.signals.strategies import (
    MomentumSignal,
    MeanReversionSignal,
    TrendFollowingSignal,
)

__all__ = [
    "BaseSignal",
    "SignalOutput",
    "SignalRegistry",
    "MomentumSignal",
    "MeanReversionSignal",
    "TrendFollowingSignal",
]
