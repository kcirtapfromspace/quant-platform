"""Signal registry and strategy framework.

Consumes features from the feature engine and produces signal scores
and target positions in the range [-1, 1].
"""
from quant.signals.base import BaseSignal, SignalOutput
from quant.signals.factors import BreakoutSignal, ReturnQualitySignal, VolatilitySignal
from quant.signals.registry import SignalRegistry
from quant.signals.strategies import (
    MeanReversionSignal,
    MomentumSignal,
    TrendFollowingSignal,
)

__all__ = [
    "BaseSignal",
    "BreakoutSignal",
    "MeanReversionSignal",
    "MomentumSignal",
    "ReturnQualitySignal",
    "SignalOutput",
    "SignalRegistry",
    "TrendFollowingSignal",
    "VolatilitySignal",
]
