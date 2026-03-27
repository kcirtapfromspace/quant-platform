"""Signal registry and strategy framework.

Consumes features from the feature engine and produces signal scores
and target positions in the range [-1, 1].
"""
from quant.signals.base import BaseSignal, SignalOutput
from quant.signals.factors import BreakoutSignal, ReturnQualitySignal, VolatilitySignal
from quant.signals.regime import (
    CorrelationRegime,
    MarketRegime,
    RegimeConfig,
    RegimeDetector,
    RegimeState,
    RegimeWeightAdapter,
    TrendRegime,
    VolRegime,
)
from quant.signals.registry import SignalRegistry
from quant.signals.strategies import (
    MeanReversionSignal,
    MomentumSignal,
    TrendFollowingSignal,
)

__all__ = [
    "BaseSignal",
    "BreakoutSignal",
    "CorrelationRegime",
    "MarketRegime",
    "MeanReversionSignal",
    "MomentumSignal",
    "RegimeConfig",
    "RegimeDetector",
    "RegimeState",
    "RegimeWeightAdapter",
    "ReturnQualitySignal",
    "SignalOutput",
    "SignalRegistry",
    "TrendFollowingSignal",
    "TrendRegime",
    "VolRegime",
    "VolatilitySignal",
]
