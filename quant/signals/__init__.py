"""Signal registry and strategy framework.

Consumes features from the feature engine and produces signal scores
and target positions in the range [-1, 1].
"""
from quant.signals.adaptive_combiner import (
    AdaptiveCombinerConfig,
    AdaptiveSignalCombiner,
    AdaptiveWeights,
)
from quant.signals.base import BaseSignal, SignalOutput
from quant.signals.cross_sectional import (
    CrossSectionalMeanReversion,
    CrossSectionalMomentum,
    CrossSectionalSignal,
    CrossSectionalVolatility,
    QuantileSelection,
    QuantileSelector,
    QuantileWeights,
    percentile_rank,
    scores_to_quantile_weights,
    winsorize,
    z_score_normalize,
)
from quant.signals.decay import (
    DecayConfig,
    DecayResult,
    HorizonIC,
    SignalDecayAnalyzer,
)
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
    "AdaptiveCombinerConfig",
    "AdaptiveSignalCombiner",
    "AdaptiveWeights",
    "BaseSignal",
    "BreakoutSignal",
    "CorrelationRegime",
    "DecayConfig",
    "DecayResult",
    "CrossSectionalMeanReversion",
    "CrossSectionalMomentum",
    "CrossSectionalSignal",
    "CrossSectionalVolatility",
    "HorizonIC",
    "MarketRegime",
    "MeanReversionSignal",
    "MomentumSignal",
    "percentile_rank",
    "QuantileSelection",
    "QuantileSelector",
    "QuantileWeights",
    "RegimeConfig",
    "RegimeDetector",
    "RegimeState",
    "RegimeWeightAdapter",
    "ReturnQualitySignal",
    "SignalDecayAnalyzer",
    "SignalOutput",
    "SignalRegistry",
    "TrendFollowingSignal",
    "TrendRegime",
    "VolRegime",
    "VolatilitySignal",
    "scores_to_quantile_weights",
    "winsorize",
    "z_score_normalize",
]
