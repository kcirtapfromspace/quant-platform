"""Quant research tools: signal analysis, alpha research, and factor studies."""
from quant.research.cointegration import (
    CointegrationConfig,
    CointegrationTester,
    HedgeMethod,
    PairResult,
    ScreenResult,
)
from quant.research.performance_analytics import (
    AnalyticsConfig,
    PerformanceAnalyzer,
    PerformanceResult,
)
from quant.research.signal_evaluator import (
    EvaluationResult,
    EvaluatorConfig,
    QuantileStats,
    SignalEvaluator,
)

__all__ = [
    "AnalyticsConfig",
    "CointegrationConfig",
    "CointegrationTester",
    "EvaluationResult",
    "EvaluatorConfig",
    "HedgeMethod",
    "PairResult",
    "PerformanceAnalyzer",
    "PerformanceResult",
    "QuantileStats",
    "ScreenResult",
    "SignalEvaluator",
]
