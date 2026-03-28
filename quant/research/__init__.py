"""Quant research tools: signal analysis, alpha research, and factor studies."""
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
    "EvaluationResult",
    "EvaluatorConfig",
    "PerformanceAnalyzer",
    "PerformanceResult",
    "QuantileStats",
    "SignalEvaluator",
]
