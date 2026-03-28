"""Portfolio construction, optimisation, and attribution.

Combines alpha signals into target portfolios using mean-variance,
risk parity, minimum variance, or maximum diversification optimisation,
with constraint enforcement and turnover-aware rebalancing.
"""
from quant.portfolio.alpha import AlphaCombiner, AlphaScore, CombinationMethod
from quant.portfolio.attribution import (
    AttributionReport,
    PerformanceAttributor,
    SectorAttribution,
    SignalAttribution,
)
from quant.portfolio.benchmark_analytics import (
    ActivePosition,
    ActiveRiskResult,
    ActiveWeightResult,
    BenchmarkAnalyzer,
    BenchmarkConfig,
    BrinsonResult,
    SectorEffect,
    TEBudgetResult,
)
from quant.portfolio.constraints import PortfolioConstraints
from quant.portfolio.dashboard import CIODashboard, StrategyLine
from quant.portfolio.engine import ConstructionResult, PortfolioConfig, PortfolioEngine
from quant.portfolio.factor_attribution import (
    FactorAttributionReport,
    FactorAttributor,
    FactorContribution,
    construct_factors,
)
from quant.portfolio.lifecycle import (
    HealthStatus,
    LifecycleConfig,
    LifecycleManager,
    LifecycleReport,
    Recommendation,
    StrategyHealth,
    StrategySnapshot,
)
from quant.portfolio.optimizers import (
    BaseOptimizer,
    MaxDiversificationOptimizer,
    MeanVarianceOptimizer,
    MinimumVarianceOptimizer,
    OptimizationMethod,
    OptimizationResult,
    RiskParityOptimizer,
    get_optimizer,
)
from quant.portfolio.position_scaler import (
    PositionScaler,
    ScaledPosition,
    ScalingConfig,
    ScalingMethod,
)
from quant.portfolio.pre_trade import (
    PreTradeConfig,
    PreTradePipeline,
    PreTradeResult,
    TradeAdjustment,
)
from quant.portfolio.rebalance_trigger import (
    CalendarFrequency,
    RebalanceTrigger,
    TriggerConfig,
    TriggerDecision,
    TriggerReason,
)
from quant.portfolio.rebalancer import RebalanceEngine, RebalanceResult, Trade
from quant.portfolio.scenario_analyzer import (
    ScenarioAnalyzer,
    ScenarioConfig,
    ScenarioResult,
    ShockResult,
)
from quant.portfolio.strategy_correlation import (
    CrowdingAlert,
    StrategyCorrelationConfig,
    StrategyCorrelationMonitor,
    StrategyCorrelationReport,
)
from quant.portfolio.strategy_ranking import (
    RankingConfig,
    RankingResult,
    StrategyMetrics,
    StrategyRank,
    StrategyRanker,
)
from quant.portfolio.walk_forward_attribution import (
    WalkForwardAttributionConfig,
    WalkForwardAttributionResult,
    WalkForwardAttributor,
    WindowSnapshot,
)

__all__ = [
    # Alpha
    "ActivePosition",
    "ActiveRiskResult",
    "ActiveWeightResult",
    "AlphaCombiner",
    "AlphaScore",
    "CombinationMethod",
    # Benchmark analytics
    "BenchmarkAnalyzer",
    "BenchmarkConfig",
    "BrinsonResult",
    # Attribution
    "AttributionReport",
    "PerformanceAttributor",
    "SectorAttribution",
    "SignalAttribution",
    # Factor attribution
    "FactorAttributionReport",
    "FactorAttributor",
    "FactorContribution",
    "construct_factors",
    # Constraints
    "PortfolioConstraints",
    # Dashboard
    "CIODashboard",
    "StrategyLine",
    # Lifecycle
    "HealthStatus",
    "LifecycleConfig",
    "LifecycleManager",
    "LifecycleReport",
    "Recommendation",
    "StrategyHealth",
    "StrategySnapshot",
    # Engine
    "ConstructionResult",
    "PortfolioConfig",
    "PortfolioEngine",
    # Optimizers
    "BaseOptimizer",
    "MaxDiversificationOptimizer",
    "MeanVarianceOptimizer",
    "MinimumVarianceOptimizer",
    "OptimizationMethod",
    "OptimizationResult",
    "RiskParityOptimizer",
    "get_optimizer",
    # Pre-trade pipeline
    "PreTradeConfig",
    "PreTradePipeline",
    "PreTradeResult",
    "TradeAdjustment",
    # Position scaler
    "PositionScaler",
    "ScaledPosition",
    "ScalingConfig",
    "ScalingMethod",
    # Rebalance trigger
    "CalendarFrequency",
    "RebalanceTrigger",
    "TriggerConfig",
    "TEBudgetResult",
    "TriggerDecision",
    "TriggerReason",
    # Rebalancer
    "RebalanceEngine",
    "RebalanceResult",
    "Trade",
    # Scenario analyzer
    "ScenarioAnalyzer",
    "ScenarioConfig",
    "ScenarioResult",
    "SectorEffect",
    "ShockResult",
    # Strategy correlation
    "CrowdingAlert",
    "StrategyCorrelationConfig",
    "StrategyCorrelationMonitor",
    "StrategyCorrelationReport",
    # Strategy ranking
    "RankingConfig",
    "RankingResult",
    "StrategyMetrics",
    "StrategyRank",
    "StrategyRanker",
    # Walk-forward attribution
    "WalkForwardAttributionConfig",
    "WalkForwardAttributionResult",
    "WalkForwardAttributor",
    "WindowSnapshot",
]
