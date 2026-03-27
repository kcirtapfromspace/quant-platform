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
from quant.portfolio.constraints import PortfolioConstraints
from quant.portfolio.engine import ConstructionResult, PortfolioConfig, PortfolioEngine
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
from quant.portfolio.rebalancer import RebalanceEngine, RebalanceResult, Trade

__all__ = [
    # Alpha
    "AlphaCombiner",
    "AlphaScore",
    "CombinationMethod",
    # Attribution
    "AttributionReport",
    "PerformanceAttributor",
    "SectorAttribution",
    "SignalAttribution",
    # Constraints
    "PortfolioConstraints",
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
    # Rebalancer
    "RebalanceEngine",
    "RebalanceResult",
    "Trade",
]
