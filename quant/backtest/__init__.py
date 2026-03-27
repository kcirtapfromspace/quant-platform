"""Backtesting framework for evaluating trading strategies.

Typical usage — single-asset
-----------------------------
from quant.backtest import BacktestEngine, BacktestConfig
from quant.backtest.strategy import Strategy

class MyCrossover(Strategy):
    def generate_signals(self, ohlcv):
        fast = ohlcv["adj_close"].rolling(10).mean()
        slow = ohlcv["adj_close"].rolling(30).mean()
        return (fast > slow).astype(int)

engine = BacktestEngine()
report = engine.run(ohlcv_df, MyCrossover(), BacktestConfig())
print(report.summary())

Typical usage — multi-asset portfolio
--------------------------------------
from quant.backtest import PortfolioBacktestEngine, PortfolioBacktestConfig
from quant.signals.factors import VolatilitySignal, ReturnQualitySignal

engine = PortfolioBacktestEngine()
report = engine.run(
    returns=daily_returns_df,
    signals=[VolatilitySignal(), ReturnQualitySignal()],
    config=PortfolioBacktestConfig(rebalance_frequency=21),
)
print(report.summary())
"""
from quant.backtest.engine import BacktestConfig, BacktestEngine
from quant.backtest.portfolio_backtest import (
    PortfolioBacktestConfig,
    PortfolioBacktestEngine,
    PortfolioBacktestReport,
    RebalanceSnapshot,
)
from quant.backtest.report import BacktestReport
from quant.backtest.strategy import Strategy
from quant.backtest.walk_forward import (
    FoldResult,
    WalkForwardAnalyzer,
    WalkForwardConfig,
    WalkForwardResult,
)

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestReport",
    "FoldResult",
    "PortfolioBacktestConfig",
    "PortfolioBacktestEngine",
    "PortfolioBacktestReport",
    "RebalanceSnapshot",
    "Strategy",
    "WalkForwardAnalyzer",
    "WalkForwardConfig",
    "WalkForwardResult",
]
