"""Backtesting framework for evaluating trading strategies.

Typical usage
-------------
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
"""
from quant.backtest.engine import BacktestConfig, BacktestEngine
from quant.backtest.report import BacktestReport
from quant.backtest.strategy import Strategy

__all__ = ["BacktestEngine", "BacktestConfig", "BacktestReport", "Strategy"]
