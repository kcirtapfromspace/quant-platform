"""Execution adapters — pluggable broker/exchange connectors implementing BrokerAdapter."""
from quant.execution.alpaca import AlpacaAdapter
from quant.execution.ccxt_adapter import CCXTAdapter
from quant.execution.ib import IBAdapter
from quant.execution.paper import PaperBrokerAdapter
from quant.execution.tca import ExecutionRecord, TCACollector, TCAReport

__all__ = [
    "AlpacaAdapter",
    "CCXTAdapter",
    "ExecutionRecord",
    "IBAdapter",
    "PaperBrokerAdapter",
    "TCACollector",
    "TCAReport",
]
