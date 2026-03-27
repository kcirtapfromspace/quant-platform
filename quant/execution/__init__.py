"""Execution adapters — pluggable broker/exchange connectors implementing BrokerAdapter."""
from quant.execution.paper import PaperBrokerAdapter
from quant.execution.alpaca import AlpacaAdapter
from quant.execution.ccxt_adapter import CCXTAdapter
from quant.execution.ib import IBAdapter

__all__ = [
    "PaperBrokerAdapter",
    "AlpacaAdapter",
    "CCXTAdapter",
    "IBAdapter",
]
