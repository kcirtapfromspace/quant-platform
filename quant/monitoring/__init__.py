"""Prometheus metrics instrumentation for the quant platform.

Exposes metrics for:
  - Data ingestion pipeline (data lag, rows inserted, errors)
  - Execution / OMS layer (order counts, fill rate, latency)
  - Risk engine (drawdown, circuit breaker state)

Usage — long-running service (HTTP scrape endpoint)::

    from quant.monitoring import start_metrics_server
    start_metrics_server(port=8000)

Usage — batch job (Pushgateway)::

    from quant.monitoring import push_metrics
    push_metrics(job="quant-ingest", gateway="http://localhost:9091")
"""

from quant.monitoring.metrics import (
    # Ingest
    INGEST_DATA_LAG,
    INGEST_ROWS_INSERTED,
    INGEST_ERRORS,
    INGEST_RECORDS_FETCHED,
    INGEST_RECORDS_INVALID,
    INGEST_DURATION,
    INGEST_LAST_RUN_TIMESTAMP,
    # Execution / OMS
    ORDERS_SUBMITTED,
    ORDERS_FILLED,
    ORDERS_REJECTED,
    FILL_RATE,
    ORDER_LATENCY,
    POSITIONS_OPEN,
    # Risk
    CIRCUIT_BREAKER_TRIPPED,
    DRAWDOWN_CURRENT,
    # Helpers
    start_metrics_server,
    push_metrics,
)

__all__ = [
    "INGEST_DATA_LAG",
    "INGEST_ROWS_INSERTED",
    "INGEST_ERRORS",
    "INGEST_RECORDS_FETCHED",
    "INGEST_RECORDS_INVALID",
    "INGEST_DURATION",
    "INGEST_LAST_RUN_TIMESTAMP",
    "ORDERS_SUBMITTED",
    "ORDERS_FILLED",
    "ORDERS_REJECTED",
    "FILL_RATE",
    "ORDER_LATENCY",
    "POSITIONS_OPEN",
    "CIRCUIT_BREAKER_TRIPPED",
    "DRAWDOWN_CURRENT",
    "start_metrics_server",
    "push_metrics",
]
