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
    # Risk
    CIRCUIT_BREAKER_TRIPPED,
    DRAWDOWN_CURRENT,
    FILL_RATE,
    # Ingest
    INGEST_DATA_LAG,
    INGEST_DURATION,
    INGEST_ERRORS,
    INGEST_LAST_RUN_TIMESTAMP,
    INGEST_RECORDS_FETCHED,
    INGEST_RECORDS_INVALID,
    INGEST_ROWS_INSERTED,
    ORDER_LATENCY,
    ORDERS_FILLED,
    ORDERS_REJECTED,
    # Execution / OMS
    ORDERS_SUBMITTED,
    POSITIONS_OPEN,
    RUNNER_LAST_RUN_TIMESTAMP,
    RUNNER_PORTFOLIO_VALUE,
    RUNNER_PORTFOLIO_VOLATILITY,
    RUNNER_PREFLIGHT_FAILURES,
    RUNNER_RUN_DURATION,
    # Runner
    RUNNER_RUNS_TOTAL,
    RUNNER_TRADES_REJECTED,
    RUNNER_TRADES_SUBMITTED,
    RUNNER_TURNOVER,
    push_metrics,
    # Helpers
    start_metrics_server,
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
    "RUNNER_RUNS_TOTAL",
    "RUNNER_RUN_DURATION",
    "RUNNER_TRADES_SUBMITTED",
    "RUNNER_TRADES_REJECTED",
    "RUNNER_PORTFOLIO_VALUE",
    "RUNNER_PORTFOLIO_VOLATILITY",
    "RUNNER_TURNOVER",
    "RUNNER_LAST_RUN_TIMESTAMP",
    "RUNNER_PREFLIGHT_FAILURES",
    "start_metrics_server",
    "push_metrics",
]
