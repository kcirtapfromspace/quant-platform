"""Prometheus metric definitions for the quant platform.

All metrics are registered in the default CollectorRegistry.  Import the
constants directly — do not create additional instances of the same metric
name, as prometheus_client will raise a ValueError on duplicate registration.
"""
from __future__ import annotations

import os

from loguru import logger

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        start_http_server,
        push_to_gateway,
        CollectorRegistry,
        REGISTRY,
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    logger.warning(
        "prometheus_client not installed — metrics will be no-ops. "
        "Install with: pip install prometheus-client"
    )


def _make_counter(name: str, doc: str, labelnames: list[str] | None = None):
    if not _PROMETHEUS_AVAILABLE:
        return _NoOpMetric()
    return Counter(name, doc, labelnames or [])


def _make_gauge(name: str, doc: str, labelnames: list[str] | None = None):
    if not _PROMETHEUS_AVAILABLE:
        return _NoOpMetric()
    return Gauge(name, doc, labelnames or [])


def _make_histogram(
    name: str,
    doc: str,
    labelnames: list[str] | None = None,
    buckets: tuple | None = None,
):
    if not _PROMETHEUS_AVAILABLE:
        return _NoOpMetric()
    kwargs = {}
    if buckets is not None:
        kwargs["buckets"] = buckets
    return Histogram(name, doc, labelnames or [], **kwargs)


class _NoOpMetric:
    """Drop-in stub when prometheus_client is unavailable."""

    def labels(self, **_):
        return self

    def inc(self, *_, **__):
        pass

    def set(self, *_, **__):
        pass

    def observe(self, *_, **__):
        pass

    def time(self):
        import contextlib
        return contextlib.nullcontext()


# ── Ingest pipeline metrics ───────────────────────────────────────────────────

INGEST_DATA_LAG = _make_gauge(
    "quant_ingest_data_lag_seconds",
    "Seconds since the last successful ingest run produced new data.",
)

INGEST_ROWS_INSERTED = _make_counter(
    "quant_ingest_rows_inserted_total",
    "Total OHLCV rows inserted into the market data store.",
)

INGEST_ERRORS = _make_counter(
    "quant_ingest_errors_total",
    "Total ingest pipeline errors (fetch failures, validation failures, store errors).",
    labelnames=["stage"],  # fetch | validate | store
)

INGEST_RECORDS_FETCHED = _make_counter(
    "quant_ingest_records_fetched_total",
    "Total raw OHLCV records fetched from the upstream data source.",
)

INGEST_RECORDS_INVALID = _make_counter(
    "quant_ingest_records_invalid_total",
    "Total OHLCV records that failed validation and were rejected.",
)

INGEST_DURATION = _make_histogram(
    "quant_ingest_duration_seconds",
    "End-to-end duration of a single ingest pipeline run.",
    buckets=(1, 5, 10, 30, 60, 120, 300, 600),
)

INGEST_LAST_RUN_TIMESTAMP = _make_gauge(
    "quant_ingest_last_run_timestamp_seconds",
    "Unix timestamp of the last completed ingest pipeline run.",
)

# ── Execution / OMS metrics ───────────────────────────────────────────────────

ORDERS_SUBMITTED = _make_counter(
    "quant_orders_submitted_total",
    "Total orders submitted to the broker.",
    labelnames=["symbol", "side"],  # side: buy | sell
)

ORDERS_FILLED = _make_counter(
    "quant_orders_filled_total",
    "Total orders fully filled.",
    labelnames=["symbol", "side"],
)

ORDERS_REJECTED = _make_counter(
    "quant_orders_rejected_total",
    "Total orders rejected by the broker.",
    labelnames=["symbol"],
)

FILL_RATE = _make_gauge(
    "quant_fill_rate",
    "Rolling fill rate: proportion of submitted orders that have been filled (0–1).",
)

ORDER_LATENCY = _make_histogram(
    "quant_order_latency_seconds",
    "Elapsed seconds from order submission to first fill.",
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
)

POSITIONS_OPEN = _make_gauge(
    "quant_positions_open",
    "Number of currently open positions tracked by the OMS.",
)

# ── Risk metrics ──────────────────────────────────────────────────────────────

CIRCUIT_BREAKER_TRIPPED = _make_gauge(
    "quant_circuit_breaker_tripped",
    "1 if the drawdown circuit breaker has halted trading, 0 otherwise.",
)

DRAWDOWN_CURRENT = _make_gauge(
    "quant_drawdown_current",
    "Current portfolio drawdown as a fraction of peak value (0–1).",
)


# ── Server / push helpers ─────────────────────────────────────────────────────

def start_metrics_server(port: int = 8000) -> None:
    """Start a Prometheus HTTP scrape endpoint on *port* (blocking-safe).

    Call once at service startup.  Grafana/Prometheus polls this endpoint.

    Args:
        port: TCP port to listen on (default 8000).
    """
    if not _PROMETHEUS_AVAILABLE:
        logger.warning("prometheus_client unavailable — metrics server not started")
        return
    start_http_server(port)
    logger.info("Prometheus metrics server started on :{}", port)


def push_metrics(
    job: str = "quant-ingest",
    gateway: str | None = None,
    grouping_key: dict | None = None,
) -> None:
    """Push all metrics to the Prometheus Pushgateway.

    Use this for short-lived batch processes (e.g. the ingest pipeline) that
    exit before Prometheus can scrape them.

    Args:
        job: Job label for the Pushgateway.
        gateway: Pushgateway URL.  Falls back to PUSHGATEWAY_URL env var,
            then "http://localhost:9091".
        grouping_key: Optional extra labels for multi-instance grouping.
    """
    if not _PROMETHEUS_AVAILABLE:
        logger.warning("prometheus_client unavailable — metrics not pushed")
        return

    url = gateway or os.environ.get("PUSHGATEWAY_URL", "http://localhost:9091")
    try:
        push_to_gateway(url, job=job, registry=REGISTRY, grouping_key=grouping_key or {})
        logger.debug("Metrics pushed to Pushgateway at {}", url)
    except Exception as exc:
        logger.warning("Failed to push metrics to Pushgateway: {}", exc)
