"""Prometheus-instrumented wrapper for the ingest pipeline.

Wraps :class:`~quant.data.pipeline.IngestionPipeline` to record:
  - ``quant_ingest_data_lag_seconds`` — time since last successful run
  - ``quant_ingest_rows_inserted_total`` — cumulative rows stored
  - ``quant_ingest_records_fetched_total`` — raw records fetched
  - ``quant_ingest_records_invalid_total`` — validation rejects
  - ``quant_ingest_errors_total`` — pipeline errors by stage
  - ``quant_ingest_duration_seconds`` — run wall-clock time
  - ``quant_ingest_last_run_timestamp_seconds`` — epoch of last run

Usage::

    from quant.data.pipeline import IngestionPipeline
    from quant.data.storage.duckdb import MarketDataStore
    from quant.data.ingest.yahoo import YahooFinanceSource
    from quant.monitoring.ingest_metrics import InstrumentedIngestionPipeline
    from quant.monitoring import push_metrics

    pipeline = InstrumentedIngestionPipeline(
        store=MarketDataStore("market_data.db"),
        source=YahooFinanceSource(),
    )
    result = pipeline.run(symbols=["AAPL", "MSFT"])
    push_metrics(job="quant-ingest")
"""
from __future__ import annotations

import time
from datetime import date
from typing import Literal, Sequence

from loguru import logger

from quant.data.pipeline import IngestionPipeline, PipelineResult
from quant.data.ingest.base import DataSource
from quant.data.storage.duckdb import MarketDataStore
from quant.monitoring.metrics import (
    INGEST_DATA_LAG,
    INGEST_ROWS_INSERTED,
    INGEST_ERRORS,
    INGEST_RECORDS_FETCHED,
    INGEST_RECORDS_INVALID,
    INGEST_DURATION,
    INGEST_LAST_RUN_TIMESTAMP,
)


class InstrumentedIngestionPipeline(IngestionPipeline):
    """IngestionPipeline subclass that publishes Prometheus metrics on each run.

    Drop-in replacement for :class:`~quant.data.pipeline.IngestionPipeline`.
    All constructor arguments are identical.
    """

    def run(
        self,
        symbols: Sequence[str],
        mode: Literal["incremental", "full"] = "incremental",
        start: date | None = None,
        end: date | None = None,
    ) -> PipelineResult:
        """Run the pipeline and record metrics.

        Raises any exception from the underlying pipeline after updating the
        error counter.
        """
        t_start = time.monotonic()
        try:
            result = super().run(symbols=symbols, mode=mode, start=start, end=end)
        except Exception as exc:
            INGEST_ERRORS.labels(stage="pipeline").inc()
            logger.error("InstrumentedIngestionPipeline: unhandled error — {}", exc)
            raise

        elapsed = time.monotonic() - t_start
        now_epoch = time.time()

        # ── Counters ─────────────────────────────────────────────────────────
        INGEST_RECORDS_FETCHED.inc(result.records_fetched)
        INGEST_RECORDS_INVALID.inc(result.records_invalid)
        INGEST_ROWS_INSERTED.inc(result.records_stored)
        if result.records_invalid > 0:
            INGEST_ERRORS.labels(stage="validate").inc(result.records_invalid)

        # ── Duration histogram ────────────────────────────────────────────────
        INGEST_DURATION.observe(elapsed)

        # ── Timestamp + lag ───────────────────────────────────────────────────
        INGEST_LAST_RUN_TIMESTAMP.set(now_epoch)
        # Data lag = time since last run (reset to 0 immediately after a
        # successful run; a background job should update this gauge periodically
        # to reflect age of the most-recent stored record).
        INGEST_DATA_LAG.set(0)

        logger.debug(
            "InstrumentedIngestionPipeline: metrics updated — "
            "stored={} invalid={} duration={:.1f}s",
            result.records_stored,
            result.records_invalid,
            elapsed,
        )
        return result


def record_ingest_data_lag(store: MarketDataStore, symbols: Sequence[str]) -> None:
    """Compute and publish the data lag gauge.

    Call this on a schedule (e.g. every 60 s) from a long-running monitor
    process to keep the lag gauge fresh even when no pipeline run is active.

    The lag is defined as ``now - max(latest_date across symbols)``.

    Args:
        store: The market data store to query for latest available dates.
        symbols: Symbols to check (typically the active universe).
    """
    import datetime

    latest_date: date | None = None
    for sym in symbols:
        d = store.latest_date(sym)
        if d is not None and (latest_date is None or d > latest_date):
            latest_date = d

    if latest_date is None:
        # No data at all — set a large lag to trigger alerts
        INGEST_DATA_LAG.set(86400)
        return

    today = datetime.date.today()
    lag_days = (today - latest_date).days
    # Convert calendar days to seconds (approximate — 1 business day = 86400 s)
    lag_seconds = lag_days * 86400
    INGEST_DATA_LAG.set(max(0, lag_seconds))
