"""Market data ingestion pipeline orchestrator.

Coordinates fetch -> validate -> store with incremental/full-refresh support.
Designed to run as a daily scheduled job (or on-demand catch-up).

Usage
-----
pipeline = IngestionPipeline(store=MarketDataStore(db_path), source=YahooFinanceSource())
result = pipeline.run(symbols=UNIVERSE, mode="incremental")
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal, Sequence

import pandas as pd
from loguru import logger

from quant.data.ingest.base import DataSource, OHLCVRecord
from quant.data.storage.duckdb import MarketDataStore
from quant.data.validation import ValidationResult, validate

# US equity market calendar approximation: Mon–Fri, skipping no holidays
# A proper implementation would use pandas_market_calendars, but we keep
# dependencies minimal here and let gaps indicate non-trading days.
_LOOKBACK_DAYS_INCREMENTAL = 5  # re-fetch last N calendar days for corrections


@dataclass
class PipelineResult:
    symbols_processed: int
    records_fetched: int
    records_valid: int
    records_invalid: int
    records_stored: int
    gaps_detected: dict[str, list[date]]

    def summary(self) -> str:
        gap_count = sum(len(v) for v in self.gaps_detected.values())
        return (
            f"symbols={self.symbols_processed} "
            f"fetched={self.records_fetched} "
            f"valid={self.records_valid} "
            f"invalid={self.records_invalid} "
            f"stored={self.records_stored} "
            f"gaps={gap_count}"
        )


class IngestionPipeline:
    """Fetch, validate, and store OHLCV data.

    Args:
        store: MarketDataStore instance (owns lifecycle).
        source: DataSource implementation to pull from.
        reject_invalid: If True, invalid records are not stored. Default True.
    """

    def __init__(
        self,
        store: MarketDataStore,
        source: DataSource,
        reject_invalid: bool = True,
    ) -> None:
        self._store = store
        self._source = source
        self._reject_invalid = reject_invalid

    def run(
        self,
        symbols: Sequence[str],
        mode: Literal["incremental", "full"] = "incremental",
        start: date | None = None,
        end: date | None = None,
    ) -> PipelineResult:
        """Run the ingestion pipeline.

        Args:
            symbols: Ticker symbols to ingest.
            mode:
                "incremental" — fetch from each symbol's latest stored date
                    (or last 5 calendar days if no history). Efficient for daily runs.
                "full" — fetch entire range defined by start/end.
            start: Used with mode="full". Defaults to 2020-01-01.
            end: Defaults to today.

        Returns:
            PipelineResult with counters and any detected gaps.
        """
        today = date.today()
        end_date = end or today

        symbols = [s.upper() for s in symbols]
        logger.info(
            "Pipeline run: mode={} symbols={} end={}",
            mode,
            len(symbols),
            end_date,
        )

        if mode == "incremental":
            records = self._run_incremental(symbols, end_date)
        else:
            start_date = start or date(2020, 1, 1)
            records = self._run_full(symbols, start_date, end_date)

        # Validate
        vr: ValidationResult = validate(records)
        records_to_store = vr.valid if self._reject_invalid else records

        # Store
        stored = self._store.upsert(records_to_store)

        # Gap detection (per-symbol, last 30 days)
        gaps = self._detect_gaps(symbols, end_date)

        result = PipelineResult(
            symbols_processed=len(symbols),
            records_fetched=len(records),
            records_valid=len(vr.valid),
            records_invalid=len(records) - len(vr.valid),
            records_stored=stored,
            gaps_detected=gaps,
        )
        logger.info("Pipeline complete: {}", result.summary())
        return result

    def _run_incremental(
        self,
        symbols: list[str],
        end_date: date,
    ) -> list[OHLCVRecord]:
        """Fetch from each symbol's latest known date."""
        # Find the earliest "start" we need across all symbols
        fallback_start = end_date - timedelta(days=_LOOKBACK_DAYS_INCREMENTAL)
        min_start = end_date  # will be updated below

        symbol_starts: dict[str, date] = {}
        for sym in symbols:
            latest = self._store.latest_date(sym)
            if latest is None:
                symbol_starts[sym] = fallback_start
            else:
                # Re-fetch from latest - 1 to catch late-arriving corrections
                symbol_starts[sym] = latest - timedelta(days=1)
            if symbol_starts[sym] < min_start:
                min_start = symbol_starts[sym]

        # Batch fetch all with the widest window, then filter per-symbol
        all_records = self._source.fetch(symbols, min_start, end_date)

        # Filter to only include dates >= each symbol's start
        filtered = [
            r for r in all_records
            if r.date >= symbol_starts.get(r.symbol, fallback_start)
        ]
        logger.debug(
            "Incremental: fetched={} filtered={}", len(all_records), len(filtered)
        )
        return filtered

    def _run_full(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date,
    ) -> list[OHLCVRecord]:
        return self._source.fetch(symbols, start_date, end_date)

    def _detect_gaps(
        self,
        symbols: list[str],
        end_date: date,
        lookback_days: int = 30,
    ) -> dict[str, list[date]]:
        """Check for missing business days in the last *lookback_days* calendar days."""
        start = end_date - timedelta(days=lookback_days)
        expected = _business_days(start, end_date)

        gaps: dict[str, list[date]] = {}
        for sym in symbols:
            missing = self._store.coverage_gaps(sym, start, end_date, expected)
            if missing:
                gaps[sym] = missing
                logger.warning(
                    "{}: {} gap(s) detected in last {} days: {}",
                    sym,
                    len(missing),
                    lookback_days,
                    missing[:5],
                )
        return gaps


def _business_days(start: date, end: date) -> list[date]:
    """Return Mon–Fri dates in [start, end]. Does not account for holidays."""
    days: list[date] = []
    current = start
    while current <= end:
        if current.weekday() < 5:  # 0=Mon, 4=Fri
            days.append(current)
        current += timedelta(days=1)
    return days
