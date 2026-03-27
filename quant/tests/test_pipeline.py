"""Tests for the ingestion pipeline (mocked data source)."""
from datetime import date
from unittest.mock import MagicMock

import pytest

from quant.data.ingest.base import DataSource, OHLCVRecord
from quant.data.pipeline import IngestionPipeline, _business_days
from quant.data.storage.duckdb import MarketDataStore


def _record(symbol: str, dt: date, close: float = 100.0) -> OHLCVRecord:
    return OHLCVRecord(
        symbol=symbol,
        date=dt,
        open=close - 1,
        high=close + 2,
        low=close - 2,
        close=close,
        volume=1_000_000.0,
        adj_close=close,
    )


class MockSource(DataSource):
    def __init__(self, records: list[OHLCVRecord]) -> None:
        self._records = records

    @property
    def name(self) -> str:
        return "mock"

    def fetch(self, symbols, start, end) -> list[OHLCVRecord]:
        return [
            r for r in self._records
            if r.symbol in [s.upper() for s in symbols]
            and start <= r.date <= end
        ]


@pytest.fixture
def store():
    with MarketDataStore(":memory:") as s:
        yield s


def test_full_run_stores_records(store):
    records = [
        _record("AAPL", date(2024, 1, 2)),
        _record("AAPL", date(2024, 1, 3)),
        _record("MSFT", date(2024, 1, 2)),
    ]
    source = MockSource(records)
    pipeline = IngestionPipeline(store=store, source=source)
    result = pipeline.run(
        symbols=["AAPL", "MSFT"],
        mode="full",
        start=date(2024, 1, 1),
        end=date(2024, 1, 5),
    )
    assert result.records_fetched == 3
    assert result.records_valid == 3
    assert result.records_stored == 3
    assert result.symbols_processed == 2


def test_incremental_run_uses_latest_date(store):
    # Pre-populate with one record
    store.upsert([_record("AAPL", date(2024, 1, 2))])

    new_records = [
        _record("AAPL", date(2024, 1, 2)),  # already stored
        _record("AAPL", date(2024, 1, 3)),  # new
    ]
    source = MockSource(new_records)
    pipeline = IngestionPipeline(store=store, source=source)
    result = pipeline.run(
        symbols=["AAPL"],
        mode="incremental",
        end=date(2024, 1, 5),
    )
    assert result.records_fetched >= 1
    # After upsert, should have both dates
    df = store.query("AAPL", date(2024, 1, 1), date(2024, 1, 5))
    assert len(df) == 2


def test_invalid_records_rejected(store):
    import math
    records = [
        _record("AAPL", date(2024, 1, 2)),
        _record("AAPL", date(2024, 1, 3), close=math.nan),  # invalid
    ]
    source = MockSource(records)
    pipeline = IngestionPipeline(store=store, source=source, reject_invalid=True)
    result = pipeline.run(
        symbols=["AAPL"],
        mode="full",
        start=date(2024, 1, 1),
        end=date(2024, 1, 5),
    )
    assert result.records_invalid == 1
    assert result.records_stored == 1


def test_business_days():
    days = _business_days(date(2024, 1, 1), date(2024, 1, 7))
    # Jan 1 (Mon), Jan 2 (Tue), Jan 3 (Wed), Jan 4 (Thu), Jan 5 (Fri)
    # Jan 6 (Sat), Jan 7 (Sun) excluded
    assert date(2024, 1, 6) not in days
    assert date(2024, 1, 7) not in days
    assert date(2024, 1, 5) in days


def test_empty_source_returns_zero(store):
    source = MockSource([])
    pipeline = IngestionPipeline(store=store, source=source)
    result = pipeline.run(
        symbols=["AAPL"],
        mode="full",
        start=date(2024, 1, 1),
        end=date(2024, 1, 5),
    )
    assert result.records_fetched == 0
    assert result.records_stored == 0
