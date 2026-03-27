"""Tests for DuckDB market data store."""
from datetime import date

import pytest

from quant.data.ingest.base import OHLCVRecord
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


@pytest.fixture
def store():
    with MarketDataStore(":memory:") as s:
        yield s


def test_upsert_and_query(store):
    records = [
        _record("AAPL", date(2024, 1, 2), 185.0),
        _record("AAPL", date(2024, 1, 3), 187.0),
    ]
    store.upsert(records)
    df = store.query("AAPL", date(2024, 1, 1), date(2024, 1, 31))
    assert len(df) == 2
    assert list(df["close"]) == [185.0, 187.0]


def test_upsert_replaces_duplicate(store):
    r1 = _record("AAPL", date(2024, 1, 2), 185.0)
    r2 = _record("AAPL", date(2024, 1, 2), 190.0)
    store.upsert([r1])
    store.upsert([r2])
    df = store.query("AAPL", date(2024, 1, 1), date(2024, 1, 31))
    assert len(df) == 1
    assert df.iloc[0]["close"] == 190.0


def test_latest_date_returns_none_for_unknown(store):
    assert store.latest_date("UNKNOWN") is None


def test_latest_date_returns_max(store):
    store.upsert([
        _record("MSFT", date(2024, 1, 2)),
        _record("MSFT", date(2024, 1, 5)),
        _record("MSFT", date(2024, 1, 3)),
    ])
    assert store.latest_date("MSFT") == date(2024, 1, 5)


def test_symbols_list(store):
    store.upsert([_record("AAPL", date(2024, 1, 2))])
    store.upsert([_record("MSFT", date(2024, 1, 2))])
    assert store.symbols() == ["AAPL", "MSFT"]


def test_count(store):
    store.upsert([
        _record("AAPL", date(2024, 1, 2)),
        _record("AAPL", date(2024, 1, 3)),
        _record("MSFT", date(2024, 1, 2)),
    ])
    assert store.count() == 3
    assert store.count("AAPL") == 2
    assert store.count("MSFT") == 1


def test_coverage_gaps(store):
    store.upsert([
        _record("AAPL", date(2024, 1, 2)),
        _record("AAPL", date(2024, 1, 4)),  # missing Jan 3
    ])
    expected = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    gaps = store.coverage_gaps("AAPL", date(2024, 1, 1), date(2024, 1, 5), expected)
    assert gaps == [date(2024, 1, 3)]


def test_query_multi(store):
    store.upsert([
        _record("AAPL", date(2024, 1, 2)),
        _record("MSFT", date(2024, 1, 2)),
        _record("GOOGL", date(2024, 1, 2)),
    ])
    df = store.query_multi(["AAPL", "MSFT"], date(2024, 1, 1), date(2024, 1, 31))
    assert set(df["symbol"].unique()) == {"AAPL", "MSFT"}


def test_empty_upsert(store):
    count = store.upsert([])
    assert count == 0
