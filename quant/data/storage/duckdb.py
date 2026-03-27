"""DuckDB-backed time-series storage for OHLCV market data.

Schema: a single `ohlcv` table partitioned logically by (symbol, date).
Upsert semantics prevent duplicates; the table is append-optimised but
supports incremental updates.

Typical usage
-------------
store = MarketDataStore("/data/market.duckdb")
store.upsert(records)
df = store.query("AAPL", date(2024, 1, 1), date(2024, 12, 31))
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Generator, Sequence

import duckdb
import pandas as pd
from loguru import logger

from quant.data.ingest.base import OHLCVRecord

_DDL = """
CREATE TABLE IF NOT EXISTS ohlcv (
    symbol   VARCHAR      NOT NULL,
    date     DATE         NOT NULL,
    open     DOUBLE       NOT NULL,
    high     DOUBLE       NOT NULL,
    low      DOUBLE       NOT NULL,
    close    DOUBLE       NOT NULL,
    volume   DOUBLE       NOT NULL,
    adj_close DOUBLE      NOT NULL,
    PRIMARY KEY (symbol, date)
);

CREATE INDEX IF NOT EXISTS idx_ohlcv_date ON ohlcv (date);
"""

_UPSERT_SQL = """
INSERT OR REPLACE INTO ohlcv
    (symbol, date, open, high, low, close, volume, adj_close)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""


class MarketDataStore:
    """Persistent DuckDB store for OHLCV time-series data.

    Thread-safety: DuckDB connections are not thread-safe. Use one
    MarketDataStore per thread, or use the context manager for short-lived ops.

    Args:
        db_path: Path to the DuckDB file. Use ":memory:" for in-memory (tests).
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._db_path = str(db_path)
        if self._db_path != ":memory:":
            os.makedirs(Path(self._db_path).parent, exist_ok=True)
        self._conn = duckdb.connect(self._db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(_DDL)

    @contextmanager
    def transaction(self) -> Generator[None, None, None]:
        self._conn.execute("BEGIN")
        try:
            yield
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def upsert(self, records: Sequence[OHLCVRecord]) -> int:
        """Insert or replace OHLCV records.

        Returns the number of records written.
        """
        if not records:
            return 0

        rows = [
            (
                r.symbol,
                r.date,
                r.open,
                r.high,
                r.low,
                r.close,
                r.volume,
                r.adj_close,
            )
            for r in records
        ]

        with self.transaction():
            self._conn.executemany(_UPSERT_SQL, rows)

        logger.debug("Upserted {} OHLCV records", len(rows))
        return len(rows)

    def query(
        self,
        symbol: str,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Return a DataFrame of OHLCV bars for *symbol* in [start, end]."""
        result = self._conn.execute(
            """
            SELECT symbol, date, open, high, low, close, volume, adj_close
            FROM ohlcv
            WHERE symbol = ?
              AND date >= ?
              AND date <= ?
            ORDER BY date
            """,
            [symbol.upper(), start, end],
        ).df()
        return result

    def query_multi(
        self,
        symbols: Sequence[str],
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Return OHLCV data for multiple symbols."""
        sym_upper = [s.upper() for s in symbols]
        placeholders = ", ".join("?" * len(sym_upper))
        result = self._conn.execute(
            f"""
            SELECT symbol, date, open, high, low, close, volume, adj_close
            FROM ohlcv
            WHERE symbol IN ({placeholders})
              AND date >= ?
              AND date <= ?
            ORDER BY symbol, date
            """,
            [*sym_upper, start, end],
        ).df()
        return result

    def latest_date(self, symbol: str) -> date | None:
        """Return the most recent date stored for *symbol*, or None."""
        row = self._conn.execute(
            "SELECT MAX(date) FROM ohlcv WHERE symbol = ?",
            [symbol.upper()],
        ).fetchone()
        if row is None or row[0] is None:
            return None
        val = row[0]
        if isinstance(val, date):
            return val
        return pd.Timestamp(val).date()

    def symbols(self) -> list[str]:
        """Return all distinct symbols in the store."""
        rows = self._conn.execute(
            "SELECT DISTINCT symbol FROM ohlcv ORDER BY symbol"
        ).fetchall()
        return [r[0] for r in rows]

    def count(self, symbol: str | None = None) -> int:
        """Return total row count, optionally filtered by symbol."""
        if symbol:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM ohlcv WHERE symbol = ?", [symbol.upper()]
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) FROM ohlcv").fetchone()
        return row[0] if row else 0

    def coverage_gaps(
        self,
        symbol: str,
        start: date,
        end: date,
        expected_dates: Sequence[date],
    ) -> list[date]:
        """Return trading dates in *expected_dates* that are missing from storage."""
        df = self.query(symbol, start, end)
        stored = set(pd.to_datetime(df["date"]).dt.date) if not df.empty else set()
        return [d for d in expected_dates if d not in stored]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "MarketDataStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
