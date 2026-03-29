use std::path::Path;

use chrono::NaiveDate;
use duckdb::{params, AccessMode, Config, Connection};

use crate::error::DataError;
use crate::models::OhlcvRecord;

// Store dates as VARCHAR (ISO 8601) to avoid DuckDB Rust binding Date32 issues.
// Range queries on VARCHAR ISO dates work correctly because the format is
// lexicographically ordered.
const DDL: &str = "
CREATE TABLE IF NOT EXISTS ohlcv (
    symbol    VARCHAR NOT NULL,
    date      VARCHAR NOT NULL,
    open      DOUBLE  NOT NULL,
    high      DOUBLE  NOT NULL,
    low       DOUBLE  NOT NULL,
    close     DOUBLE  NOT NULL,
    volume    DOUBLE  NOT NULL,
    adj_close DOUBLE  NOT NULL,
    PRIMARY KEY (symbol, date)
);
CREATE INDEX IF NOT EXISTS idx_ohlcv_date ON ohlcv (date);
";

/// Persistent DuckDB store for OHLCV time-series data.
pub struct MarketDataStore {
    conn: Connection,
}

impl MarketDataStore {
    /// Open (or create) a DuckDB database at `db_path`.
    /// Pass `":memory:"` for an in-memory store (useful in tests).
    pub fn open(db_path: impl AsRef<Path>) -> Result<Self, DataError> {
        let path = db_path.as_ref();
        let conn = if path == Path::new(":memory:") {
            Connection::open_in_memory()?
        } else {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).map_err(|e| DataError::Parse(e.to_string()))?;
            }
            Connection::open(path)?
        };
        conn.execute_batch(DDL)?;
        Ok(Self { conn })
    }

    /// Open an existing DuckDB database in read-only mode.
    /// Multiple read-only connections can coexist with a single read-write writer.
    /// Used by `quant run` and `quant-api` to avoid lock conflicts with ingest jobs.
    pub fn open_read_only(db_path: impl AsRef<Path>) -> Result<Self, DataError> {
        let config = Config::default()
            .access_mode(AccessMode::ReadOnly)
            .map_err(|e| DataError::DuckDb(e))?;
        let conn = Connection::open_with_flags(db_path, config)?;
        Ok(Self { conn })
    }

    /// Insert or replace OHLCV records.  Returns the number of rows written.
    pub fn upsert(&self, records: &[OhlcvRecord]) -> Result<usize, DataError> {
        if records.is_empty() {
            return Ok(0);
        }
        let mut stmt = self.conn.prepare(
            "INSERT OR REPLACE INTO ohlcv
             (symbol, date, open, high, low, close, volume, adj_close)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        )?;
        let mut count = 0usize;
        for r in records {
            stmt.execute(params![
                r.symbol,
                r.date.format("%Y-%m-%d").to_string(),
                r.open,
                r.high,
                r.low,
                r.close,
                r.volume,
                r.adj_close,
            ])?;
            count += 1;
        }
        Ok(count)
    }

    /// Return OHLCV bars for `symbol` in the inclusive date range [start, end].
    pub fn query(
        &self,
        symbol: &str,
        start: NaiveDate,
        end: NaiveDate,
    ) -> Result<Vec<OhlcvRecord>, DataError> {
        let mut stmt = self.conn.prepare(
            "SELECT symbol, date, open, high, low, close, volume, adj_close
             FROM ohlcv
             WHERE symbol = ? AND date >= ? AND date <= ?
             ORDER BY date",
        )?;
        let rows = stmt.query_map(
            params![
                symbol.to_uppercase(),
                start.format("%Y-%m-%d").to_string(),
                end.format("%Y-%m-%d").to_string(),
            ],
            row_to_record,
        )?;

        rows.map(|r| r.map_err(DataError::DuckDb)).collect()
    }

    /// Return the most recent date stored for `symbol`, or `None`.
    pub fn latest_date(&self, symbol: &str) -> Result<Option<NaiveDate>, DataError> {
        let mut stmt = self
            .conn
            .prepare("SELECT MAX(date) FROM ohlcv WHERE symbol = ?")?;
        let date_str: Option<String> = stmt
            .query_row(params![symbol.to_uppercase()], |row| row.get(0))
            .unwrap_or(None);
        match date_str {
            None => Ok(None),
            Some(s) => {
                let d = NaiveDate::parse_from_str(&s, "%Y-%m-%d")
                    .map_err(|e| DataError::Parse(e.to_string()))?;
                Ok(Some(d))
            }
        }
    }

    /// Return all distinct symbols present in the store.
    pub fn symbols(&self) -> Result<Vec<String>, DataError> {
        let mut stmt = self
            .conn
            .prepare("SELECT DISTINCT symbol FROM ohlcv ORDER BY symbol")?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
        rows.map(|r| r.map_err(DataError::DuckDb)).collect()
    }

    /// Return row count, optionally filtered by symbol.
    pub fn count(&self, symbol: Option<&str>) -> Result<i64, DataError> {
        match symbol {
            Some(sym) => {
                let mut stmt = self
                    .conn
                    .prepare("SELECT COUNT(*) FROM ohlcv WHERE symbol = ?")?;
                Ok(stmt.query_row(params![sym.to_uppercase()], |row| row.get(0))?)
            }
            None => {
                let mut stmt = self.conn.prepare("SELECT COUNT(*) FROM ohlcv")?;
                Ok(stmt.query_row([], |row| row.get(0))?)
            }
        }
    }

    /// Return Mon–Fri dates in `expected_dates` that are absent from storage.
    pub fn coverage_gaps(
        &self,
        symbol: &str,
        start: NaiveDate,
        end: NaiveDate,
        expected_dates: &[NaiveDate],
    ) -> Result<Vec<NaiveDate>, DataError> {
        let stored = self.query(symbol, start, end)?;
        let stored_set: std::collections::HashSet<NaiveDate> =
            stored.iter().map(|r| r.date).collect();
        Ok(expected_dates
            .iter()
            .filter(|d| !stored_set.contains(d))
            .copied()
            .collect())
    }
}

fn row_to_record(row: &duckdb::Row<'_>) -> Result<OhlcvRecord, duckdb::Error> {
    let symbol: String = row.get(0)?;
    let date_str: String = row.get(1)?;
    let open: f64 = row.get(2)?;
    let high: f64 = row.get(3)?;
    let low: f64 = row.get(4)?;
    let close: f64 = row.get(5)?;
    let volume: f64 = row.get(6)?;
    let adj_close: f64 = row.get(7)?;

    // Parse date outside of the closure to avoid Error type mismatch.
    // We'll handle the parse error in the caller.
    let date = NaiveDate::parse_from_str(&date_str, "%Y-%m-%d")
        .unwrap_or_else(|_| NaiveDate::from_ymd_opt(1970, 1, 1).unwrap());

    Ok(OhlcvRecord {
        symbol,
        date,
        open,
        high,
        low,
        close,
        volume,
        adj_close,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    fn nd(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap()
    }

    fn sample_record(symbol: &str, date: NaiveDate) -> OhlcvRecord {
        OhlcvRecord {
            symbol: symbol.to_string(),
            date,
            open: 100.0,
            high: 105.0,
            low: 99.0,
            close: 103.0,
            volume: 1_000_000.0,
            adj_close: 103.0,
        }
    }

    #[test]
    fn roundtrip_upsert_and_query() {
        let store = MarketDataStore::open(":memory:").unwrap();
        let records = vec![
            sample_record("AAPL", nd(2024, 1, 2)),
            sample_record("AAPL", nd(2024, 1, 3)),
            sample_record("MSFT", nd(2024, 1, 2)),
        ];
        let written = store.upsert(&records).unwrap();
        assert_eq!(written, 3);

        let aapl = store
            .query("AAPL", nd(2024, 1, 1), nd(2024, 1, 31))
            .unwrap();
        assert_eq!(aapl.len(), 2);
        assert_eq!(aapl[0].close, 103.0);
    }

    #[test]
    fn upsert_is_idempotent() {
        let store = MarketDataStore::open(":memory:").unwrap();
        let rec = sample_record("AAPL", nd(2024, 1, 2));
        store.upsert(std::slice::from_ref(&rec)).unwrap();
        store.upsert(&[rec]).unwrap();
        assert_eq!(store.count(Some("AAPL")).unwrap(), 1);
    }

    #[test]
    fn latest_date_none_when_empty() {
        let store = MarketDataStore::open(":memory:").unwrap();
        assert!(store.latest_date("AAPL").unwrap().is_none());
    }

    #[test]
    fn latest_date_returns_max() {
        let store = MarketDataStore::open(":memory:").unwrap();
        store
            .upsert(&[
                sample_record("AAPL", nd(2024, 1, 2)),
                sample_record("AAPL", nd(2024, 1, 5)),
            ])
            .unwrap();
        assert_eq!(store.latest_date("AAPL").unwrap(), Some(nd(2024, 1, 5)));
    }

    #[test]
    fn symbols_returns_distinct() {
        let store = MarketDataStore::open(":memory:").unwrap();
        store
            .upsert(&[
                sample_record("AAPL", nd(2024, 1, 2)),
                sample_record("MSFT", nd(2024, 1, 2)),
            ])
            .unwrap();
        let syms = store.symbols().unwrap();
        assert_eq!(syms, vec!["AAPL", "MSFT"]);
    }

    #[test]
    fn coverage_gaps_detects_missing() {
        let store = MarketDataStore::open(":memory:").unwrap();
        store
            .upsert(&[sample_record("AAPL", nd(2024, 1, 2))])
            .unwrap();
        let expected = vec![nd(2024, 1, 2), nd(2024, 1, 3)];
        let gaps = store
            .coverage_gaps("AAPL", nd(2024, 1, 1), nd(2024, 1, 31), &expected)
            .unwrap();
        assert_eq!(gaps, vec![nd(2024, 1, 3)]);
    }
}
