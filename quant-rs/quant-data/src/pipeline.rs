use chrono::{Datelike, NaiveDate, Weekday};

use crate::error::DataError;
use crate::ingest::DataSource;
use crate::models::OhlcvRecord;
use crate::store::MarketDataStore;

/// Summary of a completed ingestion run.
#[derive(Debug, Clone)]
pub struct PipelineResult {
    pub symbols_processed: usize,
    pub records_fetched: usize,
    pub records_stored: usize,
    pub gaps_detected: Vec<(String, Vec<NaiveDate>)>,
}

impl PipelineResult {
    pub fn summary(&self) -> String {
        let gap_count: usize = self.gaps_detected.iter().map(|(_, v)| v.len()).sum();
        format!(
            "symbols={} fetched={} stored={} gaps={}",
            self.symbols_processed,
            self.records_fetched,
            self.records_stored,
            gap_count
        )
    }
}

/// Ingestion mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IngestMode {
    /// Re-fetch from each symbol's latest stored date (efficient for daily crons).
    Incremental,
    /// Fetch the full date range [start, end].
    Full,
}

const LOOKBACK_DAYS_INCREMENTAL: i64 = 5;

/// Orchestrates fetch → store → gap-detection for OHLCV data.
pub struct IngestionPipeline<S: DataSource> {
    store: MarketDataStore,
    source: S,
}

impl<S: DataSource> IngestionPipeline<S> {
    pub fn new(store: MarketDataStore, source: S) -> Self {
        Self { store, source }
    }

    /// Run the pipeline.
    ///
    /// * `mode = Incremental` — fetch from each symbol's latest date.
    /// * `mode = Full` — fetch `[start, end]`.
    pub fn run(
        &self,
        symbols: &[String],
        mode: IngestMode,
        start: Option<NaiveDate>,
        end: Option<NaiveDate>,
    ) -> Result<PipelineResult, DataError> {
        let today = chrono::Local::now().date_naive();
        let end_date = end.unwrap_or(today);
        let symbols_upper: Vec<String> = symbols.iter().map(|s| s.to_uppercase()).collect();

        let records = if mode == IngestMode::Incremental {
            self.run_incremental(&symbols_upper, end_date)?
        } else {
            let start_date =
                start.unwrap_or_else(|| NaiveDate::from_ymd_opt(2020, 1, 1).unwrap());
            self.source.fetch(&symbols_upper, start_date, end_date)?
        };

        let fetched = records.len();
        let stored = self.store.upsert(&records)?;

        let gaps = self.detect_gaps(&symbols_upper, end_date, 30)?;

        Ok(PipelineResult {
            symbols_processed: symbols_upper.len(),
            records_fetched: fetched,
            records_stored: stored,
            gaps_detected: gaps,
        })
    }

    fn run_incremental(
        &self,
        symbols: &[String],
        end_date: NaiveDate,
    ) -> Result<Vec<OhlcvRecord>, DataError> {
        let fallback = end_date - chrono::Duration::days(LOOKBACK_DAYS_INCREMENTAL);
        let mut min_start = end_date;
        let mut symbol_starts: std::collections::HashMap<&str, NaiveDate> =
            std::collections::HashMap::new();

        for sym in symbols {
            let sym_start = match self.store.latest_date(sym)? {
                Some(latest) => latest - chrono::Duration::days(1),
                None => fallback,
            };
            if sym_start < min_start {
                min_start = sym_start;
            }
            symbol_starts.insert(sym.as_str(), sym_start);
        }

        let all = self.source.fetch(symbols, min_start, end_date)?;
        let filtered: Vec<OhlcvRecord> = all
            .into_iter()
            .filter(|r| {
                r.date
                    >= *symbol_starts
                        .get(r.symbol.as_str())
                        .unwrap_or(&fallback)
            })
            .collect();
        Ok(filtered)
    }

    fn detect_gaps(
        &self,
        symbols: &[String],
        end_date: NaiveDate,
        lookback_days: i64,
    ) -> Result<Vec<(String, Vec<NaiveDate>)>, DataError> {
        let start = end_date - chrono::Duration::days(lookback_days);
        let expected = business_days(start, end_date);
        let mut gaps = Vec::new();

        for sym in symbols {
            let missing = self.store.coverage_gaps(sym, start, end_date, &expected)?;
            if !missing.is_empty() {
                gaps.push((sym.clone(), missing));
            }
        }

        Ok(gaps)
    }
}

/// Return Mon–Fri dates in the inclusive range [start, end].
/// Does not account for public holidays.
pub fn business_days(start: NaiveDate, end: NaiveDate) -> Vec<NaiveDate> {
    let mut days = Vec::new();
    let mut current = start;
    while current <= end {
        match current.weekday() {
            Weekday::Sat | Weekday::Sun => {}
            _ => days.push(current),
        }
        current = current.succ_opt().unwrap_or(current);
        if current == days.last().copied().unwrap_or(current) {
            break;
        }
    }
    days
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::DataSource;
    use crate::models::OhlcvRecord;

    struct MockSource(Vec<OhlcvRecord>);

    impl DataSource for MockSource {
        fn name(&self) -> &str {
            "mock"
        }
        fn fetch(
            &self,
            _symbols: &[String],
            _start: NaiveDate,
            _end: NaiveDate,
        ) -> Result<Vec<OhlcvRecord>, DataError> {
            Ok(self.0.clone())
        }
    }

    fn nd(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap()
    }

    fn rec(sym: &str, date: NaiveDate) -> OhlcvRecord {
        OhlcvRecord {
            symbol: sym.to_string(),
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
    fn full_mode_stores_all_records() {
        let store = MarketDataStore::open(":memory:").unwrap();
        let records = vec![rec("AAPL", nd(2024, 1, 2)), rec("AAPL", nd(2024, 1, 3))];
        let source = MockSource(records);
        let pipeline = IngestionPipeline::new(store, source);

        let symbols = vec!["AAPL".to_string()];
        let result = pipeline
            .run(&symbols, IngestMode::Full, None, Some(nd(2024, 1, 3)))
            .unwrap();

        assert_eq!(result.records_fetched, 2);
        assert_eq!(result.records_stored, 2);
    }

    #[test]
    fn business_days_excludes_weekends() {
        let days = business_days(nd(2024, 1, 1), nd(2024, 1, 7));
        // 2024-01-01 is Monday; 2024-01-06 Sat, 2024-01-07 Sun
        assert!(days.iter().all(|d| {
            !matches!(d.weekday(), Weekday::Sat | Weekday::Sun)
        }));
        assert_eq!(days.len(), 5); // Mon–Fri
    }
}
