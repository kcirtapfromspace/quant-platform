use chrono::NaiveDate;
use serde::{Deserialize, Serialize};

/// A single OHLCV bar for one symbol on one trading day.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OhlcvRecord {
    pub symbol: String,
    pub date: NaiveDate,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    /// Adjusted close (split/dividend corrected).  Equal to `close` when the
    /// source has already applied adjustments (Yahoo `auto_adjust=true`).
    pub adj_close: f64,
}
