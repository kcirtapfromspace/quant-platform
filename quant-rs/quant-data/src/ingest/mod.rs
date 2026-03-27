pub mod yahoo;

use chrono::NaiveDate;

use crate::error::DataError;
use crate::models::OhlcvRecord;

/// Trait for any OHLCV data source.
pub trait DataSource: Send + Sync {
    fn name(&self) -> &str;
    fn fetch(
        &self,
        symbols: &[String],
        start: NaiveDate,
        end: NaiveDate,
    ) -> Result<Vec<OhlcvRecord>, DataError>;
}
