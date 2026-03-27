use thiserror::Error;

#[derive(Debug, Error)]
pub enum DataError {
    #[error("DuckDB error: {0}")]
    DuckDb(#[from] duckdb::Error),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("No data returned for symbols: {0:?}")]
    NoData(Vec<String>),
}
