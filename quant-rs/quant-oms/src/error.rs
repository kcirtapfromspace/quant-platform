//! Error types for the OMS.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum OmsError {
    #[error("order not found: {0}")]
    OrderNotFound(String),

    #[error("order {0} is in terminal state {1} and cannot be modified")]
    OrderTerminal(String, String),

    #[error("fill references unknown broker order id: {0}")]
    UnknownBrokerOrder(String),

    #[error("SQLite persistence error: {0}")]
    Persistence(#[from] rusqlite::Error),

    #[error("invalid order: {0}")]
    InvalidOrder(String),

    #[error("data parse error: {0}")]
    Parse(String),
}

pub type OmsResult<T> = Result<T, OmsError>;
