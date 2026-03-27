//! Error types for the portfolio engine.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum PortfolioError {
    #[error("empty universe — no symbols provided")]
    EmptyUniverse,

    #[error("optimization failed: {0}")]
    OptimizationFailed(String),

    #[error("insufficient return history: need at least {needed} bars, got {got}")]
    InsufficientHistory { needed: usize, got: usize },

    #[error("symbol mismatch: {0}")]
    SymbolMismatch(String),
}

pub type PortfolioResult<T> = Result<T, PortfolioError>;
