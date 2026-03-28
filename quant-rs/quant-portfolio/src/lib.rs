//! `quant-portfolio` — Pure-Rust portfolio construction engine.
//!
//! Provides:
//! - Alpha score combination (equal-weight, static-weight, conviction)
//! - Covariance matrix estimation from return histories
//! - Portfolio optimizers: equal-weight, risk-parity, min-variance
//! - Threshold-based rebalancing with trade generation
//!
//! Mirrors `quant.portfolio.*` in a no-std-compatible pure-Rust form.

pub mod alpha;
pub mod covariance;
pub mod error;
pub mod optimizer;
pub mod rebalancer;

pub use alpha::{AlphaCombiner, AlphaScore, CombinationMethod};
pub use covariance::estimate_covariance;
pub use error::{PortfolioError, PortfolioResult};
pub use optimizer::{OptimizationMethod, OptimizationResult, Optimizer};
pub use rebalancer::{RebalanceResult, Rebalancer, Trade};
