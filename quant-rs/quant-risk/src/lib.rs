//! Pure-Rust risk engine: position sizing, exposure limits, circuit breaker.
//!
//! Phase 1 scaffold. Full implementation in Phase 3.

pub mod circuit_breaker;
pub mod limits;
pub mod sizing;

pub use circuit_breaker::DrawdownCircuitBreaker;
pub use limits::ExposureLimits;
pub use sizing::{
    kelly_fraction, position_size_fixed_fraction, position_size_vol_target, SizingMethod,
};
