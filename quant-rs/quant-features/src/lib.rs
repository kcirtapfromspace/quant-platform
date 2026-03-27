//! Pure-Rust compute kernels for technical and statistical indicators.
//!
//! All functions operate on `&[f64]` slices and return `Vec<f64>`.
//! NaN is used for warm-up positions where the indicator is undefined.
//! No Python dependency — this crate is safe to use from any Rust context.

pub mod indicators;

pub use indicators::{
    bb_bandwidth, bb_lower, bb_mid, bb_upper, ema, log_returns, macd, macd_histogram,
    macd_signal, returns, rolling_mean, rolling_std, rsi, volume_ratio, volume_sma,
};
