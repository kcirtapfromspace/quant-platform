//! PyO3 wrappers for `quant-signals` (Phase 4: full implementation).

use pyo3::prelude::*;
use quant_signals as qs;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(momentum_signal, m)?)?;
    m.add_function(wrap_pyfunction!(mean_reversion_signal, m)?)?;
    m.add_function(wrap_pyfunction!(trend_following_signal, m)?)?;
    Ok(())
}

/// Momentum signal from RSI and return series.
///
/// Args:
///     rsi_values:   RSI time-series (NaN during warmup).
///     returns:      Simple/log return series (NaN at index 0).
///     lookback:     Bars used for confidence estimation (default 5).
///     return_scale: Absolute return mapped to confidence=1 (default 0.05).
///
/// Returns:
///     Tuple (score, confidence, target_position) each in [-1,1] / [0,1] / [-1,1].
#[pyfunction]
#[pyo3(signature = (rsi_values, returns, lookback=5, return_scale=0.05))]
fn momentum_signal(
    rsi_values: Vec<f64>,
    returns: Vec<f64>,
    lookback: usize,
    return_scale: f64,
) -> (f64, f64, f64) {
    qs::momentum_signal(&rsi_values, &returns, lookback, return_scale)
}

/// Mean-reversion signal from Bollinger Bands.
///
/// Args:
///     bb_mid:    Bollinger mid-band series.
///     bb_upper:  Bollinger upper-band series.
///     bb_lower:  Bollinger lower-band series.
///     returns:   Return series (used for last-price approximation).
///     num_std:   Band multiplier (default 2.0).
///
/// Returns:
///     Tuple (score, confidence, target_position).
#[pyfunction]
#[pyo3(signature = (bb_mid, bb_upper, bb_lower, returns, num_std=2.0))]
fn mean_reversion_signal(
    bb_mid: Vec<f64>,
    bb_upper: Vec<f64>,
    bb_lower: Vec<f64>,
    returns: Vec<f64>,
    num_std: f64,
) -> (f64, f64, f64) {
    qs::mean_reversion_signal(&bb_mid, &bb_upper, &bb_lower, &returns, num_std)
}

/// Trend-following signal from MACD histogram and moving-average crossover.
///
/// Args:
///     macd_hist: MACD histogram series (fast=12, slow=26, signal=9 by convention).
///     fast_ma:   Fast SMA series.
///     slow_ma:   Slow SMA series.
///
/// Returns:
///     Tuple (score, confidence, target_position).
#[pyfunction]
fn trend_following_signal(
    macd_hist: Vec<f64>,
    fast_ma: Vec<f64>,
    slow_ma: Vec<f64>,
) -> (f64, f64, f64) {
    qs::trend_following_signal(&macd_hist, &fast_ma, &slow_ma)
}
