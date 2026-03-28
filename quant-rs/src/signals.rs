use pyo3::prelude::*;
use quant_signals::{mean_reversion_signal, momentum_signal, trend_following_signal};

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(momentum_signal_py, m)?)?;
    m.add_function(wrap_pyfunction!(mean_reversion_signal_py, m)?)?;
    m.add_function(wrap_pyfunction!(trend_following_signal_py, m)?)?;
    Ok(())
}

/// Momentum signal: RSI-based score with return-magnitude confidence.
///
/// Returns `(score, confidence, target_position)` each in [-1, 1] / [0, 1] / [-1, 1].
#[pyfunction]
#[pyo3(name = "momentum_signal")]
#[pyo3(signature = (rsi_values, returns, lookback = 5, return_scale = 0.05))]
fn momentum_signal_py(
    rsi_values: Vec<f64>,
    returns: Vec<f64>,
    lookback: usize,
    return_scale: f64,
) -> (f64, f64, f64) {
    momentum_signal(&rsi_values, &returns, lookback, return_scale)
}

/// Mean-reversion signal: Bollinger Band z-score.
///
/// Returns `(score, confidence, target_position)`.
#[pyfunction]
#[pyo3(name = "mean_reversion_signal")]
#[pyo3(signature = (bb_mid, bb_upper, bb_lower, returns, num_std = 2.0))]
fn mean_reversion_signal_py(
    bb_mid: Vec<f64>,
    bb_upper: Vec<f64>,
    bb_lower: Vec<f64>,
    returns: Vec<f64>,
    num_std: f64,
) -> (f64, f64, f64) {
    mean_reversion_signal(&bb_mid, &bb_upper, &bb_lower, &returns, num_std)
}

/// Trend-following signal: MACD histogram normalised by rolling std.
///
/// Returns `(score, confidence, target_position)`.
#[pyfunction]
#[pyo3(name = "trend_following_signal")]
fn trend_following_signal_py(
    macd_hist: Vec<f64>,
    fast_ma: Vec<f64>,
    slow_ma: Vec<f64>,
) -> (f64, f64, f64) {
    trend_following_signal(&macd_hist, &fast_ma, &slow_ma)
}
