//! PyO3 wrappers for `quant-features` compute kernels.

use pyo3::prelude::*;
use quant_features as qf;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(returns, m)?)?;
    m.add_function(wrap_pyfunction!(log_returns, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_mean, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_std, m)?)?;
    m.add_function(wrap_pyfunction!(ema, m)?)?;
    m.add_function(wrap_pyfunction!(rsi, m)?)?;
    m.add_function(wrap_pyfunction!(macd, m)?)?;
    m.add_function(wrap_pyfunction!(macd_signal, m)?)?;
    m.add_function(wrap_pyfunction!(macd_histogram, m)?)?;
    m.add_function(wrap_pyfunction!(bb_mid, m)?)?;
    m.add_function(wrap_pyfunction!(bb_upper, m)?)?;
    m.add_function(wrap_pyfunction!(bb_lower, m)?)?;
    m.add_function(wrap_pyfunction!(bb_bandwidth, m)?)?;
    m.add_function(wrap_pyfunction!(volume_sma, m)?)?;
    m.add_function(wrap_pyfunction!(volume_ratio, m)?)?;
    Ok(())
}

/// Simple period returns `(p[t] - p[t-1]) / p[t-1]`. Index 0 is NaN.
#[pyfunction]
fn returns(prices: Vec<f64>) -> Vec<f64> {
    qf::returns(&prices)
}

/// Log returns `ln(p[t] / p[t-1])`. Index 0 is NaN.
#[pyfunction]
fn log_returns(prices: Vec<f64>) -> Vec<f64> {
    qf::log_returns(&prices)
}

/// Rolling mean over `period` bars.
#[pyfunction]
#[pyo3(signature = (prices, period=20))]
fn rolling_mean(prices: Vec<f64>, period: usize) -> Vec<f64> {
    qf::rolling_mean(&prices, period)
}

/// Rolling sample std (ddof=1) over `period` bars.
#[pyfunction]
#[pyo3(signature = (prices, period=20))]
fn rolling_std(prices: Vec<f64>, period: usize) -> Vec<f64> {
    qf::rolling_std(&prices, period)
}

/// Exponential Moving Average with `span` (adjust=False).
#[pyfunction]
#[pyo3(signature = (prices, span))]
fn ema(prices: Vec<f64>, span: usize) -> Vec<f64> {
    qf::ema(&prices, span)
}

/// RSI with Wilder smoothing. NaN during warm-up.
#[pyfunction]
#[pyo3(signature = (prices, period=14))]
fn rsi(prices: Vec<f64>, period: usize) -> Vec<f64> {
    qf::rsi(&prices, period)
}

/// MACD line: EMA(fast) - EMA(slow).
#[pyfunction]
#[pyo3(signature = (prices, fast=12, slow=26))]
fn macd(prices: Vec<f64>, fast: usize, slow: usize) -> Vec<f64> {
    qf::macd(&prices, fast, slow)
}

/// MACD signal line: EMA(signal) of MACD.
#[pyfunction]
#[pyo3(signature = (prices, fast=12, slow=26, signal=9))]
fn macd_signal(prices: Vec<f64>, fast: usize, slow: usize, signal: usize) -> Vec<f64> {
    qf::macd_signal(&prices, fast, slow, signal)
}

/// MACD histogram: MACD - signal.
#[pyfunction]
#[pyo3(signature = (prices, fast=12, slow=26, signal=9))]
fn macd_histogram(prices: Vec<f64>, fast: usize, slow: usize, signal: usize) -> Vec<f64> {
    qf::macd_histogram(&prices, fast, slow, signal)
}

/// Bollinger mid band (SMA).
#[pyfunction]
#[pyo3(signature = (prices, period=20))]
fn bb_mid(prices: Vec<f64>, period: usize) -> Vec<f64> {
    qf::bb_mid(&prices, period)
}

/// Bollinger upper band: SMA + num_std * std.
#[pyfunction]
#[pyo3(signature = (prices, period=20, num_std=2.0))]
fn bb_upper(prices: Vec<f64>, period: usize, num_std: f64) -> Vec<f64> {
    qf::bb_upper(&prices, period, num_std)
}

/// Bollinger lower band: SMA - num_std * std.
#[pyfunction]
#[pyo3(signature = (prices, period=20, num_std=2.0))]
fn bb_lower(prices: Vec<f64>, period: usize, num_std: f64) -> Vec<f64> {
    qf::bb_lower(&prices, period, num_std)
}

/// Bollinger bandwidth: (upper - lower) / mid.
#[pyfunction]
#[pyo3(signature = (prices, period=20, num_std=2.0))]
fn bb_bandwidth(prices: Vec<f64>, period: usize, num_std: f64) -> Vec<f64> {
    qf::bb_bandwidth(&prices, period, num_std)
}

/// Volume SMA.
#[pyfunction]
#[pyo3(signature = (volume, period=20))]
fn volume_sma(volume: Vec<f64>, period: usize) -> Vec<f64> {
    qf::volume_sma(&volume, period)
}

/// Volume ratio: volume / rolling mean volume.
#[pyfunction]
#[pyo3(signature = (volume, period=20))]
fn volume_ratio(volume: Vec<f64>, period: usize) -> Vec<f64> {
    qf::volume_ratio(&volume, period)
}
