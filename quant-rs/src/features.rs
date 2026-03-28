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

#[pyfunction]
fn returns(prices: Vec<f64>) -> Vec<f64> {
    qf::returns(&prices)
}

#[pyfunction]
fn log_returns(prices: Vec<f64>) -> Vec<f64> {
    qf::log_returns(&prices)
}

#[pyfunction]
fn rolling_mean(prices: Vec<f64>, period: usize) -> Vec<f64> {
    qf::rolling_mean(&prices, period)
}

#[pyfunction]
fn rolling_std(prices: Vec<f64>, period: usize) -> Vec<f64> {
    qf::rolling_std(&prices, period)
}

#[pyfunction]
fn ema(prices: Vec<f64>, span: usize) -> Vec<f64> {
    qf::ema(&prices, span)
}

#[pyfunction]
fn rsi(prices: Vec<f64>, period: usize) -> Vec<f64> {
    qf::rsi(&prices, period)
}

#[pyfunction]
fn macd(prices: Vec<f64>, fast: usize, slow: usize) -> Vec<f64> {
    qf::macd(&prices, fast, slow)
}

#[pyfunction]
fn macd_signal(prices: Vec<f64>, fast: usize, slow: usize, signal: usize) -> Vec<f64> {
    qf::macd_signal(&prices, fast, slow, signal)
}

#[pyfunction]
fn macd_histogram(prices: Vec<f64>, fast: usize, slow: usize, signal: usize) -> Vec<f64> {
    qf::macd_histogram(&prices, fast, slow, signal)
}

#[pyfunction]
fn bb_mid(prices: Vec<f64>, period: usize) -> Vec<f64> {
    qf::bb_mid(&prices, period)
}

#[pyfunction]
fn bb_upper(prices: Vec<f64>, period: usize, num_std: f64) -> Vec<f64> {
    qf::bb_upper(&prices, period, num_std)
}

#[pyfunction]
fn bb_lower(prices: Vec<f64>, period: usize, num_std: f64) -> Vec<f64> {
    qf::bb_lower(&prices, period, num_std)
}

#[pyfunction]
fn bb_bandwidth(prices: Vec<f64>, period: usize, num_std: f64) -> Vec<f64> {
    qf::bb_bandwidth(&prices, period, num_std)
}

#[pyfunction]
fn volume_sma(volume: Vec<f64>, period: usize) -> Vec<f64> {
    qf::volume_sma(&volume, period)
}

#[pyfunction]
fn volume_ratio(volume: Vec<f64>, period: usize) -> Vec<f64> {
    qf::volume_ratio(&volume, period)
}
