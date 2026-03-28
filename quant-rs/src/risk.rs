use pyo3::prelude::*;
use quant_risk::{
    kelly_fraction as kelly, position_size_fixed_fraction as size_ff,
    position_size_vol_target as size_vt, DrawdownCircuitBreaker, ExposureLimits,
};

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(position_size_fixed_fraction, m)?)?;
    m.add_function(wrap_pyfunction!(kelly_fraction, m)?)?;
    m.add_function(wrap_pyfunction!(position_size_vol_target, m)?)?;
    m.add_function(wrap_pyfunction!(check_exposure, m)?)?;
    m.add_function(wrap_pyfunction!(is_circuit_tripped, m)?)?;
    m.add_function(wrap_pyfunction!(drawdown, m)?)?;
    Ok(())
}

/// Fixed-fraction position sizing: `(capital × fraction) / price`.
#[pyfunction]
fn position_size_fixed_fraction(capital: f64, price: f64, fraction: f64) -> f64 {
    size_ff(capital, price, fraction)
}

/// Kelly criterion fraction: `win_rate − (1 − win_rate) / win_loss_ratio`, clamped to [0, 1].
#[pyfunction]
fn kelly_fraction(win_rate: f64, win_loss_ratio: f64) -> f64 {
    kelly(win_rate, win_loss_ratio)
}

/// Volatility-target sizing: `(target_vol × capital) / (volatility × price)`.
#[pyfunction]
fn position_size_vol_target(capital: f64, price: f64, volatility: f64, target_vol: f64) -> f64 {
    size_vt(capital, price, volatility, target_vol)
}

/// Exposure check using default limits (max gross 150%, max position 10%, max net 100%).
///
/// Returns `None` if approved, or a reason string if rejected.
#[pyfunction]
fn check_exposure(
    capital: f64,
    current_gross: f64,
    current_net: f64,
    order_value: f64,
) -> Option<String> {
    ExposureLimits::default().check(capital, current_gross, current_net, order_value)
}

/// Returns `True` if drawdown from `peak` to `current` meets or exceeds `threshold`.
#[pyfunction]
fn is_circuit_tripped(peak: f64, current: f64, threshold: f64) -> bool {
    DrawdownCircuitBreaker::new(threshold).is_tripped(peak, current)
}

/// Current drawdown fraction: `(peak − current) / peak`, clamped to ≥ 0.
#[pyfunction]
fn drawdown(peak: f64, current: f64) -> f64 {
    DrawdownCircuitBreaker::new(0.5).drawdown(peak, current)
}
