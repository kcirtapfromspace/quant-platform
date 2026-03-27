//! PyO3 wrappers for `quant-risk` engine.

use pyo3::prelude::*;
use quant_risk as qr;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(position_size_fixed_fraction, m)?)?;
    m.add_function(wrap_pyfunction!(kelly_fraction, m)?)?;
    m.add_function(wrap_pyfunction!(position_size_vol_target, m)?)?;
    m.add_function(wrap_pyfunction!(check_exposure, m)?)?;
    m.add_function(wrap_pyfunction!(is_circuit_tripped, m)?)?;
    m.add_function(wrap_pyfunction!(drawdown, m)?)?;
    Ok(())
}

/// Fixed-fraction sizing: `(capital * fraction) / price`.
#[pyfunction]
#[pyo3(signature = (capital, price, fraction=0.02))]
fn position_size_fixed_fraction(capital: f64, price: f64, fraction: f64) -> f64 {
    qr::position_size_fixed_fraction(capital, price, fraction)
}

/// Kelly fraction: `win_rate - (1 - win_rate) / win_loss_ratio`.
#[pyfunction]
fn kelly_fraction(win_rate: f64, win_loss_ratio: f64) -> f64 {
    qr::kelly_fraction(win_rate, win_loss_ratio)
}

/// Volatility-target sizing: `(target_vol * capital) / (volatility * price)`.
#[pyfunction]
#[pyo3(signature = (capital, price, volatility, target_vol=0.10))]
fn position_size_vol_target(capital: f64, price: f64, volatility: f64, target_vol: f64) -> f64 {
    qr::position_size_vol_target(capital, price, volatility, target_vol)
}

/// Check exposure limits. Returns `None` (approved) or a rejection reason string.
///
/// Defaults: max_gross=1.5, max_position=0.10, max_net=1.0
#[pyfunction]
#[pyo3(signature = (capital, current_gross, current_net, order_value,
                    max_gross_fraction=1.5, max_position_fraction=0.10, max_net_fraction=1.0))]
fn check_exposure(
    capital: f64,
    current_gross: f64,
    current_net: f64,
    order_value: f64,
    max_gross_fraction: f64,
    max_position_fraction: f64,
    max_net_fraction: f64,
) -> Option<String> {
    let limits = qr::ExposureLimits::new(max_gross_fraction, max_position_fraction, max_net_fraction);
    limits.check(capital, current_gross, current_net, order_value)
}

/// Returns `True` if the drawdown circuit breaker is tripped.
#[pyfunction]
fn is_circuit_tripped(peak: f64, current: f64, max_drawdown_fraction: f64) -> bool {
    let cb = qr::DrawdownCircuitBreaker::new(max_drawdown_fraction);
    cb.is_tripped(peak, current)
}

/// Current drawdown as a fraction (0.0 if peak ≤ 0).
#[pyfunction]
fn drawdown(peak: f64, current: f64) -> f64 {
    if peak <= 0.0 {
        return 0.0;
    }
    ((peak - current) / peak).max(0.0)
}
