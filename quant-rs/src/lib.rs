//! `quant_rs` — PyO3 extension module exposing Rust compute kernels to Python.
//!
//! Python usage:
//! ```python
//! import quant_rs
//!
//! prices = [100.0, 101.0, ...]
//! rsi_values = quant_rs.features.rsi(prices, 14)
//! qty = quant_rs.risk.position_size_fixed_fraction(100_000.0, 50.0, 0.02)
//! ```

use pyo3::prelude::*;

mod backtest;
mod features;
mod risk;
mod signals;

#[pymodule]
fn quant_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // features submodule
    let features_mod = PyModule::new_bound(m.py(), "features")?;
    features::register(&features_mod)?;
    m.add_submodule(&features_mod)?;

    // risk submodule
    let risk_mod = PyModule::new_bound(m.py(), "risk")?;
    risk::register(&risk_mod)?;
    m.add_submodule(&risk_mod)?;

    // signals submodule
    let signals_mod = PyModule::new_bound(m.py(), "signals")?;
    signals::register(&signals_mod)?;
    m.add_submodule(&signals_mod)?;

    // backtest submodule
    let backtest_mod = PyModule::new_bound(m.py(), "backtest")?;
    backtest::register(&backtest_mod)?;
    m.add_submodule(&backtest_mod)?;

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
