use pyo3::prelude::*;

mod backtest;
mod features;
mod risk;
mod signals;

/// PyO3 Python bindings for quant-rs Rust compute kernels.
///
/// Submodules:
///   quant_rs.features  — technical indicator kernels
///   quant_rs.risk      — position sizing and circuit-breaker utilities
///   quant_rs.signals   — momentum, mean-reversion, trend-following kernels
///   quant_rs.backtest  — vectorised single-asset backtest engine
#[pymodule]
fn quant_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;

    let features_mod = PyModule::new(m.py(), "quant_rs.features")?;
    features::register(&features_mod)?;
    m.add_submodule(&features_mod)?;

    let risk_mod = PyModule::new(m.py(), "quant_rs.risk")?;
    risk::register(&risk_mod)?;
    m.add_submodule(&risk_mod)?;

    let signals_mod = PyModule::new(m.py(), "quant_rs.signals")?;
    signals::register(&signals_mod)?;
    m.add_submodule(&signals_mod)?;

    let backtest_mod = PyModule::new(m.py(), "quant_rs.backtest")?;
    backtest::register(&backtest_mod)?;
    m.add_submodule(&backtest_mod)?;

    Ok(())
}
