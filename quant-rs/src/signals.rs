//! PyO3 wrappers for `quant-signals` (stub — full impl in Phase 4).

use pyo3::prelude::*;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__status__", "Phase 1 scaffold — full implementation in Phase 4")?;
    Ok(())
}
