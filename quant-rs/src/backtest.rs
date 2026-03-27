//! PyO3 bindings for the `quant-backtest` engine — `quant_rs.backtest` submodule.
//!
//! Python usage:
//! ```python
//! import quant_rs
//! result = quant_rs.backtest.run_backtest(adj_close, signals, 0.001, 1.0)
//! # result["equity_curve"]  -> list[tuple[float, float]]  (portfolio_value, drawdown)
//! # result["trades"]        -> list[tuple[int, int, str, float]]
//! # result["sharpe_ratio"], result["max_drawdown"], result["cagr"], ...
//! ```

use pyo3::prelude::*;
use pyo3::types::PyDict;
use quant_backtest::run_backtest as rs_run_backtest;

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_backtest, m)?)?;
    Ok(())
}

/// Run a vectorised single-asset backtest and return a results dict.
///
/// Args:
///     adj_close: Adjusted close prices in ascending date order.
///     signals: Raw position signal per bar (same length as adj_close).
///         The engine shifts these one bar forward internally to prevent
///         lookahead bias.  Conventional values: 1.0 (long), 0.0 (flat),
///         -1.0 (short); fractional values are accepted.
///     commission_pct: One-way commission as a fraction of trade value
///         (default 0.001 = 10 bps).
///     initial_capital: Starting portfolio value (default 1.0).
///
/// Returns:
///     A dict with keys:
///     - ``equity_curve``: list of (portfolio_value, drawdown) tuples, one per bar.
///     - ``trades``: list of (entry_idx, exit_idx, direction, return) tuples where
///       direction is "long" or "short".
///     - ``sharpe_ratio``, ``max_drawdown``, ``cagr``, ``win_rate``,
///       ``profit_factor``, ``total_return``: float scalars.
///     - ``n_trades``: int.
#[pyfunction]
#[pyo3(signature = (adj_close, signals, commission_pct=0.001, initial_capital=1.0))]
fn run_backtest(
    py: Python<'_>,
    adj_close: Vec<f64>,
    signals: Vec<f64>,
    commission_pct: f64,
    initial_capital: f64,
) -> PyResult<PyObject> {
    let result = rs_run_backtest(&adj_close, &signals, commission_pct, initial_capital);

    let dict = PyDict::new_bound(py);

    // equity_curve: list[tuple[float, float]] — (portfolio_value, drawdown)
    let curve: Vec<(f64, f64)> = result
        .equity_curve
        .iter()
        .zip(result.drawdown_curve.iter())
        .map(|(&pv, &dd)| (pv, dd))
        .collect();
    dict.set_item("equity_curve", curve)?;

    // trades: list[tuple[int, int, str, float]] — (entry_idx, exit_idx, direction, return)
    let trades: Vec<(usize, usize, &str, f64)> = result
        .trades
        .iter()
        .map(|t| {
            (
                t.entry_idx,
                t.exit_idx,
                if t.direction > 0 { "long" } else { "short" },
                t.ret,
            )
        })
        .collect();
    dict.set_item("trades", trades)?;

    dict.set_item("sharpe_ratio", result.sharpe_ratio)?;
    dict.set_item("max_drawdown", result.max_drawdown)?;
    dict.set_item("cagr", result.cagr)?;
    dict.set_item("win_rate", result.win_rate)?;
    dict.set_item("profit_factor", result.profit_factor)?;
    dict.set_item("total_return", result.total_return)?;
    dict.set_item("n_trades", result.trades.len())?;

    Ok(dict.into())
}
