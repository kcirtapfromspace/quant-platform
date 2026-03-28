use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::Py;
use quant_backtest::run_backtest as rb;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_backtest, m)?)?;
    Ok(())
}

/// Run a vectorised single-asset backtest.
///
/// Returns a dict with keys:
///   equity_curve  — list of (portfolio_value, drawdown) per bar
///   total_return  — decimal total return over the window
///   sharpe_ratio  — annualised Sharpe (ddof=1, rf=0)
///   max_drawdown  — peak-to-trough drawdown as a positive fraction
///   cagr          — compound annual growth rate
///   win_rate      — fraction of trades with positive return
///   profit_factor — gross profit / gross loss (inf when no losers, 0 when no trades)
///   n_trades      — number of completed round-trip trades
///   trades        — list of (entry_idx, exit_idx, direction, return)
#[pyfunction]
fn run_backtest(
    py: Python<'_>,
    prices: Vec<f64>,
    signals: Vec<f64>,
    commission_pct: f64,
    initial_capital: f64,
) -> PyResult<Py<PyDict>> {
    let result = rb(&prices, &signals, commission_pct, initial_capital);

    let d = PyDict::new(py);

    let equity_curve: Vec<(f64, f64)> = result
        .equity_curve
        .iter()
        .zip(result.drawdown_curve.iter())
        .map(|(&pv, &dd)| (pv, dd))
        .collect();
    d.set_item("equity_curve", equity_curve)?;

    d.set_item("total_return", result.total_return)?;
    d.set_item("sharpe_ratio", result.sharpe_ratio)?;
    d.set_item("max_drawdown", result.max_drawdown)?;
    d.set_item("cagr", result.cagr)?;
    d.set_item("win_rate", result.win_rate)?;
    d.set_item("profit_factor", result.profit_factor)?;
    d.set_item("n_trades", result.trades.len())?;

    let trades: Vec<(usize, usize, String, f64)> = result
        .trades
        .iter()
        .map(|t| {
            let dir = if t.direction == 1 {
                "long".to_string()
            } else {
                "short".to_string()
            };
            (t.entry_idx, t.exit_idx, dir, t.ret)
        })
        .collect();
    d.set_item("trades", trades)?;

    Ok(d.unbind())
}
