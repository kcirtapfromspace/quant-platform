//! PyO3 wrappers for `quant-portfolio`.
//!
//! Exposes `quant_rs.portfolio` submodule with:
//! - `optimize(method, symbols, alpha_scores, cov_flat, n_assets, max_weight=1.0)`
//! - `combine_alpha(method, symbols, signals_per_symbol)`
//! - `rebalance(target_weights, current_weights, portfolio_value, threshold, min_trade_weight)`

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use quant_portfolio as qp;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(optimize, m)?)?;
    m.add_function(wrap_pyfunction!(rebalance, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_covariance, m)?)?;
    Ok(())
}

/// Optimize portfolio weights.
///
/// Args:
///     method:       "equal_weight", "risk_parity", or "min_variance".
///     symbols:      List of symbol strings.
///     alpha_scores: List of alpha scores (one per symbol), same order as symbols.
///     cov_flat:     Covariance matrix as a flat list (row-major, len = n^2).
///     max_weight:   Maximum weight per asset (default 1.0).
///
/// Returns:
///     dict with keys: symbols, weights, risk, method.
#[pyfunction]
#[pyo3(signature = (method, symbols, alpha_scores, cov_flat, max_weight=1.0))]
fn optimize<'py>(
    py: Python<'py>,
    method: &str,
    symbols: Vec<String>,
    alpha_scores: Vec<f64>,
    cov_flat: Vec<f64>,
    max_weight: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let opt_method = match method {
        "equal_weight" => qp::OptimizationMethod::EqualWeight,
        "risk_parity" => qp::OptimizationMethod::RiskParity,
        "min_variance" => qp::OptimizationMethod::MinVariance,
        "mean_variance" => qp::OptimizationMethod::MeanVariance,
        _ => return Err(pyo3::exceptions::PyValueError::new_err(format!("unknown method: {method}"))),
    };

    let constraints = qp::optimizer::PortfolioConstraints {
        max_weight,
        ..Default::default()
    };

    let result = qp::optimizer::optimize(opt_method, &symbols, &alpha_scores, &cov_flat, &constraints)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let d = PyDict::new_bound(py);
    d.set_item("symbols", result.symbols)?;
    d.set_item("weights", result.weights)?;
    d.set_item("risk", result.risk)?;
    d.set_item("method", method)?;
    Ok(d)
}

/// Generate rebalancing trades.
///
/// Args:
///     target_weights:  {symbol: weight} target.
///     current_weights: {symbol: weight} current.
///     portfolio_value: Total portfolio value in dollars.
///     threshold:       Minimum turnover to trigger rebalance (default 0.01).
///     min_trade_weight: Minimum per-trade weight (default 0.001).
///
/// Returns:
///     dict with keys: trades (list of trade dicts), turnover, rebalance_triggered.
#[pyfunction]
#[pyo3(signature = (target_weights, current_weights, portfolio_value, threshold=0.01, min_trade_weight=0.001))]
fn rebalance<'py>(
    py: Python<'py>,
    target_weights: HashMap<String, f64>,
    current_weights: HashMap<String, f64>,
    portfolio_value: f64,
    threshold: f64,
    min_trade_weight: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let rebalancer = qp::Rebalancer { threshold, min_trade_weight };
    let result = rebalancer.rebalance(&target_weights, &current_weights, portfolio_value);

    let d = PyDict::new_bound(py);
    let trades = PyList::empty_bound(py);
    for t in &result.trades {
        let td = PyDict::new_bound(py);
        td.set_item("symbol", &t.symbol)?;
        td.set_item("side", &t.side)?;
        td.set_item("target_weight", t.target_weight)?;
        td.set_item("trade_weight", t.trade_weight)?;
        td.set_item("dollar_amount", t.dollar_amount)?;
        trades.append(td)?;
    }
    d.set_item("trades", trades)?;
    d.set_item("turnover", result.turnover)?;
    d.set_item("rebalance_triggered", result.rebalance_triggered)?;
    Ok(d)
}

/// Estimate a covariance matrix from return data.
///
/// Args:
///     returns_flat: Row-major returns matrix (n_bars × n_assets), flattened.
///     n_assets:     Number of assets.
///     shrinkage:    Ledoit-Wolf shrinkage (0–1).  Pass None for automatic estimate.
///
/// Returns:
///     Flat list (row-major) of the n_assets × n_assets covariance matrix.
#[pyfunction]
#[pyo3(signature = (returns_flat, n_assets, shrinkage=None))]
fn estimate_covariance(
    returns_flat: Vec<f64>,
    n_assets: usize,
    shrinkage: Option<f64>,
) -> PyResult<Vec<f64>> {
    qp::estimate_covariance(&returns_flat, n_assets, shrinkage)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}
