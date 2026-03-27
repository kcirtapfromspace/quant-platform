//! PyO3 wrappers for `quant-oms`.
//!
//! Exposes `quant_rs.oms` submodule with:
//! - `OrderManagementSystem` class
//! - Order/Fill/Position helper constructors (return Python dicts)

use pyo3::prelude::*;
use pyo3::types::PyDict;

use quant_oms as qoms;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOms>()?;
    m.add_function(wrap_pyfunction!(make_order, m)?)?;
    Ok(())
}

// ── PyOms class ───────────────────────────────────────────────────────────────

/// In-process Order Management System backed by an optional SQLite store.
///
/// Args:
///     db_path: Path to a SQLite file.  Use ":memory:" for an in-process
///              (non-persistent) store, or omit to disable persistence.
#[pyclass(name = "OrderManagementSystem")]
pub struct PyOms {
    inner: qoms::OrderManagementSystem,
}

#[pymethods]
impl PyOms {
    /// Create an OMS.
    ///
    /// Args:
    ///     db_path: SQLite path or ":memory:".  Pass None to disable persistence.
    #[new]
    #[pyo3(signature = (db_path=None))]
    fn new(db_path: Option<&str>) -> PyResult<Self> {
        let store = match db_path {
            Some(path) => {
                let s = qoms::SqliteStateStore::new(path)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                Some(s)
            }
            None => None,
        };
        Ok(Self { inner: qoms::OrderManagementSystem::new(store) })
    }

    /// Restore persisted state from the SQLite store.
    fn restore_state(&mut self) -> PyResult<()> {
        self.inner
            .restore_state()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Submit a new order.  Returns the OMS order ID string.
    ///
    /// Args:
    ///     symbol:     Asset identifier.
    ///     side:       "buy" or "sell".
    ///     quantity:   Unsigned quantity.
    ///     order_type: "market", "limit", "stop", or "stop_limit". Default "market".
    #[pyo3(signature = (symbol, side, quantity, order_type="market"))]
    fn submit_order(
        &mut self,
        symbol: &str,
        side: &str,
        quantity: f64,
        order_type: &str,
    ) -> PyResult<String> {
        let side = qoms::OrderSide::from_str(side)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("invalid side: {side}")))?;
        let order_type = qoms::OrderType::from_str(order_type)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("invalid order_type: {order_type}")))?;

        let order = qoms::Order::new(symbol, side, quantity, order_type);
        self.inner
            .submit_order(order)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Register broker's order ID after submission.
    fn mark_submitted(&mut self, order_id: &str, broker_order_id: &str) -> PyResult<()> {
        self.inner
            .mark_submitted(order_id, broker_order_id)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Process a fill event.  Returns None on success.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (fill_id, order_id, broker_order_id, symbol, side, quantity, price, commission=0.0))]
    fn on_fill(
        &mut self,
        fill_id: &str,
        order_id: &str,
        broker_order_id: &str,
        symbol: &str,
        side: &str,
        quantity: f64,
        price: f64,
        commission: f64,
    ) -> PyResult<()> {
        let side = qoms::OrderSide::from_str(side)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("invalid side: {side}")))?;

        let fill = qoms::Fill {
            fill_id: fill_id.to_string(),
            order_id: order_id.to_string(),
            broker_order_id: broker_order_id.to_string(),
            symbol: symbol.to_string(),
            side,
            quantity,
            price,
            filled_at: chrono::Utc::now(),
            commission,
        };

        self.inner
            .on_fill(fill)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Cancel an active order.
    fn cancel_order(&mut self, order_id: &str) -> PyResult<()> {
        self.inner
            .cancel_order(order_id)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Get a position dict for a symbol, or None if no position exists.
    ///
    /// Returns: dict with keys: symbol, quantity, avg_cost, market_price,
    ///          market_value, unrealized_pnl.
    fn get_position<'py>(&self, py: Python<'py>, symbol: &str) -> PyResult<Option<Bound<'py, PyDict>>> {
        match self.inner.get_position(symbol) {
            None => Ok(None),
            Some(pos) => {
                let d = PyDict::new_bound(py);
                d.set_item("symbol", &pos.symbol)?;
                d.set_item("quantity", pos.quantity)?;
                d.set_item("avg_cost", pos.avg_cost)?;
                d.set_item("market_price", pos.market_price)?;
                d.set_item("market_value", pos.market_value())?;
                d.set_item("unrealized_pnl", pos.unrealized_pnl())?;
                Ok(Some(d))
            }
        }
    }

    /// Get all positions as a dict {symbol: position_dict}.
    fn get_all_positions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let result = PyDict::new_bound(py);
        for (sym, pos) in self.inner.get_all_positions() {
            let d = PyDict::new_bound(py);
            d.set_item("symbol", &pos.symbol)?;
            d.set_item("quantity", pos.quantity)?;
            d.set_item("avg_cost", pos.avg_cost)?;
            d.set_item("market_price", pos.market_price)?;
            d.set_item("market_value", pos.market_value())?;
            d.set_item("unrealized_pnl", pos.unrealized_pnl())?;
            result.set_item(sym, d)?;
        }
        Ok(result)
    }

    /// Update the market price for a symbol (mark-to-market).
    fn update_market_price(&mut self, symbol: &str, price: f64) {
        self.inner.update_market_price(symbol, price);
    }

    fn set_cash(&mut self, amount: f64) {
        self.inner.set_cash(amount);
    }

    fn cash(&self) -> f64 {
        self.inner.cash()
    }

    fn portfolio_value(&self) -> f64 {
        self.inner.portfolio_value()
    }

    /// Number of active (non-terminal) orders.
    fn active_order_count(&self) -> usize {
        self.inner.get_active_orders().len()
    }
}

// ── Helper: make_order ────────────────────────────────────────────────────────

/// Create an order dict (does not register it in any OMS).
///
/// Useful for inspecting order structure from Python tests.
#[pyfunction]
#[pyo3(signature = (symbol, side, quantity, order_type="market"))]
fn make_order<'py>(
    py: Python<'py>,
    symbol: &str,
    side: &str,
    quantity: f64,
    order_type: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let side_enum = qoms::OrderSide::from_str(side)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("invalid side: {side}")))?;
    let type_enum = qoms::OrderType::from_str(order_type)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("invalid order_type: {order_type}")))?;

    let order = qoms::Order::new(symbol, side_enum, quantity, type_enum);
    let d = PyDict::new_bound(py);
    d.set_item("id", &order.id)?;
    d.set_item("symbol", &order.symbol)?;
    d.set_item("side", order.side.as_str())?;
    d.set_item("quantity", order.quantity)?;
    d.set_item("order_type", order.order_type.as_str())?;
    d.set_item("status", order.status.as_str())?;
    Ok(d)
}
