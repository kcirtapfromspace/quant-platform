//! Broker abstraction — defines the `Broker` trait and shared types used by
//! all broker adapter implementations (Alpaca, Interactive Brokers, etc.).

use thiserror::Error;

use crate::models::Order;

// ── Error type ────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum BrokerError {
    /// Low-level HTTP transport failure.
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    /// The broker returned a non-2xx status.
    #[error("broker API error (HTTP {status}): {message}")]
    Api { status: u16, message: String },

    /// Configuration problem (missing env var, bad header value, etc.).
    #[error("broker configuration error: {0}")]
    Config(String),
}

// ── Shared response types ─────────────────────────────────────────────────────

/// Summary of the trading account returned by the broker.
#[derive(Debug, Clone)]
pub struct AccountInfo {
    pub id: String,
    pub equity: f64,
    pub cash: f64,
    pub buying_power: f64,
    pub portfolio_value: f64,
    pub currency: String,
    pub status: String,
}

/// A single open position as reported by the broker.
#[derive(Debug, Clone)]
pub struct BrokerPosition {
    pub symbol: String,
    pub qty: f64,
    pub avg_entry_price: f64,
    pub market_value: f64,
    pub unrealized_pl: f64,
}

// ── Broker trait ──────────────────────────────────────────────────────────────

/// Minimal interface every broker adapter must implement.
///
/// All methods are synchronous (blocking).  If a concrete adapter needs async
/// IO it must wrap its own runtime internally; callers remain sync.
pub trait Broker {
    /// Submit an order to the broker and return the broker-assigned order ID.
    fn submit_order(&self, order: &Order) -> Result<String, BrokerError>;

    /// Fetch current account information (equity, cash, buying power).
    fn get_account(&self) -> Result<AccountInfo, BrokerError>;

    /// Fetch all open positions.
    fn get_positions(&self) -> Result<Vec<BrokerPosition>, BrokerError>;

    /// Cancel an open order by the broker-assigned order ID.
    fn cancel_order(&self, broker_order_id: &str) -> Result<(), BrokerError>;
}
