//! `quant-oms` — Pure-Rust Order Management System.
//!
//! Provides:
//! - `Order`, `Fill`, `Position` data models with enumerations
//! - `SqliteStateStore` — write-through SQLite persistence
//! - `OrderManagementSystem` — stateful order book + position tracker
//!
//! No Python dependency — usable from any Rust context or via PyO3 bindings.

pub mod alpaca;
pub mod broker;
pub mod error;
pub mod models;
pub mod store;
pub mod system;

pub use alpaca::AlpacaBrokerAdapter;
pub use broker::{AccountInfo, Broker, BrokerError, BrokerPosition};
pub use error::{OmsError, OmsResult};
pub use models::{Fill, Order, OrderSide, OrderStatus, OrderType, Position, TimeInForce};
pub use store::SqliteStateStore;
pub use system::OrderManagementSystem;
