//! SQLite write-through state store for OMS persistence.
//!
//! Mirrors `quant.oms.persistence.SQLiteStateStore`.
//! Uses WAL mode for safe concurrent reads from a single process.
//!
//! Schema:
//!   orders    — one row per order, updated in place on status changes
//!   fills     — append-only fill records
//!   positions — one row per symbol, upserted on every position change

use std::collections::HashMap;
use std::path::Path;

use chrono::{DateTime, NaiveDate, Utc};
use rusqlite::{params, Connection, OptionalExtension};

use crate::error::OmsResult;
use crate::models::{Fill, Order, OrderSide, OrderStatus, OrderType, Position, TimeInForce};

const SCHEMA_SQL: &str = "
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS orders (
    id              TEXT PRIMARY KEY,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL,
    quantity        REAL NOT NULL,
    order_type      TEXT NOT NULL,
    limit_price     REAL,
    stop_price      REAL,
    time_in_force   TEXT NOT NULL,
    broker_order_id TEXT,
    status          TEXT NOT NULL,
    filled_quantity REAL NOT NULL DEFAULT 0.0,
    avg_fill_price  REAL NOT NULL DEFAULT 0.0,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    strategy_id     TEXT,
    sector          TEXT
);

CREATE TABLE IF NOT EXISTS fills (
    fill_id          TEXT PRIMARY KEY,
    order_id         TEXT NOT NULL,
    broker_order_id  TEXT NOT NULL,
    symbol           TEXT NOT NULL,
    side             TEXT NOT NULL,
    quantity         REAL NOT NULL,
    price            REAL NOT NULL,
    filled_at        TEXT NOT NULL,
    commission       REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS positions (
    symbol       TEXT PRIMARY KEY,
    quantity     REAL NOT NULL DEFAULT 0.0,
    avg_cost     REAL NOT NULL DEFAULT 0.0,
    market_price REAL NOT NULL DEFAULT 0.0
);
";

fn dt_to_str(dt: DateTime<Utc>) -> String {
    dt.to_rfc3339()
}

fn str_to_dt(s: &str) -> DateTime<Utc> {
    DateTime::parse_from_rfc3339(s)
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or_else(|_| Utc::now())
}

pub struct SqliteStateStore {
    conn: Connection,
}

impl SqliteStateStore {
    /// Open (or create) the store at `db_path`.  Use `":memory:"` for tests.
    pub fn new(db_path: impl AsRef<Path>) -> OmsResult<Self> {
        let conn = Connection::open(db_path)?;
        let store = Self { conn };
        store.ensure_schema()?;
        Ok(store)
    }

    fn ensure_schema(&self) -> OmsResult<()> {
        self.conn.execute_batch(SCHEMA_SQL)?;
        Ok(())
    }

    // ── Order persistence ─────────────────────────────────────────────────

    pub fn save_order(&self, order: &Order) -> OmsResult<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO orders
             (id, symbol, side, quantity, order_type, limit_price, stop_price,
              time_in_force, broker_order_id, status, filled_quantity,
              avg_fill_price, created_at, updated_at, strategy_id, sector)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16)",
            params![
                order.id,
                order.symbol,
                order.side.as_str(),
                order.quantity,
                order.order_type.as_str(),
                order.limit_price,
                order.stop_price,
                order.time_in_force.as_str(),
                order.broker_order_id,
                order.status.as_str(),
                order.filled_quantity,
                order.avg_fill_price,
                dt_to_str(order.created_at),
                dt_to_str(order.updated_at),
                order.strategy_id,
                order.sector,
            ],
        )?;
        Ok(())
    }

    pub fn load_orders(&self) -> OmsResult<Vec<Order>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, symbol, side, quantity, order_type, limit_price, stop_price,
                    time_in_force, broker_order_id, status, filled_quantity,
                    avg_fill_price, created_at, updated_at, strategy_id, sector
             FROM orders",
        )?;

        let orders = stmt
            .query_map([], |row| {
                let side_str: String = row.get(2)?;
                let type_str: String = row.get(4)?;
                let tif_str: String = row.get(7)?;
                let status_str: String = row.get(9)?;
                let created_str: String = row.get(12)?;
                let updated_str: String = row.get(13)?;

                Ok(Order {
                    id: row.get(0)?,
                    symbol: row.get(1)?,
                    side: OrderSide::from_str(&side_str).unwrap_or(OrderSide::Buy),
                    quantity: row.get(3)?,
                    order_type: OrderType::from_str(&type_str).unwrap_or(OrderType::Market),
                    limit_price: row.get(5)?,
                    stop_price: row.get(6)?,
                    time_in_force: TimeInForce::from_str(&tif_str).unwrap_or(TimeInForce::Day),
                    broker_order_id: row.get(8)?,
                    status: OrderStatus::from_str(&status_str).unwrap_or(OrderStatus::Pending),
                    filled_quantity: row.get(10)?,
                    avg_fill_price: row.get(11)?,
                    created_at: str_to_dt(&created_str),
                    updated_at: str_to_dt(&updated_str),
                    strategy_id: row.get(14)?,
                    sector: row.get(15)?,
                })
            })?
            .filter_map(Result::ok)
            .collect();

        Ok(orders)
    }

    // ── Fill persistence ──────────────────────────────────────────────────

    pub fn save_fill(&self, fill: &Fill) -> OmsResult<()> {
        self.conn.execute(
            "INSERT OR IGNORE INTO fills
             (fill_id, order_id, broker_order_id, symbol, side, quantity,
              price, filled_at, commission)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)",
            params![
                fill.fill_id,
                fill.order_id,
                fill.broker_order_id,
                fill.symbol,
                fill.side.as_str(),
                fill.quantity,
                fill.price,
                dt_to_str(fill.filled_at),
                fill.commission,
            ],
        )?;
        Ok(())
    }

    pub fn load_fills(&self) -> OmsResult<Vec<Fill>> {
        let mut stmt = self.conn.prepare(
            "SELECT fill_id, order_id, broker_order_id, symbol, side,
                    quantity, price, filled_at, commission
             FROM fills",
        )?;

        let fills = stmt
            .query_map([], |row| {
                let side_str: String = row.get(4)?;
                let filled_at_str: String = row.get(7)?;
                Ok(Fill {
                    fill_id: row.get(0)?,
                    order_id: row.get(1)?,
                    broker_order_id: row.get(2)?,
                    symbol: row.get(3)?,
                    side: OrderSide::from_str(&side_str).unwrap_or(OrderSide::Buy),
                    quantity: row.get(5)?,
                    price: row.get(6)?,
                    filled_at: str_to_dt(&filled_at_str),
                    commission: row.get(8)?,
                })
            })?
            .filter_map(Result::ok)
            .collect();

        Ok(fills)
    }

    // ── Position persistence ──────────────────────────────────────────────

    pub fn save_position(&self, position: &Position) -> OmsResult<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO positions (symbol, quantity, avg_cost, market_price)
             VALUES (?1, ?2, ?3, ?4)",
            params![
                position.symbol,
                position.quantity,
                position.avg_cost,
                position.market_price,
            ],
        )?;
        Ok(())
    }

    pub fn load_positions(&self) -> OmsResult<HashMap<String, Position>> {
        let mut stmt = self
            .conn
            .prepare("SELECT symbol, quantity, avg_cost, market_price FROM positions")?;

        let positions = stmt
            .query_map([], |row| {
                Ok(Position {
                    symbol: row.get(0)?,
                    quantity: row.get(1)?,
                    avg_cost: row.get(2)?,
                    market_price: row.get(3)?,
                })
            })?
            .filter_map(Result::ok)
            .map(|p| (p.symbol.clone(), p))
            .collect();

        Ok(positions)
    }

    // ── Broker-ID map ─────────────────────────────────────────────────────

    pub fn load_broker_id_map(&self) -> OmsResult<HashMap<String, String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT broker_order_id, id FROM orders WHERE broker_order_id IS NOT NULL")?;

        let map = stmt
            .query_map([], |row| {
                let broker_id: String = row.get(0)?;
                let oms_id: String = row.get(1)?;
                Ok((broker_id, oms_id))
            })?
            .filter_map(Result::ok)
            .collect();

        Ok(map)
    }

    /// Returns the OMS order ID for a given broker order ID, if it exists.
    pub fn find_order_id_by_broker_id(&self, broker_order_id: &str) -> OmsResult<Option<String>> {
        let result = self
            .conn
            .query_row(
                "SELECT id FROM orders WHERE broker_order_id = ?1",
                params![broker_order_id],
                |row| row.get(0),
            )
            .optional()?;
        Ok(result)
    }

    /// Returns the date (in local time) of the most recently created order,
    /// or `None` if no orders exist in the store.
    pub fn last_order_date(&self) -> OmsResult<Option<NaiveDate>> {
        let result = self
            .conn
            .query_row(
                "SELECT created_at FROM orders ORDER BY created_at DESC LIMIT 1",
                [],
                |row| row.get::<_, String>(0),
            )
            .optional()?;

        Ok(result.and_then(|s| {
            DateTime::parse_from_rfc3339(&s)
                .ok()
                .map(|dt| dt.with_timezone(&chrono::Local).date_naive())
        }))
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Order, OrderSide, OrderType};

    fn in_memory_store() -> SqliteStateStore {
        SqliteStateStore::new(":memory:").unwrap()
    }

    #[test]
    fn test_save_and_load_order() {
        let store = in_memory_store();
        let order = Order::new("AAPL", OrderSide::Buy, 10.0, OrderType::Market);
        let id = order.id.clone();
        store.save_order(&order).unwrap();

        let loaded = store.load_orders().unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].id, id);
        assert_eq!(loaded[0].symbol, "AAPL");
    }

    #[test]
    fn test_save_and_load_position() {
        let store = in_memory_store();
        let mut pos = Position::new("MSFT");
        pos.quantity = 5.0;
        pos.avg_cost = 300.0;
        pos.market_price = 310.0;
        store.save_position(&pos).unwrap();

        let loaded = store.load_positions().unwrap();
        assert_eq!(loaded.len(), 1);
        let loaded_pos = &loaded["MSFT"];
        assert!((loaded_pos.quantity - 5.0).abs() < 1e-12);
        assert!((loaded_pos.avg_cost - 300.0).abs() < 1e-12);
    }

    #[test]
    fn test_save_and_load_fill() {
        let store = in_memory_store();
        let fill = Fill {
            fill_id: "f1".into(),
            order_id: "o1".into(),
            broker_order_id: "b1".into(),
            symbol: "AAPL".into(),
            side: OrderSide::Buy,
            quantity: 10.0,
            price: 150.0,
            filled_at: Utc::now(),
            commission: 1.0,
        };
        store.save_fill(&fill).unwrap();

        let loaded = store.load_fills().unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].fill_id, "f1");
    }

    #[test]
    fn test_order_overwrite_on_replace() {
        let store = in_memory_store();
        let mut order = Order::new("AAPL", OrderSide::Buy, 10.0, OrderType::Market);
        store.save_order(&order).unwrap();
        order.status = OrderStatus::Filled;
        order.filled_quantity = 10.0;
        store.save_order(&order).unwrap();

        let loaded = store.load_orders().unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].status, OrderStatus::Filled);
    }
}
