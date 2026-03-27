//! OMS data models: Order, Fill, Position, and their enumerations.
//!
//! Mirrors `quant.oms.models` — field names and semantics are identical.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ── Enumerations ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OrderSide {
    Buy,
    Sell,
}

impl OrderSide {
    pub fn as_str(self) -> &'static str {
        match self {
            OrderSide::Buy => "buy",
            OrderSide::Sell => "sell",
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "buy" => Some(OrderSide::Buy),
            "sell" => Some(OrderSide::Sell),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OrderType {
    Market,
    Limit,
    Stop,
    StopLimit,
}

impl OrderType {
    pub fn as_str(self) -> &'static str {
        match self {
            OrderType::Market => "market",
            OrderType::Limit => "limit",
            OrderType::Stop => "stop",
            OrderType::StopLimit => "stop_limit",
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "market" => Some(OrderType::Market),
            "limit" => Some(OrderType::Limit),
            "stop" => Some(OrderType::Stop),
            "stop_limit" => Some(OrderType::StopLimit),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OrderStatus {
    Pending,
    Submitted,
    Accepted,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
    Expired,
}

impl OrderStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            OrderStatus::Pending => "pending",
            OrderStatus::Submitted => "submitted",
            OrderStatus::Accepted => "accepted",
            OrderStatus::PartiallyFilled => "partially_filled",
            OrderStatus::Filled => "filled",
            OrderStatus::Cancelled => "cancelled",
            OrderStatus::Rejected => "rejected",
            OrderStatus::Expired => "expired",
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "pending" => Some(OrderStatus::Pending),
            "submitted" => Some(OrderStatus::Submitted),
            "accepted" => Some(OrderStatus::Accepted),
            "partially_filled" => Some(OrderStatus::PartiallyFilled),
            "filled" => Some(OrderStatus::Filled),
            "cancelled" => Some(OrderStatus::Cancelled),
            "rejected" => Some(OrderStatus::Rejected),
            "expired" => Some(OrderStatus::Expired),
            _ => None,
        }
    }

    pub fn is_active(self) -> bool {
        matches!(
            self,
            OrderStatus::Pending
                | OrderStatus::Submitted
                | OrderStatus::Accepted
                | OrderStatus::PartiallyFilled
        )
    }

    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            OrderStatus::Filled
                | OrderStatus::Cancelled
                | OrderStatus::Rejected
                | OrderStatus::Expired
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TimeInForce {
    Day,
    Gtc,
    Ioc,
    Fok,
}

impl TimeInForce {
    pub fn as_str(self) -> &'static str {
        match self {
            TimeInForce::Day => "day",
            TimeInForce::Gtc => "gtc",
            TimeInForce::Ioc => "ioc",
            TimeInForce::Fok => "fok",
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "day" => Some(TimeInForce::Day),
            "gtc" => Some(TimeInForce::Gtc),
            "ioc" => Some(TimeInForce::Ioc),
            "fok" => Some(TimeInForce::Fok),
            _ => None,
        }
    }
}

// ── Order ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    /// Internal OMS order ID (UUID v4).
    pub id: String,
    pub symbol: String,
    pub side: OrderSide,
    /// Unsigned quantity of shares/contracts/units.
    pub quantity: f64,
    pub order_type: OrderType,
    pub limit_price: Option<f64>,
    pub stop_price: Option<f64>,
    pub time_in_force: TimeInForce,

    /// ID returned by the broker after submission.
    pub broker_order_id: Option<String>,
    pub status: OrderStatus,
    pub filled_quantity: f64,
    pub avg_fill_price: f64,

    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,

    pub strategy_id: Option<String>,
    pub sector: Option<String>,
}

impl Order {
    /// Create a new pending order with an auto-generated UUID.
    pub fn new(
        symbol: impl Into<String>,
        side: OrderSide,
        quantity: f64,
        order_type: OrderType,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            symbol: symbol.into(),
            side,
            quantity,
            order_type,
            limit_price: None,
            stop_price: None,
            time_in_force: TimeInForce::Day,
            broker_order_id: None,
            status: OrderStatus::Pending,
            filled_quantity: 0.0,
            avg_fill_price: 0.0,
            created_at: now,
            updated_at: now,
            strategy_id: None,
            sector: None,
        }
    }

    pub fn remaining_quantity(&self) -> f64 {
        self.quantity - self.filled_quantity
    }

    /// Positive for buys, negative for sells.
    pub fn signed_quantity(&self) -> f64 {
        if self.side == OrderSide::Buy {
            self.quantity
        } else {
            -self.quantity
        }
    }

    pub fn is_active(&self) -> bool {
        self.status.is_active()
    }

    pub fn is_terminal(&self) -> bool {
        self.status.is_terminal()
    }
}

// ── Fill ──────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fill {
    pub fill_id: String,
    pub order_id: String,
    pub broker_order_id: String,
    pub symbol: String,
    pub side: OrderSide,
    pub quantity: f64,
    pub price: f64,
    pub filled_at: DateTime<Utc>,
    pub commission: f64,
}

impl Fill {
    pub fn gross_value(&self) -> f64 {
        self.quantity * self.price
    }
}

// ── Position ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    /// Signed quantity — positive = long, negative = short.
    pub quantity: f64,
    /// Average cost basis per unit.
    pub avg_cost: f64,
    /// Latest market price (updated externally).
    pub market_price: f64,
}

impl Position {
    pub fn new(symbol: impl Into<String>) -> Self {
        Self {
            symbol: symbol.into(),
            quantity: 0.0,
            avg_cost: 0.0,
            market_price: 0.0,
        }
    }

    pub fn market_value(&self) -> f64 {
        self.quantity * self.market_price
    }

    pub fn cost_basis(&self) -> f64 {
        self.quantity * self.avg_cost
    }

    pub fn unrealized_pnl(&self) -> f64 {
        if self.quantity == 0.0 {
            return 0.0;
        }
        (self.market_price - self.avg_cost) * self.quantity
    }

    /// Update position after a fill using weighted average cost.
    pub fn apply_fill(&mut self, fill: &Fill) {
        let filled_qty = if fill.side == OrderSide::Buy {
            fill.quantity
        } else {
            -fill.quantity
        };
        let new_qty = self.quantity + filled_qty;

        if new_qty == 0.0 {
            self.avg_cost = 0.0;
        } else if (self.quantity >= 0.0 && filled_qty > 0.0)
            || (self.quantity <= 0.0 && filled_qty < 0.0)
        {
            // Increasing or same-side — update weighted average cost.
            let total_cost = self.avg_cost * self.quantity.abs() + fill.price * fill.quantity;
            self.avg_cost = total_cost / new_qty.abs();
        }
        // Reducing: cost basis stays the same (realized P&L tracked separately).

        self.quantity = new_qty;
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_new_is_pending() {
        let o = Order::new("AAPL", OrderSide::Buy, 10.0, OrderType::Market);
        assert_eq!(o.status, OrderStatus::Pending);
        assert!(o.is_active());
        assert!(!o.is_terminal());
    }

    #[test]
    fn test_signed_quantity_buy() {
        let o = Order::new("AAPL", OrderSide::Buy, 5.0, OrderType::Market);
        assert!((o.signed_quantity() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_signed_quantity_sell() {
        let o = Order::new("AAPL", OrderSide::Sell, 5.0, OrderType::Market);
        assert!((o.signed_quantity() + 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_position_apply_fill_buy_increases_qty() {
        let mut pos = Position::new("AAPL");
        let fill = Fill {
            fill_id: "f1".into(),
            order_id: "o1".into(),
            broker_order_id: "b1".into(),
            symbol: "AAPL".into(),
            side: OrderSide::Buy,
            quantity: 10.0,
            price: 150.0,
            filled_at: Utc::now(),
            commission: 0.0,
        };
        pos.apply_fill(&fill);
        assert!((pos.quantity - 10.0).abs() < 1e-12);
        assert!((pos.avg_cost - 150.0).abs() < 1e-12);
    }

    #[test]
    fn test_position_apply_fill_sell_reduces_qty() {
        let mut pos = Position {
            symbol: "AAPL".into(),
            quantity: 10.0,
            avg_cost: 150.0,
            market_price: 160.0,
        };
        let fill = Fill {
            fill_id: "f2".into(),
            order_id: "o2".into(),
            broker_order_id: "b2".into(),
            symbol: "AAPL".into(),
            side: OrderSide::Sell,
            quantity: 5.0,
            price: 160.0,
            filled_at: Utc::now(),
            commission: 0.0,
        };
        pos.apply_fill(&fill);
        assert!((pos.quantity - 5.0).abs() < 1e-12);
        // avg_cost unchanged on reduction
        assert!((pos.avg_cost - 150.0).abs() < 1e-12);
    }

    #[test]
    fn test_position_close_resets_avg_cost() {
        let mut pos = Position {
            symbol: "AAPL".into(),
            quantity: 10.0,
            avg_cost: 150.0,
            market_price: 0.0,
        };
        let fill = Fill {
            fill_id: "f3".into(),
            order_id: "o3".into(),
            broker_order_id: "b3".into(),
            symbol: "AAPL".into(),
            side: OrderSide::Sell,
            quantity: 10.0,
            price: 160.0,
            filled_at: Utc::now(),
            commission: 0.0,
        };
        pos.apply_fill(&fill);
        assert!(pos.quantity.abs() < 1e-12);
        assert!(pos.avg_cost.abs() < 1e-12);
    }

    #[test]
    fn test_order_status_round_trip() {
        let statuses = [
            "pending",
            "submitted",
            "accepted",
            "partially_filled",
            "filled",
            "cancelled",
            "rejected",
            "expired",
        ];
        for s in statuses {
            let status = OrderStatus::from_str(s).unwrap();
            assert_eq!(status.as_str(), s);
        }
    }
}
