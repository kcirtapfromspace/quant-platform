"""OMS data models: Order, Fill, Position, and their enumerations."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    PENDING = "pending"       # created locally, not yet sent to broker
    SUBMITTED = "submitted"   # sent to broker, awaiting acknowledgment
    ACCEPTED = "accepted"     # broker acknowledged
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeInForce(str, Enum):
    DAY = "day"
    GTC = "gtc"   # good till cancelled
    IOC = "ioc"   # immediate or cancel
    FOK = "fok"   # fill or kill


@dataclass
class Order:
    """A single order in the OMS lifecycle.

    Attributes:
        symbol: Asset identifier (e.g. "AAPL", "BTC/USDT").
        side: BUY or SELL.
        quantity: Unsigned quantity of shares/contracts/units to trade.
        order_type: MARKET, LIMIT, STOP, or STOP_LIMIT.
        limit_price: Required for LIMIT and STOP_LIMIT orders.
        stop_price: Required for STOP and STOP_LIMIT orders.
        time_in_force: Order duration.
        id: Internal OMS order ID (UUID, auto-generated).
        broker_order_id: ID returned by the broker after submission.
        status: Current lifecycle status.
        filled_quantity: Cumulative filled quantity.
        avg_fill_price: Average price of fills so far.
        created_at: UTC timestamp when the order was created.
        updated_at: UTC timestamp of the last status change.
        strategy_id: Optional tag to track which strategy generated the order.
        sector: Optional sector label used by the risk engine.
    """

    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    broker_order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0

    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)

    strategy_id: Optional[str] = None
    sector: Optional[str] = None

    @property
    def is_active(self) -> bool:
        return self.status in (
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.ACCEPTED,
            OrderStatus.PARTIALLY_FILLED,
        )

    @property
    def is_terminal(self) -> bool:
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        )

    @property
    def remaining_quantity(self) -> float:
        return self.quantity - self.filled_quantity

    def signed_quantity(self) -> float:
        """Returns positive for buys, negative for sells."""
        return self.quantity if self.side == OrderSide.BUY else -self.quantity


@dataclass
class Fill:
    """A single execution report (fill) from the broker.

    Attributes:
        order_id: Internal OMS order ID this fill belongs to.
        broker_order_id: Broker-side order reference.
        fill_id: Unique fill/execution ID from the broker.
        symbol: Asset identifier.
        side: BUY or SELL.
        quantity: Quantity filled in this event.
        price: Fill price per unit.
        filled_at: UTC timestamp of the fill.
        commission: Commission charged for this fill.
    """

    order_id: str
    broker_order_id: str
    fill_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    filled_at: datetime
    commission: float = 0.0

    @property
    def gross_value(self) -> float:
        return self.quantity * self.price


@dataclass
class Position:
    """Current holding for a single symbol.

    Attributes:
        symbol: Asset identifier.
        quantity: Signed quantity — positive = long, negative = short.
        avg_cost: Average cost basis per unit.
        market_price: Latest market price per unit (updated externally).
        unrealized_pnl: Unrealized P&L based on market_price.
    """

    symbol: str
    quantity: float = 0.0
    avg_cost: float = 0.0
    market_price: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.market_price

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.avg_cost

    @property
    def unrealized_pnl(self) -> float:
        if self.quantity == 0:
            return 0.0
        return (self.market_price - self.avg_cost) * self.quantity

    def apply_fill(self, fill: Fill) -> None:
        """Update position after a fill using weighted average cost."""
        filled_qty = fill.quantity if fill.side == OrderSide.BUY else -fill.quantity
        new_qty = self.quantity + filled_qty

        if new_qty == 0.0:
            self.avg_cost = 0.0
        elif (self.quantity >= 0 and filled_qty > 0) or (
            self.quantity <= 0 and filled_qty < 0
        ):
            # Increasing or same-side — update weighted avg cost
            total_cost = self.avg_cost * abs(self.quantity) + fill.price * fill.quantity
            self.avg_cost = total_cost / abs(new_qty)
        # Reducing: cost basis stays the same (realized P&L is tracked separately)

        self.quantity = new_qty
