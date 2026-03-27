"""Paper trading broker adapter — simulates fills without real orders.

Fills are executed immediately at the provided limit price or the last
known price for market orders. Suitable for strategy testing and CI.
"""
from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from loguru import logger

from quant.oms.broker import BrokerAdapter
from quant.oms.models import Fill, Order, OrderSide, OrderStatus, OrderType, Position


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PaperBrokerAdapter(BrokerAdapter):
    """Simulated broker that fills all orders immediately.

    Useful for:
      - Paper trading (live market data, simulated execution)
      - Strategy back-testing integration tests
      - CI pipelines that cannot use real broker APIs

    Fill price logic:
      - MARKET orders: filled at *default_price* if no price map is set, or
        at the price returned by the optional price_feed callable.
      - LIMIT orders: filled at the limit_price (no slippage simulation).
      - STOP orders: filled at the stop_price.

    Args:
        initial_cash: Starting cash balance for the paper account.
        price_feed: Optional callable ``(symbol: str) -> float`` that returns
            the current market price for a symbol.  Falls back to
            *default_fill_price* when the callable returns None or is not set.
        default_fill_price: Fallback price used when price_feed is absent or
            returns None (default 100.0).
        commission_rate: Flat percentage commission per fill (default 0.0).
    """

    def __init__(
        self,
        initial_cash: float = 100_000.0,
        price_feed=None,
        default_fill_price: float = 100.0,
        commission_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self._cash = initial_cash
        self._price_feed = price_feed
        self._default_fill_price = default_fill_price
        self._commission_rate = commission_rate

        self._lock = threading.Lock()
        self._connected = False

        # broker_order_id → Order
        self._orders: Dict[str, Order] = {}
        # broker_order_id → list[Fill]
        self._fills: Dict[str, List[Fill]] = {}
        # symbol → Position
        self._positions: Dict[str, Position] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def connect(self) -> None:
        self._connected = True
        logger.info("PaperBrokerAdapter: connected (cash={:.2f})", self._cash)

    def disconnect(self) -> None:
        self._connected = False
        logger.info("PaperBrokerAdapter: disconnected")

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── Order operations ──────────────────────────────────────────────────

    def submit_order(self, order: Order) -> str:
        broker_id = str(uuid.uuid4())
        fill_price = self._resolve_fill_price(order)
        commission = fill_price * order.quantity * self._commission_rate

        fill = Fill(
            order_id=order.id,
            broker_order_id=broker_id,
            fill_id=str(uuid.uuid4()),
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            filled_at=_utcnow(),
            commission=commission,
        )

        with self._lock:
            self._orders[broker_id] = order
            self._fills[broker_id] = [fill]
            self._update_paper_position(fill)
            cash_delta = fill.gross_value + commission
            if order.side == OrderSide.BUY:
                self._cash -= cash_delta
            else:
                self._cash += fill.gross_value - commission

        logger.debug(
            "PaperBrokerAdapter: filled {} {} qty={} @ {:.4f} broker_id={}",
            order.symbol,
            order.side.value,
            order.quantity,
            fill_price,
            broker_id,
        )
        self._notify_fill(fill)
        return broker_id

    def cancel_order(self, broker_order_id: str) -> bool:
        with self._lock:
            order = self._orders.get(broker_order_id)
        if order is None:
            return False
        # Paper orders are filled immediately so they're already terminal
        logger.debug(
            "PaperBrokerAdapter: cancel_order {} — already filled (paper)", broker_order_id
        )
        return False

    # ── Query operations ──────────────────────────────────────────────────

    def get_position(self, symbol: str) -> Optional[Position]:
        with self._lock:
            return self._positions.get(symbol)

    def get_fills(self, broker_order_id: str) -> list[Fill]:
        with self._lock:
            return list(self._fills.get(broker_order_id, []))

    def get_account_cash(self) -> float:
        with self._lock:
            return self._cash

    # ── Helpers ───────────────────────────────────────────────────────────

    def _resolve_fill_price(self, order: Order) -> float:
        if order.order_type == OrderType.LIMIT and order.limit_price is not None:
            return order.limit_price
        if order.order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and order.stop_price:
            return order.stop_price
        if self._price_feed is not None:
            price = self._price_feed(order.symbol)
            if price is not None:
                return float(price)
        return self._default_fill_price

    def _update_paper_position(self, fill: Fill) -> None:
        """Update internal position (caller holds the lock)."""
        pos = self._positions.setdefault(fill.symbol, Position(symbol=fill.symbol))
        pos.apply_fill(fill)
        if pos.quantity == 0.0:
            del self._positions[fill.symbol]
