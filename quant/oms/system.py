"""Order Management System — order lifecycle orchestrator.

Coordinates risk validation, broker submission, fill processing, and
position state. Thread-safe for concurrent fill callbacks.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import datetime, timezone

from loguru import logger

from quant.oms.broker import BrokerAdapter, BrokerError
from quant.oms.models import Fill, Order, OrderStatus, Position
from quant.oms.persistence import SQLiteStateStore


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class OrderManagementSystem:
    """Central order lifecycle manager.

    Responsibilities:
      - Submit orders through the broker adapter
      - Track all orders in memory (order book)
      - Process fills and update positions
      - Maintain a live position map
      - Provide fill event hooks for downstream consumers (e.g. strategy)

    The OMS deliberately does NOT run risk checks itself — the calling code
    (e.g. strategy or execution layer) must obtain a risk engine approval
    before calling :meth:`submit_order`.

    Usage::

        oms = OrderManagementSystem(broker=AlpacaAdapter(...))
        oms.start()

        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=10)
        broker_id = oms.submit_order(order)

        # Later...
        position = oms.get_position("AAPL")
        oms.stop()
    """

    def __init__(
        self,
        broker: BrokerAdapter,
        state_store: SQLiteStateStore | None = None,
    ) -> None:
        self._broker = broker
        self._store = state_store
        self._lock = threading.Lock()

        # Order book: oms_order_id → Order
        self._orders: dict[str, Order] = {}
        # broker_order_id → oms_order_id (reverse lookup)
        self._broker_id_map: dict[str, str] = {}

        # Position map: symbol → Position
        self._positions: dict[str, Position] = {}

        # Fills that arrived before their broker_id was registered (e.g. synchronous
        # paper fills) are buffered here until the mapping is established.
        self._pending_fills: dict[str, list[Fill]] = {}

        # Optional downstream fill hooks (e.g. strategy callbacks)
        self._fill_hooks: list[Callable[[Fill], None]] = []

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """Connect to the broker and register the fill callback."""
        self._broker.connect()
        self._broker.register_fill_callback(self.on_fill)
        logger.info("OMS started — broker connected")

    def stop(self) -> None:
        """Disconnect from the broker."""
        self._broker.disconnect()
        logger.info("OMS stopped")

    def __enter__(self) -> OrderManagementSystem:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    # ── State recovery ─────────────────────────────────────────────────────

    def restore_state(self) -> None:
        """Reload orders, positions, and broker-ID map from the state store.

        Call this once after construction (and before ``start()``) to recover
        state from a previous run.  If no state store is configured this is a
        no-op.
        """
        if self._store is None:
            return

        with self._lock:
            # Positions
            self._positions = self._store.load_positions()
            logger.info(
                "OMS: restored {} positions from state store",
                len(self._positions),
            )

            # Orders (all persisted, including terminal — needed for audit)
            orders = self._store.load_orders()
            for order in orders:
                self._orders[order.id] = order
                if order.broker_order_id is not None:
                    self._broker_id_map[order.broker_order_id] = order.id
            logger.info(
                "OMS: restored {} orders ({} active) from state store",
                len(orders),
                sum(1 for o in orders if o.is_active),
            )

    def save_snapshot(self, cash: float, peak_portfolio_value: float) -> None:
        """Persist a portfolio state snapshot for circuit-breaker recovery.

        Should be called periodically (e.g. after each runner cycle) by the
        service layer.
        """
        if self._store is not None:
            self._store.save_snapshot(cash, peak_portfolio_value)

    def load_latest_snapshot(self) -> dict | None:
        """Load the most recent portfolio snapshot, or None."""
        if self._store is None:
            return None
        return self._store.load_latest_snapshot()

    # ── Order submission ──────────────────────────────────────────────────

    def submit_order(self, order: Order) -> str:
        """Submit an order to the broker and register it in the order book.

        Args:
            order: Order to submit (status must be PENDING).

        Returns:
            The broker-assigned order ID.

        Raises:
            BrokerError: If the broker rejects the order.
            ValueError: If the order is in a non-pending status.
        """
        if order.status != OrderStatus.PENDING:
            raise ValueError(
                f"Cannot submit order {order.id} with status {order.status}"
            )

        with self._lock:
            self._orders[order.id] = order
        self._persist_order(order)

        try:
            broker_id = self._broker.submit_order(order)
        except BrokerError:
            with self._lock:
                order.status = OrderStatus.REJECTED
                order.updated_at = _utcnow()
            self._persist_order(order)
            logger.error("OMS: order {} rejected by broker", order.id)
            raise

        with self._lock:
            order.broker_order_id = broker_id
            order.status = OrderStatus.SUBMITTED
            order.updated_at = _utcnow()
            self._broker_id_map[broker_id] = order.id

            # Drain any fills that arrived synchronously before the mapping existed
            buffered = self._pending_fills.pop(broker_id, [])
            for buffered_fill in buffered:
                self._apply_fill_to_order(order, buffered_fill)
                self._apply_fill_to_position(buffered_fill)
        self._persist_order(order)

        for buffered_fill in buffered:
            self._persist_fill(buffered_fill)
            self._persist_position_for(buffered_fill.symbol)
            for hook in self._fill_hooks:
                try:
                    hook(buffered_fill)
                except Exception:
                    logger.exception("OMS: fill hook raised an exception (buffered fill)")

        logger.info(
            "OMS: submitted {} {} {} qty={} broker_id={}",
            order.symbol,
            order.side.value,
            order.order_type.value,
            order.quantity,
            broker_id,
        )
        return broker_id

    # ── Order cancellation ────────────────────────────────────────────────

    def cancel_order(self, order_id: str) -> bool:
        """Request cancellation of an open order.

        Args:
            order_id: Internal OMS order ID.

        Returns:
            True if the broker accepted the cancellation request.
        """
        with self._lock:
            order = self._orders.get(order_id)
        if order is None:
            logger.warning("OMS: cancel_order called for unknown id={}", order_id)
            return False
        if order.is_terminal:
            logger.debug(
                "OMS: cancel_order skipped for terminal order {} status={}",
                order_id,
                order.status.value,
            )
            return False
        if order.broker_order_id is None:
            logger.warning(
                "OMS: order {} has no broker_order_id yet — cannot cancel", order_id
            )
            return False

        accepted = self._broker.cancel_order(order.broker_order_id)
        if accepted:
            with self._lock:
                order.status = OrderStatus.CANCELLED
                order.updated_at = _utcnow()
            self._persist_order(order)
            logger.info("OMS: order {} cancelled", order_id)
        return accepted

    # ── Fill processing ───────────────────────────────────────────────────

    def on_fill(self, fill: Fill) -> None:
        """Process an incoming fill event from the broker adapter.

        Updates the associated order's status and cumulative fill fields,
        then updates the internal position map.  Finally, dispatches the
        fill to any registered downstream hooks.

        This method is called from the broker adapter — it may be invoked
        on a background thread, so all mutations are lock-protected.
        """
        with self._lock:
            oms_id = self._broker_id_map.get(fill.broker_order_id)
            order = self._orders.get(oms_id) if oms_id else None

            if order is not None:
                self._apply_fill_to_order(order, fill)
                self._apply_fill_to_position(fill)
            else:
                # Broker_id mapping not registered yet (synchronous fill race).
                # Buffer the fill so submit_order can drain it after registration.
                self._pending_fills.setdefault(fill.broker_order_id, []).append(fill)
                return

        self._persist_fill(fill)
        if order is not None:
            self._persist_order(order)
        self._persist_position_for(fill.symbol)

        logger.info(
            "OMS: fill — {} {} qty={:.4f} @ {:.4f}",
            fill.symbol,
            fill.side.value,
            fill.quantity,
            fill.price,
        )

        for hook in self._fill_hooks:
            try:
                hook(fill)
            except Exception:
                logger.exception("OMS: fill hook raised an exception")

    def _apply_fill_to_order(self, order: Order, fill: Fill) -> None:
        """Update order cumulative fields (caller holds the lock)."""
        prev_qty = order.filled_quantity
        new_qty = prev_qty + fill.quantity

        # Update weighted average fill price
        if new_qty > 0:
            order.avg_fill_price = (
                order.avg_fill_price * prev_qty + fill.price * fill.quantity
            ) / new_qty

        order.filled_quantity = new_qty
        order.updated_at = _utcnow()

        if new_qty >= order.quantity:
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIALLY_FILLED

    def _apply_fill_to_position(self, fill: Fill) -> None:
        """Update or create a position entry (caller holds the lock)."""
        position = self._positions.setdefault(fill.symbol, Position(symbol=fill.symbol))
        position.apply_fill(fill)
        if position.quantity == 0.0:
            del self._positions[fill.symbol]

    # ── Queries ───────────────────────────────────────────────────────────

    def get_order(self, order_id: str) -> Order | None:
        """Return the OMS order record by internal ID, or None."""
        with self._lock:
            return self._orders.get(order_id)

    def get_active_orders(self) -> list[Order]:
        """Return all orders that are not in a terminal state."""
        with self._lock:
            return [o for o in self._orders.values() if o.is_active]

    def get_position(self, symbol: str) -> Position | None:
        """Return the current OMS-tracked position for *symbol*, or None."""
        with self._lock:
            return self._positions.get(symbol)

    def get_all_positions(self) -> dict[str, Position]:
        """Return a snapshot of all open positions."""
        with self._lock:
            return dict(self._positions)

    def get_fills(self, broker_order_id: str) -> list[Fill]:
        """Proxy to the broker adapter for historical fills."""
        return self._broker.get_fills(broker_order_id)

    def get_account_cash(self) -> float:
        """Return current cash balance from the broker."""
        return self._broker.get_account_cash()

    # ── Fill hooks ────────────────────────────────────────────────────────

    def register_fill_hook(self, hook: Callable[[Fill], None]) -> None:
        """Register a callback that fires after every fill is processed.

        Multiple hooks can be registered and will all be called in order.
        Exceptions inside hooks are caught and logged without stopping others.
        """
        self._fill_hooks.append(hook)

    # ── Persistence helpers ────────────────────────────────────────────────

    def _persist_order(self, order: Order) -> None:
        if self._store is not None:
            self._store.save_order(order)

    def _persist_fill(self, fill: Fill) -> None:
        if self._store is not None:
            self._store.save_fill(fill)

    def _persist_position_for(self, symbol: str) -> None:
        if self._store is None:
            return
        with self._lock:
            position = self._positions.get(symbol)
        if position is not None:
            self._store.save_position(position)
        else:
            self._store.delete_position(symbol)
