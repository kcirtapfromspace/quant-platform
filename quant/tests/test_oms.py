"""Tests for OMS models, BrokerAdapter interface, and OrderManagementSystem."""
from __future__ import annotations

import pytest

from quant.oms.models import (
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    TimeInForce,
)
from quant.oms.system import OrderManagementSystem
from quant.execution.paper import PaperBrokerAdapter


# ── Model tests ──────────────────────────────────────────────────────────────


class TestOrder:
    def test_signed_quantity_buy(self):
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=10)
        assert order.signed_quantity() == 10

    def test_signed_quantity_sell(self):
        order = Order(symbol="AAPL", side=OrderSide.SELL, quantity=10)
        assert order.signed_quantity() == -10

    def test_remaining_quantity(self):
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
        order.filled_quantity = 40
        assert order.remaining_quantity == 60

    def test_is_active_pending(self):
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=1)
        assert order.is_active is True
        assert order.is_terminal is False

    def test_is_terminal_filled(self):
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=1)
        order.status = OrderStatus.FILLED
        assert order.is_terminal is True
        assert order.is_active is False

    def test_is_terminal_cancelled(self):
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=1)
        order.status = OrderStatus.CANCELLED
        assert order.is_terminal is True


class TestPosition:
    def _make_fill(self, side: OrderSide, qty: float, price: float) -> Fill:
        from datetime import datetime, timezone
        import uuid
        return Fill(
            order_id="o1",
            broker_order_id="b1",
            fill_id=str(uuid.uuid4()),
            symbol="AAPL",
            side=side,
            quantity=qty,
            price=price,
            filled_at=datetime.now(timezone.utc),
        )

    def test_apply_buy_fill_new_position(self):
        pos = Position(symbol="AAPL")
        fill = self._make_fill(OrderSide.BUY, 10, 150.0)
        pos.apply_fill(fill)
        assert pos.quantity == 10
        assert pos.avg_cost == 150.0

    def test_apply_buy_fills_weighted_avg(self):
        pos = Position(symbol="AAPL")
        pos.apply_fill(self._make_fill(OrderSide.BUY, 10, 100.0))
        pos.apply_fill(self._make_fill(OrderSide.BUY, 10, 200.0))
        assert pos.quantity == 20
        assert pos.avg_cost == pytest.approx(150.0)

    def test_apply_sell_reduces_position(self):
        pos = Position(symbol="AAPL")
        pos.apply_fill(self._make_fill(OrderSide.BUY, 20, 100.0))
        pos.apply_fill(self._make_fill(OrderSide.SELL, 10, 120.0))
        assert pos.quantity == 10
        assert pos.avg_cost == pytest.approx(100.0)  # cost basis unchanged on reduce

    def test_apply_full_sell_clears_position(self):
        pos = Position(symbol="AAPL")
        pos.apply_fill(self._make_fill(OrderSide.BUY, 10, 100.0))
        pos.apply_fill(self._make_fill(OrderSide.SELL, 10, 110.0))
        assert pos.quantity == 0
        assert pos.avg_cost == 0.0

    def test_unrealized_pnl(self):
        pos = Position(symbol="AAPL", quantity=10, avg_cost=100.0, market_price=110.0)
        assert pos.unrealized_pnl == pytest.approx(100.0)

    def test_market_value(self):
        pos = Position(symbol="AAPL", quantity=5, market_price=200.0)
        assert pos.market_value == pytest.approx(1000.0)


# ── PaperBrokerAdapter tests ──────────────────────────────────────────────────


class TestPaperBrokerAdapter:
    def test_connect_disconnect(self):
        adapter = PaperBrokerAdapter()
        assert not adapter.is_connected
        adapter.connect()
        assert adapter.is_connected
        adapter.disconnect()
        assert not adapter.is_connected

    def test_submit_market_order_buy(self):
        adapter = PaperBrokerAdapter(default_fill_price=150.0)
        adapter.connect()
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=10)
        broker_id = adapter.submit_order(order)
        assert broker_id is not None
        pos = adapter.get_position("AAPL")
        assert pos is not None
        assert pos.quantity == 10
        assert pos.avg_cost == pytest.approx(150.0)

    def test_submit_market_order_sell(self):
        adapter = PaperBrokerAdapter(default_fill_price=150.0)
        adapter.connect()
        # Build a position first
        adapter.submit_order(Order(symbol="AAPL", side=OrderSide.BUY, quantity=10))
        adapter.submit_order(Order(symbol="AAPL", side=OrderSide.SELL, quantity=10))
        pos = adapter.get_position("AAPL")
        assert pos is None  # flat

    def test_limit_order_filled_at_limit_price(self):
        adapter = PaperBrokerAdapter(default_fill_price=200.0)
        adapter.connect()
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=5,
            order_type=OrderType.LIMIT,
            limit_price=155.0,
        )
        adapter.submit_order(order)
        pos = adapter.get_position("AAPL")
        assert pos.avg_cost == pytest.approx(155.0)

    def test_cash_decreases_after_buy(self):
        adapter = PaperBrokerAdapter(initial_cash=10_000.0, default_fill_price=100.0)
        adapter.connect()
        adapter.submit_order(Order(symbol="AAPL", side=OrderSide.BUY, quantity=10))
        assert adapter.get_account_cash() == pytest.approx(9_000.0)

    def test_cash_increases_after_sell(self):
        adapter = PaperBrokerAdapter(initial_cash=10_000.0, default_fill_price=100.0)
        adapter.connect()
        adapter.submit_order(Order(symbol="AAPL", side=OrderSide.BUY, quantity=10))
        adapter.submit_order(Order(symbol="AAPL", side=OrderSide.SELL, quantity=10))
        assert adapter.get_account_cash() == pytest.approx(10_000.0)

    def test_fill_callback_triggered(self):
        received = []
        adapter = PaperBrokerAdapter(default_fill_price=100.0)
        adapter.connect()
        adapter.register_fill_callback(lambda f: received.append(f))
        adapter.submit_order(Order(symbol="AAPL", side=OrderSide.BUY, quantity=5))
        assert len(received) == 1
        assert received[0].symbol == "AAPL"
        assert received[0].quantity == 5

    def test_get_fills_returns_fill(self):
        adapter = PaperBrokerAdapter(default_fill_price=100.0)
        adapter.connect()
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=5)
        broker_id = adapter.submit_order(order)
        fills = adapter.get_fills(broker_id)
        assert len(fills) == 1
        assert fills[0].price == pytest.approx(100.0)

    def test_context_manager(self):
        with PaperBrokerAdapter() as adapter:
            assert adapter.is_connected
        assert not adapter.is_connected

    def test_cancel_order_returns_false_paper(self):
        adapter = PaperBrokerAdapter(default_fill_price=100.0)
        adapter.connect()
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=5)
        broker_id = adapter.submit_order(order)
        # Paper fills immediately so cancel should return False
        result = adapter.cancel_order(broker_id)
        assert result is False

    def test_price_feed_used(self):
        def feed(symbol):
            return 999.0 if symbol == "TSLA" else None

        adapter = PaperBrokerAdapter(price_feed=feed, default_fill_price=1.0)
        adapter.connect()
        adapter.submit_order(Order(symbol="TSLA", side=OrderSide.BUY, quantity=1))
        pos = adapter.get_position("TSLA")
        assert pos.avg_cost == pytest.approx(999.0)


# ── OrderManagementSystem tests ───────────────────────────────────────────────


class TestOrderManagementSystem:
    def _make_oms(self, **adapter_kwargs) -> OrderManagementSystem:
        adapter = PaperBrokerAdapter(**adapter_kwargs)
        oms = OrderManagementSystem(broker=adapter)
        oms.start()
        return oms

    def test_submit_order_returns_broker_id(self):
        oms = self._make_oms(default_fill_price=100.0)
        try:
            order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=10)
            broker_id = oms.submit_order(order)
            assert broker_id is not None
        finally:
            oms.stop()

    def test_position_updated_after_fill(self):
        oms = self._make_oms(default_fill_price=200.0)
        try:
            order = Order(symbol="MSFT", side=OrderSide.BUY, quantity=5)
            oms.submit_order(order)
            pos = oms.get_position("MSFT")
            assert pos is not None
            assert pos.quantity == 5
            assert pos.avg_cost == pytest.approx(200.0)
        finally:
            oms.stop()

    def test_flat_position_removed(self):
        oms = self._make_oms(default_fill_price=100.0)
        try:
            oms.submit_order(Order(symbol="AAPL", side=OrderSide.BUY, quantity=10))
            oms.submit_order(Order(symbol="AAPL", side=OrderSide.SELL, quantity=10))
            pos = oms.get_position("AAPL")
            assert pos is None
        finally:
            oms.stop()

    def test_order_status_filled(self):
        oms = self._make_oms(default_fill_price=100.0)
        try:
            order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=10)
            oms.submit_order(order)
            retrieved = oms.get_order(order.id)
            assert retrieved.status == OrderStatus.FILLED
            assert retrieved.filled_quantity == 10
        finally:
            oms.stop()

    def test_submit_non_pending_order_raises(self):
        oms = self._make_oms()
        try:
            order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=5)
            order.status = OrderStatus.SUBMITTED
            with pytest.raises(ValueError):
                oms.submit_order(order)
        finally:
            oms.stop()

    def test_fill_hook_called(self):
        received = []
        oms = self._make_oms(default_fill_price=50.0)
        oms.register_fill_hook(lambda f: received.append(f))
        try:
            oms.submit_order(Order(symbol="NVDA", side=OrderSide.BUY, quantity=3))
            assert len(received) == 1
            assert received[0].symbol == "NVDA"
        finally:
            oms.stop()

    def test_get_active_orders_after_fill(self):
        oms = self._make_oms(default_fill_price=100.0)
        try:
            order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=10)
            oms.submit_order(order)
            # Paper fills immediately → no active orders
            active = oms.get_active_orders()
            assert len(active) == 0
        finally:
            oms.stop()

    def test_cancel_order_unknown_id(self):
        oms = self._make_oms()
        try:
            result = oms.cancel_order("nonexistent-id")
            assert result is False
        finally:
            oms.stop()

    def test_context_manager(self):
        adapter = PaperBrokerAdapter(default_fill_price=100.0)
        with OrderManagementSystem(broker=adapter) as oms:
            order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=1)
            oms.submit_order(order)
            assert oms.get_position("AAPL") is not None

    def test_multiple_positions(self):
        oms = self._make_oms(default_fill_price=100.0)
        try:
            oms.submit_order(Order(symbol="AAPL", side=OrderSide.BUY, quantity=5))
            oms.submit_order(Order(symbol="MSFT", side=OrderSide.BUY, quantity=8))
            positions = oms.get_all_positions()
            assert "AAPL" in positions
            assert "MSFT" in positions
            assert positions["AAPL"].quantity == 5
            assert positions["MSFT"].quantity == 8
        finally:
            oms.stop()
