"""Tests for OMS state persistence and recovery (QUA-28)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from quant.execution.paper import PaperBrokerAdapter
from quant.oms.models import (
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    TimeInForce,
)
from quant.oms.persistence import SQLiteStateStore
from quant.oms.system import OrderManagementSystem


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── SQLiteStateStore unit tests ──────────────────────────────────────────────


class TestSQLiteStateStoreOrders:
    def test_save_and_load_order(self):
        store = SQLiteStateStore(":memory:")
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.LIMIT,
            limit_price=150.0,
            time_in_force=TimeInForce.DAY,
            strategy_id="strat_1",
            sector="tech",
        )
        store.save_order(order)
        loaded = store.load_orders()
        assert len(loaded) == 1
        o = loaded[0]
        assert o.id == order.id
        assert o.symbol == "AAPL"
        assert o.side == OrderSide.BUY
        assert o.quantity == 10
        assert o.order_type == OrderType.LIMIT
        assert o.limit_price == 150.0
        assert o.time_in_force == TimeInForce.DAY
        assert o.status == OrderStatus.PENDING
        assert o.strategy_id == "strat_1"
        assert o.sector == "tech"
        store.close()

    def test_upsert_updates_mutable_fields(self):
        store = SQLiteStateStore(":memory:")
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=10)
        store.save_order(order)

        order.status = OrderStatus.FILLED
        order.broker_order_id = "broker-123"
        order.filled_quantity = 10
        order.avg_fill_price = 155.0
        order.updated_at = _utcnow()
        store.save_order(order)

        loaded = store.load_orders()
        assert len(loaded) == 1
        assert loaded[0].status == OrderStatus.FILLED
        assert loaded[0].broker_order_id == "broker-123"
        assert loaded[0].filled_quantity == 10
        assert loaded[0].avg_fill_price == 155.0
        store.close()

    def test_load_active_orders(self):
        store = SQLiteStateStore(":memory:")
        active = Order(symbol="AAPL", side=OrderSide.BUY, quantity=5)
        active.status = OrderStatus.SUBMITTED
        store.save_order(active)

        filled = Order(symbol="MSFT", side=OrderSide.SELL, quantity=3)
        filled.status = OrderStatus.FILLED
        store.save_order(filled)

        cancelled = Order(symbol="TSLA", side=OrderSide.BUY, quantity=1)
        cancelled.status = OrderStatus.CANCELLED
        store.save_order(cancelled)

        result = store.load_active_orders()
        assert len(result) == 1
        assert result[0].id == active.id
        store.close()

    def test_optional_fields_null(self):
        store = SQLiteStateStore(":memory:")
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=1)
        store.save_order(order)
        loaded = store.load_orders()[0]
        assert loaded.limit_price is None
        assert loaded.stop_price is None
        assert loaded.broker_order_id is None
        assert loaded.strategy_id is None
        assert loaded.sector is None
        store.close()


class TestSQLiteStateStoreFills:
    def test_save_and_load_fill(self):
        store = SQLiteStateStore(":memory:")
        now = _utcnow()
        fill = Fill(
            order_id="o1",
            broker_order_id="b1",
            fill_id="f1",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            price=150.0,
            filled_at=now,
            commission=1.50,
        )
        store.save_fill(fill)
        loaded = store.load_fills()
        assert len(loaded) == 1
        f = loaded[0]
        assert f.fill_id == "f1"
        assert f.order_id == "o1"
        assert f.broker_order_id == "b1"
        assert f.symbol == "AAPL"
        assert f.side == OrderSide.BUY
        assert f.quantity == 10
        assert f.price == pytest.approx(150.0)
        assert f.commission == pytest.approx(1.50)
        store.close()

    def test_duplicate_fill_ignored(self):
        store = SQLiteStateStore(":memory:")
        fill = Fill(
            order_id="o1",
            broker_order_id="b1",
            fill_id="f1",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            price=150.0,
            filled_at=_utcnow(),
        )
        store.save_fill(fill)
        store.save_fill(fill)  # duplicate
        assert len(store.load_fills()) == 1
        store.close()

    def test_load_fills_for_order(self):
        store = SQLiteStateStore(":memory:")
        f1 = Fill(
            order_id="o1", broker_order_id="b1", fill_id="f1",
            symbol="AAPL", side=OrderSide.BUY, quantity=5,
            price=100.0, filled_at=_utcnow(),
        )
        f2 = Fill(
            order_id="o1", broker_order_id="b1", fill_id="f2",
            symbol="AAPL", side=OrderSide.BUY, quantity=5,
            price=101.0, filled_at=_utcnow(),
        )
        f3 = Fill(
            order_id="o2", broker_order_id="b2", fill_id="f3",
            symbol="MSFT", side=OrderSide.BUY, quantity=3,
            price=300.0, filled_at=_utcnow(),
        )
        store.save_fill(f1)
        store.save_fill(f2)
        store.save_fill(f3)
        o1_fills = store.load_fills_for_order("o1")
        assert len(o1_fills) == 2
        assert all(f.order_id == "o1" for f in o1_fills)
        store.close()


class TestSQLiteStateStorePositions:
    def test_save_and_load_positions(self):
        store = SQLiteStateStore(":memory:")
        pos = Position(symbol="AAPL", quantity=10, avg_cost=150.0, market_price=155.0)
        store.save_position(pos)
        loaded = store.load_positions()
        assert "AAPL" in loaded
        p = loaded["AAPL"]
        assert p.quantity == 10
        assert p.avg_cost == pytest.approx(150.0)
        assert p.market_price == pytest.approx(155.0)
        store.close()

    def test_upsert_position(self):
        store = SQLiteStateStore(":memory:")
        pos = Position(symbol="AAPL", quantity=10, avg_cost=150.0)
        store.save_position(pos)
        pos.quantity = 20
        pos.avg_cost = 145.0
        store.save_position(pos)
        loaded = store.load_positions()
        assert loaded["AAPL"].quantity == 20
        assert loaded["AAPL"].avg_cost == pytest.approx(145.0)
        store.close()

    def test_delete_position(self):
        store = SQLiteStateStore(":memory:")
        store.save_position(Position(symbol="AAPL", quantity=10, avg_cost=100.0))
        store.delete_position("AAPL")
        loaded = store.load_positions()
        assert "AAPL" not in loaded
        store.close()

    def test_multiple_positions(self):
        store = SQLiteStateStore(":memory:")
        store.save_position(Position(symbol="AAPL", quantity=10, avg_cost=100.0))
        store.save_position(Position(symbol="MSFT", quantity=5, avg_cost=300.0))
        loaded = store.load_positions()
        assert len(loaded) == 2
        assert "AAPL" in loaded
        assert "MSFT" in loaded
        store.close()


class TestSQLiteStateStoreSnapshots:
    def test_save_and_load_snapshot(self):
        store = SQLiteStateStore(":memory:")
        store.save_snapshot(cash=50_000.0, peak_portfolio_value=120_000.0)
        snap = store.load_latest_snapshot()
        assert snap is not None
        assert snap["cash"] == pytest.approx(50_000.0)
        assert snap["peak_portfolio_value"] == pytest.approx(120_000.0)
        assert snap["timestamp"] is not None
        store.close()

    def test_latest_snapshot_is_most_recent(self):
        store = SQLiteStateStore(":memory:")
        store.save_snapshot(cash=10_000.0, peak_portfolio_value=50_000.0)
        store.save_snapshot(cash=20_000.0, peak_portfolio_value=80_000.0)
        snap = store.load_latest_snapshot()
        assert snap["cash"] == pytest.approx(20_000.0)
        assert snap["peak_portfolio_value"] == pytest.approx(80_000.0)
        store.close()

    def test_no_snapshots_returns_none(self):
        store = SQLiteStateStore(":memory:")
        assert store.load_latest_snapshot() is None
        store.close()


class TestPurgeTerminalOrders:
    def test_purge_removes_old_terminal_orders_and_fills(self):
        store = SQLiteStateStore(":memory:")
        old_time = _utcnow() - timedelta(days=30)

        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=10)
        order.status = OrderStatus.FILLED
        order.updated_at = old_time
        store.save_order(order)

        fill = Fill(
            order_id=order.id, broker_order_id="b1", fill_id="f1",
            symbol="AAPL", side=OrderSide.BUY, quantity=10,
            price=150.0, filled_at=old_time,
        )
        store.save_fill(fill)

        cutoff = _utcnow() - timedelta(days=7)
        deleted = store.purge_terminal_orders(before=cutoff)
        assert deleted == 1
        assert len(store.load_orders()) == 0
        assert len(store.load_fills()) == 0
        store.close()

    def test_purge_keeps_recent_and_active_orders(self):
        store = SQLiteStateStore(":memory:")

        # Recent filled order
        recent = Order(symbol="AAPL", side=OrderSide.BUY, quantity=5)
        recent.status = OrderStatus.FILLED
        recent.updated_at = _utcnow()
        store.save_order(recent)

        # Active order
        active = Order(symbol="MSFT", side=OrderSide.BUY, quantity=3)
        active.status = OrderStatus.SUBMITTED
        active.updated_at = _utcnow() - timedelta(days=30)
        store.save_order(active)

        cutoff = _utcnow() - timedelta(days=7)
        deleted = store.purge_terminal_orders(before=cutoff)
        assert deleted == 0
        assert len(store.load_orders()) == 2
        store.close()


# ── OMS + Persistence integration tests ─────────────────────────────────────


class TestOMSPersistenceIntegration:
    def _make_oms(
        self, store: SQLiteStateStore, **adapter_kwargs
    ) -> OrderManagementSystem:
        adapter = PaperBrokerAdapter(**adapter_kwargs)
        oms = OrderManagementSystem(broker=adapter, state_store=store)
        oms.start()
        return oms

    def test_orders_persisted_on_submit(self):
        store = SQLiteStateStore(":memory:")
        oms = self._make_oms(store, default_fill_price=100.0)
        try:
            order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=10)
            oms.submit_order(order)
            db_orders = store.load_orders()
            assert len(db_orders) == 1
            assert db_orders[0].id == order.id
            assert db_orders[0].status == OrderStatus.FILLED
        finally:
            oms.stop()
            store.close()

    def test_fills_persisted_on_submit(self):
        store = SQLiteStateStore(":memory:")
        oms = self._make_oms(store, default_fill_price=150.0)
        try:
            order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=5)
            oms.submit_order(order)
            fills = store.load_fills()
            assert len(fills) == 1
            assert fills[0].symbol == "AAPL"
            assert fills[0].price == pytest.approx(150.0)
        finally:
            oms.stop()
            store.close()

    def test_positions_persisted_on_fill(self):
        store = SQLiteStateStore(":memory:")
        oms = self._make_oms(store, default_fill_price=200.0)
        try:
            oms.submit_order(Order(symbol="AAPL", side=OrderSide.BUY, quantity=10))
            positions = store.load_positions()
            assert "AAPL" in positions
            assert positions["AAPL"].quantity == 10
            assert positions["AAPL"].avg_cost == pytest.approx(200.0)
        finally:
            oms.stop()
            store.close()

    def test_closed_position_deleted_from_store(self):
        store = SQLiteStateStore(":memory:")
        oms = self._make_oms(store, default_fill_price=100.0)
        try:
            oms.submit_order(Order(symbol="AAPL", side=OrderSide.BUY, quantity=10))
            assert "AAPL" in store.load_positions()
            oms.submit_order(Order(symbol="AAPL", side=OrderSide.SELL, quantity=10))
            assert "AAPL" not in store.load_positions()
        finally:
            oms.stop()
            store.close()

    def test_snapshot_round_trip(self):
        store = SQLiteStateStore(":memory:")
        adapter = PaperBrokerAdapter()
        oms = OrderManagementSystem(broker=adapter, state_store=store)
        oms.save_snapshot(cash=75_000.0, peak_portfolio_value=110_000.0)
        snap = oms.load_latest_snapshot()
        assert snap is not None
        assert snap["cash"] == pytest.approx(75_000.0)
        assert snap["peak_portfolio_value"] == pytest.approx(110_000.0)
        store.close()


class TestOMSStateRecovery:
    def test_restore_positions(self):
        store = SQLiteStateStore(":memory:")
        store.save_position(Position(symbol="AAPL", quantity=10, avg_cost=150.0))
        store.save_position(Position(symbol="MSFT", quantity=5, avg_cost=300.0))

        adapter = PaperBrokerAdapter()
        oms = OrderManagementSystem(broker=adapter, state_store=store)
        oms.restore_state()

        assert oms.get_position("AAPL").quantity == 10
        assert oms.get_position("MSFT").quantity == 5
        assert len(oms.get_all_positions()) == 2
        store.close()

    def test_restore_orders_and_broker_id_map(self):
        store = SQLiteStateStore(":memory:")
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=10)
        order.broker_order_id = "broker-abc"
        order.status = OrderStatus.FILLED
        order.filled_quantity = 10
        order.avg_fill_price = 155.0
        store.save_order(order)

        adapter = PaperBrokerAdapter()
        oms = OrderManagementSystem(broker=adapter, state_store=store)
        oms.restore_state()

        restored = oms.get_order(order.id)
        assert restored is not None
        assert restored.status == OrderStatus.FILLED
        assert restored.broker_order_id == "broker-abc"
        store.close()

    def test_restore_empty_store_is_noop(self):
        store = SQLiteStateStore(":memory:")
        adapter = PaperBrokerAdapter()
        oms = OrderManagementSystem(broker=adapter, state_store=store)
        oms.restore_state()
        assert len(oms.get_all_positions()) == 0
        assert len(oms.get_active_orders()) == 0
        store.close()

    def test_full_round_trip_submit_restart_verify(self):
        """Simulate: submit orders → crash → restart → verify state."""
        import os
        import tempfile

        db_path = os.path.join(tempfile.mkdtemp(), "test_oms.db")

        # --- Session 1: submit orders ---
        store1 = SQLiteStateStore(db_path)
        adapter1 = PaperBrokerAdapter(default_fill_price=100.0)
        oms1 = OrderManagementSystem(broker=adapter1, state_store=store1)
        oms1.start()
        oms1.submit_order(Order(symbol="AAPL", side=OrderSide.BUY, quantity=10))
        oms1.submit_order(Order(symbol="MSFT", side=OrderSide.BUY, quantity=5))
        oms1.save_snapshot(cash=adapter1.get_account_cash(), peak_portfolio_value=101_500.0)
        oms1.stop()
        store1.close()

        # --- Session 2: restart and verify ---
        store2 = SQLiteStateStore(db_path)
        adapter2 = PaperBrokerAdapter(default_fill_price=100.0)
        oms2 = OrderManagementSystem(broker=adapter2, state_store=store2)
        oms2.restore_state()

        positions = oms2.get_all_positions()
        assert "AAPL" in positions
        assert "MSFT" in positions
        assert positions["AAPL"].quantity == 10
        assert positions["MSFT"].quantity == 5

        snap = oms2.load_latest_snapshot()
        assert snap is not None
        assert snap["peak_portfolio_value"] == pytest.approx(101_500.0)

        fills = store2.load_fills()
        assert len(fills) == 2

        orders = store2.load_orders()
        assert len(orders) == 2
        assert all(o.status == OrderStatus.FILLED for o in orders)

        store2.close()

    def test_no_store_restore_is_noop(self):
        adapter = PaperBrokerAdapter()
        oms = OrderManagementSystem(broker=adapter)
        oms.restore_state()  # should not raise
        assert len(oms.get_all_positions()) == 0

    def test_no_store_snapshot_is_noop(self):
        adapter = PaperBrokerAdapter()
        oms = OrderManagementSystem(broker=adapter)
        oms.save_snapshot(cash=1000.0, peak_portfolio_value=2000.0)  # no-op
        assert oms.load_latest_snapshot() is None
