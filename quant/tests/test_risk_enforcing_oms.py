"""Tests for the risk-enforcing OMS middleware (QUA-117)."""
from __future__ import annotations

import pytest

from quant.execution.paper import PaperBrokerAdapter
from quant.oms.models import Order, OrderSide, OrderType, Position
from quant.oms.risk_enforcing import RiskEnforcingOMS, RiskRejectionError
from quant.oms.system import OrderManagementSystem
from quant.risk.circuit_breaker import DrawdownCircuitBreaker
from quant.risk.engine import RiskConfig, RiskEngine
from quant.risk.limits import ExposureLimits


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_stack(
    *,
    initial_cash: float = 100_000.0,
    fill_price: float = 100.0,
    limits: ExposureLimits | None = None,
    circuit_breaker: DrawdownCircuitBreaker | None = None,
    sector_map: dict[str, str] | None = None,
    capital_override: float | None = None,
    on_rejection: object | None = None,
) -> tuple[RiskEnforcingOMS, OrderManagementSystem, PaperBrokerAdapter]:
    """Create a full RiskEnforcingOMS stack for testing.

    Automatically provides a price_feed that returns fill_price for all
    symbols, so market orders are properly priced for risk validation.
    """
    adapter = PaperBrokerAdapter(
        initial_cash=initial_cash,
        default_fill_price=fill_price,
    )
    oms = OrderManagementSystem(broker=adapter)
    oms.start()

    risk_config = RiskConfig(
        limits=limits or ExposureLimits(),
        circuit_breaker=circuit_breaker or DrawdownCircuitBreaker(),
    )
    risk_engine = RiskEngine(risk_config)

    risk_oms = RiskEnforcingOMS(
        oms=oms,
        risk_engine=risk_engine,
        sector_map=sector_map,
        capital_override=capital_override,
        price_feed=lambda _sym: fill_price,
        on_rejection=on_rejection,
    )
    return risk_oms, oms, adapter


# ── Basic approval flow ─────────────────────────────────────────────────────


class TestBasicApproval:
    def test_small_order_approved(self):
        risk_oms, oms, _ = _make_stack(
            initial_cash=100_000.0, fill_price=100.0
        )
        try:
            order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=5)
            broker_id = risk_oms.submit_order(order)
            assert broker_id is not None
            assert risk_oms.get_position("AAPL") is not None
            assert risk_oms.get_position("AAPL").quantity == 5
        finally:
            oms.stop()

    def test_stats_updated_on_approval(self):
        risk_oms, oms, _ = _make_stack()
        try:
            order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=1)
            risk_oms.submit_order(order)
            stats = risk_oms.stats
            assert stats.total_submitted == 1
            assert stats.total_approved == 1
            assert stats.total_rejected == 0
        finally:
            oms.stop()

    def test_multiple_orders_approved(self):
        risk_oms, oms, _ = _make_stack(
            initial_cash=1_000_000.0, fill_price=100.0
        )
        try:
            for sym in ["AAPL", "MSFT", "GOOG"]:
                order = Order(symbol=sym, side=OrderSide.BUY, quantity=10)
                risk_oms.submit_order(order)
            positions = risk_oms.get_all_positions()
            assert len(positions) == 3
            assert risk_oms.stats.total_approved == 3
        finally:
            oms.stop()


# ── Rejection scenarios ─────────────────────────────────────────────────────


class TestRejection:
    def test_order_exceeding_max_order_size_rejected(self):
        # max_order_fraction=0.10 → max order = $10k on $100k capital
        # Order: 150 shares @ $100 = $15,000 → exceeds 10%
        risk_oms, oms, _ = _make_stack(
            initial_cash=100_000.0,
            fill_price=100.0,
            limits=ExposureLimits(max_order_fraction=0.10),
            capital_override=100_000.0,
        )
        try:
            order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=150)
            with pytest.raises(RiskRejectionError) as exc_info:
                risk_oms.submit_order(order)
            assert "order size" in exc_info.value.result.reason.lower() or \
                   "max_order_size" in exc_info.value.result.checks_failed
            # Order should NOT have been submitted
            assert risk_oms.get_position("AAPL") is None
        finally:
            oms.stop()

    def test_position_limit_rejection(self):
        # max_position_fraction=0.05 → max position = $5k on $100k
        # Order: 60 shares @ $100 = $6,000 → exceeds 5%
        risk_oms, oms, _ = _make_stack(
            initial_cash=100_000.0,
            fill_price=100.0,
            limits=ExposureLimits(max_position_fraction=0.05),
            capital_override=100_000.0,
        )
        try:
            order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=60)
            with pytest.raises(RiskRejectionError) as exc_info:
                risk_oms.submit_order(order)
            assert "position_limit" in exc_info.value.result.checks_failed
        finally:
            oms.stop()

    def test_rejection_stats_tracked(self):
        risk_oms, oms, _ = _make_stack(
            initial_cash=100_000.0,
            fill_price=100.0,
            limits=ExposureLimits(max_order_fraction=0.01),
            capital_override=100_000.0,
        )
        try:
            order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=50)
            with pytest.raises(RiskRejectionError):
                risk_oms.submit_order(order)
            stats = risk_oms.stats
            assert stats.total_submitted == 1
            assert stats.total_rejected == 1
            assert stats.total_approved == 0
            assert "max_order_size" in stats.rejections_by_check
        finally:
            oms.stop()

    def test_rejection_callback_fired(self):
        received = []

        def on_reject(order, result):
            received.append((order.symbol, result.reason))

        risk_oms, oms, _ = _make_stack(
            initial_cash=100_000.0,
            fill_price=100.0,
            limits=ExposureLimits(max_order_fraction=0.01),
            capital_override=100_000.0,
            on_rejection=on_reject,
        )
        try:
            with pytest.raises(RiskRejectionError):
                risk_oms.submit_order(
                    Order(symbol="TSLA", side=OrderSide.BUY, quantity=50)
                )
            assert len(received) == 1
            assert received[0][0] == "TSLA"
        finally:
            oms.stop()

    def test_rejected_order_stays_pending(self):
        risk_oms, oms, _ = _make_stack(
            initial_cash=100_000.0,
            fill_price=100.0,
            limits=ExposureLimits(max_order_fraction=0.01),
            capital_override=100_000.0,
        )
        try:
            order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=50)
            with pytest.raises(RiskRejectionError):
                risk_oms.submit_order(order)
            # Order was never submitted so it's not in the OMS order book
            # at all (submit_order was never called on the inner OMS)
            assert oms.get_order(order.id) is None
        finally:
            oms.stop()


# ── Sector exposure checks ──────────────────────────────────────────────────


class TestSectorExposure:
    def test_sector_map_populated_in_risk_order(self):
        risk_oms, oms, _ = _make_stack(
            initial_cash=1_000_000.0,
            fill_price=100.0,
            sector_map={"AAPL": "Technology", "XOM": "Energy"},
            limits=ExposureLimits(max_sector_fraction=0.05),
            capital_override=1_000_000.0,
        )
        try:
            # $5,000 order on $1M capital with 5% sector limit = fine
            order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=50)
            risk_oms.submit_order(order)
            assert risk_oms.get_position("AAPL").quantity == 50
        finally:
            oms.stop()

    def test_order_sector_field_overrides_map(self):
        # If the order already has a sector, it takes precedence
        risk_oms, oms, _ = _make_stack(
            initial_cash=1_000_000.0,
            fill_price=100.0,
            sector_map={"AAPL": "Technology"},
            capital_override=1_000_000.0,
        )
        try:
            order = Order(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=5,
                sector="Custom Sector",
            )
            # Should work — sector from order.sector is used
            risk_oms.submit_order(order)
            assert risk_oms.get_position("AAPL") is not None
        finally:
            oms.stop()


# ── Emergency bypass ────────────────────────────────────────────────────────


class TestBypass:
    def test_bypass_skips_risk_checks(self):
        # Set extremely tight limits that would normally reject
        risk_oms, oms, _ = _make_stack(
            initial_cash=100_000.0,
            fill_price=100.0,
            limits=ExposureLimits(max_order_fraction=0.001),
            capital_override=100_000.0,
        )
        try:
            order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=50)
            with risk_oms.bypass_risk("emergency liquidation test"):
                broker_id = risk_oms.submit_order(order)
            assert broker_id is not None
            assert risk_oms.get_position("AAPL").quantity == 50
        finally:
            oms.stop()

    def test_bypass_stats_tracked(self):
        risk_oms, oms, _ = _make_stack(
            initial_cash=100_000.0,
            fill_price=100.0,
            limits=ExposureLimits(max_order_fraction=0.001),
            capital_override=100_000.0,
        )
        try:
            with risk_oms.bypass_risk("test"):
                risk_oms.submit_order(
                    Order(symbol="AAPL", side=OrderSide.BUY, quantity=50)
                )
            stats = risk_oms.stats
            assert stats.total_bypassed == 1
            assert stats.total_submitted == 1
            # Bypassed orders are not counted as approved
            assert stats.total_approved == 0
        finally:
            oms.stop()

    def test_bypass_deactivated_after_context(self):
        risk_oms, oms, _ = _make_stack(
            initial_cash=100_000.0,
            fill_price=100.0,
            limits=ExposureLimits(max_order_fraction=0.001),
            capital_override=100_000.0,
        )
        try:
            with risk_oms.bypass_risk("test"):
                risk_oms.submit_order(
                    Order(symbol="AAPL", side=OrderSide.BUY, quantity=50)
                )

            # After context, bypass should be deactivated — new order rejected
            with pytest.raises(RiskRejectionError):
                risk_oms.submit_order(
                    Order(symbol="MSFT", side=OrderSide.BUY, quantity=50)
                )
        finally:
            oms.stop()

    def test_bypass_deactivated_on_exception(self):
        risk_oms, oms, _ = _make_stack(
            initial_cash=100_000.0,
            fill_price=100.0,
            limits=ExposureLimits(max_order_fraction=0.001),
            capital_override=100_000.0,
        )
        try:
            with pytest.raises(RuntimeError):
                with risk_oms.bypass_risk("test"):
                    raise RuntimeError("simulated failure")

            # Bypass should still be deactivated
            with pytest.raises(RiskRejectionError):
                risk_oms.submit_order(
                    Order(symbol="AAPL", side=OrderSide.BUY, quantity=50)
                )
        finally:
            oms.stop()


# ── Delegated query methods ──────────────────────────────────────────────────


class TestDelegation:
    def test_get_position_delegates(self):
        risk_oms, oms, _ = _make_stack(fill_price=150.0)
        try:
            risk_oms.submit_order(
                Order(symbol="AAPL", side=OrderSide.BUY, quantity=10)
            )
            pos = risk_oms.get_position("AAPL")
            assert pos is not None
            assert pos.quantity == 10
            assert pos.avg_cost == pytest.approx(150.0)
        finally:
            oms.stop()

    def test_get_all_positions_delegates(self):
        risk_oms, oms, _ = _make_stack(
            initial_cash=1_000_000.0, fill_price=100.0
        )
        try:
            risk_oms.submit_order(
                Order(symbol="AAPL", side=OrderSide.BUY, quantity=5)
            )
            risk_oms.submit_order(
                Order(symbol="MSFT", side=OrderSide.BUY, quantity=3)
            )
            positions = risk_oms.get_all_positions()
            assert len(positions) == 2
        finally:
            oms.stop()

    def test_get_account_cash_delegates(self):
        risk_oms, oms, _ = _make_stack(
            initial_cash=50_000.0, fill_price=100.0
        )
        try:
            risk_oms.submit_order(
                Order(symbol="AAPL", side=OrderSide.BUY, quantity=10)
            )
            cash = risk_oms.get_account_cash()
            assert cash == pytest.approx(49_000.0)
        finally:
            oms.stop()

    def test_cancel_order_delegates(self):
        risk_oms, oms, _ = _make_stack()
        try:
            result = risk_oms.cancel_order("nonexistent")
            assert result is False
        finally:
            oms.stop()

    def test_fill_hook_delegates(self):
        received = []
        risk_oms, oms, _ = _make_stack(fill_price=100.0)
        risk_oms.register_fill_hook(lambda f: received.append(f))
        try:
            risk_oms.submit_order(
                Order(symbol="AAPL", side=OrderSide.BUY, quantity=5)
            )
            assert len(received) == 1
            assert received[0].symbol == "AAPL"
        finally:
            oms.stop()

    def test_context_manager(self):
        adapter = PaperBrokerAdapter(
            initial_cash=100_000.0, default_fill_price=100.0
        )
        inner_oms = OrderManagementSystem(broker=adapter)
        risk_engine = RiskEngine()

        with RiskEnforcingOMS(oms=inner_oms, risk_engine=risk_engine) as risk_oms:
            risk_oms.submit_order(
                Order(symbol="AAPL", side=OrderSide.BUY, quantity=1)
            )
            assert risk_oms.get_position("AAPL") is not None


# ── Portfolio state construction ─────────────────────────────────────────────


class TestPortfolioState:
    def test_capital_override_used(self):
        risk_oms, oms, _ = _make_stack(
            initial_cash=100_000.0,
            fill_price=100.0,
            capital_override=500_000.0,
            limits=ExposureLimits(max_order_fraction=0.10),
        )
        try:
            # With $500k capital and 10% order limit, $9k order should pass
            order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=90)
            risk_oms.submit_order(order)
            assert risk_oms.get_position("AAPL").quantity == 90
        finally:
            oms.stop()

    def test_capital_derived_from_cash_plus_positions(self):
        # Without capital_override, capital = cash + sum(abs(position values))
        risk_oms, oms, _ = _make_stack(
            initial_cash=100_000.0,
            fill_price=100.0,
        )
        try:
            # First order: small, passes easily
            risk_oms.submit_order(
                Order(symbol="AAPL", side=OrderSide.BUY, quantity=5)
            )
            # Capital should now be cash (99,500) + abs(500) = ~100,000
            stats = risk_oms.stats
            assert stats.total_approved == 1
        finally:
            oms.stop()


# ── Sell orders / closing positions ──────────────────────────────────────────


class TestSellOrders:
    def test_sell_order_passes_risk(self):
        risk_oms, oms, _ = _make_stack(
            initial_cash=100_000.0, fill_price=100.0
        )
        try:
            risk_oms.submit_order(
                Order(symbol="AAPL", side=OrderSide.BUY, quantity=10)
            )
            risk_oms.submit_order(
                Order(symbol="AAPL", side=OrderSide.SELL, quantity=10)
            )
            assert risk_oms.get_position("AAPL") is None
            assert risk_oms.stats.total_approved == 2
        finally:
            oms.stop()

    def test_limit_order_uses_limit_price(self):
        risk_oms, oms, _ = _make_stack(
            initial_cash=100_000.0,
            fill_price=50.0,
            capital_override=100_000.0,
        )
        try:
            order = Order(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=5,
                order_type=OrderType.LIMIT,
                limit_price=50.0,
            )
            risk_oms.submit_order(order)
            assert risk_oms.get_position("AAPL").quantity == 5
        finally:
            oms.stop()


# ── RiskRejectionError ───────────────────────────────────────────────────────


class TestRiskRejectionError:
    def test_error_message_includes_details(self):
        from quant.risk.engine import RiskCheckResult

        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
        result = RiskCheckResult(
            approved=False,
            adjusted_quantity=0.0,
            reason="order too large",
            checks_failed=["max_order_size"],
        )
        err = RiskRejectionError(order, result)
        assert "AAPL" in str(err)
        assert "buy" in str(err)
        assert "order too large" in str(err)
        assert err.order is order
        assert err.result is result

    def test_error_is_exception(self):
        assert issubclass(RiskRejectionError, Exception)
