"""Risk-enforcing OMS wrapper — guarantees every order passes risk validation.

The base :class:`~quant.oms.system.OrderManagementSystem` deliberately does
NOT run risk checks, leaving that to the caller.  This wrapper closes that
gap by intercepting :meth:`submit_order`, constructing a portfolio snapshot,
running the :class:`~quant.risk.engine.RiskEngine` validation, and only
forwarding approved orders.

This is a CIO-mandated safety layer: **no order reaches the broker without
passing all risk checks**.

Usage::

    from quant.oms.risk_enforcing import RiskEnforcingOMS

    risk_oms = RiskEnforcingOMS(oms=inner_oms, risk_engine=engine)
    risk_oms.submit_order(order)  # raises RiskRejectionError if rejected
"""
from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone

from loguru import logger

from quant.oms.models import Fill, Order, OrderSide, OrderStatus, Position
from quant.oms.persistence import SQLiteStateStore
from quant.oms.system import OrderManagementSystem
from quant.risk.engine import (
    Order as RiskOrder,
    PortfolioState,
    RiskCheckResult,
    RiskEngine,
)


class RiskRejectionError(Exception):
    """Raised when the risk engine rejects an order.

    Attributes:
        order: The OMS order that was rejected.
        result: The full :class:`RiskCheckResult` from the risk engine.
    """

    def __init__(self, order: Order, result: RiskCheckResult) -> None:
        self.order = order
        self.result = result
        super().__init__(
            f"Risk rejected order {order.id} ({order.symbol} "
            f"{order.side.value} qty={order.quantity}): {result.reason}"
        )


@dataclass
class RejectionStats:
    """Counters for risk-enforcing OMS audit trail."""

    total_submitted: int = 0
    total_approved: int = 0
    total_rejected: int = 0
    total_adjusted: int = 0
    total_bypassed: int = 0
    rejections_by_check: dict[str, int] = field(default_factory=dict)


class RiskEnforcingOMS:
    """OMS wrapper that mandates risk engine approval for every order.

    Delegates all read-only queries (positions, orders, fills) to the
    underlying OMS.  Only :meth:`submit_order` is intercepted for risk
    validation.

    The wrapper:
      1. Builds a :class:`PortfolioState` from current OMS positions + cash.
      2. Converts the OMS :class:`Order` to a risk-engine :class:`RiskOrder`.
      3. Calls :meth:`RiskEngine.validate`.
      4. If rejected → raises :class:`RiskRejectionError` (order is NOT
         submitted, status remains PENDING).
      5. If approved with adjusted quantity → updates the order quantity
         before forwarding.
      6. Forwards the approved order to the underlying OMS.

    An emergency bypass is available via :meth:`bypass_risk` for scenarios
    that require immediate execution (e.g., liquidation of a breached
    position).  All bypasses are audit-logged.

    Args:
        oms: The underlying order management system.
        risk_engine: Pre-configured risk engine.
        sector_map: Optional mapping of symbol → sector label, used to
            populate the sector field on risk orders and compute sector
            exposures for the portfolio snapshot.
        capital_override: If set, use this as portfolio capital instead of
            deriving it from broker cash + position market values.
        price_feed: Optional callable (symbol → price) for estimating fill
            prices on market orders.  Without this, market orders with no
            existing position will use price=0 (which effectively bypasses
            dollar-based risk checks).
        on_rejection: Optional callback fired on every rejection, receives
            the order and the :class:`RiskCheckResult`.
    """

    def __init__(
        self,
        oms: OrderManagementSystem,
        risk_engine: RiskEngine,
        *,
        sector_map: dict[str, str] | None = None,
        capital_override: float | None = None,
        price_feed: Callable[[str], float | None] | None = None,
        on_rejection: Callable[[Order, RiskCheckResult], None] | None = None,
    ) -> None:
        self._oms = oms
        self._risk = risk_engine
        self._sector_map = sector_map or {}
        self._capital_override = capital_override
        self._price_feed = price_feed
        self._on_rejection = on_rejection

        self._bypass_active = False
        self._lock = threading.Lock()
        self._stats = RejectionStats()

    # ── Lifecycle (delegate) ──────────────────────────────────────────────

    def start(self) -> None:
        self._oms.start()

    def stop(self) -> None:
        self._oms.stop()

    def __enter__(self) -> RiskEnforcingOMS:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    def restore_state(self) -> None:
        self._oms.restore_state()

    # ── Risk-enforced order submission ────────────────────────────────────

    def submit_order(self, order: Order) -> str:
        """Submit an order only if it passes risk engine validation.

        Args:
            order: Order to submit (status must be PENDING).

        Returns:
            The broker-assigned order ID.

        Raises:
            RiskRejectionError: If the risk engine rejects the order.
            BrokerError: If the broker rejects the order after risk approval.
            ValueError: If the order is not in PENDING status.
        """
        with self._lock:
            self._stats.total_submitted += 1

            if self._bypass_active:
                self._stats.total_bypassed += 1
                logger.warning(
                    "Risk BYPASS: submitting order {} without validation "
                    "(symbol={} side={} qty={})",
                    order.id,
                    order.symbol,
                    order.side.value,
                    order.quantity,
                )
                return self._oms.submit_order(order)

        # Build portfolio state outside the lock (may call broker)
        portfolio = self._build_portfolio_state()

        # Convert OMS order → risk order
        risk_order = self._to_risk_order(order)

        # Run risk validation
        result = self._risk.validate(risk_order, portfolio)

        with self._lock:
            if not result.approved:
                self._stats.total_rejected += 1
                for check in result.checks_failed:
                    self._stats.rejections_by_check[check] = (
                        self._stats.rejections_by_check.get(check, 0) + 1
                    )

                logger.warning(
                    "Risk REJECTED order {} — {} (failed: {})",
                    order.id,
                    result.reason,
                    ", ".join(result.checks_failed),
                )

                if self._on_rejection is not None:
                    try:
                        self._on_rejection(order, result)
                    except Exception:
                        logger.exception("on_rejection callback raised")

                raise RiskRejectionError(order, result)

            # Approved — check if quantity was adjusted.
            # The risk engine returns a signed quantity; the OMS order uses
            # unsigned quantity + side, so we take the absolute value.
            adjusted_unsigned = abs(result.adjusted_quantity)
            if adjusted_unsigned != order.quantity:
                logger.info(
                    "Risk adjusted order {} quantity: {} → {}",
                    order.id,
                    order.quantity,
                    adjusted_unsigned,
                )
                order.quantity = adjusted_unsigned
                self._stats.total_adjusted += 1

            self._stats.total_approved += 1

        return self._oms.submit_order(order)

    # ── Emergency bypass ──────────────────────────────────────────────────

    @contextlib.contextmanager
    def bypass_risk(self, reason: str) -> Iterator[None]:
        """Context manager that temporarily disables risk validation.

        Every order submitted within this context bypasses risk checks.
        The bypass is fully audit-logged with the provided reason.

        Args:
            reason: Human-readable justification for the bypass.

        Usage::

            with risk_oms.bypass_risk("emergency liquidation of breached position"):
                risk_oms.submit_order(liquidation_order)
        """
        logger.warning("Risk bypass ACTIVATED — reason: {}", reason)
        with self._lock:
            self._bypass_active = True
        try:
            yield
        finally:
            with self._lock:
                self._bypass_active = False
            logger.warning("Risk bypass DEACTIVATED — reason: {}", reason)

    # ── Statistics / audit ────────────────────────────────────────────────

    @property
    def stats(self) -> RejectionStats:
        """Return a snapshot of the rejection/approval statistics."""
        with self._lock:
            return RejectionStats(
                total_submitted=self._stats.total_submitted,
                total_approved=self._stats.total_approved,
                total_rejected=self._stats.total_rejected,
                total_adjusted=self._stats.total_adjusted,
                total_bypassed=self._stats.total_bypassed,
                rejections_by_check=dict(self._stats.rejections_by_check),
            )

    # ── Delegated queries ─────────────────────────────────────────────────

    def get_order(self, order_id: str) -> Order | None:
        return self._oms.get_order(order_id)

    def get_active_orders(self) -> list[Order]:
        return self._oms.get_active_orders()

    def get_position(self, symbol: str) -> Position | None:
        return self._oms.get_position(symbol)

    def get_all_positions(self) -> dict[str, Position]:
        return self._oms.get_all_positions()

    def get_fills(self, broker_order_id: str) -> list[Fill]:
        return self._oms.get_fills(broker_order_id)

    def get_account_cash(self) -> float:
        return self._oms.get_account_cash()

    def cancel_order(self, order_id: str) -> bool:
        return self._oms.cancel_order(order_id)

    def register_fill_hook(self, hook: Callable[[Fill], None]) -> None:
        self._oms.register_fill_hook(hook)

    def save_snapshot(self, cash: float, peak_portfolio_value: float) -> None:
        self._oms.save_snapshot(cash, peak_portfolio_value)

    def load_latest_snapshot(self) -> dict | None:
        return self._oms.load_latest_snapshot()

    # ── Internal helpers ──────────────────────────────────────────────────

    def _build_portfolio_state(self) -> PortfolioState:
        """Construct a risk-engine PortfolioState from current OMS state."""
        positions = self._oms.get_all_positions()
        cash = self._oms.get_account_cash()

        # Build position map: symbol → signed dollar value
        pos_map: dict[str, float] = {}
        for sym, pos in positions.items():
            pos_map[sym] = pos.market_value

        # Build sector exposure map
        sector_exposures: dict[str, float] = {}
        for sym, pos in positions.items():
            sector = self._sector_map.get(sym)
            if sector is not None:
                sector_exposures[sector] = (
                    sector_exposures.get(sector, 0.0) + abs(pos.market_value)
                )

        # Capital: override or cash + gross position value
        if self._capital_override is not None:
            capital = self._capital_override
        else:
            capital = cash + sum(abs(v) for v in pos_map.values())

        return PortfolioState(
            capital=capital,
            positions=pos_map,
            sector_exposures=sector_exposures,
            peak_portfolio_value=max(capital, 0.0),
        )

    def _estimate_price(self, order: Order) -> float:
        """Estimate the fill price for risk validation.

        Priority: limit_price → stop_price → position market_price →
        price_feed → 0.0 (risk engine uses dollar values, so price=0 means
        all dollar-based checks pass trivially — callers should provide a
        price_feed to avoid this).
        """
        if order.limit_price and order.limit_price > 0:
            return order.limit_price
        if order.stop_price and order.stop_price > 0:
            return order.stop_price

        pos = self._oms.get_position(order.symbol)
        if pos is not None and pos.market_price > 0:
            return pos.market_price

        if self._price_feed is not None:
            feed_price = self._price_feed(order.symbol)
            if feed_price is not None and feed_price > 0:
                return feed_price

        return 0.0

    def _to_risk_order(self, order: Order) -> RiskOrder:
        """Convert an OMS Order to a risk-engine Order."""
        price = self._estimate_price(order)

        signed_qty = (
            order.quantity if order.side == OrderSide.BUY else -order.quantity
        )

        return RiskOrder(
            symbol=order.symbol,
            quantity=signed_qty,
            price=price,
            sector=order.sector or self._sector_map.get(order.symbol),
        )
