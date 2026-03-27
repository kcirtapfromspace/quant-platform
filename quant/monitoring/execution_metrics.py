"""Prometheus-instrumented wrappers for the OMS and risk engine.

Provides:
  - :class:`InstrumentedOMS` — wraps :class:`~quant.oms.system.OrderManagementSystem`
    to track order counts, fill rates, and latency.
  - :func:`update_risk_metrics` — call on each portfolio valuation tick to
    keep the drawdown and circuit-breaker gauges current.

Usage::

    from quant.execution.paper import PaperBrokerAdapter
    from quant.monitoring.execution_metrics import InstrumentedOMS, update_risk_metrics
    from quant.monitoring import start_metrics_server

    start_metrics_server(port=8000)

    broker = PaperBrokerAdapter(starting_cash=100_000)
    oms = InstrumentedOMS(broker=broker)
    oms.start()

    order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=10)
    oms.submit_order(order)

    # In your portfolio update loop:
    update_risk_metrics(circuit_breaker, current_portfolio_value)
"""
from __future__ import annotations

import time
from typing import Callable, Optional

from loguru import logger

from quant.oms.broker import BrokerAdapter, BrokerError
from quant.oms.models import Fill, Order, OrderStatus
from quant.oms.system import OrderManagementSystem
from quant.risk.circuit_breaker import DrawdownCircuitBreaker
from quant.monitoring.metrics import (
    ORDERS_SUBMITTED,
    ORDERS_FILLED,
    ORDERS_REJECTED,
    FILL_RATE,
    ORDER_LATENCY,
    POSITIONS_OPEN,
    CIRCUIT_BREAKER_TRIPPED,
    DRAWDOWN_CURRENT,
)


class InstrumentedOMS(OrderManagementSystem):
    """OrderManagementSystem subclass that records Prometheus metrics.

    Records:
      - ``quant_orders_submitted_total`` (labels: symbol, side)
      - ``quant_orders_filled_total`` (labels: symbol, side)
      - ``quant_orders_rejected_total`` (labels: symbol)
      - ``quant_fill_rate`` — rolling ratio filled / submitted
      - ``quant_order_latency_seconds`` — submit → first-fill latency
      - ``quant_positions_open`` — number of open positions

    All other OMS behaviour is unchanged.
    """

    def __init__(self, broker: BrokerAdapter) -> None:
        super().__init__(broker=broker)
        # Track submission timestamps for latency calculation: order_id → epoch
        self._submit_times: dict[str, float] = {}
        # Running totals for fill-rate gauge
        self._total_submitted: int = 0
        self._total_filled: int = 0

    # ── Order submission ──────────────────────────────────────────────────

    def submit_order(self, order: Order) -> str:
        side_label = order.side.value.lower()
        try:
            broker_id = super().submit_order(order)
        except BrokerError:
            ORDERS_REJECTED.labels(symbol=order.symbol).inc()
            self._refresh_fill_rate()
            raise

        ORDERS_SUBMITTED.labels(symbol=order.symbol, side=side_label).inc()
        self._total_submitted += 1
        self._submit_times[order.id] = time.monotonic()
        self._refresh_fill_rate()
        return broker_id

    # ── Fill processing ───────────────────────────────────────────────────

    def on_fill(self, fill: Fill) -> None:
        super().on_fill(fill)

        # Resolve order to get side label and track fill completion
        with self._lock:
            oms_id = self._broker_id_map.get(fill.broker_order_id)
            order = self._orders.get(oms_id) if oms_id else None

        if order is not None:
            side_label = order.side.value.lower()
            ORDERS_FILLED.labels(symbol=fill.symbol, side=side_label).inc()

            if order.status == OrderStatus.FILLED:
                self._total_filled += 1
                self._refresh_fill_rate()
                # Record submit-to-fill latency
                submit_time = self._submit_times.pop(order.id, None)
                if submit_time is not None:
                    latency = time.monotonic() - submit_time
                    ORDER_LATENCY.observe(latency)
                    logger.debug(
                        "InstrumentedOMS: fill latency {:.3f}s for order {}",
                        latency,
                        order.id,
                    )

        # Refresh position count after every fill
        POSITIONS_OPEN.set(len(self.get_all_positions()))

    # ── Internal helpers ──────────────────────────────────────────────────

    def _refresh_fill_rate(self) -> None:
        if self._total_submitted > 0:
            FILL_RATE.set(self._total_filled / self._total_submitted)


def update_risk_metrics(
    circuit_breaker: DrawdownCircuitBreaker,
    current_portfolio_value: float,
) -> None:
    """Update the circuit-breaker and drawdown gauges.

    Call this on every portfolio valuation tick (bar close, tick event, etc.)
    so the Prometheus gauges stay current.

    Args:
        circuit_breaker: The active :class:`~quant.risk.circuit_breaker.DrawdownCircuitBreaker`.
        current_portfolio_value: Current total portfolio value in base currency.
    """
    circuit_breaker.update(current_portfolio_value)

    CIRCUIT_BREAKER_TRIPPED.set(1 if circuit_breaker.is_tripped() else 0)

    peak = circuit_breaker._peak_value
    if peak > 0 and current_portfolio_value > 0:
        drawdown = max(0.0, (peak - current_portfolio_value) / peak)
    else:
        drawdown = 0.0
    DRAWDOWN_CURRENT.set(drawdown)
