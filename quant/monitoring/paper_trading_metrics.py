"""Prometheus metrics for paper trading operations.

Covers:
  - Portfolio P&L (cumulative, unrealized, daily)
  - Sector exposure
  - Position-level tracking
  - Adapter connectivity
  - Daily trade log counter

These complement the existing quant.monitoring.metrics module which tracks
execution/OMS and ingest pipeline metrics.

Usage::

    from quant.monitoring.paper_trading_metrics import (
        record_portfolio_state,
        record_position,
        record_trade,
        set_adapter_connected,
    )
"""
from __future__ import annotations

import os

from loguru import logger

try:
    from prometheus_client import Counter, Gauge, Histogram
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    logger.warning(
        "prometheus_client not installed — paper trading metrics will be no-ops."
    )


def _make_counter(name: str, doc: str, labelnames: list[str] | None = None):
    if not _PROMETHEUS_AVAILABLE:
        return _NoOpMetric()
    return Counter(name, doc, labelnames or [])


def _make_gauge(name: str, doc: str, labelnames: list[str] | None = None):
    if not _PROMETHEUS_AVAILABLE:
        return _NoOpMetric()
    return Gauge(name, doc, labelnames or [])


def _make_histogram(
    name: str,
    doc: str,
    labelnames: list[str] | None = None,
    buckets: tuple | None = None,
):
    if not _PROMETHEUS_AVAILABLE:
        return _NoOpMetric()
    kwargs = {}
    if buckets is not None:
        kwargs["buckets"] = buckets
    return Histogram(name, doc, labelnames or [], **kwargs)


class _NoOpMetric:
    def labels(self, **_):
        return self
    def inc(self, *_, **__): pass
    def set(self, *_, **__): pass
    def observe(self, *_, **__): pass


# ── Portfolio-level metrics ───────────────────────────────────────────────────

PAPER_NOTIONAL = _make_gauge(
    "quant_paper_notional_dollars",
    "Starting notional capital for paper trading (constant, $1M).",
)

PAPER_PORTFOLIO_VALUE = _make_gauge(
    "quant_paper_portfolio_value_dollars",
    "Current mark-to-market portfolio value in dollars.",
)

PAPER_REALIZED_PNL = _make_gauge(
    "quant_paper_realized_pnl_dollars",
    "Cumulative realized P&L in dollars since paper trading inception.",
)

PAPER_UNREALIZED_PNL = _make_gauge(
    "quant_paper_unrealized_pnl_dollars",
    "Current unrealized P&L across all open positions.",
)

PAPER_DAILY_PNL = _make_gauge(
    "quant_paper_daily_pnl_dollars",
    "P&L for the current trading day (resets at market open).",
)

PAPER_DAILY_PNL_PCT = _make_gauge(
    "quant_paper_daily_pnl_pct",
    "Daily P&L as a fraction of starting notional (0-based, negative for losses).",
)

PAPER_DRAWDOWN = _make_gauge(
    "quant_paper_drawdown_current",
    "Current drawdown from peak portfolio value (0–1 fraction). "
    "CRO thresholds: 0.10 yellow, 0.15 orange, 0.20 red.",
)

PAPER_PEAK_VALUE = _make_gauge(
    "quant_paper_peak_value_dollars",
    "Highest recorded portfolio value (used for drawdown calculation).",
)

# ── Sector exposure ───────────────────────────────────────────────────────────

PAPER_SECTOR_EXPOSURE = _make_gauge(
    "quant_paper_sector_exposure_dollars",
    "Gross dollar exposure per GICS sector.",
    labelnames=["sector"],
)

PAPER_SECTOR_EXPOSURE_PCT = _make_gauge(
    "quant_paper_sector_exposure_pct",
    "Sector exposure as fraction of total portfolio value (0–1).",
    labelnames=["sector"],
)

# ── Position-level metrics ────────────────────────────────────────────────────

PAPER_POSITION_VALUE = _make_gauge(
    "quant_paper_position_value_dollars",
    "Current market value of a single position.",
    labelnames=["symbol", "sector"],
)

PAPER_POSITION_UNREALIZED_PNL = _make_gauge(
    "quant_paper_position_unrealized_pnl_dollars",
    "Unrealized P&L for a single position.",
    labelnames=["symbol", "sector"],
)

PAPER_POSITIONS_OPEN = _make_gauge(
    "quant_paper_positions_open",
    "Number of currently open paper trading positions.",
)

# ── Trade log / execution ─────────────────────────────────────────────────────

PAPER_TRADES_TODAY = _make_counter(
    "quant_paper_trades_today_total",
    "Number of paper trades submitted today (resets each day via external reset).",
    labelnames=["side"],  # buy | sell
)

PAPER_TRADE_VALUE = _make_histogram(
    "quant_paper_trade_value_dollars",
    "Dollar value of each paper trade.",
    labelnames=["side"],
    buckets=(1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000),
)

PAPER_FILLS_TOTAL = _make_counter(
    "quant_paper_fills_total",
    "Total fill events received from the paper trading adapter.",
    labelnames=["symbol", "side"],
)

# ── Adapter connectivity ──────────────────────────────────────────────────────

PAPER_ADAPTER_CONNECTED = _make_gauge(
    "quant_paper_adapter_connected",
    "1 if the Alpaca paper trading adapter is connected, 0 otherwise.",
)

PAPER_ADAPTER_ERRORS = _make_counter(
    "quant_paper_adapter_errors_total",
    "Total adapter errors (connection drops, submission failures).",
    labelnames=["error_type"],  # connect | submit | cancel | stream
)

PAPER_ADAPTER_RECONNECTS = _make_counter(
    "quant_paper_adapter_reconnects_total",
    "Total Alpaca paper adapter reconnection attempts.",
)


# ── Convenience helpers ───────────────────────────────────────────────────────

def record_portfolio_state(
    *,
    portfolio_value: float,
    realized_pnl: float,
    unrealized_pnl: float,
    daily_pnl: float,
    peak_value: float,
    notional: float | None = None,
) -> None:
    """Update all portfolio-level gauges in one call.

    Args:
        portfolio_value: Current mark-to-market value in dollars.
        realized_pnl: Cumulative realized P&L since inception.
        unrealized_pnl: Current unrealized P&L.
        daily_pnl: Today's P&L in dollars.
        peak_value: Highest recorded portfolio value (for drawdown).
        notional: Starting notional (defaults to QUANT_PAPER_NOTIONAL env var or $1M).
    """
    if notional is None:
        notional = float(os.environ.get("QUANT_PAPER_NOTIONAL", 1_000_000))

    PAPER_NOTIONAL.set(notional)
    PAPER_PORTFOLIO_VALUE.set(portfolio_value)
    PAPER_REALIZED_PNL.set(realized_pnl)
    PAPER_UNREALIZED_PNL.set(unrealized_pnl)
    PAPER_DAILY_PNL.set(daily_pnl)
    PAPER_PEAK_VALUE.set(peak_value)

    if notional > 0:
        PAPER_DAILY_PNL_PCT.set(daily_pnl / notional)

    if peak_value > 0:
        drawdown = max(0.0, (peak_value - portfolio_value) / peak_value)
        PAPER_DRAWDOWN.set(drawdown)


def record_position(
    *,
    symbol: str,
    sector: str,
    market_value: float,
    unrealized_pnl: float,
) -> None:
    """Update position-level gauges for a single holding."""
    PAPER_POSITION_VALUE.labels(symbol=symbol, sector=sector).set(market_value)
    PAPER_POSITION_UNREALIZED_PNL.labels(symbol=symbol, sector=sector).set(unrealized_pnl)


def record_sector_exposure(sector_exposures: dict[str, float], portfolio_value: float) -> None:
    """Update sector exposure gauges.

    Args:
        sector_exposures: Dict mapping sector name → gross dollar exposure.
        portfolio_value: Total portfolio value (for pct calculation).
    """
    for sector, exposure in sector_exposures.items():
        PAPER_SECTOR_EXPOSURE.labels(sector=sector).set(exposure)
        if portfolio_value > 0:
            PAPER_SECTOR_EXPOSURE_PCT.labels(sector=sector).set(exposure / portfolio_value)


def record_trade(*, symbol: str, side: str, value: float) -> None:
    """Record a paper trade for daily trade log and histogram."""
    side_key = side.lower()
    PAPER_TRADES_TODAY.labels(side=side_key).inc()
    PAPER_TRADE_VALUE.labels(side=side_key).observe(value)


def set_adapter_connected(connected: bool) -> None:
    """Update adapter connectivity gauge."""
    PAPER_ADAPTER_CONNECTED.set(1 if connected else 0)
