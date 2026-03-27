"""Transaction Cost Analysis (TCA) — execution quality measurement.

Measures the gap between intended and realised execution prices to quantify
slippage, market impact, and total execution cost.

Key concepts:
  * **Arrival price**: the market price at the moment the trading decision was
    made (before the order hit the broker).  This is the most meaningful
    benchmark for measuring implementation shortfall.
  * **Implementation shortfall**: ``(fill_price - arrival_price) / arrival_price``
    for buys (negative = saved vs. arrival).  Sign convention is *positive =
    cost*, so a negative value means we got a better price than expected.
  * **VWAP slippage**: same formula but using period VWAP as benchmark.

Usage::

    from quant.execution.tca import TCACollector, TCAReport

    collector = TCACollector()
    # Register arrival price BEFORE submitting the order
    collector.register_decision("order-id-1", arrival_price=150.25)
    # ... order fills at 150.30 ...
    collector.record_fill("order-id-1", fill_price=150.30, quantity=100,
                          side="buy", symbol="AAPL")
    report = collector.report()
    print(report.summary())
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Benchmark(str, Enum):
    """Benchmark type for slippage computation."""

    ARRIVAL = "arrival"  # decision / arrival price
    VWAP = "vwap"  # volume-weighted average price during window
    CLOSE = "close"  # bar close price
    OPEN = "open"  # next-bar open price


@dataclass
class ExecutionRecord:
    """Single-order execution quality record.

    All prices are absolute (not returns).  Sign convention for cost
    fields: **positive = cost / drag**, negative = improvement.
    """

    order_id: str
    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    fill_price: float

    arrival_price: float | None = None
    vwap_price: float | None = None
    close_price: float | None = None
    open_price: float | None = None

    filled_at: datetime = field(default_factory=_utcnow)
    strategy_id: str | None = None

    # ── Computed slippage ──────────────────────────────────────────────

    @property
    def implementation_shortfall(self) -> float | None:
        """Fill vs. arrival price as a signed fraction (positive = cost)."""
        return self._slippage(self.arrival_price)

    @property
    def vwap_slippage(self) -> float | None:
        """Fill vs. VWAP as a signed fraction."""
        return self._slippage(self.vwap_price)

    @property
    def close_slippage(self) -> float | None:
        """Fill vs. close price as a signed fraction."""
        return self._slippage(self.close_price)

    @property
    def open_slippage(self) -> float | None:
        """Fill vs. next-bar open as a signed fraction."""
        return self._slippage(self.open_price)

    @property
    def dollar_cost(self) -> float | None:
        """Absolute dollar cost of implementation shortfall."""
        is_val = self.implementation_shortfall
        if is_val is None:
            return None
        return is_val * self.fill_price * self.quantity

    def _slippage(self, benchmark: float | None) -> float | None:
        """Compute signed slippage vs. a benchmark price.

        For buys:  ``(fill - bench) / bench``  → positive = we paid more.
        For sells: ``(bench - fill) / bench``  → positive = we received less.
        """
        if benchmark is None or benchmark == 0.0:
            return None
        if self.side == "buy":
            return (self.fill_price - benchmark) / benchmark
        else:
            return (benchmark - self.fill_price) / benchmark


@dataclass
class _Decision:
    """Pending decision — arrival price captured before order submission."""

    arrival_price: float | None = None
    vwap_price: float | None = None
    close_price: float | None = None
    open_price: float | None = None
    strategy_id: str | None = None


class TCACollector:
    """Collects execution records and computes TCA metrics.

    Typical flow:
      1. ``register_decision(order_id, arrival_price=...)`` — before submission
      2. ``record_fill(order_id, ...)`` — after the fill arrives
      3. ``report()`` — aggregate analysis

    The collector can also be attached as an OMS fill hook via
    :meth:`as_fill_hook` for automatic capture.
    """

    def __init__(self) -> None:
        self._decisions: dict[str, _Decision] = {}
        self._records: list[ExecutionRecord] = []

    # ── Pre-trade ──────────────────────────────────────────────────────

    def register_decision(
        self,
        order_id: str,
        *,
        arrival_price: float | None = None,
        vwap_price: float | None = None,
        close_price: float | None = None,
        open_price: float | None = None,
        strategy_id: str | None = None,
    ) -> None:
        """Capture benchmark prices at decision time."""
        self._decisions[order_id] = _Decision(
            arrival_price=arrival_price,
            vwap_price=vwap_price,
            close_price=close_price,
            open_price=open_price,
            strategy_id=strategy_id,
        )

    # ── Post-trade ─────────────────────────────────────────────────────

    def record_fill(
        self,
        order_id: str,
        *,
        fill_price: float,
        quantity: float,
        side: str,
        symbol: str,
        filled_at: datetime | None = None,
        vwap_price: float | None = None,
        close_price: float | None = None,
        open_price: float | None = None,
    ) -> ExecutionRecord:
        """Record a fill and compute slippage vs. pre-registered benchmarks.

        Benchmark prices provided here override any set at decision time
        (useful when VWAP/close are only known after the fact).
        """
        decision = self._decisions.pop(order_id, _Decision())

        record = ExecutionRecord(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            fill_price=fill_price,
            arrival_price=decision.arrival_price,
            vwap_price=vwap_price or decision.vwap_price,
            close_price=close_price or decision.close_price,
            open_price=open_price or decision.open_price,
            filled_at=filled_at or _utcnow(),
            strategy_id=decision.strategy_id,
        )
        self._records.append(record)
        return record

    # ── Analysis ───────────────────────────────────────────────────────

    @property
    def records(self) -> list[ExecutionRecord]:
        return list(self._records)

    def report(self) -> TCAReport:
        """Build an aggregate TCA report from all recorded fills."""
        return TCAReport(records=list(self._records))

    def clear(self) -> None:
        """Reset all collected records and pending decisions."""
        self._decisions.clear()
        self._records.clear()


@dataclass
class TCAReport:
    """Aggregated transaction cost analysis report."""

    records: list[ExecutionRecord]

    @property
    def n_fills(self) -> int:
        return len(self.records)

    @property
    def total_notional(self) -> float:
        """Sum of ``fill_price * quantity`` across all records."""
        return sum(r.fill_price * r.quantity for r in self.records)

    # ── Implementation shortfall ───────────────────────────────────────

    @property
    def mean_implementation_shortfall(self) -> float | None:
        """Notional-weighted mean implementation shortfall."""
        return self._weighted_mean(lambda r: r.implementation_shortfall)

    @property
    def total_dollar_cost(self) -> float:
        """Sum of dollar implementation shortfall across all records."""
        return sum(r.dollar_cost for r in self.records if r.dollar_cost is not None)

    # ── VWAP slippage ──────────────────────────────────────────────────

    @property
    def mean_vwap_slippage(self) -> float | None:
        """Notional-weighted mean VWAP slippage."""
        return self._weighted_mean(lambda r: r.vwap_slippage)

    # ── Close slippage ─────────────────────────────────────────────────

    @property
    def mean_close_slippage(self) -> float | None:
        """Notional-weighted mean close slippage."""
        return self._weighted_mean(lambda r: r.close_slippage)

    # ── By strategy ────────────────────────────────────────────────────

    def by_strategy(self) -> dict[str | None, TCAReport]:
        """Split the report by ``strategy_id``."""
        groups: dict[str | None, list[ExecutionRecord]] = {}
        for r in self.records:
            groups.setdefault(r.strategy_id, []).append(r)
        return {k: TCAReport(records=v) for k, v in groups.items()}

    def by_symbol(self) -> dict[str, TCAReport]:
        """Split the report by symbol."""
        groups: dict[str, list[ExecutionRecord]] = {}
        for r in self.records:
            groups.setdefault(r.symbol, []).append(r)
        return {k: TCAReport(records=v) for k, v in groups.items()}

    # ── Summary ────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Human-readable TCA summary."""
        lines = [
            f"TCA Report — {self.n_fills} fills, ${self.total_notional:,.0f} notional",
            "─" * 55,
        ]

        mis = self.mean_implementation_shortfall
        if mis is not None:
            lines.append(f"  Impl. shortfall (wtd mean) : {mis:+.4%}")
        lines.append(f"  Total dollar cost          : ${self.total_dollar_cost:+,.2f}")

        mvs = self.mean_vwap_slippage
        if mvs is not None:
            lines.append(f"  VWAP slippage (wtd mean)   : {mvs:+.4%}")

        mcs = self.mean_close_slippage
        if mcs is not None:
            lines.append(f"  Close slippage (wtd mean)  : {mcs:+.4%}")

        # Per-symbol breakdown
        by_sym = self.by_symbol()
        if len(by_sym) > 1:
            lines.append("")
            lines.append("  Per-symbol breakdown:")
            for sym, sub in sorted(by_sym.items()):
                is_val = sub.mean_implementation_shortfall
                is_str = f"{is_val:+.4%}" if is_val is not None else "n/a"
                lines.append(
                    f"    {sym:8s}  fills={sub.n_fills:3d}  "
                    f"notional=${sub.total_notional:>12,.0f}  "
                    f"shortfall={is_str}"
                )

        return "\n".join(lines)

    # ── Internal ───────────────────────────────────────────────────────

    def _weighted_mean(self, fn) -> float | None:
        """Compute notional-weighted mean of a per-record metric."""
        total_weight = 0.0
        total_value = 0.0
        for r in self.records:
            val = fn(r)
            if val is None:
                continue
            notional = r.fill_price * r.quantity
            total_weight += notional
            total_value += val * notional
        if total_weight == 0.0:
            return None
        return total_value / total_weight
