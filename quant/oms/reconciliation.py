"""Position reconciliation — OMS vs broker state.

Compares internally tracked positions with authoritative broker positions
and produces a structured report of discrepancies. Optionally applies
corrections to bring the OMS back in sync.

Reconciliation is essential for production trading where fills can be
missed, network issues cause state drift, or partial fills arrive out
of order.

Key concepts:

  * **Break**: A discrepancy between OMS and broker position quantity
    that exceeds the configured tolerance.
  * **Phantom position**: A position tracked by the OMS that the broker
    does not recognise.
  * **Missing position**: A broker position that the OMS has no record of.
  * **Quantity break**: Both sides hold a position, but quantities
    disagree beyond tolerance.

Usage::

    from quant.oms.reconciliation import (
        PositionReconciler,
        ReconciliationConfig,
    )

    reconciler = PositionReconciler(config=ReconciliationConfig())
    report = reconciler.reconcile(oms_positions, broker_positions)
    print(report.summary())

    if report.has_breaks:
        corrections = reconciler.compute_corrections(report)
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ReconciliationConfig:
    """Configuration for position reconciliation.

    Attributes:
        quantity_tolerance: Absolute tolerance for quantity comparison.
            Differences below this threshold are considered matched.
        price_tolerance_pct: Relative tolerance for avg cost comparison
            (as a fraction, e.g. 0.01 = 1%).  Used for informational
            flagging only — does not trigger a break.
        flag_price_drift: Whether to flag avg cost divergence in the
            report even when quantity matches.
    """

    quantity_tolerance: float = 1e-6
    price_tolerance_pct: float = 0.01
    flag_price_drift: bool = True


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class BreakType(str, enum.Enum):
    """Classification of a reconciliation discrepancy."""

    MATCHED = "matched"
    QUANTITY_BREAK = "quantity_break"
    PHANTOM = "phantom"      # OMS has position, broker does not
    MISSING = "missing"      # Broker has position, OMS does not


class CorrectionAction(str, enum.Enum):
    """Action to apply to resolve a break."""

    NONE = "none"
    SET_QUANTITY = "set_quantity"     # Override OMS quantity to broker
    CREATE_POSITION = "create"       # Create position in OMS
    REMOVE_POSITION = "remove"       # Remove phantom from OMS


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PositionBreak:
    """A single reconciliation item (matched or broken).

    Attributes:
        symbol:         Asset identifier.
        break_type:     Classification of the discrepancy.
        oms_quantity:   Quantity tracked by the OMS (0 if missing).
        broker_quantity: Quantity reported by the broker (0 if phantom).
        quantity_diff:  broker_quantity - oms_quantity.
        oms_avg_cost:   OMS average cost (0 if missing).
        broker_avg_cost: Broker average cost (0 if phantom).
        price_drift_pct: |oms_avg_cost - broker_avg_cost| / broker_avg_cost.
    """

    symbol: str
    break_type: BreakType
    oms_quantity: float
    broker_quantity: float
    quantity_diff: float
    oms_avg_cost: float
    broker_avg_cost: float
    price_drift_pct: float


@dataclass(frozen=True, slots=True)
class Correction:
    """A proposed correction to resolve a break.

    Attributes:
        symbol:     Asset identifier.
        action:     What to do.
        target_qty: Target quantity after correction (for SET/CREATE).
        target_cost: Target avg cost after correction (for CREATE).
        reason:     Human-readable explanation.
    """

    symbol: str
    action: CorrectionAction
    target_qty: float
    target_cost: float
    reason: str


@dataclass
class ReconciliationReport:
    """Full reconciliation report.

    Attributes:
        items:           All reconciliation items (matched + breaks).
        timestamp:       When the reconciliation was performed.
        n_matched:       Number of fully matched positions.
        n_quantity_breaks: Number of quantity breaks.
        n_phantom:       Number of phantom positions.
        n_missing:       Number of missing positions.
    """

    items: list[PositionBreak] = field(default_factory=list)
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    n_matched: int = 0
    n_quantity_breaks: int = 0
    n_phantom: int = 0
    n_missing: int = 0

    @property
    def has_breaks(self) -> bool:
        """True if any breaks were found."""
        return (self.n_quantity_breaks + self.n_phantom + self.n_missing) > 0

    @property
    def n_total(self) -> int:
        """Total items reconciled."""
        return len(self.items)

    @property
    def breaks(self) -> list[PositionBreak]:
        """Return only items with breaks (non-matched)."""
        return [i for i in self.items if i.break_type != BreakType.MATCHED]

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            "Position Reconciliation Report",
            "=" * 55,
            "",
            f"Timestamp          : {self.timestamp:%Y-%m-%d %H:%M:%S UTC}",
            f"Total positions    : {self.n_total}",
            f"Matched            : {self.n_matched}",
            f"Quantity breaks    : {self.n_quantity_breaks}",
            f"Phantom (OMS only) : {self.n_phantom}",
            f"Missing (broker)   : {self.n_missing}",
            f"Status             : {'CLEAN' if not self.has_breaks else 'BREAKS FOUND'}",
        ]
        if self.has_breaks:
            lines.append("")
            lines.append(
                f"{'Symbol':<8} {'Type':<16} {'OMS Qty':>10} "
                f"{'Broker Qty':>10} {'Diff':>10}"
            )
            lines.append("-" * 55)
            for item in self.breaks:
                lines.append(
                    f"{item.symbol:<8} {item.break_type.value:<16} "
                    f"{item.oms_quantity:>10.4f} {item.broker_quantity:>10.4f} "
                    f"{item.quantity_diff:>+10.4f}"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reconciler
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PositionSnapshot:
    """Lightweight position snapshot for reconciliation input.

    Decoupled from the OMS ``Position`` model so callers can build
    snapshots from any source (broker API, database, CSV, etc.).

    Attributes:
        symbol:   Asset identifier.
        quantity: Signed quantity (positive = long, negative = short).
        avg_cost: Average cost per unit.
    """

    symbol: str
    quantity: float
    avg_cost: float = 0.0


class PositionReconciler:
    """Compares OMS positions against broker positions.

    Args:
        config: Reconciliation parameters.
    """

    def __init__(self, config: ReconciliationConfig | None = None) -> None:
        self._config = config or ReconciliationConfig()

    @property
    def config(self) -> ReconciliationConfig:
        return self._config

    def reconcile(
        self,
        oms_positions: list[PositionSnapshot],
        broker_positions: list[PositionSnapshot],
    ) -> ReconciliationReport:
        """Compare OMS vs broker positions and produce a report.

        Args:
            oms_positions:    Current OMS position snapshots.
            broker_positions: Authoritative broker position snapshots.

        Returns:
            :class:`ReconciliationReport` with all items and break counts.
        """
        oms_map: dict[str, PositionSnapshot] = {p.symbol: p for p in oms_positions}
        broker_map: dict[str, PositionSnapshot] = {p.symbol: p for p in broker_positions}

        all_symbols = sorted(set(oms_map) | set(broker_map))
        tol = self._config.quantity_tolerance

        items: list[PositionBreak] = []
        n_matched = 0
        n_qty_break = 0
        n_phantom = 0
        n_missing = 0

        for sym in all_symbols:
            oms_pos = oms_map.get(sym)
            broker_pos = broker_map.get(sym)

            oms_qty = oms_pos.quantity if oms_pos else 0.0
            broker_qty = broker_pos.quantity if broker_pos else 0.0
            oms_cost = oms_pos.avg_cost if oms_pos else 0.0
            broker_cost = broker_pos.avg_cost if broker_pos else 0.0
            diff = broker_qty - oms_qty

            # Price drift
            price_drift = 0.0
            if broker_cost > 0 and oms_cost > 0:
                price_drift = abs(oms_cost - broker_cost) / broker_cost

            # Classify
            if oms_pos is not None and broker_pos is None:
                break_type = BreakType.PHANTOM
                n_phantom += 1
            elif oms_pos is None and broker_pos is not None:
                break_type = BreakType.MISSING
                n_missing += 1
            elif abs(diff) > tol:
                break_type = BreakType.QUANTITY_BREAK
                n_qty_break += 1
            else:
                break_type = BreakType.MATCHED
                n_matched += 1

            items.append(PositionBreak(
                symbol=sym,
                break_type=break_type,
                oms_quantity=oms_qty,
                broker_quantity=broker_qty,
                quantity_diff=diff,
                oms_avg_cost=oms_cost,
                broker_avg_cost=broker_cost,
                price_drift_pct=price_drift,
            ))

        return ReconciliationReport(
            items=items,
            n_matched=n_matched,
            n_quantity_breaks=n_qty_break,
            n_phantom=n_phantom,
            n_missing=n_missing,
        )

    def compute_corrections(
        self,
        report: ReconciliationReport,
    ) -> list[Correction]:
        """Generate correction actions to resolve all breaks.

        The corrections assume the broker is the source of truth:
          - Quantity breaks → set OMS quantity to broker quantity.
          - Phantom positions → remove from OMS.
          - Missing positions → create in OMS with broker values.

        Args:
            report: A reconciliation report (from :meth:`reconcile`).

        Returns:
            List of :class:`Correction` objects, one per break.
        """
        corrections: list[Correction] = []

        for item in report.items:
            if item.break_type == BreakType.MATCHED:
                continue

            if item.break_type == BreakType.QUANTITY_BREAK:
                corrections.append(Correction(
                    symbol=item.symbol,
                    action=CorrectionAction.SET_QUANTITY,
                    target_qty=item.broker_quantity,
                    target_cost=item.broker_avg_cost,
                    reason=(
                        f"Quantity mismatch: OMS={item.oms_quantity:.4f}, "
                        f"broker={item.broker_quantity:.4f}, "
                        f"diff={item.quantity_diff:+.4f}"
                    ),
                ))
            elif item.break_type == BreakType.PHANTOM:
                corrections.append(Correction(
                    symbol=item.symbol,
                    action=CorrectionAction.REMOVE_POSITION,
                    target_qty=0.0,
                    target_cost=0.0,
                    reason=(
                        f"Phantom position: OMS holds {item.oms_quantity:.4f} "
                        f"but broker reports no position"
                    ),
                ))
            elif item.break_type == BreakType.MISSING:
                corrections.append(Correction(
                    symbol=item.symbol,
                    action=CorrectionAction.CREATE_POSITION,
                    target_qty=item.broker_quantity,
                    target_cost=item.broker_avg_cost,
                    reason=(
                        f"Missing position: broker holds {item.broker_quantity:.4f} "
                        f"but OMS has no record"
                    ),
                ))

        return corrections

    def price_drift_flags(
        self,
        report: ReconciliationReport,
    ) -> list[PositionBreak]:
        """Return matched items where avg cost has drifted beyond tolerance.

        These are informational — they don't represent a break, but may
        indicate that the OMS cost basis has diverged from the broker's
        calculation (e.g. due to different corporate action handling).

        Args:
            report: A reconciliation report.

        Returns:
            List of matched items with price drift above threshold.
        """
        if not self._config.flag_price_drift:
            return []

        pct = self._config.price_tolerance_pct
        return [
            item for item in report.items
            if item.break_type == BreakType.MATCHED
            and item.price_drift_pct > pct
        ]
