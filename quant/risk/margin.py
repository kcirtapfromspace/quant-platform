"""Leverage and margin management.

Tracks portfolio-level leverage, margin requirements, financing costs,
and margin call thresholds for leveraged hedge fund strategies.

Key concepts:

  * **Gross leverage**: sum(|position_values|) / NAV.
  * **Net leverage**: sum(position_values) / NAV.
  * **Margin requirement**: Reg-T initial (50%) and maintenance (25%)
    margins applied to gross exposure, with per-asset overrides.
  * **Financing cost**: daily accrual on borrowed capital at a
    configurable borrow rate (annualised).
  * **Margin call**: triggered when available margin drops below zero
    (equity < maintenance requirement).

Usage::

    from quant.risk.margin import MarginManager, MarginConfig

    mgr = MarginManager(MarginConfig(
        initial_margin_rate=0.50,
        maintenance_margin_rate=0.25,
        borrow_rate_annual=0.055,
    ))
    status = mgr.compute(positions, nav)
    print(status.summary())
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MarginConfig:
    """Margin and leverage parameters.

    Attributes:
        initial_margin_rate:     Fraction of position value required as
            initial margin (Reg-T default 50%).
        maintenance_margin_rate: Fraction required for maintenance margin
            (Reg-T default 25%).
        max_gross_leverage:      Hard ceiling on gross leverage. Positions
            beyond this are flagged.
        borrow_rate_annual:      Annualised rate charged on borrowed capital.
        trading_days:            Trading days per year for daily cost accrual.
        margin_overrides:        Per-symbol margin rate overrides (e.g. for
            concentrated or volatile names). Maps symbol → maintenance rate.
    """

    initial_margin_rate: float = 0.50
    maintenance_margin_rate: float = 0.25
    max_gross_leverage: float = 4.0
    borrow_rate_annual: float = 0.055
    trading_days: int = 252
    margin_overrides: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PositionMargin:
    """Per-position margin analytics.

    Attributes:
        symbol:                Asset identifier.
        market_value:          Signed market value.
        abs_value:             Absolute market value (contribution to gross).
        margin_rate:           Effective maintenance margin rate used.
        maintenance_required:  Margin required for this position.
        leverage_contrib:      This position's contribution to gross leverage.
    """

    symbol: str
    market_value: float
    abs_value: float
    margin_rate: float
    maintenance_required: float
    leverage_contrib: float


@dataclass
class MarginStatus:
    """Portfolio-level margin and leverage status.

    Attributes:
        nav:                    Net asset value (equity).
        gross_exposure:         Sum of |position values|.
        net_exposure:           Sum of position values.
        long_exposure:          Sum of positive position values.
        short_exposure:         Sum of |negative position values|.
        gross_leverage:         gross_exposure / nav.
        net_leverage:           net_exposure / nav.
        initial_margin_req:     Total initial margin required.
        maintenance_margin_req: Total maintenance margin required.
        excess_margin:          nav - maintenance_margin_req.
        margin_utilisation:     maintenance_margin_req / nav.
        margin_call:            True if excess_margin < 0.
        distance_to_call_pct:  (nav - maintenance_req) / nav.
        daily_financing_cost:   Daily cost of borrowed capital.
        annual_financing_cost:  Annualised financing cost.
        leverage_headroom:      How much more gross exposure is allowed
            before hitting max_gross_leverage.
        position_margins:       Per-position margin breakdown.
    """

    nav: float
    gross_exposure: float
    net_exposure: float
    long_exposure: float
    short_exposure: float
    gross_leverage: float
    net_leverage: float
    initial_margin_req: float
    maintenance_margin_req: float
    excess_margin: float
    margin_utilisation: float
    margin_call: bool
    distance_to_call_pct: float
    daily_financing_cost: float
    annual_financing_cost: float
    leverage_headroom: float
    position_margins: list[PositionMargin]

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            "Margin & Leverage Status",
            "=" * 55,
            "",
            f"NAV                : ${self.nav:,.2f}",
            f"Gross exposure     : ${self.gross_exposure:,.2f}",
            f"Net exposure       : ${self.net_exposure:,.2f}",
            f"  Long             : ${self.long_exposure:,.2f}",
            f"  Short            : ${self.short_exposure:,.2f}",
            "",
            f"Gross leverage     : {self.gross_leverage:.2f}x",
            f"Net leverage       : {self.net_leverage:+.2f}x",
            f"Leverage headroom  : ${self.leverage_headroom:,.2f}",
            "",
            f"Initial margin req : ${self.initial_margin_req:,.2f}",
            f"Maint. margin req  : ${self.maintenance_margin_req:,.2f}",
            f"Excess margin      : ${self.excess_margin:,.2f}",
            f"Margin utilisation  : {self.margin_utilisation:.1%}",
            f"Distance to call   : {self.distance_to_call_pct:.1%}",
            f"MARGIN CALL        : {'YES' if self.margin_call else 'No'}",
            "",
            f"Daily financing    : ${self.daily_financing_cost:,.2f}",
            f"Annual financing   : ${self.annual_financing_cost:,.2f}",
        ]
        if self.position_margins:
            lines.append("")
            lines.append(
                f"{'Symbol':<8} {'MktVal':>12} {'Rate':>6} {'Maint':>12} {'Lev':>6}"
            )
            lines.append("-" * 55)
            sorted_positions = sorted(
                self.position_margins, key=lambda p: p.abs_value, reverse=True,
            )
            for pm in sorted_positions[:15]:
                lines.append(
                    f"{pm.symbol:<8} ${pm.market_value:>10,.2f} "
                    f"{pm.margin_rate:>5.0%} ${pm.maintenance_required:>10,.2f} "
                    f"{pm.leverage_contrib:>5.2f}x"
                )
            if len(self.position_margins) > 15:
                lines.append(f"  ... and {len(self.position_margins) - 15} more")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class MarginManager:
    """Computes leverage, margin requirements, and financing costs.

    Args:
        config: Margin configuration parameters.
    """

    def __init__(self, config: MarginConfig | None = None) -> None:
        self._config = config or MarginConfig()

    @property
    def config(self) -> MarginConfig:
        return self._config

    def compute(
        self,
        positions: dict[str, float],
        nav: float,
    ) -> MarginStatus:
        """Compute full margin and leverage status.

        Args:
            positions: Map of symbol → signed market value (not quantity).
                Positive = long, negative = short.
            nav: Current net asset value (equity).

        Returns:
            :class:`MarginStatus` with comprehensive margin analytics.

        Raises:
            ValueError: If NAV is not positive.
        """
        if nav <= 0:
            raise ValueError(f"NAV must be positive, got {nav}")

        cfg = self._config

        long_exp = 0.0
        short_exp = 0.0
        total_maint = 0.0
        total_init = 0.0
        position_margins: list[PositionMargin] = []

        for sym, mkt_val in positions.items():
            abs_val = abs(mkt_val)
            if mkt_val >= 0:
                long_exp += mkt_val
            else:
                short_exp += abs_val

            # Per-position margin rate (override or default)
            maint_rate = cfg.margin_overrides.get(sym, cfg.maintenance_margin_rate)
            init_rate = max(maint_rate, cfg.initial_margin_rate)

            maint_req = abs_val * maint_rate
            init_req = abs_val * init_rate
            total_maint += maint_req
            total_init += init_req

            position_margins.append(PositionMargin(
                symbol=sym,
                market_value=mkt_val,
                abs_value=abs_val,
                margin_rate=maint_rate,
                maintenance_required=maint_req,
                leverage_contrib=abs_val / nav,
            ))

        gross_exp = long_exp + short_exp
        net_exp = long_exp - short_exp
        gross_lev = gross_exp / nav
        net_lev = net_exp / nav

        excess = nav - total_maint
        util = total_maint / nav if nav > 0 else 0.0
        dist_to_call = excess / nav if nav > 0 else 0.0

        # Financing: cost of borrowed capital
        # Borrowed = max(0, gross_exposure - nav)
        borrowed = max(0.0, gross_exp - nav)
        annual_cost = borrowed * cfg.borrow_rate_annual
        daily_cost = annual_cost / cfg.trading_days

        # Leverage headroom
        max_gross = cfg.max_gross_leverage * nav
        headroom = max_gross - gross_exp

        return MarginStatus(
            nav=nav,
            gross_exposure=gross_exp,
            net_exposure=net_exp,
            long_exposure=long_exp,
            short_exposure=short_exp,
            gross_leverage=gross_lev,
            net_leverage=net_lev,
            initial_margin_req=total_init,
            maintenance_margin_req=total_maint,
            excess_margin=excess,
            margin_utilisation=util,
            margin_call=excess < 0,
            distance_to_call_pct=dist_to_call,
            daily_financing_cost=daily_cost,
            annual_financing_cost=annual_cost,
            leverage_headroom=headroom,
            position_margins=position_margins,
        )

    def financing_impact(
        self,
        positions: dict[str, float],
        nav: float,
        holding_days: int = 252,
    ) -> pd.Series:
        """Project cumulative financing cost over a holding period.

        Args:
            positions: Symbol → signed market value.
            nav:       Current NAV.
            holding_days: Number of trading days to project.

        Returns:
            Series of cumulative financing cost indexed by day number.
        """
        status = self.compute(positions, nav)
        daily = status.daily_financing_cost
        days = list(range(1, holding_days + 1))
        costs = [daily * d for d in days]
        return pd.Series(costs, index=days, name="cumulative_financing_cost")

    def liquidation_value(
        self,
        positions: dict[str, float],
        nav: float,
        haircuts: dict[str, float] | None = None,
    ) -> float:
        """Estimate NAV after liquidation with haircuts.

        Args:
            positions:  Symbol → signed market value.
            nav:        Current NAV.
            haircuts:   Optional symbol → haircut fraction (e.g. 0.02 = 2%).
                Defaults to 0 (no haircut) if not specified.

        Returns:
            Estimated post-liquidation NAV.
        """
        haircuts = haircuts or {}
        total_cost = 0.0
        for sym, mkt_val in positions.items():
            hc = haircuts.get(sym, 0.0)
            total_cost += abs(mkt_val) * hc
        return nav - total_cost
