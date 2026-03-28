"""Realistic execution simulator for portfolio rebalancing.

Models how portfolio trades get filled in practice, accounting for
market impact, volume participation limits, and spread costs.  Bridges
the gap between the portfolio optimizer (which estimates theoretical
costs) and realistic backtesting (which needs actual execution prices).

Fill models:

  * **Instant** — fill at mid-price plus half-spread (baseline).
  * **Participation** — fills limited by a maximum fraction of average
    daily volume.  Orders exceeding the cap are partially filled.
  * **MarketImpact** — applies Almgren-Chriss square-root temporary
    impact on top of spread.  Slippage scales with sqrt(participation).

Key outputs:

  * **OrderFill** — per-order execution details (fill price, slippage,
    participation rate, fill fraction).
  * **ExecutionSummary** — aggregate statistics (implementation
    shortfall, realised cost decomposition, fill rates).
  * **CapacityEstimate** — strategy capacity analysis showing how
    execution costs scale with AUM.

Usage::

    from quant.execution.fill_simulator import (
        ExecutionSimulator,
        SimulatorConfig,
    )

    simulator = ExecutionSimulator(SimulatorConfig(
        fill_model=FillModel.MARKET_IMPACT,
        max_participation=0.10,
    ))
    result = simulator.simulate_rebalance(
        target_weights=target,
        current_weights=current,
        prices=price_series,
        volumes=volume_series,
        volatilities=vol_series,
    )
    print(result.summary())

    # Estimate strategy capacity
    capacity = simulator.estimate_capacity(
        target_weights=target,
        volumes=volume_series,
        volatilities=vol_series,
        expected_alpha_bps=50.0,
    )
    print(capacity.summary())
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class FillModel(str, Enum):
    """Execution fill model for simulation."""

    INSTANT = "instant"
    PARTICIPATION = "participation"
    MARKET_IMPACT = "market_impact"


@dataclass
class SimulatorConfig:
    """Configuration for execution simulation.

    Attributes:
        fill_model:         Which fill model to use.
        max_participation:  Maximum fraction of ADV to trade per period.
        impact_coefficient: η in the square-root impact model.
        impact_exponent:    β exponent (0.5 = square-root law).
        spread_bps:         Half-spread cost in basis points.
        commission_bps:     Per-trade commission in basis points.
        aum:                Portfolio AUM in dollars for notional sizing.
        annualisation:      Trading days per year for vol conversion.
    """

    fill_model: FillModel = FillModel.MARKET_IMPACT
    max_participation: float = 0.10
    impact_coefficient: float = 0.10
    impact_exponent: float = 0.50
    spread_bps: float = 5.0
    commission_bps: float = 1.0
    aum: float = 100_000_000
    annualisation: int = 252


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OrderFill:
    """Execution result for a single order.

    Attributes:
        symbol:             Ticker.
        side:               ``"buy"`` or ``"sell"``.
        target_notional:    Desired trade notional (USD).
        filled_notional:    Actually filled notional (USD).
        fill_fraction:      Fraction of target that was filled [0, 1].
        arrival_price:      Mid-price at order arrival.
        fill_price:         Realised average fill price.
        slippage_bps:       Total slippage from arrival price in bps.
        impact_cost_bps:    Market impact component in bps.
        spread_cost_bps:    Half-spread component in bps.
        commission_cost_bps: Commission component in bps.
        total_cost_bps:     Total execution cost in bps.
        participation_rate: Filled volume as fraction of ADV.
    """

    symbol: str
    side: str
    target_notional: float
    filled_notional: float
    fill_fraction: float
    arrival_price: float
    fill_price: float
    slippage_bps: float
    impact_cost_bps: float
    spread_cost_bps: float
    commission_cost_bps: float
    total_cost_bps: float
    participation_rate: float


@dataclass
class ExecutionSummary:
    """Aggregate execution simulation results.

    Attributes:
        fills:              Per-order fill details.
        n_orders:           Total orders generated.
        n_fully_filled:     Orders filled at 100%.
        n_partially_filled: Orders with partial fills.
        n_unfilled:         Orders with zero fill.
        total_notional:     Sum of target notional across all orders.
        filled_notional:    Sum of actually filled notional.
        total_cost_bps:     Notional-weighted average total cost.
        avg_slippage_bps:   Notional-weighted average slippage.
        avg_participation:  Notional-weighted average participation rate.
        avg_fill_fraction:  Overall fill rate (filled / target notional).
        implementation_shortfall_bps: Estimated implementation shortfall.
        impact_cost_bps:    Notional-weighted average impact cost.
        spread_cost_bps:    Notional-weighted average spread cost.
        commission_cost_bps: Notional-weighted average commission cost.
    """

    fills: list[OrderFill] = field(repr=False)
    n_orders: int = 0
    n_fully_filled: int = 0
    n_partially_filled: int = 0
    n_unfilled: int = 0
    total_notional: float = 0.0
    filled_notional: float = 0.0
    total_cost_bps: float = 0.0
    avg_slippage_bps: float = 0.0
    avg_participation: float = 0.0
    avg_fill_fraction: float = 0.0
    implementation_shortfall_bps: float = 0.0
    impact_cost_bps: float = 0.0
    spread_cost_bps: float = 0.0
    commission_cost_bps: float = 0.0

    def summary(self) -> str:
        """Return a human-readable execution summary."""
        lines = [
            f"Execution Summary ({self.n_orders} orders)",
            "=" * 60,
            "",
            f"Orders              : {self.n_orders}",
            f"  Fully filled      : {self.n_fully_filled}",
            f"  Partially filled  : {self.n_partially_filled}",
            f"  Unfilled          : {self.n_unfilled}",
            "",
            f"Total notional      : ${self.total_notional:,.0f}",
            f"Filled notional     : ${self.filled_notional:,.0f}",
            f"Fill fraction       : {self.avg_fill_fraction:.1%}",
            "",
            "Cost decomposition (notional-weighted bps):",
            f"  Market impact     : {self.impact_cost_bps:.1f} bps",
            f"  Spread            : {self.spread_cost_bps:.1f} bps",
            f"  Commission        : {self.commission_cost_bps:.1f} bps",
            f"  Total cost        : {self.total_cost_bps:.1f} bps",
            "",
            f"Impl. shortfall     : {self.implementation_shortfall_bps:.1f} bps",
            f"Avg participation   : {self.avg_participation:.2%}",
        ]

        if self.fills:
            lines.extend(["", "Per-order fills (top 5 by notional):"])
            top = sorted(self.fills, key=lambda f: f.target_notional, reverse=True)[:5]
            for f in top:
                lines.append(
                    f"  {f.symbol:<8s} {f.side:<4s}: "
                    f"${f.filled_notional:>12,.0f} / ${f.target_notional:>12,.0f} "
                    f"({f.fill_fraction:.0%}) cost={f.total_cost_bps:.1f}bps"
                )

        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class CapacityEstimate:
    """Strategy capacity estimation result.

    Attributes:
        max_aum:            Maximum AUM before cost exceeds threshold.
        cost_at_max:        Execution cost at max AUM in bps.
        breakeven_aum:      AUM where cost equals expected alpha.
        aum_levels:         AUM values tested.
        cost_curve_bps:     Cost at each AUM level.
        fill_curve:         Average fill fraction at each AUM level.
        expected_alpha_bps: Expected alpha used for analysis.
        max_cost_ratio:     Cost / alpha threshold for max AUM.
    """

    max_aum: float
    cost_at_max: float
    breakeven_aum: float
    aum_levels: tuple[float, ...]
    cost_curve_bps: tuple[float, ...]
    fill_curve: tuple[float, ...]
    expected_alpha_bps: float
    max_cost_ratio: float

    def summary(self) -> str:
        """Return a human-readable capacity summary."""
        lines = [
            "Capacity Estimate",
            "=" * 60,
            "",
            f"Expected alpha       : {self.expected_alpha_bps:.1f} bps",
            f"Cost threshold       : {self.max_cost_ratio:.0%} of alpha",
            f"Max AUM              : ${self.max_aum:,.0f}",
            f"Cost at max AUM      : {self.cost_at_max:.1f} bps",
            f"Breakeven AUM        : ${self.breakeven_aum:,.0f}",
            "",
            "Cost curve:",
        ]
        for aum, cost, fill in zip(
            self.aum_levels, self.cost_curve_bps, self.fill_curve, strict=True,
        ):
            ratio = cost / self.expected_alpha_bps if self.expected_alpha_bps > 0 else 0.0
            lines.append(
                f"  ${aum:>15,.0f}: {cost:6.1f} bps "
                f"({ratio:5.0%} of alpha, {fill:.0%} filled)"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


class ExecutionSimulator:
    """Realistic execution simulator for portfolio rebalancing.

    Simulates how a set of portfolio trades would be filled in practice,
    modelling volume participation limits, spread costs, and market impact.

    Args:
        config: Simulation configuration.
    """

    def __init__(self, config: SimulatorConfig | None = None) -> None:
        self._config = config or SimulatorConfig()

    @property
    def config(self) -> SimulatorConfig:
        return self._config

    # ── Main entry point ───────────────────────────────────────────

    def simulate_rebalance(
        self,
        target_weights: dict[str, float],
        current_weights: dict[str, float] | None = None,
        prices: pd.Series | None = None,
        volumes: pd.Series | None = None,
        volatilities: pd.Series | None = None,
    ) -> ExecutionSummary:
        """Simulate execution of a portfolio rebalance.

        Args:
            target_weights:  Target portfolio weights ``{symbol: weight}``.
            current_weights: Current portfolio weights.  Defaults to zero.
            prices:          Current asset prices (pd.Series).  Defaults to
                             100 per asset when not provided.
            volumes:         Average daily volume in USD (pd.Series).  Defaults
                             to 1e9 per asset.
            volatilities:    Annualised volatility (pd.Series).  Defaults to
                             0.20 per asset.

        Returns:
            :class:`ExecutionSummary` with fill details and cost breakdown.
        """
        current = current_weights or {}

        # All symbols involved in the rebalance
        all_symbols = sorted(set(target_weights) | set(current))

        fills: list[OrderFill] = []
        for sym in all_symbols:
            tw = target_weights.get(sym, 0.0)
            cw = current.get(sym, 0.0)
            delta = tw - cw

            if abs(delta) < 1e-10:
                continue

            side = "buy" if delta > 0 else "sell"

            # Market data with sensible defaults
            price = _series_get(prices, sym, 100.0)
            adv = _series_get(volumes, sym, 1e9)
            vol = _series_get(volatilities, sym, 0.20)

            target_notional = abs(delta) * self._config.aum

            fill = self._simulate_fill(sym, side, target_notional, price, adv, vol)
            fills.append(fill)

        return self._build_summary(fills)

    # ── Capacity estimation ────────────────────────────────────────

    def estimate_capacity(
        self,
        target_weights: dict[str, float],
        volumes: pd.Series,
        volatilities: pd.Series,
        expected_alpha_bps: float = 50.0,
        max_cost_ratio: float = 0.25,
        prices: pd.Series | None = None,
        n_points: int = 10,
    ) -> CapacityEstimate:
        """Estimate maximum AUM before execution costs erode alpha.

        Sweeps a log-spaced range of AUM values, simulating execution
        at each level.  Reports the maximum AUM at which execution cost
        stays below ``max_cost_ratio × expected_alpha_bps``.

        Args:
            target_weights:     Strategy target weights.
            volumes:            Average daily volume per asset (USD).
            volatilities:       Annualised volatility per asset.
            expected_alpha_bps: Expected strategy alpha in bps per trade.
            max_cost_ratio:     Maximum acceptable cost / alpha ratio.
            prices:             Asset prices (optional).
            n_points:           Number of AUM levels to test.

        Returns:
            :class:`CapacityEstimate` with cost curve and capacity limits.
        """
        cost_threshold = max_cost_ratio * expected_alpha_bps

        # Log-spaced AUM from $1M to $10B
        aum_levels = np.logspace(6, 10, n_points)

        cost_curve: list[float] = []
        fill_curve: list[float] = []
        max_aum = aum_levels[-1]
        cost_at_max = 0.0
        breakeven_aum = aum_levels[-1]
        found_max = False
        found_breakeven = False

        for aum in aum_levels:
            cfg = SimulatorConfig(
                fill_model=self._config.fill_model,
                max_participation=self._config.max_participation,
                impact_coefficient=self._config.impact_coefficient,
                impact_exponent=self._config.impact_exponent,
                spread_bps=self._config.spread_bps,
                commission_bps=self._config.commission_bps,
                aum=float(aum),
                annualisation=self._config.annualisation,
            )
            sim = ExecutionSimulator(cfg)
            result = sim.simulate_rebalance(
                target_weights, prices=prices,
                volumes=volumes, volatilities=volatilities,
            )
            cost_curve.append(result.total_cost_bps)
            fill_curve.append(result.avg_fill_fraction)

            if not found_max and result.total_cost_bps > cost_threshold:
                # Interpolate between this and previous level
                if len(cost_curve) >= 2:
                    prev_cost = cost_curve[-2]
                    prev_aum = aum_levels[len(cost_curve) - 2]
                    frac = (cost_threshold - prev_cost) / (result.total_cost_bps - prev_cost)
                    max_aum = prev_aum + frac * (aum - prev_aum)
                else:
                    max_aum = float(aum)
                cost_at_max = cost_threshold
                found_max = True

            if not found_breakeven and result.total_cost_bps > expected_alpha_bps:
                if len(cost_curve) >= 2:
                    prev_cost = cost_curve[-2]
                    prev_aum = aum_levels[len(cost_curve) - 2]
                    frac = (expected_alpha_bps - prev_cost) / (result.total_cost_bps - prev_cost)
                    breakeven_aum = prev_aum + frac * (aum - prev_aum)
                else:
                    breakeven_aum = float(aum)
                found_breakeven = True

        if not found_max:
            cost_at_max = cost_curve[-1] if cost_curve else 0.0

        return CapacityEstimate(
            max_aum=max_aum,
            cost_at_max=cost_at_max,
            breakeven_aum=breakeven_aum,
            aum_levels=tuple(float(a) for a in aum_levels),
            cost_curve_bps=tuple(cost_curve),
            fill_curve=tuple(fill_curve),
            expected_alpha_bps=expected_alpha_bps,
            max_cost_ratio=max_cost_ratio,
        )

    # ── Fill models ────────────────────────────────────────────────

    def _simulate_fill(
        self,
        symbol: str,
        side: str,
        target_notional: float,
        price: float,
        adv: float,
        volatility: float,
    ) -> OrderFill:
        """Dispatch to the configured fill model."""
        cfg = self._config

        if cfg.fill_model == FillModel.INSTANT:
            return self._fill_instant(symbol, side, target_notional, price)
        elif cfg.fill_model == FillModel.PARTICIPATION:
            return self._fill_participation(symbol, side, target_notional, price, adv)
        else:
            return self._fill_market_impact(
                symbol, side, target_notional, price, adv, volatility,
            )

    def _fill_instant(
        self,
        symbol: str,
        side: str,
        target_notional: float,
        price: float,
    ) -> OrderFill:
        """Fill instantly at mid-price plus half-spread."""
        cfg = self._config
        spread = cfg.spread_bps
        commission = cfg.commission_bps

        direction = 1.0 if side == "buy" else -1.0
        fill_price = price * (1.0 + direction * spread / 10_000)

        return OrderFill(
            symbol=symbol,
            side=side,
            target_notional=target_notional,
            filled_notional=target_notional,
            fill_fraction=1.0,
            arrival_price=price,
            fill_price=fill_price,
            slippage_bps=spread,
            impact_cost_bps=0.0,
            spread_cost_bps=spread,
            commission_cost_bps=commission,
            total_cost_bps=spread + commission,
            participation_rate=0.0,
        )

    def _fill_participation(
        self,
        symbol: str,
        side: str,
        target_notional: float,
        price: float,
        adv: float,
    ) -> OrderFill:
        """Fill limited by maximum volume participation rate."""
        cfg = self._config

        # Cap fill at max_participation × ADV
        max_fillable = cfg.max_participation * adv
        filled_notional = min(target_notional, max_fillable)
        fill_frac = filled_notional / target_notional if target_notional > 0 else 1.0

        participation = filled_notional / adv if adv > 0 else 0.0

        spread = cfg.spread_bps
        commission = cfg.commission_bps
        direction = 1.0 if side == "buy" else -1.0
        fill_price = price * (1.0 + direction * spread / 10_000)

        return OrderFill(
            symbol=symbol,
            side=side,
            target_notional=target_notional,
            filled_notional=filled_notional,
            fill_fraction=fill_frac,
            arrival_price=price,
            fill_price=fill_price,
            slippage_bps=spread,
            impact_cost_bps=0.0,
            spread_cost_bps=spread,
            commission_cost_bps=commission,
            total_cost_bps=spread + commission,
            participation_rate=participation,
        )

    def _fill_market_impact(
        self,
        symbol: str,
        side: str,
        target_notional: float,
        price: float,
        adv: float,
        volatility: float,
    ) -> OrderFill:
        """Fill with Almgren-Chriss square-root market impact.

        Impact model::

            impact_bps = η · σ_daily · participation^β · 10_000

        where ``participation = filled_notional / ADV``.
        """
        cfg = self._config

        # Volume-limited fill
        max_fillable = cfg.max_participation * adv
        filled_notional = min(target_notional, max_fillable)
        fill_frac = filled_notional / target_notional if target_notional > 0 else 1.0

        participation = filled_notional / adv if adv > 0 else 0.0

        # Market impact: η · σ_daily · participation^β · 10_000
        sigma_daily = volatility / math.sqrt(cfg.annualisation)
        impact_bps = 0.0
        if participation > 0:
            impact_bps = (
                cfg.impact_coefficient
                * sigma_daily
                * (participation ** cfg.impact_exponent)
                * 10_000
            )

        spread = cfg.spread_bps
        commission = cfg.commission_bps
        total_slippage = impact_bps + spread

        direction = 1.0 if side == "buy" else -1.0
        fill_price = price * (1.0 + direction * total_slippage / 10_000)

        return OrderFill(
            symbol=symbol,
            side=side,
            target_notional=target_notional,
            filled_notional=filled_notional,
            fill_fraction=fill_frac,
            arrival_price=price,
            fill_price=fill_price,
            slippage_bps=total_slippage,
            impact_cost_bps=impact_bps,
            spread_cost_bps=spread,
            commission_cost_bps=commission,
            total_cost_bps=impact_bps + spread + commission,
            participation_rate=participation,
        )

    # ── Summary builder ────────────────────────────────────────────

    def _build_summary(self, fills: list[OrderFill]) -> ExecutionSummary:
        """Aggregate individual fills into an execution summary."""
        if not fills:
            return ExecutionSummary(fills=[])

        n_orders = len(fills)
        n_fully = sum(1 for f in fills if f.fill_fraction >= 1.0 - 1e-9)
        n_partial = sum(
            1 for f in fills if 1e-9 < f.fill_fraction < 1.0 - 1e-9
        )
        n_unfilled = sum(1 for f in fills if f.fill_fraction < 1e-9)

        total_notional = sum(f.target_notional for f in fills)
        filled_notional = sum(f.filled_notional for f in fills)

        if filled_notional > 0:
            # Notional-weighted averages
            weights = [f.filled_notional / filled_notional for f in fills]
            wf = list(zip(weights, fills, strict=True))
            avg_slippage = sum(w * f.slippage_bps for w, f in wf)
            avg_total_cost = sum(w * f.total_cost_bps for w, f in wf)
            avg_impact = sum(w * f.impact_cost_bps for w, f in wf)
            avg_spread = sum(w * f.spread_cost_bps for w, f in wf)
            avg_commission = sum(w * f.commission_cost_bps for w, f in wf)
            avg_participation = sum(w * f.participation_rate for w, f in wf)
        else:
            avg_slippage = 0.0
            avg_total_cost = 0.0
            avg_impact = 0.0
            avg_spread = 0.0
            avg_commission = 0.0
            avg_participation = 0.0

        avg_fill = filled_notional / total_notional if total_notional > 0 else 0.0

        # Implementation shortfall = total cost + opportunity cost
        # Opportunity cost from unfilled orders:
        unfilled_notional = total_notional - filled_notional
        opportunity_bps = 0.0
        if total_notional > 0 and unfilled_notional > 0:
            # Approximate: unfilled portion loses expected alpha
            opportunity_bps = (unfilled_notional / total_notional) * avg_total_cost

        is_bps = avg_total_cost + opportunity_bps

        return ExecutionSummary(
            fills=fills,
            n_orders=n_orders,
            n_fully_filled=n_fully,
            n_partially_filled=n_partial,
            n_unfilled=n_unfilled,
            total_notional=total_notional,
            filled_notional=filled_notional,
            total_cost_bps=avg_total_cost,
            avg_slippage_bps=avg_slippage,
            avg_participation=avg_participation,
            avg_fill_fraction=avg_fill,
            implementation_shortfall_bps=is_bps,
            impact_cost_bps=avg_impact,
            spread_cost_bps=avg_spread,
            commission_cost_bps=avg_commission,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _series_get(s: pd.Series | None, key: str, default: float) -> float:
    """Safely extract a value from a nullable Series."""
    if s is not None and key in s.index:
        return float(s[key])
    return default
