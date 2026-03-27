"""Multi-component transaction cost model.

Decomposes expected execution costs into three additive layers:

  1. **Spread cost** — half the bid-ask spread, paid on every trade.
  2. **Market impact** — temporary + permanent price impact, modelled
     using the square-root law: ``impact = η · σ · (Q / ADV)^β``.
  3. **Commission** — fixed per-share, per-trade, or percentage fee.

Total expected cost in basis points::

    cost_bps = spread_bps + impact_bps + commission_bps

The model can be used in three modes:

  * **Backtest mode** — estimate total round-trip cost for a rebalance,
    given target weight changes and portfolio value.
  * **Pre-trade mode** — estimate cost of a proposed order to decide
    whether the trade is worth executing (break-even analysis).
  * **Per-fill mode** — compute expected cost for a single fill to
    compare against realised slippage in TCA.

Usage::

    from quant.execution.cost_model import TransactionCostModel, CostModelConfig

    model = TransactionCostModel(CostModelConfig(
        default_spread_bps=5.0,
        impact_coefficient=0.1,
        commission_per_share=0.005,
    ))

    # Estimate cost for a rebalance
    cost = model.estimate_rebalance_cost(weight_changes, portfolio_value, adv, volatility)

    # Estimate cost for a single order
    cost = model.estimate_order_cost("AAPL", notional=100_000, adv=50_000_000, volatility=0.25)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CostModelConfig:
    """Configuration for the transaction cost model.

    Attributes:
        default_spread_bps: Default half-spread in basis points, used when
            per-asset spread data is not available.
        spread_overrides:   Per-symbol spread overrides in basis points.
        impact_coefficient: Market impact coefficient (η).  Typical range
            0.05–0.20 for liquid US equities.
        impact_exponent:    Participation rate exponent (β).  Square-root
            law uses 0.5.
        volatility_scaling: If True, scale impact by annualised volatility.
        commission_per_share: Fixed commission per share (e.g. $0.005).
        commission_pct:     Percentage commission as a decimal (e.g. 0.0001
            for 1 bps).  Additive with per-share commission.
        min_commission:     Minimum commission per order in dollars.
        annualisation:      Trading days per year for vol scaling.
    """

    default_spread_bps: float = 5.0
    spread_overrides: dict[str, float] = field(default_factory=dict)
    impact_coefficient: float = 0.10
    impact_exponent: float = 0.50
    volatility_scaling: bool = True
    commission_per_share: float = 0.005
    commission_pct: float = 0.0
    min_commission: float = 0.0
    annualisation: int = 252


# ---------------------------------------------------------------------------
# Cost breakdown
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """Breakdown of estimated transaction cost for a single trade.

    All values are in basis points unless otherwise noted.

    Attributes:
        symbol:          Ticker symbol.
        notional:        Dollar notional of the trade.
        spread_bps:      Half-spread cost component.
        impact_bps:      Market impact cost component.
        commission_bps:  Commission cost component.
        total_bps:       Total one-way cost (spread + impact + commission).
        total_dollars:   Total cost in dollars.
    """

    symbol: str
    notional: float
    spread_bps: float
    impact_bps: float
    commission_bps: float
    total_bps: float
    total_dollars: float


@dataclass(frozen=True, slots=True)
class RebalanceCostEstimate:
    """Aggregate cost estimate for a full portfolio rebalance.

    Attributes:
        per_asset:       Per-asset cost breakdowns.
        total_spread_bps: Portfolio-level weighted spread cost.
        total_impact_bps: Portfolio-level weighted impact cost.
        total_commission_bps: Portfolio-level weighted commission cost.
        total_cost_bps:  Total portfolio-level one-way cost.
        total_cost_dollars: Total cost in dollars.
        portfolio_value: Portfolio value used for the estimate.
        turnover:        Sum of absolute weight changes.
    """

    per_asset: list[CostEstimate]
    total_spread_bps: float
    total_impact_bps: float
    total_commission_bps: float
    total_cost_bps: float
    total_cost_dollars: float
    portfolio_value: float
    turnover: float


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class TransactionCostModel:
    """Multi-component transaction cost estimator.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: CostModelConfig | None = None) -> None:
        self._config = config or CostModelConfig()

    @property
    def config(self) -> CostModelConfig:
        return self._config

    # ── Single order cost ─────────────────────────────────────────

    def estimate_order_cost(
        self,
        symbol: str,
        notional: float,
        adv: float | None = None,
        volatility: float | None = None,
        price: float | None = None,
        quantity: float | None = None,
    ) -> CostEstimate:
        """Estimate cost for a single order.

        Args:
            symbol:     Ticker symbol.
            notional:   Dollar notional of the order.
            adv:        Average daily dollar volume.  Required for impact.
            volatility: Annualised volatility.  Used if volatility_scaling
                        is enabled.
            price:      Price per share (for per-share commission calc).
            quantity:   Number of shares (for per-share commission calc).

        Returns:
            :class:`CostEstimate` with component breakdown.
        """
        if notional <= 0:
            return CostEstimate(
                symbol=symbol,
                notional=0.0,
                spread_bps=0.0,
                impact_bps=0.0,
                commission_bps=0.0,
                total_bps=0.0,
                total_dollars=0.0,
            )

        spread = self._spread_cost(symbol)
        impact = self._impact_cost(notional, adv, volatility)
        commission = self._commission_cost(notional, price, quantity)

        total_bps = spread + impact + commission
        total_dollars = notional * total_bps / 10_000

        return CostEstimate(
            symbol=symbol,
            notional=notional,
            spread_bps=spread,
            impact_bps=impact,
            commission_bps=commission,
            total_bps=total_bps,
            total_dollars=total_dollars,
        )

    # ── Rebalance cost ────────────────────────────────────────────

    def estimate_rebalance_cost(
        self,
        weight_changes: dict[str, float],
        portfolio_value: float,
        adv: dict[str, float] | None = None,
        volatility: dict[str, float] | None = None,
        prices: dict[str, float] | None = None,
    ) -> RebalanceCostEstimate:
        """Estimate total cost for a portfolio rebalance.

        Args:
            weight_changes:  ``{symbol: abs_weight_change}`` for each asset
                being traded.
            portfolio_value: Current portfolio value in dollars.
            adv:             ``{symbol: average_daily_volume}`` in dollars.
            volatility:      ``{symbol: annualised_volatility}``.
            prices:          ``{symbol: price_per_share}``.

        Returns:
            :class:`RebalanceCostEstimate` with per-asset and aggregate costs.
        """
        adv = adv or {}
        volatility = volatility or {}
        prices = prices or {}

        per_asset: list[CostEstimate] = []
        total_dollars = 0.0
        turnover = 0.0

        for symbol, abs_dw in weight_changes.items():
            notional = abs(abs_dw) * portfolio_value
            turnover += abs(abs_dw)

            px = prices.get(symbol)
            qty = notional / px if px and px > 0 else None

            est = self.estimate_order_cost(
                symbol=symbol,
                notional=notional,
                adv=adv.get(symbol),
                volatility=volatility.get(symbol),
                price=px,
                quantity=qty,
            )
            per_asset.append(est)
            total_dollars += est.total_dollars

        # Portfolio-level bps = total_dollars / portfolio_value * 10_000
        total_bps = total_dollars / portfolio_value * 10_000 if portfolio_value > 0 else 0.0

        # Decompose by component
        total_spread_dollars = sum(
            e.notional * e.spread_bps / 10_000 for e in per_asset
        )
        total_impact_dollars = sum(
            e.notional * e.impact_bps / 10_000 for e in per_asset
        )
        total_comm_dollars = sum(
            e.notional * e.commission_bps / 10_000 for e in per_asset
        )

        if portfolio_value > 0:
            spread_bps = total_spread_dollars / portfolio_value * 10_000
            impact_bps = total_impact_dollars / portfolio_value * 10_000
            comm_bps = total_comm_dollars / portfolio_value * 10_000
        else:
            spread_bps = impact_bps = comm_bps = 0.0

        return RebalanceCostEstimate(
            per_asset=per_asset,
            total_spread_bps=spread_bps,
            total_impact_bps=impact_bps,
            total_commission_bps=comm_bps,
            total_cost_bps=total_bps,
            total_cost_dollars=total_dollars,
            portfolio_value=portfolio_value,
            turnover=turnover,
        )

    # ── Break-even analysis ───────────────────────────────────────

    def break_even_alpha(
        self,
        symbol: str,
        notional: float,
        adv: float | None = None,
        volatility: float | None = None,
        holding_period_days: int = 1,
    ) -> float:
        """Minimum alpha (in bps) required to justify a round-trip trade.

        Args:
            symbol:              Ticker symbol.
            notional:            Dollar notional.
            adv:                 Average daily dollar volume.
            volatility:          Annualised volatility.
            holding_period_days: Expected holding period.

        Returns:
            Break-even alpha in basis points (round-trip cost / holding period).
        """
        one_way = self.estimate_order_cost(
            symbol=symbol, notional=notional, adv=adv, volatility=volatility
        )
        round_trip_bps = one_way.total_bps * 2.0
        if holding_period_days > 0:
            return round_trip_bps / holding_period_days
        return round_trip_bps

    # ── Flat bps equivalent ───────────────────────────────────────

    def as_flat_bps(
        self,
        symbol: str,
        notional: float = 100_000,
        adv: float | None = None,
        volatility: float | None = None,
    ) -> float:
        """Return the total one-way cost as a single flat bps number.

        Useful for backwards-compatible integration with engines that
        accept a flat ``commission_bps`` parameter.
        """
        est = self.estimate_order_cost(
            symbol=symbol, notional=notional, adv=adv, volatility=volatility
        )
        return est.total_bps

    # ── Component helpers ─────────────────────────────────────────

    def _spread_cost(self, symbol: str) -> float:
        """Half-spread cost in basis points."""
        return self._config.spread_overrides.get(
            symbol, self._config.default_spread_bps
        )

    def _impact_cost(
        self,
        notional: float,
        adv: float | None,
        volatility: float | None,
    ) -> float:
        """Market impact in basis points using square-root model.

        impact = η · σ_daily · (Q / ADV)^β · 10_000

        where σ_daily = σ_annual / sqrt(252).
        """
        if adv is None or adv <= 0:
            return 0.0

        participation = notional / adv
        if participation <= 0:
            return 0.0

        eta = self._config.impact_coefficient
        beta = self._config.impact_exponent

        if self._config.volatility_scaling and volatility is not None and volatility > 0:
            sigma_daily = volatility / math.sqrt(self._config.annualisation)
        else:
            sigma_daily = 1.0  # unitless when vol not available

        impact = eta * sigma_daily * (participation ** beta) * 10_000
        return impact

    def _commission_cost(
        self,
        notional: float,
        price: float | None,
        quantity: float | None,
    ) -> float:
        """Commission cost in basis points."""
        comm_dollars = 0.0

        # Percentage commission
        if self._config.commission_pct > 0:
            comm_dollars += notional * self._config.commission_pct

        # Per-share commission
        if self._config.commission_per_share > 0 and quantity is not None:
            comm_dollars += abs(quantity) * self._config.commission_per_share

        # Minimum commission
        if comm_dollars < self._config.min_commission:
            comm_dollars = self._config.min_commission

        if notional > 0:
            return comm_dollars / notional * 10_000
        return 0.0
