"""Rebalancing engine: target portfolio → trade list.

Converts a target weight vector into a list of trades needed to move from
the current portfolio to the target, respecting turnover constraints and
producing executable orders compatible with the OMS.
"""
from __future__ import annotations

from dataclasses import dataclass

from quant.portfolio.constraints import PortfolioConstraints


@dataclass(frozen=True, slots=True)
class Trade:
    """A single trade required to move towards the target portfolio.

    Attributes:
        symbol:         Asset identifier.
        current_weight: Current portfolio weight.
        target_weight:  Desired portfolio weight after rebalance.
        trade_weight:   Weight delta to execute (positive = buy, negative = sell).
        dollar_amount:  Approximate dollar value of the trade.
        side:           "BUY" or "SELL".
    """

    symbol: str
    current_weight: float
    target_weight: float
    trade_weight: float
    dollar_amount: float
    side: str


@dataclass(frozen=True, slots=True)
class RebalanceResult:
    """Output of a rebalance computation.

    Attributes:
        trades:            List of trades to execute.
        target_weights:    Final target weights (post-constraint adjustment).
        turnover:          Total one-way turnover (sum of |trade_weight|).
        n_buys:            Number of buy trades.
        n_sells:           Number of sell trades.
        turnover_capped:   True if turnover was reduced to meet constraint.
    """

    trades: list[Trade]
    target_weights: dict[str, float]
    turnover: float
    n_buys: int
    n_sells: int
    turnover_capped: bool


class RebalanceEngine:
    """Converts target portfolio weights into an executable trade list.

    Handles:
    - Dead-band filtering (skip trades below a minimum threshold)
    - Turnover budgeting (proportionally scale trades if over budget)
    - Dollar amount estimation for OMS submission

    Usage::

        engine = RebalanceEngine(min_trade_weight=0.005)
        result = engine.rebalance(
            current_weights={"AAPL": 0.3, "GOOG": 0.2},
            target_weights={"AAPL": 0.25, "GOOG": 0.25, "MSFT": 0.1},
            portfolio_value=1_000_000,
            constraints=PortfolioConstraints(max_turnover=0.3),
        )
        for trade in result.trades:
            print(f"{trade.side} {trade.symbol}: ${trade.dollar_amount:,.0f}")

    Args:
        min_trade_weight: Minimum absolute trade weight to execute (dead band).
            Trades smaller than this are dropped to avoid excessive churn.
    """

    def __init__(self, min_trade_weight: float = 0.001) -> None:
        self._min_trade_weight = min_trade_weight

    def rebalance(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        portfolio_value: float,
        constraints: PortfolioConstraints | None = None,
    ) -> RebalanceResult:
        """Compute trades to move from current to target weights.

        Args:
            current_weights:  {symbol: weight} of current portfolio.
            target_weights:   {symbol: weight} of desired portfolio.
            portfolio_value:  Total portfolio value in dollars.
            constraints:      Optional constraints (primarily for turnover budget).

        Returns:
            RebalanceResult with trade list and summary statistics.
        """
        all_symbols = sorted(set(current_weights) | set(target_weights))

        # Compute raw deltas
        deltas: dict[str, float] = {}
        for sym in all_symbols:
            curr = current_weights.get(sym, 0.0)
            tgt = target_weights.get(sym, 0.0)
            delta = tgt - curr
            deltas[sym] = delta

        raw_turnover = sum(abs(d) for d in deltas.values())

        # Apply turnover budget if constrained
        turnover_capped = False
        if constraints and constraints.max_turnover is not None and raw_turnover > constraints.max_turnover + 1e-9:
            scale = constraints.max_turnover / raw_turnover
            deltas = {sym: d * scale for sym, d in deltas.items()}
            turnover_capped = True

        # Build trade list, filtering by dead band
        trades: list[Trade] = []
        final_weights: dict[str, float] = {}

        for sym in all_symbols:
            delta = deltas[sym]
            curr = current_weights.get(sym, 0.0)
            actual_target = curr + delta

            final_weights[sym] = actual_target

            if abs(delta) < self._min_trade_weight:
                continue

            trades.append(
                Trade(
                    symbol=sym,
                    current_weight=curr,
                    target_weight=actual_target,
                    trade_weight=delta,
                    dollar_amount=abs(delta) * portfolio_value,
                    side="BUY" if delta > 0 else "SELL",
                )
            )

        # Remove zero-weight positions from final weights
        final_weights = {s: w for s, w in final_weights.items() if abs(w) > 1e-9}

        actual_turnover = sum(abs(t.trade_weight) for t in trades)
        n_buys = sum(1 for t in trades if t.side == "BUY")
        n_sells = sum(1 for t in trades if t.side == "SELL")

        return RebalanceResult(
            trades=trades,
            target_weights=final_weights,
            turnover=actual_turnover,
            n_buys=n_buys,
            n_sells=n_sells,
            turnover_capped=turnover_capped,
        )
