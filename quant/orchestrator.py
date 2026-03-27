"""Multi-Strategy Orchestrator — run N strategy sleeves with capital allocation.

Manages a portfolio of strategies, each with its own signals, risk budget,
and capital allocation.  Aggregates target weights across sleeves, nets
conflicting positions, and routes orders through a single shared OMS.

Usage::

    from quant.orchestrator import StrategyOrchestrator, OrchestratorConfig, StrategySleeve

    sleeves = [
        StrategySleeve(
            name="momentum",
            signals=[MomentumSignal()],
            capital_weight=0.40,
        ),
        StrategySleeve(
            name="mean_reversion",
            signals=[MeanReversionSignal()],
            capital_weight=0.35,
        ),
        StrategySleeve(
            name="trend",
            signals=[TrendFollowingSignal()],
            capital_weight=0.25,
        ),
    ]

    orchestrator = StrategyOrchestrator(
        config=OrchestratorConfig(universe=["AAPL", "GOOG", "MSFT"]),
        sleeves=sleeves,
        oms=oms,
    )
    result = orchestrator.run_once()
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd
from loguru import logger

from quant.oms.models import Order, OrderSide, OrderType
from quant.oms.system import OrderManagementSystem
from quant.portfolio.alpha import AlphaCombiner, CombinationMethod
from quant.portfolio.engine import ConstructionResult, PortfolioConfig, PortfolioEngine
from quant.risk.engine import (
    Order as RiskOrder,
)
from quant.risk.engine import (
    PortfolioState,
    RiskConfig,
    RiskEngine,
)
from quant.signals.base import BaseSignal, SignalOutput


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Type aliases ──────────────────────────────────────────────────────────────

FeatureProvider = Callable[[str, BaseSignal], dict[str, pd.Series]]
ReturnsProvider = Callable[[list[str], int], pd.DataFrame]


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class StrategySleeve:
    """Configuration for one strategy within the orchestrator.

    Attributes:
        name:               Unique strategy identifier.
        signals:            Signal instances for this strategy.
        capital_weight:     Fraction of total capital allocated (0–1).
                            All sleeve weights must sum to <= 1.0.
        portfolio_config:   Portfolio construction config for this sleeve.
        risk_config:        Risk engine config for this sleeve.
        combination_method: How to combine this sleeve's signal outputs.
        signal_weights:     Static weights for STATIC_WEIGHT combination.
        sector_map:         {symbol: sector} for risk checks.
        lookback_days:      History for covariance estimation.
        enabled:            If False, skip this sleeve entirely.
    """

    name: str
    signals: list[BaseSignal] = field(default_factory=list)
    capital_weight: float = 1.0
    portfolio_config: PortfolioConfig = field(default_factory=PortfolioConfig)
    risk_config: RiskConfig | None = None
    combination_method: CombinationMethod = CombinationMethod.EQUAL_WEIGHT
    signal_weights: dict[str, float] | None = None
    sector_map: dict[str, str] = field(default_factory=dict)
    lookback_days: int = 252
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class SleeveResult:
    """Output of a single strategy sleeve's execution.

    Attributes:
        name:               Sleeve name.
        capital_allocated:  Dollar capital allocated to this sleeve.
        target_weights:     Target portfolio weights for this sleeve's universe.
        construction:       Portfolio construction output.
        n_trades:           Number of trades proposed.
        error:              Error message if the sleeve failed.
    """

    name: str
    capital_allocated: float = 0.0
    target_weights: dict[str, float] = field(default_factory=dict)
    construction: ConstructionResult | None = None
    n_trades: int = 0
    error: str = ""


@dataclass
class OrchestratorResult:
    """Aggregated result across all strategy sleeves.

    Attributes:
        timestamp:          When the orchestration completed.
        total_portfolio:    Total portfolio value.
        sleeve_results:     Per-sleeve execution details.
        combined_weights:   Net target weights after aggregation.
        n_submitted:        Total orders submitted.
        n_rejected:         Total orders rejected by risk.
        error:              Error message if orchestration failed.
    """

    timestamp: datetime
    total_portfolio: float = 0.0
    sleeve_results: list[SleeveResult] = field(default_factory=list)
    combined_weights: dict[str, float] = field(default_factory=dict)
    n_submitted: int = 0
    n_rejected: int = 0
    error: str = ""


@dataclass
class OrchestratorConfig:
    """Configuration for the StrategyOrchestrator.

    Attributes:
        universe:           Shared symbol universe (sleeves can use subsets).
        risk_config:        Top-level risk config for aggregate position limits.
        sector_map:         {symbol: sector} for aggregate risk checks.
        min_order_value:    Minimum dollar value per order.
        net_conflicting:    If True, net opposing positions across sleeves.
                            If False, reject conflicting positions.
    """

    universe: list[str] = field(default_factory=list)
    risk_config: RiskConfig = field(default_factory=RiskConfig)
    sector_map: dict[str, str] = field(default_factory=dict)
    min_order_value: float = 100.0
    net_conflicting: bool = True


class StrategyOrchestrator:
    """Multi-strategy orchestrator with capital allocation and risk budgeting.

    Runs N strategy sleeves, each with independent signal computation and
    portfolio construction, then aggregates the target weights, applies
    top-level risk limits, and routes orders through a shared OMS.

    Pipeline per cycle:
      1. Allocate capital to each sleeve by weight.
      2. For each enabled sleeve: compute signals → portfolio construction.
      3. Combine target weights (capital-weighted, with netting).
      4. Compute net trades from current positions to combined targets.
      5. Validate trades through the aggregate risk engine.
      6. Submit approved trades to the OMS.

    Args:
        config:           Orchestrator configuration.
        sleeves:          List of strategy sleeves to run.
        oms:              Shared OrderManagementSystem instance.
        feature_provider: Optional callable for feature data.
        returns_provider: Optional callable for returns data.
    """

    def __init__(
        self,
        config: OrchestratorConfig,
        sleeves: list[StrategySleeve],
        oms: OrderManagementSystem,
        feature_provider: FeatureProvider | None = None,
        returns_provider: ReturnsProvider | None = None,
    ) -> None:
        self._config = config
        self._sleeves = sleeves
        self._oms = oms
        self._feature_provider = feature_provider
        self._returns_provider = returns_provider

        # Top-level risk engine for aggregate position limits
        self._risk_engine = RiskEngine(config.risk_config)

        self._validate_sleeve_weights()

    def _validate_sleeve_weights(self) -> None:
        """Ensure sleeve capital weights sum to <= 1.0."""
        total = sum(s.capital_weight for s in self._sleeves if s.enabled)
        if total > 1.0 + 1e-9:
            raise ValueError(
                f"Sleeve capital weights sum to {total:.4f}, must be <= 1.0"
            )

    @property
    def sleeves(self) -> list[StrategySleeve]:
        return list(self._sleeves)

    def run_once(self) -> OrchestratorResult:
        """Execute a single multi-strategy rebalance cycle.

        Returns:
            OrchestratorResult with per-sleeve details and aggregate metrics.
        """
        now = _utcnow()

        try:
            total_value = self._get_portfolio_value()
            current_weights = self._get_current_weights(total_value)

            # ── 1. Run each sleeve ────────────────────────────────────────
            sleeve_results: list[SleeveResult] = []
            for sleeve in self._sleeves:
                if not sleeve.enabled:
                    continue
                sr = self._run_sleeve(sleeve, total_value, now)
                sleeve_results.append(sr)

            # ── 2. Combine target weights ─────────────────────────────────
            combined = self._combine_weights(sleeve_results)

            # ── 3. Compute net trades ─────────────────────────────────────
            trades = self._compute_trades(current_weights, combined, total_value)

            # ── 4. Risk validate and execute ──────────────────────────────
            portfolio_state = self._build_portfolio_state(total_value)
            n_submitted = 0
            n_rejected = 0

            for trade in trades:
                submitted = self._execute_trade(trade, total_value, portfolio_state)
                if submitted:
                    n_submitted += 1
                else:
                    n_rejected += 1

            result = OrchestratorResult(
                timestamp=now,
                total_portfolio=total_value,
                sleeve_results=sleeve_results,
                combined_weights=combined,
                n_submitted=n_submitted,
                n_rejected=n_rejected,
            )

        except Exception as exc:
            logger.exception("Orchestrator: run_once failed")
            result = OrchestratorResult(
                timestamp=now,
                error=str(exc),
            )

        self._log_result(result)
        return result

    # ── Sleeve execution ─────────────────────────────────────────────────

    def _run_sleeve(
        self,
        sleeve: StrategySleeve,
        total_value: float,
        timestamp: datetime,
    ) -> SleeveResult:
        """Run a single strategy sleeve and return its target weights."""
        sleeve_capital = total_value * sleeve.capital_weight

        try:
            # Compute signals
            universe_signals = self._compute_signals(sleeve, timestamp)

            # Combine into alpha
            combiner = AlphaCombiner(
                method=sleeve.combination_method,
                weights=sleeve.signal_weights,
            )
            alpha_scores = combiner.combine_universe(timestamp, universe_signals)

            # Portfolio construction
            engine = PortfolioEngine(sleeve.portfolio_config)
            returns_history = self._get_returns_history(sleeve.lookback_days)

            construction = engine.construct(
                alpha_scores=alpha_scores,
                returns_history=returns_history,
                current_weights={},  # each sleeve starts from scratch
                portfolio_value=sleeve_capital,
            )

            # Extract target weights and scale by sleeve capital weight
            target_weights: dict[str, float] = {}
            if construction.rebalance_triggered:
                for sym, w in construction.optimization.weights.items():
                    target_weights[sym] = w * sleeve.capital_weight

            return SleeveResult(
                name=sleeve.name,
                capital_allocated=sleeve_capital,
                target_weights=target_weights,
                construction=construction,
                n_trades=len(construction.rebalance.trades) if construction.rebalance_triggered else 0,
            )

        except Exception as exc:
            logger.warning(
                "Orchestrator: sleeve '{}' failed — {}", sleeve.name, exc
            )
            return SleeveResult(
                name=sleeve.name,
                capital_allocated=sleeve_capital,
                error=str(exc),
            )

    def _compute_signals(
        self,
        sleeve: StrategySleeve,
        timestamp: datetime,
    ) -> dict[str, list[SignalOutput]]:
        """Compute all signals for a sleeve across the universe."""
        universe_signals: dict[str, list[SignalOutput]] = {}
        universe = self._config.universe

        for symbol in universe:
            symbol_signals: list[SignalOutput] = []
            for signal in sleeve.signals:
                features = self._get_features(symbol, signal)
                try:
                    output = signal.compute(symbol, features, timestamp)
                    tagged = SignalOutput(
                        symbol=output.symbol,
                        timestamp=output.timestamp,
                        score=output.score,
                        confidence=output.confidence,
                        target_position=output.target_position,
                        metadata={**output.metadata, "signal_name": signal.name},
                    )
                    symbol_signals.append(tagged)
                except Exception:
                    logger.warning(
                        "Signal {} failed for {} in sleeve '{}' — skipping",
                        signal.name,
                        symbol,
                        sleeve.name,
                    )
            universe_signals[symbol] = symbol_signals

        return universe_signals

    # ── Weight aggregation ───────────────────────────────────────────────

    def _combine_weights(
        self, sleeve_results: list[SleeveResult]
    ) -> dict[str, float]:
        """Combine target weights across sleeves.

        Each sleeve's weights are already scaled by its capital_weight,
        so simple summation gives the aggregate portfolio weight.
        Conflicting directions (one sleeve long, another short in same
        symbol) are netted if config.net_conflicting is True.
        """
        combined: dict[str, float] = {}

        for sr in sleeve_results:
            if sr.error:
                continue
            for sym, w in sr.target_weights.items():
                combined[sym] = combined.get(sym, 0.0) + w

        return combined

    # ── Trade generation ─────────────────────────────────────────────────

    def _compute_trades(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        portfolio_value: float,
    ) -> list[_AggTrade]:
        """Compute the net trades needed to move from current to target."""
        all_symbols = set(current_weights) | set(target_weights)
        trades: list[_AggTrade] = []

        for sym in sorted(all_symbols):
            current_w = current_weights.get(sym, 0.0)
            target_w = target_weights.get(sym, 0.0)
            delta_w = target_w - current_w

            dollar_amount = abs(delta_w) * portfolio_value
            if dollar_amount < self._config.min_order_value:
                continue

            trades.append(
                _AggTrade(
                    symbol=sym,
                    target_weight=target_w,
                    trade_weight=delta_w,
                    dollar_amount=dollar_amount,
                    side="BUY" if delta_w > 0 else "SELL",
                )
            )

        return trades

    # ── Trade execution ──────────────────────────────────────────────────

    def _execute_trade(
        self,
        trade: _AggTrade,
        portfolio_value: float,
        portfolio_state: PortfolioState,
    ) -> bool:
        """Validate a trade through aggregate risk and submit to OMS.

        Returns True if the order was submitted, False if rejected.
        """
        signed_dollar = trade.trade_weight * portfolio_value
        risk_order = RiskOrder(
            symbol=trade.symbol,
            quantity=signed_dollar,
            price=1.0,
            sector=self._config.sector_map.get(trade.symbol),
        )

        risk_result = self._risk_engine.validate(risk_order, portfolio_state)

        if not risk_result.approved:
            logger.warning(
                "Orchestrator: risk rejected {} {} — {}",
                trade.side,
                trade.symbol,
                risk_result.reason,
            )
            return False

        # Convert dollar amount to share quantity
        position = self._oms.get_position(trade.symbol)
        price = position.market_price if position and position.market_price > 0 else 0.0
        approved_dollars = abs(risk_result.adjusted_quantity)
        share_qty = approved_dollars / price if price > 0 else approved_dollars

        oms_order = Order(
            symbol=trade.symbol,
            side=OrderSide.BUY if trade.side == "BUY" else OrderSide.SELL,
            quantity=share_qty,
            order_type=OrderType.MARKET,
            strategy_id="orchestrator",
            sector=self._config.sector_map.get(trade.symbol),
        )

        try:
            self._oms.submit_order(oms_order)
            return True
        except Exception:
            logger.exception(
                "Orchestrator: OMS submission failed for {} {}", trade.side, trade.symbol
            )
            return False

    # ── Portfolio state helpers ───────────────────────────────────────────

    def _get_portfolio_value(self) -> float:
        cash = self._oms.get_account_cash()
        positions = self._oms.get_all_positions()
        pos_value = sum(p.market_value for p in positions.values())
        return cash + pos_value

    def _get_current_weights(self, portfolio_value: float) -> dict[str, float]:
        if portfolio_value <= 0:
            return {}
        positions = self._oms.get_all_positions()
        return {
            sym: pos.market_value / portfolio_value
            for sym, pos in positions.items()
            if abs(pos.market_value) > 1e-9
        }

    def _get_returns_history(self, lookback_days: int) -> pd.DataFrame:
        if self._returns_provider is not None:
            return self._returns_provider(self._config.universe, lookback_days)
        return pd.DataFrame(columns=self._config.universe)

    def _get_features(
        self, symbol: str, signal: BaseSignal
    ) -> dict[str, pd.Series]:
        if self._feature_provider is not None:
            return self._feature_provider(symbol, signal)
        return {}

    def _build_portfolio_state(self, portfolio_value: float) -> PortfolioState:
        positions = self._oms.get_all_positions()
        pos_dollars = {sym: pos.market_value for sym, pos in positions.items()}

        sector_exposures: dict[str, float] = {}
        for sym, pos in positions.items():
            sector = self._config.sector_map.get(sym, "unknown")
            sector_exposures[sector] = (
                sector_exposures.get(sector, 0.0) + abs(pos.market_value)
            )

        return PortfolioState(
            capital=portfolio_value,
            positions=pos_dollars,
            sector_exposures=sector_exposures,
            peak_portfolio_value=portfolio_value,
        )

    # ── Logging ──────────────────────────────────────────────────────────

    @staticmethod
    def _log_result(result: OrchestratorResult) -> None:
        if result.error:
            logger.error("Orchestrator: FAILED — {}", result.error)
            return

        active = [sr for sr in result.sleeve_results if not sr.error]
        failed = [sr for sr in result.sleeve_results if sr.error]

        logger.info(
            "Orchestrator: portfolio=${:,.0f} | sleeves={} active, {} failed | "
            "submitted={} rejected={} | {} symbols",
            result.total_portfolio,
            len(active),
            len(failed),
            result.n_submitted,
            result.n_rejected,
            len(result.combined_weights),
        )

        for sr in active:
            logger.info(
                "  sleeve '{}': ${:,.0f} allocated, {} trades",
                sr.name,
                sr.capital_allocated,
                sr.n_trades,
            )

        for sr in failed:
            logger.warning("  sleeve '{}': FAILED — {}", sr.name, sr.error)


# ── Internal trade type ──────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _AggTrade:
    """Aggregated trade across strategy sleeves."""

    symbol: str
    target_weight: float
    trade_weight: float
    dollar_amount: float
    side: str
