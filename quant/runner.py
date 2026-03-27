"""Strategy Runner — live orchestration loop.

Ties all platform modules together into a single execution pipeline:

  Features → Signals → Alpha Combiner → Portfolio Optimizer →
  Risk Validation → OMS Execution → Position Tracking

Supports two modes:
  - **Live**: runs on a schedule (e.g. daily at market close), pulling real
    market data and submitting orders through the configured broker adapter.
  - **Paper**: identical pipeline but routed through PaperBrokerAdapter for
    simulation and validation before going live.

Usage::

    from quant.runner import StrategyRunner, RunnerConfig

    config = RunnerConfig(
        universe=["AAPL", "GOOG", "MSFT", "JPM", "XOM"],
        signals=[MomentumSignal(), MeanReversionSignal(), TrendFollowingSignal()],
        portfolio_config=PortfolioConfig(
            optimization_method=OptimizationMethod.RISK_PARITY,
            constraints=PortfolioConstraints(long_only=True, max_weight=0.25),
        ),
        risk_config=RiskConfig(
            limits=ExposureLimits(max_position_fraction=0.25),
        ),
    )
    runner = StrategyRunner(config, oms=oms)
    runner.run_once()  # single rebalance pass
"""
from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd
from loguru import logger

from quant.oms.models import Order, OrderSide, OrderType
from quant.oms.system import OrderManagementSystem
from quant.portfolio.alpha import AlphaCombiner, CombinationMethod
from quant.portfolio.engine import ConstructionResult, PortfolioConfig, PortfolioEngine
from quant.portfolio.rebalancer import Trade
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


class RunnerState(enum.Enum):
    IDLE = "idle"
    COMPUTING_SIGNALS = "computing_signals"
    CONSTRUCTING_PORTFOLIO = "constructing_portfolio"
    VALIDATING_RISK = "validating_risk"
    EXECUTING = "executing"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    """Record of a single trade execution attempt.

    Attributes:
        symbol:         Asset identifier.
        side:           BUY or SELL.
        target_weight:  Target portfolio weight.
        dollar_amount:  Target dollar amount.
        risk_approved:  Whether the risk engine approved the trade.
        risk_reason:    Reason for risk rejection (empty if approved).
        order_id:       OMS order ID if submitted, None if rejected.
    """

    symbol: str
    side: str
    target_weight: float
    dollar_amount: float
    risk_approved: bool
    risk_reason: str
    order_id: str | None


@dataclass
class RunResult:
    """Output of a single runner execution pass.

    Attributes:
        timestamp:      When the run completed.
        state:          Final runner state.
        construction:   Portfolio construction output (if successful).
        executions:     Per-trade execution records.
        n_submitted:    Number of orders submitted to OMS.
        n_rejected:     Number of orders rejected by risk engine.
        portfolio_value: Total portfolio value at run time.
        error:          Error message if the run failed.
    """

    timestamp: datetime
    state: RunnerState
    construction: ConstructionResult | None = None
    executions: list[ExecutionRecord] = field(default_factory=list)
    n_submitted: int = 0
    n_rejected: int = 0
    portfolio_value: float = 0.0
    error: str = ""


@dataclass
class RunnerConfig:
    """Configuration for the StrategyRunner.

    Attributes:
        universe:           List of symbols to trade.
        signals:            List of signal instances to evaluate.
        portfolio_config:   Portfolio construction configuration.
        risk_config:        Risk engine configuration.
        combination_method: How to combine signal outputs into alpha.
        signal_weights:     Static weights for STATIC_WEIGHT combination.
        sector_map:         {symbol: sector} for risk and constraint checks.
        lookback_days:      Number of trading days of history for covariance.
        min_order_value:    Minimum dollar value per order (skip smaller trades).
    """

    universe: list[str] = field(default_factory=list)
    signals: list[BaseSignal] = field(default_factory=list)
    portfolio_config: PortfolioConfig = field(default_factory=PortfolioConfig)
    risk_config: RiskConfig = field(default_factory=RiskConfig)
    combination_method: CombinationMethod = CombinationMethod.EQUAL_WEIGHT
    signal_weights: dict[str, float] | None = None
    sector_map: dict[str, str] = field(default_factory=dict)
    lookback_days: int = 252
    min_order_value: float = 100.0


class StrategyRunner:
    """Live strategy execution orchestrator.

    Connects signals, portfolio construction, risk management, and order
    management into a single pipeline.  Each call to :meth:`run_once`
    performs a full rebalance cycle.

    The runner does NOT schedule itself — the caller is responsible for
    invoking ``run_once()`` at the desired frequency (e.g. via cron,
    APScheduler, or a simple while-loop).

    Args:
        config: Runner configuration.
        oms:    Initialised and started OrderManagementSystem.
        feature_provider: Optional callable that returns feature data for
            signals.  Signature: ``(symbol, signal) -> dict[str, pd.Series]``.
            If None, signals receive empty feature dicts (useful when signals
            are self-contained or receive data through other means).
        returns_provider: Optional callable that returns a DataFrame of
            daily asset returns for the universe.
            Signature: ``(symbols, lookback_days) -> pd.DataFrame``.
            Required for portfolio optimisation (covariance estimation).
    """

    def __init__(
        self,
        config: RunnerConfig,
        oms: OrderManagementSystem,
        feature_provider: FeatureProvider | None = None,
        returns_provider: ReturnsProvider | None = None,
    ) -> None:
        self._config = config
        self._oms = oms
        self._feature_provider = feature_provider
        self._returns_provider = returns_provider

        self._alpha_combiner = AlphaCombiner(
            method=config.combination_method,
            weights=config.signal_weights,
        )
        self._portfolio_engine = PortfolioEngine(config.portfolio_config)
        self._risk_engine = RiskEngine(config.risk_config)

        self._state = RunnerState.IDLE
        self._last_result: RunResult | None = None

    @property
    def state(self) -> RunnerState:
        return self._state

    @property
    def last_result(self) -> RunResult | None:
        return self._last_result

    def run_once(self) -> RunResult:
        """Execute a single full rebalance cycle.

        Steps:
        1. Compute signals for all symbols in the universe.
        2. Combine signal outputs into composite alpha scores.
        3. Build the return history and run portfolio optimisation.
        4. Validate each proposed trade through the risk engine.
        5. Submit approved trades to the OMS.

        Returns:
            RunResult with execution details and diagnostics.
        """
        now = _utcnow()

        try:
            # ── 1. Compute signals ────────────────────────────────────────
            self._state = RunnerState.COMPUTING_SIGNALS
            universe_signals = self._compute_signals(now)

            # ── 2. Combine into alpha scores ──────────────────────────────
            alpha_scores = self._alpha_combiner.combine_universe(
                now, universe_signals
            )

            # ── 3. Portfolio construction ─────────────────────────────────
            self._state = RunnerState.CONSTRUCTING_PORTFOLIO
            portfolio_value = self._get_portfolio_value()
            current_weights = self._get_current_weights(portfolio_value)

            returns_history = self._get_returns_history()
            construction = self._portfolio_engine.construct(
                alpha_scores=alpha_scores,
                returns_history=returns_history,
                current_weights=current_weights,
                portfolio_value=portfolio_value,
            )

            # ── 4. Risk validation + 5. Execution ────────────────────────
            self._state = RunnerState.VALIDATING_RISK
            executions: list[ExecutionRecord] = []
            n_submitted = 0
            n_rejected = 0

            if construction.rebalance_triggered:
                self._state = RunnerState.EXECUTING
                portfolio_snapshot = self._build_portfolio_state(portfolio_value)

                for trade in construction.rebalance.trades:
                    record = self._execute_trade(
                        trade, portfolio_value, portfolio_snapshot
                    )
                    executions.append(record)
                    if record.risk_approved and record.order_id:
                        n_submitted += 1
                    elif not record.risk_approved:
                        n_rejected += 1

            self._state = RunnerState.IDLE
            result = RunResult(
                timestamp=now,
                state=RunnerState.IDLE,
                construction=construction,
                executions=executions,
                n_submitted=n_submitted,
                n_rejected=n_rejected,
                portfolio_value=portfolio_value,
            )

        except Exception as exc:
            self._state = RunnerState.ERROR
            logger.exception("StrategyRunner: run_once failed")
            result = RunResult(
                timestamp=now,
                state=RunnerState.ERROR,
                error=str(exc),
            )

        self._last_result = result
        self._log_result(result)
        return result

    # ── Private: Signal computation ───────────────────────────────────────

    def _compute_signals(
        self, timestamp: datetime
    ) -> dict[str, list[SignalOutput]]:
        """Run all configured signals for each symbol in the universe."""
        universe_signals: dict[str, list[SignalOutput]] = {}

        for symbol in self._config.universe:
            symbol_signals: list[SignalOutput] = []

            for signal in self._config.signals:
                features = self._get_features(symbol, signal)
                try:
                    output = signal.compute(symbol, features, timestamp)
                    # Tag the signal name into metadata for the combiner
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
                        "Signal {} failed for {} — skipping",
                        signal.name,
                        symbol,
                    )

            universe_signals[symbol] = symbol_signals

        return universe_signals

    def _get_features(
        self, symbol: str, signal: BaseSignal
    ) -> dict[str, pd.Series]:
        """Get features for a signal, using the feature provider if available."""
        if self._feature_provider is not None:
            return self._feature_provider(symbol, signal)
        return {}

    # ── Private: Portfolio state ──────────────────────────────────────────

    def _get_portfolio_value(self) -> float:
        """Compute total portfolio value: cash + sum of position market values."""
        cash = self._oms.get_account_cash()
        positions = self._oms.get_all_positions()
        pos_value = sum(p.market_value for p in positions.values())
        return cash + pos_value

    def _get_current_weights(
        self, portfolio_value: float
    ) -> dict[str, float]:
        """Convert current OMS positions to weight-space."""
        if portfolio_value <= 0:
            return {}
        positions = self._oms.get_all_positions()
        return {
            sym: pos.market_value / portfolio_value
            for sym, pos in positions.items()
            if abs(pos.market_value) > 1e-9
        }

    def _get_returns_history(self) -> pd.DataFrame:
        """Get historical returns for covariance estimation."""
        if self._returns_provider is not None:
            return self._returns_provider(
                self._config.universe, self._config.lookback_days
            )
        # Fallback: return empty DataFrame (optimizer will raise if needed)
        return pd.DataFrame(columns=self._config.universe)

    def _build_portfolio_state(self, portfolio_value: float) -> PortfolioState:
        """Build a PortfolioState snapshot from OMS positions."""
        positions = self._oms.get_all_positions()
        pos_dollars = {
            sym: pos.market_value for sym, pos in positions.items()
        }

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

    # ── Private: Trade execution ──────────────────────────────────────────

    def _execute_trade(
        self,
        trade: Trade,
        portfolio_value: float,
        portfolio_state: PortfolioState,
    ) -> ExecutionRecord:
        """Validate a single trade through risk and submit to OMS if approved."""
        if trade.dollar_amount < self._config.min_order_value:
            return ExecutionRecord(
                symbol=trade.symbol,
                side=trade.side,
                target_weight=trade.target_weight,
                dollar_amount=trade.dollar_amount,
                risk_approved=False,
                risk_reason=f"Below minimum order value ${self._config.min_order_value}",
                order_id=None,
            )

        # Risk order uses dollar amounts: quantity=dollar_amount, price=1.0
        # so that RiskOrder.dollar_value = dollar_amount, matching risk limits.
        signed_dollar = trade.trade_weight * portfolio_value
        risk_order = RiskOrder(
            symbol=trade.symbol,
            quantity=signed_dollar,
            price=1.0,
            sector=self._config.sector_map.get(trade.symbol),
        )

        # Risk validation
        risk_result = self._risk_engine.validate(risk_order, portfolio_state)

        if not risk_result.approved:
            logger.warning(
                "Runner: risk rejected {} {} — {}",
                trade.side,
                trade.symbol,
                risk_result.reason,
            )
            return ExecutionRecord(
                symbol=trade.symbol,
                side=trade.side,
                target_weight=trade.target_weight,
                dollar_amount=trade.dollar_amount,
                risk_approved=False,
                risk_reason=risk_result.reason,
                order_id=None,
            )

        # Convert dollar amount to share quantity for OMS
        position = self._oms.get_position(trade.symbol)
        price = position.market_price if position and position.market_price > 0 else 0.0
        approved_dollars = abs(risk_result.adjusted_quantity)
        share_qty = approved_dollars / price if price > 0 else approved_dollars

        oms_order = Order(
            symbol=trade.symbol,
            side=OrderSide.BUY if trade.side == "BUY" else OrderSide.SELL,
            quantity=share_qty,
            order_type=OrderType.MARKET,
            strategy_id="strategy_runner",
            sector=self._config.sector_map.get(trade.symbol),
        )

        try:
            self._oms.submit_order(oms_order)
            order_id = oms_order.id
        except Exception:
            logger.exception(
                "Runner: OMS submission failed for {} {}",
                trade.side,
                trade.symbol,
            )
            order_id = None

        return ExecutionRecord(
            symbol=trade.symbol,
            side=trade.side,
            target_weight=trade.target_weight,
            dollar_amount=trade.dollar_amount,
            risk_approved=True,
            risk_reason="",
            order_id=order_id,
        )

    # ── Private: Logging ──────────────────────────────────────────────────

    @staticmethod
    def _log_result(result: RunResult) -> None:
        if result.state == RunnerState.ERROR:
            logger.error("Runner: FAILED — {}", result.error)
            return

        if result.construction and result.construction.rebalance_triggered:
            logger.info(
                "Runner: rebalanced | portfolio=${:,.0f} | "
                "submitted={} rejected={} | "
                "vol={:.1%} turnover={:.1%}",
                result.portfolio_value,
                result.n_submitted,
                result.n_rejected,
                result.construction.optimization.risk,
                result.construction.rebalance.turnover,
            )
        else:
            logger.info(
                "Runner: no rebalance needed | portfolio=${:,.0f}",
                result.portfolio_value,
            )



# Type aliases for provider callables
FeatureProvider = Callable[[str, BaseSignal], dict[str, pd.Series]]
ReturnsProvider = Callable[[list[str], int], pd.DataFrame]
