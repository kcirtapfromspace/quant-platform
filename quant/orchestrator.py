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

from quant.execution.quality_tracker import ExecutionQualityTracker
from quant.oms.models import Order, OrderSide, OrderType
from quant.oms.system import OrderManagementSystem
from quant.portfolio.alpha import AlphaCombiner, CombinationMethod
from quant.portfolio.engine import ConstructionResult, PortfolioConfig, PortfolioEngine
from quant.portfolio.lifecycle import (
    LifecycleConfig,
    LifecycleManager,
    LifecycleReport,
    StrategySnapshot,
)
from quant.portfolio.position_scaler import PositionScaler, ScalingConfig
from quant.portfolio.pre_trade import PreTradeConfig, PreTradePipeline, PreTradeResult
from quant.portfolio.strategy_correlation import (
    StrategyCorrelationMonitor,
    StrategyCorrelationReport,
)
from quant.risk.circuit_breaker import DrawdownCircuitBreaker
from quant.risk.engine import (
    Order as RiskOrder,
)
from quant.risk.engine import (
    PortfolioState,
    RiskConfig,
    RiskEngine,
)
from quant.risk.reporting import RiskReport, RiskReporter
from quant.risk.strategy_monitor import StrategyMonitor
from quant.signals.adaptive_combiner import AdaptiveCombinerConfig, AdaptiveSignalCombiner
from quant.signals.base import BaseSignal, SignalOutput
from quant.signals.decay import DecayConfig, DecayResult, SignalDecayAnalyzer
from quant.signals.regime import RegimeDetector, RegimeState, RegimeWeightAdapter


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _rank(xs: list[float]) -> list[float]:
    """Compute average ranks (1-based) for a list of values."""
    indexed = sorted(enumerate(xs), key=lambda t: t[1])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 1) / 2  # average of 1-based ranks i+1..j
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


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
        adaptive_combiner_config: Config for IC-adaptive signal combination.
                            When set, overrides combination_method with adaptive
                            IC-weighted combination that learns from recent
                            signal performance.
        scaling_config:     Position scaling config (conviction / vol / Kelly).
                            None to skip scaling.
        sector_map:         {symbol: sector} for risk checks.
        lookback_days:      History for covariance estimation.
        enabled:            If False, skip this sleeve entirely.
    """

    name: str
    signals: list[BaseSignal] = field(default_factory=list)
    capital_weight: float = 1.0
    strategy_type: str = ""
    portfolio_config: PortfolioConfig = field(default_factory=PortfolioConfig)
    risk_config: RiskConfig | None = None
    combination_method: CombinationMethod = CombinationMethod.EQUAL_WEIGHT
    signal_weights: dict[str, float] | None = None
    adaptive_combiner_config: AdaptiveCombinerConfig | None = None
    scaling_config: ScalingConfig | None = None
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
        alpha_scores:       Combined alpha scores before portfolio construction.
                            Used for signal IC computation across cycles.
        construction:       Portfolio construction output.
        n_trades:           Number of trades proposed.
        error:              Error message if the sleeve failed.
    """

    name: str
    capital_allocated: float = 0.0
    target_weights: dict[str, float] = field(default_factory=dict)
    alpha_scores: dict[str, float] = field(default_factory=dict)
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
    pre_trade_result: PreTradeResult | None = None
    lifecycle_report: LifecycleReport | None = None
    risk_report: RiskReport | None = None
    correlation_report: StrategyCorrelationReport | None = None
    circuit_breaker_tripped: bool = False
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
    pre_trade_config: PreTradeConfig | None = None
    lifecycle_config: LifecycleConfig | None = None
    apply_lifecycle_realloc: bool = False
    regime_detector: RegimeDetector | None = None
    regime_adapter: RegimeWeightAdapter | None = None
    regime_lookback_days: int = 252
    circuit_breaker: DrawdownCircuitBreaker | None = None
    risk_reporter: RiskReporter | None = None
    strategy_correlation: StrategyCorrelationMonitor | None = None
    strategy_monitor: StrategyMonitor | None = None
    quality_tracker: ExecutionQualityTracker | None = None
    decay_config: DecayConfig | None = None
    decay_eval_interval: int = 21


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

        # Per-sleeve adaptive combiners (stateful — persist across cycles)
        self._adaptive_combiners: dict[str, AdaptiveSignalCombiner] = {}
        for sleeve in sleeves:
            if sleeve.adaptive_combiner_config is not None:
                self._adaptive_combiners[sleeve.name] = AdaptiveSignalCombiner(
                    sleeve.adaptive_combiner_config
                )

        # Strategy lifecycle manager (persists across cycles)
        self._lifecycle_mgr: LifecycleManager | None = None
        self._lifecycle_weights: dict[str, float] | None = None
        if config.lifecycle_config is not None:
            self._lifecycle_mgr = LifecycleManager(config.lifecycle_config)

        # Signal IC tracking: stores alpha scores from previous cycle
        # and accumulates IC history for lifecycle health evaluation
        self._last_alpha_scores: dict[str, dict[str, float]] = {}
        self._sleeve_ic_history: dict[str, list[float]] = {}

        # Signal decay tracking: accumulate alpha scores over time
        # for periodic decay analysis (half-life computation)
        self._decay_analyzer: SignalDecayAnalyzer | None = None
        self._alpha_score_history: dict[str, list[tuple[datetime, dict[str, float]]]] = {}
        self._sleeve_half_life: dict[str, int | None] = {}
        self._cycles_since_decay: int = 0
        if config.decay_config is not None:
            self._decay_analyzer = SignalDecayAnalyzer(config.decay_config)

        # Regime state
        self._last_regime: RegimeState | None = None

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

            # ── 0. Circuit breaker check ───────────────────────────────
            cb = self._config.circuit_breaker
            if cb is not None:
                approved, reason = cb.check(total_value)
                if not approved:
                    logger.warning("Orchestrator: {}", reason)
                    result = OrchestratorResult(
                        timestamp=now,
                        total_portfolio=total_value,
                        circuit_breaker_tripped=True,
                    )
                    self._log_result(result)
                    return result

            # ── 1. Detect regime and adjust capital weights ──────────────
            effective_weights = self._regime_adjusted_weights()

            # ── 1b. Apply strategy monitor scaling ────────────────────────
            monitor = self._config.strategy_monitor
            if monitor is not None:
                for name, w in list(effective_weights.items()):
                    scale = monitor.capital_scale(name)
                    if scale < 1.0:
                        logger.info(
                            "  monitor: sleeve '{}' scaled {:.0%} (health={})",
                            name, scale, monitor.status(name).health.value
                            if name in monitor.strategy_names else "unknown",
                        )
                    effective_weights[name] = w * scale

            # ── 1c. Apply execution quality scaling ───────────────────────
            quality = self._config.quality_tracker
            if quality is not None:
                for name, w in list(effective_weights.items()):
                    q_score = quality.quality_score(name)
                    if q_score < 1.0:
                        logger.info(
                            "  quality: sleeve '{}' scaled {:.2f}",
                            name, q_score,
                        )
                        effective_weights[name] = w * q_score

            # ── 1d. Apply lifecycle reallocation from previous cycle ───────
            if (
                self._config.apply_lifecycle_realloc
                and self._lifecycle_weights is not None
            ):
                for name in list(effective_weights):
                    if name in self._lifecycle_weights:
                        prev = effective_weights[name]
                        effective_weights[name] = self._lifecycle_weights[name]
                        if abs(effective_weights[name] - prev) > 1e-6:
                            logger.info(
                                "  lifecycle: sleeve '{}' {:.1%} → {:.1%}",
                                name, prev, effective_weights[name],
                            )

            # ── 2. Run each sleeve ────────────────────────────────────────
            sleeve_results: list[SleeveResult] = []
            for sleeve in self._sleeves:
                if not sleeve.enabled:
                    continue
                cap_w = effective_weights.get(sleeve.name, sleeve.capital_weight)
                sr = self._run_sleeve(sleeve, total_value, now, capital_weight_override=cap_w)
                sleeve_results.append(sr)

            # ── 2b. Update monitor and compute signal IC ──────────────
            if monitor is not None:
                for sr in sleeve_results:
                    if not sr.error:
                        monitor.update(sr.name, sr.capital_allocated)

            # Compute signal IC: compare last cycle's alpha scores
            # against the most recent day's realised returns.
            recent_returns = self._get_returns_history(1)
            if not recent_returns.empty:
                last_day = recent_returns.iloc[-1]
                for sr in sleeve_results:
                    if sr.error:
                        continue
                    prev_scores = self._last_alpha_scores.get(sr.name)
                    if prev_scores:
                        ic = self._compute_signal_ic(prev_scores, last_day)
                        if ic is not None:
                            if sr.name not in self._sleeve_ic_history:
                                self._sleeve_ic_history[sr.name] = []
                            self._sleeve_ic_history[sr.name].append(ic)

            # Store current alpha scores for next cycle's IC computation
            for sr in sleeve_results:
                if not sr.error and sr.alpha_scores:
                    self._last_alpha_scores[sr.name] = dict(sr.alpha_scores)
                    # Accumulate for decay analysis
                    if self._decay_analyzer is not None:
                        if sr.name not in self._alpha_score_history:
                            self._alpha_score_history[sr.name] = []
                        self._alpha_score_history[sr.name].append(
                            (now, dict(sr.alpha_scores))
                        )

            # ── 2b2. Periodic signal decay analysis ────────────────────
            self._cycles_since_decay += 1
            if (
                self._decay_analyzer is not None
                and self._cycles_since_decay >= self._config.decay_eval_interval
            ):
                self._run_decay_analysis()
                self._cycles_since_decay = 0

            # ── 2c. Lifecycle evaluation ────────────────────────────────
            lifecycle_report: LifecycleReport | None = None
            if self._lifecycle_mgr is not None:
                eval_window = self._config.lifecycle_config.eval_window if self._config.lifecycle_config else 63
                returns_df = self._get_returns_history(eval_window)
                for sr in sleeve_results:
                    if sr.error:
                        continue
                    strategy_returns = self._compute_sleeve_returns(
                        sr, returns_df
                    )
                    cap_w = effective_weights.get(sr.name, 0.0)

                    # Build IC data for lifecycle snapshot
                    ic_hist = self._sleeve_ic_history.get(sr.name, [])
                    signal_ic = ic_hist[-1] if ic_hist else None
                    ic_history = pd.Series(ic_hist) if ic_hist else None
                    half_life = self._sleeve_half_life.get(sr.name)

                    self._lifecycle_mgr.update(
                        StrategySnapshot(
                            name=sr.name,
                            returns_series=strategy_returns,
                            current_weight=cap_w,
                            signal_ic=signal_ic,
                            ic_history=ic_history,
                            signal_half_life=half_life,
                        )
                    )
                lifecycle_report = self._lifecycle_mgr.evaluate()
                if lifecycle_report.has_critical:
                    logger.warning(
                        "Lifecycle: {} critical strategies detected",
                        lifecycle_report.n_critical,
                    )
                # Store recommended weights for next cycle
                if self._config.apply_lifecycle_realloc and lifecycle_report.recommendations:
                    self._lifecycle_weights = {
                        r.strategy: r.recommended_weight
                        for r in lifecycle_report.recommendations
                    }

            # ── 2d. Strategy correlation evaluation ────────────────────
            correlation_report: StrategyCorrelationReport | None = None
            corr_monitor = self._config.strategy_correlation
            if corr_monitor is not None:
                corr_window = corr_monitor.config.window
                corr_returns_df = self._get_returns_history(corr_window)
                strat_return_lists: dict[str, list[float]] = {}
                for sr in sleeve_results:
                    if sr.error:
                        continue
                    ret_series = self._compute_sleeve_returns(sr, corr_returns_df)
                    if not ret_series.empty:
                        strat_return_lists[sr.name] = ret_series.tolist()
                if len(strat_return_lists) >= 2:
                    correlation_report = corr_monitor.evaluate(
                        strategy_returns=strat_return_lists,
                        capital_weights=effective_weights,
                        timestamp=now,
                    )
                    if correlation_report.level == "critical":
                        logger.warning(
                            "Strategy correlation CRITICAL: avg={:.3f} eff_N={:.1f}",
                            correlation_report.avg_pairwise_corr,
                            correlation_report.effective_strategies,
                        )
                    elif correlation_report.level == "elevated":
                        logger.info(
                            "Strategy correlation elevated: avg={:.3f} eff_N={:.1f}",
                            correlation_report.avg_pairwise_corr,
                            correlation_report.effective_strategies,
                        )
                    for alert in correlation_report.crowding_alerts:
                        logger.warning("Crowding: {}", alert.message)

            # ── 3. Combine target weights ─────────────────────────────────
            combined = self._combine_weights(sleeve_results)

            # ── 3b. Pre-trade pipeline (limits + cost + min filter) ───────
            pre_trade_result: PreTradeResult | None = None
            if self._config.pre_trade_config is not None:
                pipeline = PreTradePipeline(self._config.pre_trade_config)
                pre_trade_result = pipeline.process(
                    target_weights=combined,
                    current_weights=current_weights,
                    portfolio_value=total_value,
                    sector_map=self._config.sector_map or None,
                )
                if pre_trade_result.was_modified:
                    logger.info(
                        "  pre-trade: {} adjustments, {} trades filtered",
                        pre_trade_result.n_adjustments,
                        pre_trade_result.trades_filtered,
                    )
                combined = pre_trade_result.adjusted_weights

            # ── 4. Compute net trades ─────────────────────────────────────
            trades = self._compute_trades(current_weights, combined, total_value)

            # ── 5. Risk validate and execute ──────────────────────────────
            portfolio_state = self._build_portfolio_state(total_value)
            n_submitted = 0
            n_rejected = 0

            for trade in trades:
                submitted = self._execute_trade(trade, total_value, portfolio_state)
                if submitted:
                    n_submitted += 1
                else:
                    n_rejected += 1

            # ── 6. Risk reporting ────────────────────────────────────────
            risk_report: RiskReport | None = None
            if self._config.risk_reporter is not None:
                risk_report = self._generate_risk_report(total_value)

            result = OrchestratorResult(
                timestamp=now,
                total_portfolio=total_value,
                sleeve_results=sleeve_results,
                combined_weights=combined,
                pre_trade_result=pre_trade_result,
                lifecycle_report=lifecycle_report,
                risk_report=risk_report,
                correlation_report=correlation_report,
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

    @property
    def last_regime(self) -> RegimeState | None:
        """Most recently detected regime state, or None if not yet run."""
        return self._last_regime

    @property
    def strategy_monitor(self) -> StrategyMonitor | None:
        """Strategy performance monitor, or None if not configured."""
        return self._config.strategy_monitor

    @property
    def quality_tracker(self) -> ExecutionQualityTracker | None:
        """Execution quality tracker, or None if not configured."""
        return self._config.quality_tracker

    @property
    def lifecycle_manager(self) -> LifecycleManager | None:
        """Lifecycle manager, or None if not configured."""
        return self._lifecycle_mgr

    @property
    def lifecycle_weights(self) -> dict[str, float] | None:
        """Recommended weights from the last lifecycle evaluation, or None."""
        return self._lifecycle_weights

    @property
    def strategy_correlation(self) -> StrategyCorrelationMonitor | None:
        """Strategy correlation monitor, or None if not configured."""
        return self._config.strategy_correlation

    @property
    def circuit_breaker(self) -> DrawdownCircuitBreaker | None:
        """Circuit breaker, or None if not configured."""
        return self._config.circuit_breaker

    # ── Regime-aware capital allocation ────────────────────────────────

    def _regime_adjusted_weights(self) -> dict[str, float]:
        """Detect regime and return adjusted sleeve capital weights.

        If no regime detector is configured, returns the base weights
        unchanged.

        Returns:
            ``{sleeve_name: adjusted_capital_weight}``.
        """
        detector = self._config.regime_detector
        adapter = self._config.regime_adapter
        if detector is None or adapter is None:
            return {s.name: s.capital_weight for s in self._sleeves if s.enabled}

        # Get returns for regime detection
        returns_df = self._get_returns_history(self._config.regime_lookback_days)
        if returns_df.empty:
            return {s.name: s.capital_weight for s in self._sleeves if s.enabled}

        # Convert DataFrame to 2D list for the pure-Python detector
        returns_2d: list[list[float]] = []
        for _, row in returns_df.iterrows():
            returns_2d.append([float(v) if not pd.isna(v) else 0.0 for v in row])

        if len(returns_2d) < 10:
            return {s.name: s.capital_weight for s in self._sleeves if s.enabled}

        regime = detector.detect(returns=returns_2d)
        self._last_regime = regime

        logger.info(
            "Orchestrator: regime={} confidence={:.2f} (vol={}, trend={}, corr={})",
            regime.regime.value,
            regime.confidence,
            regime.vol_regime.value,
            regime.trend_regime.value,
            regime.corr_regime.value,
        )

        # Build base weights and strategy type mapping
        base_weights = {s.name: s.capital_weight for s in self._sleeves if s.enabled}
        strategy_types = {s.name: s.strategy_type for s in self._sleeves if s.enabled}

        adjusted = adapter.adapt(regime, base_weights, strategy_types)

        # Log adjustments
        for name in base_weights:
            base = base_weights[name]
            adj = adjusted.get(name, base)
            if abs(adj - base) > 0.001:
                logger.info(
                    "  sleeve '{}': {:.1%} → {:.1%} ({:+.1%})",
                    name, base, adj, adj - base,
                )

        return adjusted

    # ── Sleeve execution ─────────────────────────────────────────────────

    def _run_sleeve(
        self,
        sleeve: StrategySleeve,
        total_value: float,
        timestamp: datetime,
        *,
        capital_weight_override: float | None = None,
    ) -> SleeveResult:
        """Run a single strategy sleeve and return its target weights."""
        cap_w = capital_weight_override if capital_weight_override is not None else sleeve.capital_weight
        sleeve_capital = total_value * cap_w

        try:
            # Compute signals
            universe_signals = self._compute_signals(sleeve, timestamp)

            # Combine into alpha
            returns_history = self._get_returns_history(sleeve.lookback_days)
            adaptive = self._adaptive_combiners.get(sleeve.name)

            if adaptive is not None:
                # Update IC history with most recent returns
                if not returns_history.empty:
                    last_returns = returns_history.iloc[-1]
                    signal_scores: dict[str, pd.Series] = {}
                    for sym, sig_list in universe_signals.items():
                        for sig in sig_list:
                            sig_name = sig.metadata.get("signal_name", "unknown")
                            if sig_name not in signal_scores:
                                signal_scores[sig_name] = pd.Series(dtype=float)
                            signal_scores[sig_name] = pd.concat([
                                signal_scores[sig_name],
                                pd.Series([sig.score], index=[sym]),
                            ])
                    adaptive.update(signal_scores, last_returns, timestamp)

                # Combine using adaptive IC-weighted method
                alpha_scores = adaptive.combine_universe(
                    timestamp, universe_signals
                )
            else:
                combiner = AlphaCombiner(
                    method=sleeve.combination_method,
                    weights=sleeve.signal_weights,
                )
                alpha_scores = combiner.combine_universe(
                    timestamp, universe_signals
                )

            # Position scaling (conviction / vol-adjusted / Kelly)
            if sleeve.scaling_config is not None:
                scaler = PositionScaler(sleeve.scaling_config)
                alpha_scores = scaler.scale_to_alpha_dict(
                    alpha_scores, returns_history
                )

            # Portfolio construction
            engine = PortfolioEngine(sleeve.portfolio_config)

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
                    target_weights[sym] = w * cap_w

            # Extract float scores from AlphaScore objects for IC tracking
            float_scores = {
                sym: (a.score if hasattr(a, "score") else float(a))
                for sym, a in alpha_scores.items()
            }

            return SleeveResult(
                name=sleeve.name,
                capital_allocated=sleeve_capital,
                target_weights=target_weights,
                alpha_scores=float_scores,
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

    # ── Risk reporting ─────────────────────────────────────────────────

    def _generate_risk_report(self, portfolio_value: float) -> RiskReport | None:
        """Generate a risk report using the configured RiskReporter."""
        reporter = self._config.risk_reporter
        if reporter is None:
            return None

        # Portfolio returns from returns history
        returns_df = self._get_returns_history(252)
        current_weights = self._get_current_weights(portfolio_value)

        if returns_df.empty or not current_weights:
            # Approximate portfolio returns from equal-weight if no positions
            if returns_df.empty:
                return None
            portfolio_returns = returns_df.mean(axis=1)
        else:
            # Weight-implied portfolio returns
            common = [s for s in current_weights if s in returns_df.columns]
            if not common:
                portfolio_returns = returns_df.mean(axis=1)
            else:
                w = pd.Series({s: current_weights[s] for s in common})
                portfolio_returns = returns_df[common].mul(w).sum(axis=1)

        # Position dollar values
        positions = self._oms.get_all_positions()
        pos_dollars = {sym: pos.market_value for sym, pos in positions.items()}

        return reporter.generate_report(
            returns=portfolio_returns,
            positions=pos_dollars,
            portfolio_value=portfolio_value,
        )

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

    @staticmethod
    def _compute_signal_ic(
        alpha_scores: dict[str, float],
        returns: pd.Series,
    ) -> float | None:
        """Compute Spearman rank IC between signal scores and realised returns.

        Args:
            alpha_scores: ``{symbol: alpha_score}`` from the previous cycle.
            returns: Most recent day's returns, indexed by symbol.

        Returns:
            Rank IC (Spearman correlation) or None if insufficient overlap.
        """
        common = [s for s in alpha_scores if s in returns.index]
        if len(common) < 3:
            return None

        scores = [alpha_scores[s] for s in common]
        rets = [float(returns[s]) for s in common]

        # Spearman rank correlation (pure Python — avoid scipy dependency)
        n = len(common)
        score_ranks = _rank(scores)
        ret_ranks = _rank(rets)

        # Pearson correlation of ranks
        mean_sr = sum(score_ranks) / n
        mean_rr = sum(ret_ranks) / n
        num = sum((score_ranks[i] - mean_sr) * (ret_ranks[i] - mean_rr) for i in range(n))
        denom_s = sum((score_ranks[i] - mean_sr) ** 2 for i in range(n))
        denom_r = sum((ret_ranks[i] - mean_rr) ** 2 for i in range(n))
        denom = (denom_s * denom_r) ** 0.5
        if denom < 1e-12:
            return 0.0
        return num / denom

    def _run_decay_analysis(self) -> None:
        """Run signal decay analysis on accumulated alpha score history.

        Computes half-life per sleeve and stores it for lifecycle health
        evaluation.  Requires at least ``min_periods`` cross-sections
        (from :class:`DecayConfig`) to produce a valid result.
        """
        assert self._decay_analyzer is not None
        returns_df = self._get_returns_history(
            max(self._decay_analyzer._config.horizons) + len(next(iter(self._alpha_score_history.values()), []))
            if self._alpha_score_history
            else 252
        )
        if returns_df.empty:
            return

        for sleeve_name, history in self._alpha_score_history.items():
            if len(history) < (self._decay_analyzer._config.min_periods or 20):
                continue

            # Build signal_scores DataFrame: DatetimeIndex × symbols
            timestamps = [ts for ts, _ in history]
            all_symbols = sorted(
                {sym for _, scores in history for sym in scores}
            )
            rows: list[dict[str, float]] = []
            for _, scores in history:
                rows.append({sym: scores.get(sym, float("nan")) for sym in all_symbols})

            signal_df = pd.DataFrame(rows, index=pd.DatetimeIndex(timestamps), columns=all_symbols)

            # Align returns to signal dates
            common_cols = [c for c in all_symbols if c in returns_df.columns]
            if len(common_cols) < self._decay_analyzer._config.min_assets:
                continue
            aligned_returns = returns_df[common_cols].reindex(signal_df.index).fillna(0.0)
            signal_aligned = signal_df[common_cols]

            try:
                result: DecayResult = self._decay_analyzer.analyze(
                    signal_scores=signal_aligned,
                    asset_returns=aligned_returns,
                    signal_name=sleeve_name,
                )
                self._sleeve_half_life[sleeve_name] = result.half_life
                logger.info(
                    "Decay analysis '{}': half_life={}d peak_ic={:.4f} optimal={}d",
                    sleeve_name,
                    result.half_life,
                    result.peak_ic,
                    result.optimal_horizon,
                )
            except (ValueError, Exception):
                # Insufficient data or computation error — skip
                pass

    @staticmethod
    def _compute_sleeve_returns(
        sleeve_result: SleeveResult, returns_df: pd.DataFrame
    ) -> pd.Series:
        """Approximate strategy returns from asset returns × target weights."""
        if returns_df.empty or not sleeve_result.target_weights:
            return pd.Series(dtype=float)
        weights = sleeve_result.target_weights
        common = [s for s in weights if s in returns_df.columns]
        if not common:
            return pd.Series(dtype=float)
        w = pd.Series({s: weights[s] for s in common})
        return returns_df[common].mul(w).sum(axis=1)

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
