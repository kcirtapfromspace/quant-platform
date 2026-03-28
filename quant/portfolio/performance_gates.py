"""Strategy performance gates for automated lifecycle promotion.

Defines quantitative criteria for promoting strategies through lifecycle
stages (Research → Paper → Live) and for demoting or pausing strategies
that fail to meet ongoing requirements.

Gate stages:

  1. **RESEARCH** → **PAPER**: Strategy must pass backtest quality gates.
  2. **PAPER** → **LIVE**: Strategy must demonstrate live-consistent
     performance during paper trading.
  3. **LIVE** → **PAUSE** (demotion): Strategy fails ongoing monitoring.

Each gate is a set of pass/fail criteria evaluated against a
:class:`StrategyPerformance` snapshot.  A strategy must pass **all**
criteria for a gate to open.

Usage::

    from quant.portfolio.performance_gates import (
        GateEvaluator,
        GateConfig,
        StrategyPerformance,
    )

    evaluator = GateEvaluator()
    perf = StrategyPerformance(
        sharpe=1.5, max_drawdown=-0.12, n_trades=200, ...
    )
    result = evaluator.evaluate_promotion("paper_to_live", perf)
    if result.passed:
        print("Strategy approved for live trading")
    else:
        for f in result.failures:
            print(f"  FAIL: {f}")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


class StrategyStage(Enum):
    """Strategy lifecycle stage."""

    RESEARCH = "research"
    PAPER = "paper"
    LIVE = "live"
    PAUSED = "paused"


# ---------------------------------------------------------------------------
# Performance snapshot
# ---------------------------------------------------------------------------


@dataclass
class StrategyPerformance:
    """Performance snapshot used for gate evaluation.

    Attributes:
        sharpe:               Annualised Sharpe ratio.
        cagr:                 Compound annual growth rate.
        max_drawdown:         Maximum drawdown (negative number).
        calmar:               CAGR / |max_drawdown|.
        win_rate:             Fraction of profitable periods.
        profit_factor:        Gross profits / gross losses.
        n_trades:             Total number of trades executed.
        avg_turnover:         Average one-way turnover per rebalance.
        tracking_error:       Tracking error vs backtest (paper/live only).
        live_backtest_drift:  Z-score of live vs backtest drift.
        n_days:               Number of trading days evaluated.
        information_ratio:    IR if benchmark available.
        signal_ic:            Most recent signal IC.
    """

    sharpe: float = 0.0
    cagr: float = 0.0
    max_drawdown: float = 0.0
    calmar: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    n_trades: int = 0
    avg_turnover: float = 0.0
    tracking_error: float | None = None
    live_backtest_drift: float | None = None
    n_days: int = 0
    information_ratio: float | None = None
    signal_ic: float | None = None


# ---------------------------------------------------------------------------
# Gate criteria
# ---------------------------------------------------------------------------


@dataclass
class GateCriterion:
    """Single pass/fail criterion for a gate.

    Attributes:
        name:       Human-readable criterion name.
        field:      StrategyPerformance attribute to check.
        operator:   Comparison operator (">=", "<=", ">", "<").
        threshold:  Required value.
    """

    name: str
    field_name: str
    operator: str
    threshold: float

    def evaluate(self, perf: StrategyPerformance) -> tuple[bool, str]:
        """Evaluate this criterion against performance data.

        Returns:
            (passed, message) tuple.
        """
        value = getattr(perf, self.field_name, None)
        if value is None:
            return True, f"{self.name}: skipped (no data)"

        ops = {
            ">=": lambda a, b: a >= b,
            "<=": lambda a, b: a <= b,
            ">": lambda a, b: a > b,
            "<": lambda a, b: a < b,
        }
        op_fn = ops.get(self.operator)
        if op_fn is None:
            return False, f"{self.name}: unknown operator '{self.operator}'"

        passed = op_fn(value, self.threshold)
        status = "PASS" if passed else "FAIL"
        return passed, (
            f"{self.name}: {status} "
            f"({self.field_name}={value:.4f} {self.operator} {self.threshold:.4f})"
        )


# ---------------------------------------------------------------------------
# Gate configuration
# ---------------------------------------------------------------------------


@dataclass
class GateConfig:
    """Configuration for strategy performance gates.

    Attributes:
        research_to_paper:  Criteria for research → paper promotion.
        paper_to_live:      Criteria for paper → live promotion.
        live_demotion:      Criteria for live → paused demotion (inverted:
                            failing these means the strategy should be paused).
        min_paper_days:     Minimum paper trading days before live promotion.
        min_research_trades: Minimum backtest trades for research gate.
    """

    research_to_paper: list[GateCriterion] = field(default_factory=list)
    paper_to_live: list[GateCriterion] = field(default_factory=list)
    live_demotion: list[GateCriterion] = field(default_factory=list)
    min_paper_days: int = 60
    min_research_trades: int = 100

    @classmethod
    def default(cls) -> GateConfig:
        """Return sensible default gate criteria."""
        return cls(
            research_to_paper=[
                GateCriterion("Min Sharpe", "sharpe", ">=", 1.0),
                GateCriterion("Max Drawdown", "max_drawdown", ">=", -0.25),
                GateCriterion("Min Trades", "n_trades", ">=", 100),
                GateCriterion("Min Win Rate", "win_rate", ">=", 0.40),
                GateCriterion("Min Profit Factor", "profit_factor", ">=", 1.1),
            ],
            paper_to_live=[
                GateCriterion("Min Sharpe", "sharpe", ">=", 0.8),
                GateCriterion("Max Drawdown", "max_drawdown", ">=", -0.20),
                GateCriterion("Max Tracking Error", "tracking_error", "<=", 0.05),
                GateCriterion("Max Drift Z", "live_backtest_drift", "<=", 2.0),
                GateCriterion("Min Profit Factor", "profit_factor", ">=", 1.0),
            ],
            live_demotion=[
                GateCriterion("Min Sharpe", "sharpe", ">=", 0.0),
                GateCriterion("Max Drawdown", "max_drawdown", ">=", -0.30),
                GateCriterion("Max Drift Z", "live_backtest_drift", "<=", 3.0),
            ],
        )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


@dataclass
class GateResult:
    """Result of evaluating a single gate.

    Attributes:
        gate_name:    Name of the gate evaluated.
        passed:       Whether all criteria passed.
        n_criteria:   Total number of criteria.
        n_passed:     Number of criteria that passed.
        n_failed:     Number of criteria that failed.
        n_skipped:    Number of criteria skipped (no data).
        details:      Per-criterion result messages.
        failures:     List of failure messages only.
    """

    gate_name: str
    passed: bool
    n_criteria: int = 0
    n_passed: int = 0
    n_failed: int = 0
    n_skipped: int = 0
    details: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable summary."""
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            f"Gate: {self.gate_name} — {status}",
            f"  Criteria: {self.n_criteria} total, "
            f"{self.n_passed} passed, {self.n_failed} failed, "
            f"{self.n_skipped} skipped",
        ]
        for detail in self.details:
            lines.append(f"  {detail}")
        return "\n".join(lines)


@dataclass
class PromotionDecision:
    """Complete promotion/demotion decision for a strategy.

    Attributes:
        strategy_name:   Strategy identifier.
        current_stage:   Current lifecycle stage.
        recommended_stage: Recommended stage after evaluation.
        promotion_gate:  Gate result for promotion (None if not evaluated).
        demotion_gate:   Gate result for demotion check (None if not evaluated).
        should_promote:  Whether promotion is recommended.
        should_demote:   Whether demotion is recommended.
    """

    strategy_name: str
    current_stage: StrategyStage
    recommended_stage: StrategyStage
    promotion_gate: GateResult | None = None
    demotion_gate: GateResult | None = None
    should_promote: bool = False
    should_demote: bool = False

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"Strategy: {self.strategy_name}",
            f"  Current stage    : {self.current_stage.value}",
            f"  Recommended stage: {self.recommended_stage.value}",
        ]
        if self.should_promote:
            lines.append("  Action: PROMOTE")
        elif self.should_demote:
            lines.append("  Action: DEMOTE")
        else:
            lines.append("  Action: HOLD")
        if self.promotion_gate:
            lines.append("")
            lines.append(self.promotion_gate.summary())
        if self.demotion_gate:
            lines.append("")
            lines.append(self.demotion_gate.summary())
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class GateEvaluator:
    """Evaluates strategy performance against promotion/demotion gates.

    Args:
        config: Gate configuration.  Uses :meth:`GateConfig.default` if None.
    """

    def __init__(self, config: GateConfig | None = None) -> None:
        self._config = config or GateConfig.default()

    @property
    def config(self) -> GateConfig:
        return self._config

    def evaluate_gate(
        self,
        gate_name: str,
        criteria: list[GateCriterion],
        perf: StrategyPerformance,
    ) -> GateResult:
        """Evaluate a set of criteria against performance data.

        Args:
            gate_name: Name for reporting.
            criteria:  List of criteria to check.
            perf:      Strategy performance snapshot.

        Returns:
            :class:`GateResult` with pass/fail and details.
        """
        details: list[str] = []
        failures: list[str] = []
        n_passed = 0
        n_failed = 0
        n_skipped = 0

        for criterion in criteria:
            passed, msg = criterion.evaluate(perf)
            details.append(msg)
            if "skipped" in msg:
                n_skipped += 1
            elif passed:
                n_passed += 1
            else:
                n_failed += 1
                failures.append(msg)

        return GateResult(
            gate_name=gate_name,
            passed=n_failed == 0,
            n_criteria=len(criteria),
            n_passed=n_passed,
            n_failed=n_failed,
            n_skipped=n_skipped,
            details=details,
            failures=failures,
        )

    def evaluate_promotion(
        self,
        strategy_name: str,
        current_stage: StrategyStage,
        perf: StrategyPerformance,
    ) -> PromotionDecision:
        """Evaluate whether a strategy should be promoted or demoted.

        Args:
            strategy_name: Strategy identifier.
            current_stage: Current lifecycle stage.
            perf:          Performance snapshot.

        Returns:
            :class:`PromotionDecision` with recommendation.
        """
        cfg = self._config
        promotion_gate = None
        demotion_gate = None
        should_promote = False
        should_demote = False
        recommended = current_stage

        if current_stage == StrategyStage.RESEARCH:
            # Check research → paper gate
            extra_criteria = [
                GateCriterion("Min Trades", "n_trades", ">=", cfg.min_research_trades),
            ]
            all_criteria = cfg.research_to_paper + [
                c for c in extra_criteria
                if not any(ec.field_name == c.field_name for ec in cfg.research_to_paper)
            ]
            promotion_gate = self.evaluate_gate(
                "research_to_paper", all_criteria, perf,
            )
            if promotion_gate.passed:
                should_promote = True
                recommended = StrategyStage.PAPER

        elif current_stage == StrategyStage.PAPER:
            # Check paper → live gate
            extra_criteria = []
            if perf.n_days < cfg.min_paper_days:
                extra_criteria.append(
                    GateCriterion(
                        "Min Paper Days", "n_days", ">=", cfg.min_paper_days,
                    ),
                )
            all_criteria = cfg.paper_to_live + extra_criteria
            promotion_gate = self.evaluate_gate(
                "paper_to_live", all_criteria, perf,
            )
            if promotion_gate.passed:
                should_promote = True
                recommended = StrategyStage.LIVE

            # Also check demotion
            demotion_gate = self.evaluate_gate(
                "live_demotion", cfg.live_demotion, perf,
            )
            if not demotion_gate.passed:
                should_demote = True
                recommended = StrategyStage.PAUSED

        elif current_stage == StrategyStage.LIVE:
            # Check ongoing demotion criteria
            demotion_gate = self.evaluate_gate(
                "live_demotion", cfg.live_demotion, perf,
            )
            if not demotion_gate.passed:
                should_demote = True
                recommended = StrategyStage.PAUSED

        return PromotionDecision(
            strategy_name=strategy_name,
            current_stage=current_stage,
            recommended_stage=recommended,
            promotion_gate=promotion_gate,
            demotion_gate=demotion_gate,
            should_promote=should_promote,
            should_demote=should_demote,
        )
