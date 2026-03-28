"""Tests for strategy performance gates (QUA-99)."""
from __future__ import annotations

from quant.portfolio.performance_gates import (
    GateConfig,
    GateCriterion,
    GateEvaluator,
    GateResult,
    PromotionDecision,
    StrategyPerformance,
    StrategyStage,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strong_backtest() -> StrategyPerformance:
    """Strategy that passes all research gates comfortably."""
    return StrategyPerformance(
        sharpe=1.8, cagr=0.15, max_drawdown=-0.12, calmar=1.25,
        win_rate=0.55, profit_factor=1.6, n_trades=300,
        avg_turnover=0.30, n_days=500,
    )


def _weak_backtest() -> StrategyPerformance:
    """Strategy that fails most research gates."""
    return StrategyPerformance(
        sharpe=0.3, cagr=0.01, max_drawdown=-0.35, calmar=0.03,
        win_rate=0.38, profit_factor=0.9, n_trades=50,
        avg_turnover=0.80, n_days=100,
    )


def _strong_paper() -> StrategyPerformance:
    """Strategy that passes paper → live gates."""
    return StrategyPerformance(
        sharpe=1.2, cagr=0.10, max_drawdown=-0.10, calmar=1.0,
        win_rate=0.52, profit_factor=1.3, n_trades=150,
        tracking_error=0.02, live_backtest_drift=0.8,
        n_days=90,
    )


def _drifting_paper() -> StrategyPerformance:
    """Paper strategy with high drift from backtest."""
    return StrategyPerformance(
        sharpe=0.9, cagr=0.06, max_drawdown=-0.15, calmar=0.4,
        win_rate=0.48, profit_factor=1.1, n_trades=100,
        tracking_error=0.08, live_backtest_drift=3.5,
        n_days=90,
    )


def _failing_live() -> StrategyPerformance:
    """Live strategy that should be paused."""
    return StrategyPerformance(
        sharpe=-0.5, cagr=-0.05, max_drawdown=-0.35, calmar=-0.14,
        win_rate=0.35, profit_factor=0.7, n_trades=200,
        live_backtest_drift=4.0, n_days=180,
    )


# ---------------------------------------------------------------------------
# Gate criterion
# ---------------------------------------------------------------------------


class TestGateCriterion:
    def test_pass_gte(self):
        c = GateCriterion("Test", "sharpe", ">=", 1.0)
        passed, msg = c.evaluate(StrategyPerformance(sharpe=1.5))
        assert passed
        assert "PASS" in msg

    def test_fail_gte(self):
        c = GateCriterion("Test", "sharpe", ">=", 1.0)
        passed, msg = c.evaluate(StrategyPerformance(sharpe=0.5))
        assert not passed
        assert "FAIL" in msg

    def test_pass_lte(self):
        c = GateCriterion("Test", "tracking_error", "<=", 0.05)
        passed, _ = c.evaluate(StrategyPerformance(tracking_error=0.03))
        assert passed

    def test_fail_lte(self):
        c = GateCriterion("Test", "tracking_error", "<=", 0.05)
        passed, _ = c.evaluate(StrategyPerformance(tracking_error=0.10))
        assert not passed

    def test_skip_none_value(self):
        c = GateCriterion("Test", "tracking_error", "<=", 0.05)
        passed, msg = c.evaluate(StrategyPerformance())  # tracking_error=None
        assert passed  # Skipped counts as pass
        assert "skipped" in msg

    def test_exact_threshold_passes_gte(self):
        c = GateCriterion("Test", "sharpe", ">=", 1.0)
        passed, _ = c.evaluate(StrategyPerformance(sharpe=1.0))
        assert passed


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------


class TestGateEvaluation:
    def test_all_pass(self):
        evaluator = GateEvaluator()
        result = evaluator.evaluate_gate(
            "test", evaluator.config.research_to_paper, _strong_backtest(),
        )
        assert isinstance(result, GateResult)
        assert result.passed

    def test_some_fail(self):
        evaluator = GateEvaluator()
        result = evaluator.evaluate_gate(
            "test", evaluator.config.research_to_paper, _weak_backtest(),
        )
        assert not result.passed
        assert result.n_failed > 0

    def test_failures_listed(self):
        evaluator = GateEvaluator()
        result = evaluator.evaluate_gate(
            "test", evaluator.config.research_to_paper, _weak_backtest(),
        )
        assert len(result.failures) == result.n_failed

    def test_details_populated(self):
        evaluator = GateEvaluator()
        result = evaluator.evaluate_gate(
            "test", evaluator.config.research_to_paper, _strong_backtest(),
        )
        assert len(result.details) == result.n_criteria

    def test_empty_criteria_passes(self):
        evaluator = GateEvaluator()
        result = evaluator.evaluate_gate("empty", [], _strong_backtest())
        assert result.passed
        assert result.n_criteria == 0


# ---------------------------------------------------------------------------
# Research → Paper promotion
# ---------------------------------------------------------------------------


class TestResearchToPaper:
    def test_strong_promoted(self):
        decision = GateEvaluator().evaluate_promotion(
            "MomentumV2", StrategyStage.RESEARCH, _strong_backtest(),
        )
        assert isinstance(decision, PromotionDecision)
        assert decision.should_promote
        assert decision.recommended_stage == StrategyStage.PAPER

    def test_weak_not_promoted(self):
        decision = GateEvaluator().evaluate_promotion(
            "BadStrat", StrategyStage.RESEARCH, _weak_backtest(),
        )
        assert not decision.should_promote
        assert decision.recommended_stage == StrategyStage.RESEARCH

    def test_min_trades_enforced(self):
        perf = _strong_backtest()
        perf.n_trades = 20  # Below default min_research_trades=100
        decision = GateEvaluator().evaluate_promotion(
            "FewTrades", StrategyStage.RESEARCH, perf,
        )
        assert not decision.should_promote

    def test_promotion_gate_populated(self):
        decision = GateEvaluator().evaluate_promotion(
            "Test", StrategyStage.RESEARCH, _strong_backtest(),
        )
        assert decision.promotion_gate is not None


# ---------------------------------------------------------------------------
# Paper → Live promotion
# ---------------------------------------------------------------------------


class TestPaperToLive:
    def test_strong_paper_promoted(self):
        decision = GateEvaluator().evaluate_promotion(
            "MomentumV2", StrategyStage.PAPER, _strong_paper(),
        )
        assert decision.should_promote
        assert decision.recommended_stage == StrategyStage.LIVE

    def test_drifting_paper_not_promoted(self):
        decision = GateEvaluator().evaluate_promotion(
            "DriftStrat", StrategyStage.PAPER, _drifting_paper(),
        )
        assert not decision.should_promote

    def test_min_paper_days_enforced(self):
        perf = _strong_paper()
        perf.n_days = 30  # Below default min_paper_days=60
        decision = GateEvaluator().evaluate_promotion(
            "TooEarly", StrategyStage.PAPER, perf,
        )
        assert not decision.should_promote

    def test_paper_demotion_checked(self):
        """Paper strategies also get checked for demotion."""
        decision = GateEvaluator().evaluate_promotion(
            "BadPaper", StrategyStage.PAPER, _failing_live(),
        )
        assert decision.should_demote
        assert decision.recommended_stage == StrategyStage.PAUSED


# ---------------------------------------------------------------------------
# Live demotion
# ---------------------------------------------------------------------------


class TestLiveDemotion:
    def test_healthy_live_holds(self):
        decision = GateEvaluator().evaluate_promotion(
            "GoodLive", StrategyStage.LIVE, _strong_paper(),
        )
        assert not decision.should_demote
        assert decision.recommended_stage == StrategyStage.LIVE

    def test_failing_live_demoted(self):
        decision = GateEvaluator().evaluate_promotion(
            "BadLive", StrategyStage.LIVE, _failing_live(),
        )
        assert decision.should_demote
        assert decision.recommended_stage == StrategyStage.PAUSED

    def test_demotion_gate_populated(self):
        decision = GateEvaluator().evaluate_promotion(
            "Test", StrategyStage.LIVE, _failing_live(),
        )
        assert decision.demotion_gate is not None
        assert not decision.demotion_gate.passed


# ---------------------------------------------------------------------------
# Paused stage
# ---------------------------------------------------------------------------


class TestPausedStage:
    def test_paused_no_action(self):
        decision = GateEvaluator().evaluate_promotion(
            "Paused", StrategyStage.PAUSED, _strong_backtest(),
        )
        assert not decision.should_promote
        assert not decision.should_demote
        assert decision.recommended_stage == StrategyStage.PAUSED


# ---------------------------------------------------------------------------
# Custom gates
# ---------------------------------------------------------------------------


class TestCustomGates:
    def test_custom_criteria(self):
        cfg = GateConfig(
            research_to_paper=[
                GateCriterion("Min Sharpe", "sharpe", ">=", 2.0),
            ],
        )
        decision = GateEvaluator(cfg).evaluate_promotion(
            "Test", StrategyStage.RESEARCH,
            StrategyPerformance(sharpe=1.5, n_trades=200),
        )
        assert not decision.should_promote

    def test_relaxed_criteria(self):
        cfg = GateConfig(
            research_to_paper=[
                GateCriterion("Min Sharpe", "sharpe", ">=", 0.5),
            ],
            min_research_trades=10,
        )
        decision = GateEvaluator(cfg).evaluate_promotion(
            "Test", StrategyStage.RESEARCH,
            StrategyPerformance(sharpe=0.8, n_trades=20),
        )
        assert decision.should_promote


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------


class TestDefaultConfig:
    def test_default_has_research_gates(self):
        cfg = GateConfig.default()
        assert len(cfg.research_to_paper) >= 3

    def test_default_has_paper_gates(self):
        cfg = GateConfig.default()
        assert len(cfg.paper_to_live) >= 3

    def test_default_has_demotion_gates(self):
        cfg = GateConfig.default()
        assert len(cfg.live_demotion) >= 2

    def test_default_min_paper_days(self):
        cfg = GateConfig.default()
        assert cfg.min_paper_days > 0


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_gate_result_summary(self):
        evaluator = GateEvaluator()
        result = evaluator.evaluate_gate(
            "test", evaluator.config.research_to_paper, _strong_backtest(),
        )
        summary = result.summary()
        assert "Gate:" in summary
        assert "PASSED" in summary or "FAILED" in summary

    def test_promotion_decision_summary(self):
        decision = GateEvaluator().evaluate_promotion(
            "MomentumV2", StrategyStage.RESEARCH, _strong_backtest(),
        )
        summary = decision.summary()
        assert "MomentumV2" in summary
        assert "PROMOTE" in summary or "HOLD" in summary or "DEMOTE" in summary
