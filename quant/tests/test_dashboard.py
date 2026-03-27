"""Tests for the CIO Dashboard (QUA-64)."""
from __future__ import annotations

from datetime import datetime, timezone

from quant.portfolio.dashboard import CIODashboard
from quant.portfolio.lifecycle import (
    HealthStatus,
    LifecycleReport,
    Recommendation,
    StrategyHealth,
)
from quant.risk.reporting import (
    ConcentrationMetrics,
    RiskReport,
    StressResult,
    VaRMethod,
    VaRResult,
)

# ── Helpers ───────────────────────────────────────────────────────────────


def _make_lifecycle_report() -> LifecycleReport:
    return LifecycleReport(
        timestamp=datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc),
        strategy_health=[
            StrategyHealth(
                name="momentum",
                status=HealthStatus.HEALTHY,
                current_weight=0.60,
                rolling_sharpe=1.5,
                rolling_vol=0.12,
                max_drawdown=0.04,
                current_drawdown=0.01,
                signal_ic=0.08,
                ic_trend=0.001,
            ),
            StrategyHealth(
                name="mean_rev",
                status=HealthStatus.WATCH,
                current_weight=0.40,
                rolling_sharpe=0.3,
                rolling_vol=0.15,
                max_drawdown=0.08,
                current_drawdown=0.06,
                signal_ic=0.02,
                ic_trend=-0.001,
                reasons=["low Sharpe"],
            ),
        ],
        recommendations=[
            Recommendation(
                strategy="momentum",
                current_weight=0.60,
                recommended_weight=0.65,
                delta=0.05,
                reason="healthy — increase allocation",
            ),
            Recommendation(
                strategy="mean_rev",
                current_weight=0.40,
                recommended_weight=0.35,
                delta=-0.05,
                reason="watch — reduce allocation",
            ),
        ],
        n_healthy=1,
        n_watch=1,
        n_degraded=0,
        n_critical=0,
    )


def _make_risk_report() -> RiskReport:
    return RiskReport(
        var_results=[
            VaRResult(confidence=0.95, method=VaRMethod.HISTORICAL, var=0.018, cvar=0.025, n_observations=252),
            VaRResult(confidence=0.95, method=VaRMethod.PARAMETRIC, var=0.016, cvar=0.022, n_observations=252),
            VaRResult(confidence=0.99, method=VaRMethod.HISTORICAL, var=0.032, cvar=0.045, n_observations=252),
            VaRResult(confidence=0.99, method=VaRMethod.PARAMETRIC, var=0.028, cvar=0.038, n_observations=252),
        ],
        stress_results=[
            StressResult(scenario_name="2008 GFC", portfolio_pnl=-380_000, portfolio_return=-0.38, per_asset={}),
            StressResult(scenario_name="COVID crash", portfolio_pnl=-340_000, portfolio_return=-0.34, per_asset={}),
        ],
        concentration=ConcentrationMetrics(hhi=0.15, effective_n=6.7, top1_weight=0.20, top5_weight=0.65, n_positions=10),
        annualised_volatility=0.14,
        max_drawdown=0.08,
        portfolio_value=1_000_000,
    )


class _FakeResult:
    """Stub OrchestratorResult for testing."""

    def __init__(
        self,
        lifecycle_report=None,
        risk_report=None,
        circuit_breaker_tripped=False,
    ):
        self.timestamp = datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc)
        self.total_portfolio = 1_000_000
        self.sleeve_results = [object(), object()]
        self.n_submitted = 5
        self.n_rejected = 1
        self.circuit_breaker_tripped = circuit_breaker_tripped
        self.lifecycle_report = lifecycle_report
        self.risk_report = risk_report


# ── Tests ─────────────────────────────────────────────────────────────────


class TestCIODashboardConstruction:
    def test_from_full_result(self):
        result = _FakeResult(
            lifecycle_report=_make_lifecycle_report(),
            risk_report=_make_risk_report(),
        )
        dash = CIODashboard.from_orchestrator_result(result)
        assert dash.portfolio_value == 1_000_000
        assert dash.n_sleeves == 2
        assert dash.n_submitted == 5
        assert dash.n_rejected == 1

    def test_strategy_lines_populated(self):
        result = _FakeResult(lifecycle_report=_make_lifecycle_report())
        dash = CIODashboard.from_orchestrator_result(result)
        assert len(dash.strategy_lines) == 2
        assert dash.strategy_lines[0].name == "momentum"
        assert dash.strategy_lines[0].health == HealthStatus.HEALTHY
        assert dash.strategy_lines[1].health == HealthStatus.WATCH

    def test_recommendations_in_lines(self):
        result = _FakeResult(lifecycle_report=_make_lifecycle_report())
        dash = CIODashboard.from_orchestrator_result(result)
        mom = dash.strategy_lines[0]
        assert abs(mom.recommended_weight - 0.65) < 1e-6
        assert abs(mom.delta - 0.05) < 1e-6

    def test_risk_metrics_populated(self):
        result = _FakeResult(risk_report=_make_risk_report())
        dash = CIODashboard.from_orchestrator_result(result)
        assert dash.var_95 is not None
        assert abs(dash.var_95 - 0.018) < 1e-6
        assert dash.var_99 is not None
        assert dash.annualised_vol is not None
        assert dash.hhi is not None

    def test_worst_stress(self):
        result = _FakeResult(risk_report=_make_risk_report())
        dash = CIODashboard.from_orchestrator_result(result)
        assert dash.worst_stress_name == "2008 GFC"
        assert dash.worst_stress_pnl == -380_000

    def test_circuit_breaker_flag(self):
        result = _FakeResult(circuit_breaker_tripped=True)
        dash = CIODashboard.from_orchestrator_result(result)
        assert dash.circuit_breaker_tripped

    def test_empty_result(self):
        result = _FakeResult()
        dash = CIODashboard.from_orchestrator_result(result)
        assert len(dash.strategy_lines) == 0
        assert dash.var_95 is None
        assert dash.annualised_vol is None


class TestCIODashboardRender:
    def test_render_full(self):
        result = _FakeResult(
            lifecycle_report=_make_lifecycle_report(),
            risk_report=_make_risk_report(),
        )
        dash = CIODashboard.from_orchestrator_result(result)
        text = dash.render()
        assert "CIO Dashboard" in text
        assert "Strategy Health" in text
        assert "Risk Metrics" in text
        assert "Worst Stress" in text
        assert "momentum" in text
        assert "mean_rev" in text

    def test_render_risk_only(self):
        result = _FakeResult(risk_report=_make_risk_report())
        dash = CIODashboard.from_orchestrator_result(result)
        text = dash.render()
        assert "Risk Metrics" in text
        assert "Strategy Health" not in text

    def test_render_lifecycle_only(self):
        result = _FakeResult(lifecycle_report=_make_lifecycle_report())
        dash = CIODashboard.from_orchestrator_result(result)
        text = dash.render()
        assert "Strategy Health" in text

    def test_render_circuit_breaker_warning(self):
        result = _FakeResult(circuit_breaker_tripped=True)
        dash = CIODashboard.from_orchestrator_result(result)
        text = dash.render()
        assert "CIRCUIT BREAKER TRIPPED" in text

    def test_render_empty_result(self):
        result = _FakeResult()
        dash = CIODashboard.from_orchestrator_result(result)
        text = dash.render()
        assert "CIO Dashboard" in text
        assert len(text) > 50
