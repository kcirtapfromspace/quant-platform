"""CIO Dashboard — unified investment report across lifecycle, risk, and performance.

Aggregates outputs from the orchestrator's lifecycle manager, risk reporter,
and sleeve execution into a single structured report for CIO decision-making.

Usage::

    from quant.portfolio.dashboard import CIODashboard

    dashboard = CIODashboard.from_orchestrator_result(result)
    print(dashboard.render())
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from quant.portfolio.lifecycle import HealthStatus, LifecycleReport
from quant.risk.reporting import RiskReport


@dataclass(frozen=True, slots=True)
class StrategyLine:
    """Per-strategy summary line for the dashboard."""

    name: str
    health: HealthStatus
    capital_weight: float
    recommended_weight: float
    delta: float
    sharpe: float
    max_drawdown: float
    signal_ic: float | None


@dataclass
class CIODashboard:
    """Unified CIO investment dashboard.

    Combines lifecycle, risk, and execution data into a single
    decision-support report.
    """

    timestamp: datetime
    portfolio_value: float
    n_sleeves: int
    n_submitted: int
    n_rejected: int
    circuit_breaker_tripped: bool

    # Strategy health
    strategy_lines: list[StrategyLine] = field(default_factory=list)
    n_healthy: int = 0
    n_watch: int = 0
    n_degraded: int = 0
    n_critical: int = 0
    total_reallocation: float = 0.0

    # Risk
    var_95: float | None = None
    var_99: float | None = None
    cvar_95: float | None = None
    cvar_99: float | None = None
    annualised_vol: float | None = None
    max_drawdown: float | None = None
    hhi: float | None = None
    effective_n: float | None = None

    # Stress
    worst_stress_name: str = ""
    worst_stress_pnl: float = 0.0
    worst_stress_return: float = 0.0

    @classmethod
    def from_orchestrator_result(cls, result: object) -> CIODashboard:
        """Build a dashboard from an OrchestratorResult.

        Args:
            result: An OrchestratorResult instance (untyped to avoid circular
                import — duck-typed on attribute access).
        """
        lifecycle: LifecycleReport | None = getattr(result, "lifecycle_report", None)
        risk: RiskReport | None = getattr(result, "risk_report", None)

        # Strategy lines from lifecycle
        strategy_lines: list[StrategyLine] = []
        rec_map: dict[str, tuple[float, float]] = {}
        if lifecycle is not None:
            for rec in lifecycle.recommendations:
                rec_map[rec.strategy] = (rec.recommended_weight, rec.delta)
            for h in lifecycle.strategy_health:
                rw, delta = rec_map.get(h.name, (h.current_weight, 0.0))
                strategy_lines.append(
                    StrategyLine(
                        name=h.name,
                        health=h.status,
                        capital_weight=h.current_weight,
                        recommended_weight=rw,
                        delta=delta,
                        sharpe=h.rolling_sharpe,
                        max_drawdown=h.max_drawdown,
                        signal_ic=h.signal_ic,
                    )
                )

        # Risk metrics
        var_95 = var_99 = cvar_95 = cvar_99 = None
        ann_vol = mdd = hhi = eff_n = None
        worst_name = ""
        worst_pnl = 0.0
        worst_ret = 0.0

        if risk is not None:
            ann_vol = risk.annualised_volatility
            mdd = risk.max_drawdown
            if risk.concentration is not None:
                hhi = risk.concentration.hhi
                eff_n = risk.concentration.effective_n

            for v in risk.var_results:
                if v.method.value == "historical":
                    if abs(v.confidence - 0.95) < 0.001:
                        var_95 = v.var
                        cvar_95 = v.cvar
                    elif abs(v.confidence - 0.99) < 0.001:
                        var_99 = v.var
                        cvar_99 = v.cvar

            if risk.stress_results:
                worst = min(risk.stress_results, key=lambda s: s.portfolio_pnl)
                worst_name = worst.scenario_name
                worst_pnl = worst.portfolio_pnl
                worst_ret = worst.portfolio_return

        sleeve_results = getattr(result, "sleeve_results", [])

        return cls(
            timestamp=getattr(result, "timestamp", datetime.min),
            portfolio_value=getattr(result, "total_portfolio", 0.0),
            n_sleeves=len(sleeve_results),
            n_submitted=getattr(result, "n_submitted", 0),
            n_rejected=getattr(result, "n_rejected", 0),
            circuit_breaker_tripped=getattr(result, "circuit_breaker_tripped", False),
            strategy_lines=strategy_lines,
            n_healthy=lifecycle.n_healthy if lifecycle else 0,
            n_watch=lifecycle.n_watch if lifecycle else 0,
            n_degraded=lifecycle.n_degraded if lifecycle else 0,
            n_critical=lifecycle.n_critical if lifecycle else 0,
            total_reallocation=lifecycle.total_reallocation if lifecycle else 0.0,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            annualised_vol=ann_vol,
            max_drawdown=mdd,
            hhi=hhi,
            effective_n=eff_n,
            worst_stress_name=worst_name,
            worst_stress_pnl=worst_pnl,
            worst_stress_return=worst_ret,
        )

    def render(self) -> str:
        """Render a human-readable CIO dashboard."""
        lines = [
            f"╔══ CIO Dashboard — {self.timestamp:%Y-%m-%d %H:%M} ══╗",
            f"  Portfolio     : ${self.portfolio_value:,.0f}",
            f"  Sleeves       : {self.n_sleeves}",
            f"  Orders        : {self.n_submitted} submitted, {self.n_rejected} rejected",
        ]

        if self.circuit_breaker_tripped:
            lines.append("  ⚠ CIRCUIT BREAKER TRIPPED — all trading halted")

        # Strategy health
        if self.strategy_lines:
            lines.append("")
            lines.append("── Strategy Health ──────────────────────────────")
            lines.append(
                f"  Healthy={self.n_healthy}  Watch={self.n_watch}  "
                f"Degraded={self.n_degraded}  Critical={self.n_critical}"
            )
            lines.append("")
            lines.append(
                f"  {'Strategy':<18s} {'Health':<10s} {'Weight':>7s} "
                f"{'Target':>7s} {'Delta':>7s} {'Sharpe':>7s} {'DD':>6s}"
            )
            lines.append("  " + "-" * 62)
            for s in self.strategy_lines:
                lines.append(
                    f"  {s.name:<18s} {s.health.value:<10s} "
                    f"{s.capital_weight:>6.1%} "
                    f"{s.recommended_weight:>6.1%} "
                    f"{s.delta:>+6.1%} "
                    f"{s.sharpe:>+7.2f} "
                    f"{s.max_drawdown:>5.1%}"
                )
            if self.total_reallocation > 1e-6:
                lines.append(f"  Total reallocation turnover: {self.total_reallocation:.1%}")

        # Risk metrics
        if self.annualised_vol is not None:
            lines.append("")
            lines.append("── Risk Metrics ────────────────────────────────")
            lines.append(f"  Ann. volatility  : {self.annualised_vol:.2%}")
            if self.max_drawdown is not None:
                lines.append(f"  Max drawdown     : {self.max_drawdown:.2%}")
            if self.var_95 is not None:
                lines.append(f"  VaR  95%         : {self.var_95:.4%}")
            if self.cvar_95 is not None:
                lines.append(f"  CVaR 95%         : {self.cvar_95:.4%}")
            if self.var_99 is not None:
                lines.append(f"  VaR  99%         : {self.var_99:.4%}")
            if self.cvar_99 is not None:
                lines.append(f"  CVaR 99%         : {self.cvar_99:.4%}")
            if self.hhi is not None:
                lines.append(f"  HHI              : {self.hhi:.4f}")
            if self.effective_n is not None:
                lines.append(f"  Effective N      : {self.effective_n:.1f}")

        # Worst stress scenario
        if self.worst_stress_name:
            lines.append("")
            lines.append("── Worst Stress Scenario ───────────────────────")
            lines.append(
                f"  {self.worst_stress_name}: "
                f"${self.worst_stress_pnl:>+,.0f} ({self.worst_stress_return:+.2%})"
            )

        lines.append("")
        lines.append("╚═══════════════════════════════════════════════╝")
        return "\n".join(lines)
