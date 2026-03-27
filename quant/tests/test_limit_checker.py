"""Tests for portfolio risk limit checker (QUA-53)."""
from __future__ import annotations

from datetime import datetime, timezone

from quant.risk.limit_checker import (
    BreachSeverity,
    LimitCheckReport,
    LimitConfig,
    RiskLimitChecker,
)

NOW = datetime(2024, 6, 15, tzinfo=timezone.utc)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _equal_weights(n: int = 10, total: float = 1.0) -> dict[str, float]:
    w = total / n
    return {f"S{i:02d}": w for i in range(n)}


def _sector_map(n: int = 10) -> dict[str, str]:
    sectors = ["Tech", "Finance", "Health"]
    return {f"S{i:02d}": sectors[i % len(sectors)] for i in range(n)}


# ── Tests: Basic check ────────────────────────────────────────────────────


class TestBasicCheck:
    def test_all_clear(self):
        checker = RiskLimitChecker(LimitConfig(
            max_position_weight=0.20,
            max_net_exposure=1.50,
            max_concentration_hhi=0.20,
        ))
        weights = _equal_weights(10, total=1.0)
        report = checker.check(weights, timestamp=NOW)
        assert isinstance(report, LimitCheckReport)
        assert not report.has_any_breach
        assert not report.has_hard_breach

    def test_empty_portfolio(self):
        checker = RiskLimitChecker()
        report = checker.check({}, timestamp=NOW)
        assert not report.has_any_breach
        assert report.n_positions == 0

    def test_metrics_computed(self):
        checker = RiskLimitChecker()
        weights = {"A": 0.3, "B": -0.2, "C": 0.1}
        report = checker.check(weights, timestamp=NOW)
        assert abs(report.gross_leverage - 0.6) < 1e-6
        assert abs(report.net_exposure - 0.2) < 1e-6
        assert report.n_positions == 3

    def test_timestamp_stored(self):
        checker = RiskLimitChecker()
        report = checker.check({"A": 0.1}, timestamp=NOW)
        assert report.timestamp == NOW


# ── Tests: Position limits ────────────────────────────────────────────────


class TestPositionLimits:
    def test_position_breach(self):
        checker = RiskLimitChecker(LimitConfig(max_position_weight=0.05))
        weights = {"A": 0.10, "B": 0.03}
        report = checker.check(weights, timestamp=NOW)
        assert report.has_any_breach
        pos_breaches = [b for b in report.breaches if "position(A)" in b.limit_name]
        assert len(pos_breaches) == 1

    def test_no_breach_under_limit(self):
        checker = RiskLimitChecker(LimitConfig(max_position_weight=0.20))
        weights = _equal_weights(10, total=1.0)
        report = checker.check(weights, timestamp=NOW)
        pos_breaches = [b for b in report.breaches if "position" in b.limit_name]
        assert len(pos_breaches) == 0

    def test_negative_position_checked(self):
        checker = RiskLimitChecker(LimitConfig(max_position_weight=0.05))
        weights = {"A": -0.10}
        report = checker.check(weights, timestamp=NOW)
        assert report.has_any_breach


# ── Tests: Sector limits ─────────────────────────────────────────────────


class TestSectorLimits:
    def test_sector_breach(self):
        checker = RiskLimitChecker(LimitConfig(max_sector_weight=0.15))
        weights = _equal_weights(10, total=1.0)
        sectors = _sector_map(10)
        report = checker.check(weights, sectors, timestamp=NOW)
        sector_breaches = [b for b in report.breaches if "sector" in b.limit_name]
        assert len(sector_breaches) > 0

    def test_no_sector_breach(self):
        checker = RiskLimitChecker(LimitConfig(max_sector_weight=0.50))
        weights = _equal_weights(10, total=1.0)
        sectors = _sector_map(10)
        report = checker.check(weights, sectors, timestamp=NOW)
        sector_breaches = [b for b in report.breaches if "sector" in b.limit_name]
        assert len(sector_breaches) == 0

    def test_no_sector_map(self):
        checker = RiskLimitChecker(LimitConfig(max_sector_weight=0.01))
        weights = {"A": 0.50}
        report = checker.check(weights, timestamp=NOW)
        sector_breaches = [b for b in report.breaches if "sector" in b.limit_name]
        assert len(sector_breaches) == 0


# ── Tests: Gross leverage ────────────────────────────────────────────────


class TestGrossLeverage:
    def test_leverage_breach(self):
        checker = RiskLimitChecker(LimitConfig(max_gross_leverage=1.0))
        weights = {"A": 0.6, "B": -0.6}
        report = checker.check(weights, timestamp=NOW)
        lev_breaches = [b for b in report.breaches if "gross" in b.limit_name]
        assert len(lev_breaches) == 1

    def test_no_leverage_breach(self):
        checker = RiskLimitChecker(LimitConfig(max_gross_leverage=2.0))
        weights = _equal_weights(10, total=1.0)
        report = checker.check(weights, timestamp=NOW)
        lev_breaches = [b for b in report.breaches if "gross" in b.limit_name]
        assert len(lev_breaches) == 0


# ── Tests: Net exposure ──────────────────────────────────────────────────


class TestNetExposure:
    def test_net_exposure_breach(self):
        checker = RiskLimitChecker(LimitConfig(max_net_exposure=0.20))
        weights = {"A": 0.30, "B": 0.10}
        report = checker.check(weights, timestamp=NOW)
        net_breaches = [b for b in report.breaches if "net" in b.limit_name]
        assert len(net_breaches) == 1

    def test_balanced_no_net_breach(self):
        checker = RiskLimitChecker(LimitConfig(max_net_exposure=0.10))
        weights = {"A": 0.50, "B": -0.50}
        report = checker.check(weights, timestamp=NOW)
        net_breaches = [b for b in report.breaches if "net" in b.limit_name]
        assert len(net_breaches) == 0


# ── Tests: Concentration (HHI) ───────────────────────────────────────────


class TestConcentration:
    def test_concentrated_portfolio(self):
        checker = RiskLimitChecker(LimitConfig(max_concentration_hhi=0.10))
        weights = {"A": 0.80, "B": 0.10, "C": 0.10}
        report = checker.check(weights, timestamp=NOW)
        hhi_breaches = [b for b in report.breaches if "hhi" in b.limit_name]
        assert len(hhi_breaches) == 1

    def test_diversified_portfolio(self):
        checker = RiskLimitChecker(LimitConfig(max_concentration_hhi=0.20))
        weights = _equal_weights(10, total=1.0)
        report = checker.check(weights, timestamp=NOW)
        hhi_breaches = [b for b in report.breaches if "hhi" in b.limit_name]
        assert len(hhi_breaches) == 0

    def test_hhi_disabled(self):
        checker = RiskLimitChecker(LimitConfig(max_concentration_hhi=None))
        weights = {"A": 1.0}
        report = checker.check(weights, timestamp=NOW)
        hhi_breaches = [b for b in report.breaches if "hhi" in b.limit_name]
        assert len(hhi_breaches) == 0

    def test_hhi_computed(self):
        checker = RiskLimitChecker()
        weights = _equal_weights(10, total=1.0)
        report = checker.check(weights, timestamp=NOW)
        assert abs(report.hhi - 0.10) < 1e-6


# ── Tests: Severity levels ───────────────────────────────────────────────


class TestSeverity:
    def test_warning_at_buffer(self):
        checker = RiskLimitChecker(LimitConfig(
            max_position_weight=0.10,
            warning_buffer=0.90,
            max_concentration_hhi=None,  # disable HHI for this test
        ))
        weights = {"A": 0.095}
        report = checker.check(weights, timestamp=NOW)
        assert report.has_warning
        assert not report.has_any_breach

    def test_soft_breach(self):
        checker = RiskLimitChecker(LimitConfig(max_position_weight=0.10))
        weights = {"A": 0.105}
        report = checker.check(weights, timestamp=NOW)
        pos = [b for b in report.breaches if "position" in b.limit_name]
        assert pos[0].severity == BreachSeverity.SOFT_BREACH

    def test_hard_breach(self):
        checker = RiskLimitChecker(LimitConfig(max_position_weight=0.10))
        weights = {"A": 0.15}
        report = checker.check(weights, timestamp=NOW)
        pos = [b for b in report.breaches if "position" in b.limit_name]
        assert pos[0].severity == BreachSeverity.HARD_BREACH

    def test_worst_severity(self):
        checker = RiskLimitChecker(LimitConfig(
            max_position_weight=0.10, max_gross_leverage=0.50
        ))
        weights = {"A": 0.15, "B": 0.40}
        report = checker.check(weights, timestamp=NOW)
        assert report.worst_severity == BreachSeverity.HARD_BREACH


# ── Tests: Enforcement ────────────────────────────────────────────────────


class TestEnforcement:
    def test_clamp_position(self):
        checker = RiskLimitChecker(LimitConfig(max_position_weight=0.10))
        weights = {"A": 0.20, "B": -0.15, "C": 0.05}
        enforced = checker.enforce(weights)
        assert abs(enforced["A"]) <= 0.10 + 1e-10
        assert abs(enforced["B"]) <= 0.10 + 1e-10
        assert abs(enforced["C"] - 0.05) < 1e-10

    def test_clamp_preserves_sign(self):
        checker = RiskLimitChecker(LimitConfig(max_position_weight=0.05))
        weights = {"A": 0.20, "B": -0.15}
        enforced = checker.enforce(weights)
        assert enforced["A"] > 0
        assert enforced["B"] < 0

    def test_scale_gross_leverage(self):
        checker = RiskLimitChecker(LimitConfig(
            max_position_weight=1.0, max_gross_leverage=1.0
        ))
        weights = {"A": 0.80, "B": -0.60}
        enforced = checker.enforce(weights)
        gross = sum(abs(w) for w in enforced.values())
        assert gross <= 1.0 + 1e-10

    def test_sector_scaling(self):
        checker = RiskLimitChecker(LimitConfig(
            max_position_weight=1.0, max_sector_weight=0.20,
            max_gross_leverage=10.0
        ))
        sector_map = {"A": "Tech", "B": "Tech", "C": "Finance"}
        weights = {"A": 0.15, "B": 0.15, "C": 0.10}
        enforced = checker.enforce(weights, sector_map)
        tech_gross = abs(enforced["A"]) + abs(enforced["B"])
        assert tech_gross <= 0.20 + 1e-10

    def test_compliant_unchanged(self):
        checker = RiskLimitChecker(LimitConfig(
            max_position_weight=0.20, max_gross_leverage=2.0
        ))
        weights = {"A": 0.10, "B": -0.05}
        enforced = checker.enforce(weights)
        for sym in weights:
            assert abs(enforced[sym] - weights[sym]) < 1e-10


# ── Tests: Audit trail ───────────────────────────────────────────────────


class TestAuditTrail:
    def test_audit_grows(self):
        checker = RiskLimitChecker()
        checker.check({"A": 0.05}, timestamp=NOW)
        checker.check({"A": 0.05}, timestamp=NOW)
        assert len(checker.audit_trail) == 2

    def test_audit_returns_copy(self):
        checker = RiskLimitChecker()
        checker.check({"A": 0.05}, timestamp=NOW)
        trail = checker.audit_trail
        trail.clear()
        assert len(checker.audit_trail) == 1


# ── Tests: Summary ────────────────────────────────────────────────────────


class TestSummary:
    def test_summary_all_clear(self):
        report = LimitCheckReport()
        assert "all clear" in report.summary()

    def test_summary_with_breaches(self):
        checker = RiskLimitChecker(LimitConfig(max_position_weight=0.05))
        report = checker.check({"A": 0.10}, timestamp=NOW)
        summary = report.summary()
        assert "Risk Limit Check" in summary
        assert "position(A)" in summary


# ── Tests: Config ─────────────────────────────────────────────────────────


class TestConfig:
    def test_default_config(self):
        config = LimitConfig()
        assert config.max_position_weight == 0.10
        assert config.max_gross_leverage == 1.50
        assert config.warning_buffer == 0.90

    def test_config_exposed(self):
        checker = RiskLimitChecker(LimitConfig(max_position_weight=0.05))
        assert checker.config.max_position_weight == 0.05
