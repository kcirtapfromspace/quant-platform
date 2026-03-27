"""Portfolio-level risk limit checking, enforcement, and audit.

Validates an entire portfolio weight vector against a configurable set
of limits and provides:

  * **Breach detection** with severity levels (warning / soft / hard).
  * **Position clamping** — scale weights to bring the portfolio back
    into compliance.
  * **Concentration check** — HHI-based diversification floor.
  * **Audit trail** — per-check history with timestamps.

Builds on :class:`ExposureLimits` (which handles individual checks)
by running all limits against a full portfolio snapshot and producing
a structured report.

Usage::

    from quant.risk.limit_checker import RiskLimitChecker, LimitConfig

    checker = RiskLimitChecker(LimitConfig(
        max_position_weight=0.10,
        max_gross_leverage=1.5,
        max_concentration_hhi=0.15,
    ))
    report = checker.check(weights, sector_map, capital)
    if report.has_hard_breach:
        weights = checker.enforce(weights, sector_map, capital)
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class LimitConfig:
    """Risk limit configuration.

    Attributes:
        max_position_weight:    Maximum absolute weight for any single
                                position.
        max_sector_weight:      Maximum gross weight per sector.
        max_gross_leverage:     Maximum sum of absolute weights.
        max_net_exposure:       Maximum absolute net exposure (sum of
                                signed weights).
        max_concentration_hhi:  Maximum Herfindahl-Hirschman Index of
                                weight squares (higher = more concentrated).
                                Set to None to disable.
        warning_buffer:         Fraction of limit at which to issue a
                                warning (e.g. 0.90 = warn at 90% of limit).
    """

    max_position_weight: float = 0.10
    max_sector_weight: float = 0.30
    max_gross_leverage: float = 1.50
    max_net_exposure: float = 0.50
    max_concentration_hhi: float | None = 0.15
    warning_buffer: float = 0.90


# ---------------------------------------------------------------------------
# Breach types
# ---------------------------------------------------------------------------


class BreachSeverity(enum.Enum):
    """Severity of a risk limit breach."""

    OK = "ok"
    WARNING = "warning"
    SOFT_BREACH = "soft_breach"
    HARD_BREACH = "hard_breach"


@dataclass(frozen=True, slots=True)
class LimitBreach:
    """A single limit breach or warning.

    Attributes:
        limit_name:   Which limit was tested.
        severity:     Breach severity.
        current:      Current value.
        limit:        Limit threshold.
        utilisation:  current / limit (1.0 = at limit, >1.0 = breach).
        detail:       Human-readable description.
    """

    limit_name: str
    severity: BreachSeverity
    current: float
    limit: float
    utilisation: float
    detail: str


# ---------------------------------------------------------------------------
# Check report
# ---------------------------------------------------------------------------


@dataclass
class LimitCheckReport:
    """Result of checking all limits against a portfolio.

    Attributes:
        breaches:       List of all breaches and warnings found.
        timestamp:      When the check was performed.
        n_positions:    Number of positions in the portfolio.
        gross_leverage: Current gross leverage.
        net_exposure:   Current net exposure.
        hhi:            Herfindahl-Hirschman Index.
    """

    breaches: list[LimitBreach] = field(default_factory=list)
    timestamp: datetime | None = None
    n_positions: int = 0
    gross_leverage: float = 0.0
    net_exposure: float = 0.0
    hhi: float = 0.0

    @property
    def has_hard_breach(self) -> bool:
        return any(b.severity == BreachSeverity.HARD_BREACH for b in self.breaches)

    @property
    def has_any_breach(self) -> bool:
        return any(
            b.severity in (BreachSeverity.SOFT_BREACH, BreachSeverity.HARD_BREACH)
            for b in self.breaches
        )

    @property
    def has_warning(self) -> bool:
        return any(b.severity == BreachSeverity.WARNING for b in self.breaches)

    @property
    def worst_severity(self) -> BreachSeverity:
        if not self.breaches:
            return BreachSeverity.OK
        severity_order = {
            BreachSeverity.OK: 0,
            BreachSeverity.WARNING: 1,
            BreachSeverity.SOFT_BREACH: 2,
            BreachSeverity.HARD_BREACH: 3,
        }
        return max(self.breaches, key=lambda b: severity_order[b.severity]).severity

    def summary(self) -> str:
        """Human-readable limit check summary."""
        if not self.breaches:
            return "Risk Limits: all clear"

        lines = [
            "Risk Limit Check",
            "=" * 60,
            f"  Positions: {self.n_positions}  Gross: {self.gross_leverage:.2f}"
            f"  Net: {self.net_exposure:+.2f}  HHI: {self.hhi:.4f}",
            "",
            f"  {'Limit':<25}{'Severity':<15}{'Current':>10}{'Limit':>10}{'Util':>8}",
            "-" * 60,
        ]

        for b in self.breaches:
            lines.append(
                f"  {b.limit_name:<25}{b.severity.value:<15}"
                f"{b.current:>10.4f}{b.limit:>10.4f}{b.utilisation:>7.0%}"
            )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Checker
# ---------------------------------------------------------------------------


class RiskLimitChecker:
    """Check and enforce portfolio-level risk limits.

    Args:
        config: Limit configuration.
    """

    def __init__(self, config: LimitConfig | None = None) -> None:
        self._config = config or LimitConfig()
        self._audit: list[LimitCheckReport] = []

    @property
    def config(self) -> LimitConfig:
        return self._config

    @property
    def audit_trail(self) -> list[LimitCheckReport]:
        """Historical check reports."""
        return list(self._audit)

    def check(
        self,
        weights: dict[str, float],
        sector_map: dict[str, str] | None = None,
        timestamp: datetime | None = None,
    ) -> LimitCheckReport:
        """Check all limits against the current portfolio weights.

        Args:
            weights:     ``{symbol: weight}`` — signed position weights.
            sector_map:  ``{symbol: sector}`` — sector classification.
            timestamp:   Check timestamp (defaults to now).

        Returns:
            :class:`LimitCheckReport` with all breaches and warnings.
        """
        timestamp = timestamp or datetime.now()
        sector_map = sector_map or {}

        breaches: list[LimitBreach] = []

        # Portfolio-level metrics
        gross = sum(abs(w) for w in weights.values())
        net = sum(weights.values())
        hhi = sum((w / gross) ** 2 for w in weights.values()) if gross > 0 else 0.0
        n = sum(1 for w in weights.values() if abs(w) > 1e-10)

        # 1. Per-position limits
        for sym, w in weights.items():
            aw = abs(w)
            if aw < 1e-10:
                continue
            breaches.extend(
                self._check_threshold(
                    f"position({sym})",
                    aw,
                    self._config.max_position_weight,
                )
            )

        # 2. Sector limits
        if sector_map:
            sector_gross: dict[str, float] = {}
            for sym, w in weights.items():
                sector = sector_map.get(sym)
                if sector:
                    sector_gross[sector] = sector_gross.get(sector, 0.0) + abs(w)

            for sector, sw in sector_gross.items():
                breaches.extend(
                    self._check_threshold(
                        f"sector({sector})",
                        sw,
                        self._config.max_sector_weight,
                    )
                )

        # 3. Gross leverage
        breaches.extend(
            self._check_threshold(
                "gross_leverage", gross, self._config.max_gross_leverage
            )
        )

        # 4. Net exposure
        breaches.extend(
            self._check_threshold(
                "net_exposure", abs(net), self._config.max_net_exposure
            )
        )

        # 5. Concentration (HHI)
        if self._config.max_concentration_hhi is not None and gross > 0:
            breaches.extend(
                self._check_threshold(
                    "concentration_hhi",
                    hhi,
                    self._config.max_concentration_hhi,
                )
            )

        report = LimitCheckReport(
            breaches=breaches,
            timestamp=timestamp,
            n_positions=n,
            gross_leverage=gross,
            net_exposure=net,
            hhi=hhi,
        )

        self._audit.append(report)
        return report

    def enforce(
        self,
        weights: dict[str, float],
        sector_map: dict[str, str] | None = None,
    ) -> dict[str, float]:
        """Enforce limits by clamping weights.

        Applies the following adjustments in order:

        1. Clamp individual positions to ``max_position_weight``.
        2. Scale down sector exposures exceeding ``max_sector_weight``.
        3. Scale entire portfolio if gross leverage exceeds limit.

        Args:
            weights:     ``{symbol: signed_weight}``.
            sector_map:  ``{symbol: sector}``.

        Returns:
            Adjusted weights dict.
        """
        sector_map = sector_map or {}
        result = dict(weights)

        # 1. Clamp individual positions
        max_pos = self._config.max_position_weight
        for sym in result:
            if abs(result[sym]) > max_pos:
                sign = 1.0 if result[sym] > 0 else -1.0
                result[sym] = sign * max_pos

        # 2. Scale down breaching sectors
        if sector_map:
            max_sec = self._config.max_sector_weight
            sector_members: dict[str, list[str]] = {}
            for sym in result:
                sec = sector_map.get(sym)
                if sec:
                    sector_members.setdefault(sec, []).append(sym)

            for _sector, members in sector_members.items():
                sector_gross = sum(abs(result[m]) for m in members)
                if sector_gross > max_sec:
                    scale = max_sec / sector_gross
                    for m in members:
                        result[m] *= scale

        # 3. Scale gross leverage
        max_gross = self._config.max_gross_leverage
        gross = sum(abs(w) for w in result.values())
        if gross > max_gross:
            scale = max_gross / gross
            result = {s: w * scale for s, w in result.items()}

        return result

    # ── Internal ──────────────────────────────────────────────────

    def _check_threshold(
        self, name: str, current: float, limit: float
    ) -> list[LimitBreach]:
        """Check a single metric against its threshold.

        Returns 0 or 1 breach depending on current value vs limit.
        """
        if limit <= 0:
            return []

        utilisation = current / limit

        if utilisation > 1.0:
            # Hard breach if >10% over, soft otherwise
            severity = (
                BreachSeverity.HARD_BREACH
                if utilisation > 1.10
                else BreachSeverity.SOFT_BREACH
            )
            return [
                LimitBreach(
                    limit_name=name,
                    severity=severity,
                    current=current,
                    limit=limit,
                    utilisation=utilisation,
                    detail=f"{name}: {current:.4f} exceeds limit {limit:.4f}",
                )
            ]

        if utilisation >= self._config.warning_buffer:
            return [
                LimitBreach(
                    limit_name=name,
                    severity=BreachSeverity.WARNING,
                    current=current,
                    limit=limit,
                    utilisation=utilisation,
                    detail=f"{name}: {current:.4f} approaching limit {limit:.4f}",
                )
            ]

        return []
