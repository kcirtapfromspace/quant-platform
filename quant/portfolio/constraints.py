"""Portfolio constraints for optimization.

Constraints are applied during portfolio construction to enforce:
- Long-only or long-short bounds
- Per-asset position caps
- Sector/group exposure limits
- Turnover budgets (max fraction of portfolio traded per rebalance)
- Gross and net exposure limits
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PortfolioConstraints:
    """Constraint set for portfolio optimisation.

    Attributes:
        long_only:          If True, all weights must be >= 0.
        min_weight:         Minimum weight per asset (ignored if long_only is True,
                            in which case the effective minimum is 0).
        max_weight:         Maximum weight per asset.
        max_sector_weight:  Maximum total weight in any single sector.
        max_gross_exposure: Maximum sum of |weights| (e.g. 1.0 for unlevered).
        max_net_exposure:   Maximum |sum of weights|.
        max_turnover:       Maximum one-way turnover per rebalance (sum of |delta|).
        sector_map:         Dict mapping symbol → sector label.
    """

    long_only: bool = False
    min_weight: float = -1.0
    max_weight: float = 1.0
    max_sector_weight: float = 1.0
    max_gross_exposure: float = 1.0
    max_net_exposure: float = 1.0
    max_turnover: float | None = None
    sector_map: dict[str, str] = field(default_factory=dict)

    def effective_min(self) -> float:
        """Return the effective minimum weight, accounting for long-only."""
        return 0.0 if self.long_only else self.min_weight

    def validate_weights(
        self,
        weights: dict[str, float],
        current_weights: dict[str, float] | None = None,
    ) -> tuple[bool, list[str]]:
        """Check whether a weight vector satisfies all constraints.

        Args:
            weights:         Proposed {symbol: weight} allocation.
            current_weights: Current portfolio weights (for turnover check).

        Returns:
            (is_valid, list_of_violations) — empty list means all constraints pass.
        """
        violations: list[str] = []
        eff_min = self.effective_min()

        for sym, w in weights.items():
            if w < eff_min - 1e-9:
                violations.append(
                    f"{sym}: weight {w:.4f} below minimum {eff_min:.4f}"
                )
            if w > self.max_weight + 1e-9:
                violations.append(
                    f"{sym}: weight {w:.4f} above maximum {self.max_weight:.4f}"
                )

        # Sector exposure
        if self.sector_map:
            sector_totals: dict[str, float] = {}
            for sym, w in weights.items():
                sector = self.sector_map.get(sym, "unknown")
                sector_totals[sector] = sector_totals.get(sector, 0.0) + abs(w)
            for sector, total in sector_totals.items():
                if total > self.max_sector_weight + 1e-9:
                    violations.append(
                        f"sector {sector}: exposure {total:.4f} above limit "
                        f"{self.max_sector_weight:.4f}"
                    )

        # Gross exposure
        gross = sum(abs(w) for w in weights.values())
        if gross > self.max_gross_exposure + 1e-9:
            violations.append(
                f"gross exposure {gross:.4f} above limit {self.max_gross_exposure:.4f}"
            )

        # Net exposure
        net = abs(sum(weights.values()))
        if net > self.max_net_exposure + 1e-9:
            violations.append(
                f"net exposure {net:.4f} above limit {self.max_net_exposure:.4f}"
            )

        # Turnover
        if self.max_turnover is not None and current_weights is not None:
            all_syms = set(weights) | set(current_weights)
            turnover = sum(
                abs(weights.get(s, 0.0) - current_weights.get(s, 0.0))
                for s in all_syms
            )
            if turnover > self.max_turnover + 1e-9:
                violations.append(
                    f"turnover {turnover:.4f} above limit {self.max_turnover:.4f}"
                )

        return (len(violations) == 0, violations)

    def clip_weights(self, weights: dict[str, float]) -> dict[str, float]:
        """Clip weights to satisfy per-asset bounds.

        Does NOT enforce sector, gross, or turnover constraints — those require
        solving an optimisation problem.  This is a fast heuristic for the
        simple bound constraints only.
        """
        eff_min = self.effective_min()
        return {
            sym: max(eff_min, min(self.max_weight, w))
            for sym, w in weights.items()
        }

    def scale_to_gross_exposure(
        self, weights: dict[str, float]
    ) -> dict[str, float]:
        """Proportionally scale weights so gross exposure equals max_gross_exposure."""
        gross = sum(abs(w) for w in weights.values())
        if gross < 1e-12:
            return weights
        scale = self.max_gross_exposure / gross
        if scale >= 1.0:
            return weights
        return {sym: w * scale for sym, w in weights.items()}
