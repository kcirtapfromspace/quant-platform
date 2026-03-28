"""Benchmark-relative risk and performance analytics.

Institutional portfolio management requires measuring performance and risk
relative to a benchmark. This module provides:

  * **Active risk**: tracking error, information ratio, active share.
  * **Active weight decomposition**: over/underweight positions with risk
    contribution.
  * **Brinson-style attribution**: sector allocation vs stock selection
    effects.
  * **Tracking error budget**: ex-ante TE from a covariance matrix and
    marginal contribution to TE for each position.

Usage::

    from quant.portfolio.benchmark_analytics import (
        BenchmarkAnalyzer,
        BenchmarkConfig,
    )

    analyzer = BenchmarkAnalyzer(BenchmarkConfig())
    result = analyzer.active_risk(portfolio_returns, benchmark_returns)
    print(result.summary())

    te = analyzer.tracking_error_budget(
        covariance, portfolio_weights, benchmark_weights,
    )
    print(te.summary())
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark-relative analytics.

    Attributes:
        annualise:     Whether to annualise risk/return metrics.
        trading_days:  Number of trading days per year.
        risk_free_rate: Annual risk-free rate for excess-return calculations.
    """

    annualise: bool = True
    trading_days: int = 252
    risk_free_rate: float = 0.0


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ActiveRiskResult:
    """Active risk and relative performance metrics.

    Attributes:
        active_return:        Annualised mean active return.
        tracking_error:       Annualised tracking error (std of active returns).
        information_ratio:    Active return / tracking error.
        active_share:         Sum of |w_p - w_b| / 2 (if weights provided).
        hit_rate:             Fraction of periods with positive active return.
        max_relative_dd:      Maximum relative drawdown (peak-to-trough of
                              cumulative active return).
        active_return_series: Full active return time series.
        n_periods:            Number of observation periods.
    """

    active_return: float
    tracking_error: float
    information_ratio: float
    active_share: float
    hit_rate: float
    max_relative_dd: float
    active_return_series: pd.Series
    n_periods: int

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            "Active Risk Analytics",
            "=" * 50,
            "",
            f"Active return      : {self.active_return:+.2f}%",
            f"Tracking error     : {self.tracking_error:.2f}%",
            f"Information ratio  : {self.information_ratio:+.3f}",
            f"Active share       : {self.active_share:.2%}" if self.active_share >= 0 else "",
            f"Hit rate           : {self.hit_rate:.1%}",
            f"Max relative DD    : {self.max_relative_dd:.2f}%",
            f"Periods            : {self.n_periods}",
        ]
        return "\n".join(line for line in lines if line is not None)


@dataclass(frozen=True, slots=True)
class ActivePosition:
    """Single position's benchmark-relative analytics.

    Attributes:
        symbol:           Asset identifier.
        portfolio_weight: Weight in the portfolio.
        benchmark_weight: Weight in the benchmark.
        active_weight:    portfolio_weight - benchmark_weight.
        mcte:             Marginal contribution to tracking error.
        risk_contrib_pct: Percentage of total tracking error variance.
    """

    symbol: str
    portfolio_weight: float
    benchmark_weight: float
    active_weight: float
    mcte: float
    risk_contrib_pct: float


@dataclass
class ActiveWeightResult:
    """Active weight decomposition across all positions.

    Attributes:
        positions:    List of ActivePosition, sorted by |active_weight|.
        active_share: Sum of |w_p - w_b| / 2.
        n_overweight: Number of overweight positions.
        n_underweight: Number of underweight positions.
    """

    positions: list[ActivePosition]
    active_share: float
    n_overweight: int
    n_underweight: int

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            "Active Weight Decomposition",
            "=" * 60,
            "",
            f"Active share  : {self.active_share:.2%}",
            f"Overweight    : {self.n_overweight} positions",
            f"Underweight   : {self.n_underweight} positions",
            "",
            f"{'Symbol':<8} {'Port':>8} {'Bench':>8} {'Active':>8} {'MCTE':>8} {'%TE²':>8}",
            "-" * 60,
        ]
        for p in self.positions[:15]:
            lines.append(
                f"{p.symbol:<8} {p.portfolio_weight:>7.2%} {p.benchmark_weight:>7.2%} "
                f"{p.active_weight:>+7.2%} {p.mcte:>7.4f} {p.risk_contrib_pct:>7.1%}"
            )
        if len(self.positions) > 15:
            lines.append(f"  ... and {len(self.positions) - 15} more positions")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class SectorEffect:
    """Attribution effect for a single sector.

    Attributes:
        sector:     Sector label.
        allocation: Return from over/under-weighting this sector.
        selection:  Return from stock selection within this sector.
        interaction: Interaction effect.
        total:      Total active return from this sector.
    """

    sector: str
    allocation: float
    selection: float
    interaction: float
    total: float


@dataclass
class BrinsonResult:
    """Brinson-Fachler single-period attribution decomposition.

    Attributes:
        allocation_total:  Aggregate allocation effect.
        selection_total:   Aggregate selection effect.
        interaction_total: Aggregate interaction effect.
        active_return:     Total active return (should equal sum of effects).
        sector_effects:    Per-sector breakdown.
    """

    allocation_total: float
    selection_total: float
    interaction_total: float
    active_return: float
    sector_effects: list[SectorEffect]

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            "Brinson-Fachler Attribution",
            "=" * 60,
            "",
            f"Active return  : {self.active_return:+.4%}",
            f"  Allocation   : {self.allocation_total:+.4%}",
            f"  Selection    : {self.selection_total:+.4%}",
            f"  Interaction  : {self.interaction_total:+.4%}",
            "",
            f"{'Sector':<12} {'Alloc':>9} {'Select':>9} {'Inter':>9} {'Total':>9}",
            "-" * 60,
        ]
        for se in sorted(self.sector_effects, key=lambda s: abs(s.total), reverse=True):
            lines.append(
                f"{se.sector:<12} {se.allocation:>+8.4%} {se.selection:>+8.4%} "
                f"{se.interaction:>+8.4%} {se.total:>+8.4%}"
            )
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class TEBudgetResult:
    """Ex-ante tracking error budget.

    Attributes:
        tracking_error: Annualised ex-ante tracking error.
        active_weights: Active weight vector.
        mcte:           Marginal contribution to TE per asset.
        risk_contrib:   Percentage of TE variance per asset.
        symbols:        Asset symbols in order.
    """

    tracking_error: float
    active_weights: np.ndarray
    mcte: np.ndarray
    risk_contrib: np.ndarray
    symbols: list[str]

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            "Tracking Error Budget",
            "=" * 50,
            "",
            f"Ex-ante TE : {self.tracking_error:.2f}%",
            "",
            f"{'Symbol':<8} {'Active':>8} {'MCTE':>8} {'%TE²':>8}",
            "-" * 40,
        ]
        order = np.argsort(-np.abs(self.risk_contrib))
        for idx in order[:15]:
            lines.append(
                f"{self.symbols[idx]:<8} {self.active_weights[idx]:>+7.2%} "
                f"{self.mcte[idx]:>7.4f} {self.risk_contrib[idx]:>7.1%}"
            )
        if len(self.symbols) > 15:
            lines.append(f"  ... and {len(self.symbols) - 15} more")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class BenchmarkAnalyzer:
    """Benchmark-relative analytics engine.

    Args:
        config: Configuration parameters.
    """

    def __init__(self, config: BenchmarkConfig | None = None) -> None:
        self._config = config or BenchmarkConfig()

    @property
    def config(self) -> BenchmarkConfig:
        return self._config

    # ── Active risk from return series ─────────────────────────────

    def active_risk(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        portfolio_weights: pd.Series | None = None,
        benchmark_weights: pd.Series | None = None,
    ) -> ActiveRiskResult:
        """Compute active risk metrics from return time series.

        Args:
            portfolio_returns: Portfolio return series (daily).
            benchmark_returns: Benchmark return series (daily).
            portfolio_weights: Optional current portfolio weights for
                active share calculation.
            benchmark_weights: Optional current benchmark weights.

        Returns:
            :class:`ActiveRiskResult` with active risk analytics.

        Raises:
            ValueError: If series have fewer than 2 overlapping periods.
        """
        # Align on common dates
        combined = pd.concat(
            [portfolio_returns.rename("p"), benchmark_returns.rename("b")],
            axis=1, sort=True,
        ).dropna()

        if len(combined) < 2:
            raise ValueError(
                f"Need at least 2 overlapping periods, got {len(combined)}"
            )

        active = combined["p"] - combined["b"]
        n = len(active)
        cfg = self._config
        ann = math.sqrt(cfg.trading_days) if cfg.annualise else 1.0
        ann_ret = cfg.trading_days if cfg.annualise else 1.0

        mean_active = float(active.mean()) * ann_ret * 100  # in %
        te = float(active.std(ddof=1)) * ann * 100  # in %
        ir = mean_active / te if te > 1e-15 else 0.0

        # Hit rate
        hit = float((active > 0).sum()) / n

        # Max relative drawdown
        cum_active = active.cumsum()
        running_max = cum_active.cummax()
        dd = cum_active - running_max
        max_rel_dd = float(dd.min()) * 100  # in % (negative)

        # Active share (if weights provided)
        active_share = -1.0  # sentinel for "not computed"
        if portfolio_weights is not None and benchmark_weights is not None:
            active_share = self._compute_active_share(
                portfolio_weights, benchmark_weights,
            )

        return ActiveRiskResult(
            active_return=mean_active,
            tracking_error=te,
            information_ratio=ir,
            active_share=active_share,
            hit_rate=hit,
            max_relative_dd=max_rel_dd,
            active_return_series=active,
            n_periods=n,
        )

    # ── Active weight decomposition ───────────────────────────────

    def active_weights(
        self,
        portfolio_weights: pd.Series,
        benchmark_weights: pd.Series,
        covariance: pd.DataFrame | None = None,
    ) -> ActiveWeightResult:
        """Decompose active weights and their risk contributions.

        Args:
            portfolio_weights: Portfolio weights indexed by symbol.
            benchmark_weights: Benchmark weights indexed by symbol.
            covariance:        Optional covariance matrix for MCTE
                computation. If not provided, MCTE is set to 0.

        Returns:
            :class:`ActiveWeightResult` with per-position analytics.

        Raises:
            ValueError: If weights are empty.
        """
        all_syms = sorted(set(portfolio_weights.index) | set(benchmark_weights.index))
        if not all_syms:
            raise ValueError("Weights are empty")

        wp = portfolio_weights.reindex(all_syms, fill_value=0.0)
        wb = benchmark_weights.reindex(all_syms, fill_value=0.0)
        active_w = wp - wb

        active_share = float(np.abs(active_w).sum()) / 2.0

        # MCTE and risk contribution if covariance provided
        mcte_vals = pd.Series(0.0, index=all_syms)
        risk_contrib_vals = pd.Series(0.0, index=all_syms)
        if covariance is not None:
            cov_syms = [s for s in all_syms if s in covariance.index]
            if len(cov_syms) >= 2:
                cov_sub = covariance.loc[cov_syms, cov_syms].values
                aw = active_w.reindex(cov_syms, fill_value=0.0).values

                cfg = self._config
                ann_factor = cfg.trading_days if cfg.annualise else 1.0

                te_var = float(aw @ cov_sub @ aw) * ann_factor
                te = math.sqrt(max(te_var, 0.0))

                if te > 1e-15:
                    cov_aw = cov_sub @ aw * ann_factor
                    mcte_raw = cov_aw / te
                    rc = aw * mcte_raw  # risk contribution (variance units)
                    rc_pct = rc / te_var if te_var > 1e-15 else rc * 0.0

                    for i, sym in enumerate(cov_syms):
                        mcte_vals[sym] = mcte_raw[i]
                        risk_contrib_vals[sym] = rc_pct[i]

        positions = []
        n_over = 0
        n_under = 0
        for sym in all_syms:
            aw = float(active_w[sym])
            if aw > 1e-10:
                n_over += 1
            elif aw < -1e-10:
                n_under += 1
            positions.append(ActivePosition(
                symbol=sym,
                portfolio_weight=float(wp[sym]),
                benchmark_weight=float(wb[sym]),
                active_weight=aw,
                mcte=float(mcte_vals[sym]),
                risk_contrib_pct=float(risk_contrib_vals[sym]),
            ))

        # Sort by |active_weight| descending
        positions.sort(key=lambda p: abs(p.active_weight), reverse=True)

        return ActiveWeightResult(
            positions=positions,
            active_share=active_share,
            n_overweight=n_over,
            n_underweight=n_under,
        )

    # ── Brinson-Fachler attribution ───────────────────────────────

    def brinson_attribution(
        self,
        portfolio_weights: pd.Series,
        benchmark_weights: pd.Series,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        sector_map: dict[str, str],
    ) -> BrinsonResult:
        """Single-period Brinson-Fachler attribution.

        Decomposes single-period active return into allocation,
        selection, and interaction effects per sector.

        Args:
            portfolio_weights: Portfolio weights indexed by symbol.
            benchmark_weights: Benchmark weights indexed by symbol.
            portfolio_returns: Single-period asset returns indexed by symbol.
            benchmark_returns: Single-period benchmark asset returns.
            sector_map:        Mapping symbol → sector name.

        Returns:
            :class:`BrinsonResult` with per-sector effects.

        Raises:
            ValueError: If sector_map is empty.
        """
        if not sector_map:
            raise ValueError("sector_map cannot be empty")

        all_syms = sorted(
            set(portfolio_weights.index) | set(benchmark_weights.index)
        )

        wp = portfolio_weights.reindex(all_syms, fill_value=0.0)
        wb = benchmark_weights.reindex(all_syms, fill_value=0.0)
        rp = portfolio_returns.reindex(all_syms, fill_value=0.0)
        rb = benchmark_returns.reindex(all_syms, fill_value=0.0)

        # Total benchmark return
        rb_total = float((wb * rb).sum())

        # Build sector-level aggregates
        sectors = sorted(set(sector_map.values()))
        effects = []
        alloc_total = 0.0
        select_total = 0.0
        inter_total = 0.0

        for sector in sectors:
            syms = [s for s in all_syms if sector_map.get(s) == sector]
            if not syms:
                continue

            # Sector weights
            w_ps = float(wp[syms].sum())
            w_bs = float(wb[syms].sum())

            # Sector returns (weighted average within sector)
            r_ps = float((wp[syms] * rp[syms]).sum()) / w_ps if w_ps > 1e-15 else 0.0
            r_bs = float((wb[syms] * rb[syms]).sum()) / w_bs if w_bs > 1e-15 else 0.0

            # Brinson-Fachler
            allocation = (w_ps - w_bs) * (r_bs - rb_total)
            selection = w_bs * (r_ps - r_bs)
            interaction = (w_ps - w_bs) * (r_ps - r_bs)

            alloc_total += allocation
            select_total += selection
            inter_total += interaction

            effects.append(SectorEffect(
                sector=sector,
                allocation=allocation,
                selection=selection,
                interaction=interaction,
                total=allocation + selection + interaction,
            ))

        active_return = alloc_total + select_total + inter_total

        return BrinsonResult(
            allocation_total=alloc_total,
            selection_total=select_total,
            interaction_total=inter_total,
            active_return=active_return,
            sector_effects=effects,
        )

    # ── Ex-ante tracking error budget ─────────────────────────────

    def tracking_error_budget(
        self,
        covariance: pd.DataFrame,
        portfolio_weights: pd.Series,
        benchmark_weights: pd.Series,
    ) -> TEBudgetResult:
        """Compute ex-ante tracking error and risk contribution per asset.

        Args:
            covariance:        Asset return covariance matrix.
            portfolio_weights: Portfolio weights indexed by symbol.
            benchmark_weights: Benchmark weights indexed by symbol.

        Returns:
            :class:`TEBudgetResult` with TE decomposition.

        Raises:
            ValueError: If covariance matrix is too small.
        """
        if covariance.shape[0] < 2:
            raise ValueError("Covariance matrix must have at least 2 assets")

        symbols = list(covariance.index)
        wp = portfolio_weights.reindex(symbols, fill_value=0.0).values
        wb = benchmark_weights.reindex(symbols, fill_value=0.0).values
        aw = wp - wb

        cov = covariance.values
        cfg = self._config
        ann_factor = cfg.trading_days if cfg.annualise else 1.0

        te_var = float(aw @ cov @ aw) * ann_factor
        te = math.sqrt(max(te_var, 0.0)) * 100  # in %

        # Marginal contribution to TE
        cov_aw = cov @ aw * ann_factor
        mcte = cov_aw / (te / 100) if te > 1e-15 else np.zeros(len(symbols))

        # Risk contribution (fraction of TE variance)
        rc = aw * cov_aw
        rc_pct = rc / te_var if te_var > 1e-15 else np.zeros(len(symbols))

        return TEBudgetResult(
            tracking_error=te,
            active_weights=aw,
            mcte=mcte,
            risk_contrib=rc_pct,
            symbols=symbols,
        )

    # ── Internal ──────────────────────────────────────────────────

    @staticmethod
    def _compute_active_share(
        portfolio_weights: pd.Series,
        benchmark_weights: pd.Series,
    ) -> float:
        """Compute active share: sum(|w_p - w_b|) / 2."""
        all_syms = sorted(set(portfolio_weights.index) | set(benchmark_weights.index))
        wp = portfolio_weights.reindex(all_syms, fill_value=0.0)
        wb = benchmark_weights.reindex(all_syms, fill_value=0.0)
        return float(np.abs(wp - wb).sum()) / 2.0
