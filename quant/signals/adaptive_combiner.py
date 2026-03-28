"""Adaptive signal combination using rolling IC-based weights.

Dynamically weights signals based on their recent predictive power,
measured by cross-sectional Information Coefficient (rank IC).  This
replaces naive equal-weight or static-weight combination with a
data-driven feedback loop:

  1. Each cycle, record cross-sectional IC for every active signal.
  2. Compute exponentially-weighted mean IC over a lookback window.
  3. Derive signal weights proportional to IC (with shrinkage toward
     equal weights for stability).
  4. Combine signals using these adaptive weights.

Signals with IC below a configurable threshold are zeroed out, so
the portfolio naturally concentrates on signals that are currently
working.

Usage::

    from quant.signals.adaptive_combiner import (
        AdaptiveSignalCombiner,
        AdaptiveCombinerConfig,
    )

    combiner = AdaptiveSignalCombiner()

    # Each rebalance cycle:
    combiner.update(signal_scores, forward_returns, timestamp)
    alpha = combiner.combine("AAPL", timestamp, signal_outputs)
    # Or for the whole universe:
    alphas = combiner.combine_universe(timestamp, universe_signals)
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from quant.signals.base import SignalOutput

if TYPE_CHECKING:
    from quant.portfolio.alpha import AlphaScore


def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AdaptiveCombinerConfig:
    """Configuration for adaptive IC-weighted signal combination.

    Attributes:
        ic_lookback:    Maximum number of IC observations to retain per signal.
        min_ic_periods: Minimum IC observations before switching from equal
                        weights to IC-weighted.  Below this threshold the
                        combiner falls back to equal weighting.
        min_ic:         Minimum exponentially-weighted mean IC for a signal
                        to receive nonzero weight.  Set to 0 to include all.
        ic_halflife:    Half-life (in observations) for exponential decay
                        when averaging IC.  Recent IC matters more.
        shrinkage:      Shrinkage toward equal weights.  0.0 = pure IC
                        weights, 1.0 = pure equal weights.  Typical 0.2–0.4.
        min_assets:     Minimum assets with both signal and return data for
                        a valid cross-sectional IC observation.
    """

    ic_lookback: int = 126
    min_ic_periods: int = 20
    min_ic: float = 0.0
    ic_halflife: int = 21
    shrinkage: float = 0.3
    min_assets: int = 3


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AdaptiveWeights:
    """Snapshot of current adaptive signal weights.

    Attributes:
        weights:      Signal name → normalised weight (sums to 1.0).
        ic_stats:     Signal name → exponentially-weighted mean IC.
        method_used:  'ic_weighted' or 'equal_fallback'.
        n_signals:    Number of signals with nonzero weight.
    """

    weights: dict[str, float]
    ic_stats: dict[str, float]
    method_used: str
    n_signals: int


# ---------------------------------------------------------------------------
# Combiner
# ---------------------------------------------------------------------------


class AdaptiveSignalCombiner:
    """Combine signals with IC-adaptive weights.

    Args:
        config: Combiner configuration.
    """

    def __init__(self, config: AdaptiveCombinerConfig | None = None) -> None:
        self._config = config or AdaptiveCombinerConfig()
        # Per-signal rolling IC history: signal_name → list of (timestamp, ic)
        self._ic_history: dict[str, list[tuple[datetime, float]]] = defaultdict(
            list
        )

    @property
    def config(self) -> AdaptiveCombinerConfig:
        return self._config

    # ── Update IC history ─────────────────────────────────────────

    def update(
        self,
        signal_scores: dict[str, pd.Series],
        asset_returns: pd.Series,
        timestamp: datetime,
    ) -> dict[str, float]:
        """Record one cross-sectional IC observation per signal.

        Args:
            signal_scores: ``{signal_name: Series}`` where each Series is
                indexed by symbol with today's signal score.
            asset_returns:  Series indexed by symbol with today's (or
                forward) returns.
            timestamp:      Current observation timestamp.

        Returns:
            Dict of ``{signal_name: ic}`` for this observation.
        """
        ics: dict[str, float] = {}

        for name, scores in signal_scores.items():
            common = scores.dropna().index.intersection(
                asset_returns.dropna().index
            )
            if len(common) < self._config.min_assets:
                continue

            s = scores.reindex(common).values
            r = asset_returns.reindex(common).values

            if np.std(s) < 1e-10 or np.std(r) < 1e-10:
                continue

            ic = _spearman_r(s, r)
            if np.isnan(ic):
                continue

            self._ic_history[name].append((timestamp, ic))

            # Trim to lookback
            if len(self._ic_history[name]) > self._config.ic_lookback:
                self._ic_history[name] = self._ic_history[name][
                    -self._config.ic_lookback :
                ]

            ics[name] = ic

        return ics

    # ── Compute adaptive weights ──────────────────────────────────

    def get_weights(
        self, signal_names: list[str] | None = None
    ) -> AdaptiveWeights:
        """Compute current adaptive weights based on IC history.

        Args:
            signal_names: If provided, compute weights only for these
                signals.  Otherwise use all signals with IC history.

        Returns:
            :class:`AdaptiveWeights` with normalised weights.
        """
        if signal_names is None:
            signal_names = list(self._ic_history.keys())

        if not signal_names:
            return AdaptiveWeights(
                weights={},
                ic_stats={},
                method_used="equal_fallback",
                n_signals=0,
            )

        # Check if we have enough IC data for any signal
        has_enough = any(
            len(self._ic_history.get(n, [])) >= self._config.min_ic_periods
            for n in signal_names
        )

        if not has_enough:
            # Fall back to equal weights
            w = 1.0 / len(signal_names)
            return AdaptiveWeights(
                weights=dict.fromkeys(signal_names, w),
                ic_stats=dict.fromkeys(signal_names, 0.0),
                method_used="equal_fallback",
                n_signals=len(signal_names),
            )

        # Compute exponentially-weighted mean IC per signal
        ic_means: dict[str, float] = {}
        for name in signal_names:
            history = self._ic_history.get(name, [])
            if len(history) < self._config.min_ic_periods:
                ic_means[name] = 0.0
            else:
                ics = np.array([ic for _, ic in history])
                ic_means[name] = self._ewm_mean(ics)

        # Apply min_ic threshold
        active: dict[str, float] = {
            n: ic for n, ic in ic_means.items() if ic >= self._config.min_ic
        }

        if not active:
            # All signals below threshold → equal weight among all
            w = 1.0 / len(signal_names)
            return AdaptiveWeights(
                weights=dict.fromkeys(signal_names, w),
                ic_stats=ic_means,
                method_used="equal_fallback",
                n_signals=len(signal_names),
            )

        # IC-proportional weights (only positive IC contributes)
        raw_ic_weights: dict[str, float] = {}
        for name in signal_names:
            if name in active and active[name] > 0:
                raw_ic_weights[name] = active[name]
            else:
                raw_ic_weights[name] = 0.0

        total_ic = sum(raw_ic_weights.values())
        if total_ic < 1e-12:
            # All IC ≤ 0 → equal weight
            w = 1.0 / len(signal_names)
            return AdaptiveWeights(
                weights=dict.fromkeys(signal_names, w),
                ic_stats=ic_means,
                method_used="equal_fallback",
                n_signals=len(signal_names),
            )

        ic_weights = {n: v / total_ic for n, v in raw_ic_weights.items()}

        # Shrink toward equal weights
        eq_w = 1.0 / len(signal_names)
        shrink = self._config.shrinkage
        final_weights = {
            n: (1.0 - shrink) * ic_weights[n] + shrink * eq_w
            for n in signal_names
        }

        # Renormalise (shrinkage can shift total slightly)
        total = sum(final_weights.values())
        if total > 1e-12:
            final_weights = {n: v / total for n, v in final_weights.items()}

        n_active = sum(1 for v in final_weights.values() if v > 1e-8)

        return AdaptiveWeights(
            weights=final_weights,
            ic_stats=ic_means,
            method_used="ic_weighted",
            n_signals=n_active,
        )

    # ── Combine signals ───────────────────────────────────────────

    def combine(
        self,
        symbol: str,
        timestamp: datetime,
        signals: list[SignalOutput],
    ) -> AlphaScore:
        """Combine signal outputs using adaptive IC-based weights.

        Args:
            symbol:    Ticker symbol.
            timestamp: Current bar timestamp.
            signals:   Signal outputs to combine.

        Returns:
            :class:`AlphaScore` with weighted score and contributions.
        """
        from quant.portfolio.alpha import AlphaScore  # noqa: PLC0415

        if not signals:
            return AlphaScore(
                symbol=symbol,
                timestamp=timestamp,
                score=0.0,
                confidence=0.0,
            )

        signal_names = [
            sig.metadata.get("signal_name", f"signal_{i}")
            for i, sig in enumerate(signals)
        ]
        aw = self.get_weights(signal_names)

        contributions: dict[str, float] = {}
        weighted_score = 0.0
        weighted_conf = 0.0

        for sig, name in zip(signals, signal_names, strict=True):
            w = aw.weights.get(name, 0.0)
            contributions[name] = sig.score * w
            weighted_score += sig.score * w
            weighted_conf += sig.confidence * w

        return AlphaScore(
            symbol=symbol,
            timestamp=timestamp,
            score=_clamp(weighted_score),
            confidence=_clamp(weighted_conf, 0.0, 1.0),
            signal_contributions=contributions,
        )

    def combine_universe(
        self,
        timestamp: datetime,
        universe_signals: dict[str, list[SignalOutput]],
    ) -> dict[str, AlphaScore]:
        """Combine signals for every symbol in the universe.

        Args:
            timestamp:        Current bar timestamp.
            universe_signals: ``{symbol: [SignalOutput, ...]}``.

        Returns:
            ``{symbol: AlphaScore}``.
        """
        return {
            sym: self.combine(sym, timestamp, sigs)
            for sym, sigs in universe_signals.items()
        }

    # ── IC history accessors ──────────────────────────────────────

    def ic_history(self, signal_name: str) -> pd.Series:
        """Return the IC time series for a signal.

        Returns:
            Series indexed by timestamp with IC values.
        """
        history = self._ic_history.get(signal_name, [])
        if not history:
            return pd.Series(dtype=float, name=signal_name)
        timestamps, ics = zip(*history, strict=True)
        return pd.Series(ics, index=timestamps, name=signal_name)

    @property
    def tracked_signals(self) -> list[str]:
        """Signal names with IC history."""
        return list(self._ic_history.keys())

    def reset(self) -> None:
        """Clear all IC history."""
        self._ic_history.clear()

    # ── Summary ───────────────────────────────────────────────────

    def summary(self, signal_names: list[str] | None = None) -> str:
        """Human-readable summary of current adaptive weights."""
        aw = self.get_weights(signal_names)

        if not aw.weights:
            return "Adaptive Signal Combiner: no signals tracked"

        lines = [
            "Adaptive Signal Combiner",
            "=" * 60,
            f"  Method: {aw.method_used}",
            f"  Active signals: {aw.n_signals}",
            f"  Shrinkage: {self._config.shrinkage:.2f}",
            f"  IC half-life: {self._config.ic_halflife} obs",
            "",
            f"  {'Signal':<25}{'Weight':>10}{'Mean IC':>12}{'N obs':>8}",
            "-" * 60,
        ]

        for name in sorted(aw.weights, key=lambda n: aw.weights[n], reverse=True):
            w = aw.weights[name]
            ic = aw.ic_stats.get(name, 0.0)
            n = len(self._ic_history.get(name, []))
            lines.append(f"  {name:<25}{w:>9.1%}{ic:>+12.4f}{n:>8}")

        return "\n".join(lines)

    # ── Internal ──────────────────────────────────────────────────

    def _ewm_mean(self, values: np.ndarray) -> float:
        """Exponentially-weighted mean with configured half-life."""
        n = len(values)
        if n == 0:
            return 0.0
        alpha = 1.0 - np.exp(-np.log(2.0) / self._config.ic_halflife)
        weights = np.array(
            [(1.0 - alpha) ** (n - 1 - i) for i in range(n)]
        )
        weights /= weights.sum()
        return float(weights @ values)


# ---------------------------------------------------------------------------
# Bayesian IC estimator — Normal-Gamma conjugate posterior
# ---------------------------------------------------------------------------


class _NormalGammaTracker:
    """O(1) Normal-Gamma conjugate posterior for online IC estimation.

    Hyperpriors: mu0=0, kappa0=1, alpha0=2, beta0=1 (weakly informative,
    centred on zero IC).  Posterior mean converges to sample mean as n→∞.
    Prior shrinks early estimates toward zero, reducing whipsaw during
    combiner warmup compared to EWM.
    """

    __slots__ = ("mu_n", "kappa_n", "alpha_n", "beta_n", "n")

    def __init__(self) -> None:
        self.mu_n: float = 0.0
        self.kappa_n: float = 1.0
        self.alpha_n: float = 2.0
        self.beta_n: float = 1.0
        self.n: int = 0

    def update(self, x: float) -> None:
        kappa_prev = self.kappa_n
        mu_prev = self.mu_n
        self.kappa_n = kappa_prev + 1.0
        self.mu_n = (kappa_prev * mu_prev + x) / self.kappa_n
        self.alpha_n += 0.5
        residual = x - mu_prev
        self.beta_n += kappa_prev * residual * residual / (2.0 * self.kappa_n)
        self.n += 1

    @property
    def posterior_mean(self) -> float:
        return self.mu_n


# ---------------------------------------------------------------------------
# Bayesian config and combiner
# ---------------------------------------------------------------------------


@dataclass
class BayesianAdaptiveCombinerConfig(AdaptiveCombinerConfig):
    """AdaptiveCombinerConfig variant that selects Bayesian IC weighting.

    Uses Normal-Gamma conjugate posterior mean as the IC estimate instead
    of EWM mean.  The prior (mu0=0) provides automatic shrinkage toward
    zero IC during the warmup period, reducing false-positive weight
    concentration before sufficient data has accumulated.

    All other parameters (ic_lookback, min_ic_periods, shrinkage, etc.)
    are inherited from AdaptiveCombinerConfig and have the same effect.
    """


class BayesianAdaptiveSignalCombiner(AdaptiveSignalCombiner):
    """AdaptiveSignalCombiner that uses Normal-Gamma posterior IC estimation.

    Replaces the EWM IC mean with the Normal-Gamma posterior mean.
    The posterior mean is a Bayesian shrinkage estimator: with few
    observations it stays close to zero (the prior), and with many
    observations it converges to the sample mean.

    Interface is identical to AdaptiveSignalCombiner; swap in by passing
    BayesianAdaptiveCombinerConfig to SleeveConfig.adaptive_combiner_config.
    """

    def __init__(
        self, config: BayesianAdaptiveCombinerConfig | None = None
    ) -> None:
        super().__init__(config or BayesianAdaptiveCombinerConfig())
        self._ng_trackers: dict[str, _NormalGammaTracker] = {}

    def update(
        self,
        signal_scores: dict[str, pd.Series],
        asset_returns: pd.Series,
        timestamp: datetime,
    ) -> dict[str, float]:
        """Record IC and update Normal-Gamma trackers per signal."""
        ics = super().update(signal_scores, asset_returns, timestamp)
        for name, ic in ics.items():
            if name not in self._ng_trackers:
                self._ng_trackers[name] = _NormalGammaTracker()
            self._ng_trackers[name].update(ic)
        return ics

    def get_weights(
        self, signal_names: list[str] | None = None
    ) -> AdaptiveWeights:
        """Compute adaptive weights using Normal-Gamma posterior IC means."""
        if signal_names is None:
            signal_names = list(self._ic_history.keys())

        if not signal_names:
            return AdaptiveWeights(
                weights={},
                ic_stats={},
                method_used="equal_fallback",
                n_signals=0,
            )

        has_enough = any(
            len(self._ic_history.get(n, [])) >= self._config.min_ic_periods
            for n in signal_names
        )

        if not has_enough:
            w = 1.0 / len(signal_names)
            return AdaptiveWeights(
                weights=dict.fromkeys(signal_names, w),
                ic_stats=dict.fromkeys(signal_names, 0.0),
                method_used="equal_fallback",
                n_signals=len(signal_names),
            )

        # Use NormalGamma posterior means instead of EWM
        ic_means: dict[str, float] = {}
        for name in signal_names:
            tracker = self._ng_trackers.get(name)
            if tracker is None or tracker.n < self._config.min_ic_periods:
                ic_means[name] = 0.0
            else:
                ic_means[name] = tracker.posterior_mean

        active = {n: ic for n, ic in ic_means.items() if ic >= self._config.min_ic}

        if not active:
            w = 1.0 / len(signal_names)
            return AdaptiveWeights(
                weights=dict.fromkeys(signal_names, w),
                ic_stats=ic_means,
                method_used="equal_fallback",
                n_signals=len(signal_names),
            )

        raw_ic_weights: dict[str, float] = {
            n: (active[n] if n in active and active[n] > 0 else 0.0)
            for n in signal_names
        }
        total_ic = sum(raw_ic_weights.values())

        if total_ic < 1e-12:
            w = 1.0 / len(signal_names)
            return AdaptiveWeights(
                weights=dict.fromkeys(signal_names, w),
                ic_stats=ic_means,
                method_used="equal_fallback",
                n_signals=len(signal_names),
            )

        ic_weights = {n: v / total_ic for n, v in raw_ic_weights.items()}
        eq_w = 1.0 / len(signal_names)
        shrink = self._config.shrinkage
        final_weights = {
            n: (1.0 - shrink) * ic_weights[n] + shrink * eq_w
            for n in signal_names
        }
        total = sum(final_weights.values())
        if total > 1e-12:
            final_weights = {n: v / total for n, v in final_weights.items()}

        n_active = sum(1 for v in final_weights.values() if v > 1e-8)
        return AdaptiveWeights(
            weights=final_weights,
            ic_stats=ic_means,
            method_used="bayesian_ng",
            n_signals=n_active,
        )

    def reset(self) -> None:
        super().reset()
        self._ng_trackers.clear()


# ---------------------------------------------------------------------------
# Pure-numpy Spearman (shared with decay module)
# ---------------------------------------------------------------------------


def _spearman_r(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Spearman rank correlation between two 1-D arrays."""
    n = len(a)
    if n < 2:
        return float("nan")

    rank_a = np.empty(n, dtype=float)
    rank_b = np.empty(n, dtype=float)
    rank_a[np.argsort(a)] = np.arange(n, dtype=float)
    rank_b[np.argsort(b)] = np.arange(n, dtype=float)

    ra = rank_a - rank_a.mean()
    rb = rank_b - rank_b.mean()
    denom = np.sqrt((ra @ ra) * (rb @ rb))
    if denom < 1e-12:
        return 0.0
    return float(ra @ rb / denom)
