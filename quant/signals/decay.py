"""Signal decay analysis — measure how predictive power fades over time.

Computes the Information Coefficient (IC) between cross-sectional signal
scores and forward returns at multiple horizons.  This reveals:

  * **Half-life**: how many days until IC drops to half its peak.
  * **Optimal rebalance frequency**: the horizon where IC is maximised.
  * **Alpha crowding**: rapid IC decay suggests crowded signals.

The IC at horizon *h* is the rank correlation between today's signal
scores and the cumulative return from *t+1* to *t+h* across all assets.

Usage::

    from quant.signals.decay import SignalDecayAnalyzer, DecayConfig

    analyzer = SignalDecayAnalyzer()
    result = analyzer.analyze(signal_scores, asset_returns)
    print(result.summary())
    print(f"Half-life: {result.half_life} days")
    print(f"Optimal horizon: {result.optimal_horizon} days")
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DecayConfig:
    """Configuration for signal decay analysis.

    Attributes:
        horizons:       Forward return horizons (in trading days) to evaluate.
        min_assets:     Minimum number of assets with both signal and return
                        data required per cross-section.
        min_periods:    Minimum number of cross-sections for a valid IC estimate.
    """

    horizons: list[int] = field(
        default_factory=lambda: [1, 2, 3, 5, 10, 21, 42, 63]
    )
    min_assets: int = 3
    min_periods: int = 20


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HorizonIC:
    """IC statistics at a single forward horizon.

    Attributes:
        horizon:     Forward return horizon in trading days.
        mean_ic:     Mean rank IC across all cross-sections.
        std_ic:      Standard deviation of IC across cross-sections.
        ir:          Information ratio (mean_ic / std_ic).
        t_stat:      t-statistic for mean IC ≠ 0.
        hit_rate:    Fraction of cross-sections with positive IC.
        n_periods:   Number of valid cross-sections.
    """

    horizon: int
    mean_ic: float
    std_ic: float
    ir: float
    t_stat: float
    hit_rate: float
    n_periods: int


@dataclass
class DecayResult:
    """Complete signal decay analysis result.

    Attributes:
        signal_name:      Name of the signal analysed.
        horizon_ics:      IC statistics per horizon, ordered by horizon.
        ic_series:        Per-horizon time series of daily IC values.
        optimal_horizon:  Horizon with highest mean IC.
        half_life:        Days until IC drops to half of peak (None if
                          IC never drops below half).
        peak_ic:          Highest mean IC across horizons.
    """

    signal_name: str
    horizon_ics: list[HorizonIC] = field(default_factory=list)
    ic_series: dict[int, pd.Series] = field(default_factory=dict)
    optimal_horizon: int = 1
    half_life: int | None = None
    peak_ic: float = 0.0

    def ic_curve(self) -> pd.Series:
        """Mean IC as a function of horizon.

        Returns:
            Series indexed by horizon (int), values are mean IC.
        """
        return pd.Series(
            {h.horizon: h.mean_ic for h in self.horizon_ics},
            name="mean_ic",
        )

    def summary(self) -> str:
        """Human-readable decay analysis summary."""
        if not self.horizon_ics:
            return f"Signal Decay ({self.signal_name}): no data"

        lines = [
            f"Signal Decay Analysis: {self.signal_name}",
            "=" * 60,
            f"  Peak IC           : {self.peak_ic:+.4f}",
            f"  Optimal horizon   : {self.optimal_horizon}d",
            f"  Half-life         : {self.half_life}d"
            if self.half_life
            else "  Half-life         : >max horizon",
            "",
            f"  {'Horizon':>8}  {'Mean IC':>10}  {'Std':>8}  {'IR':>8}"
            f"  {'t-stat':>8}  {'Hit%':>6}  {'N':>5}",
            "-" * 60,
        ]

        for h in self.horizon_ics:
            lines.append(
                f"  {h.horizon:>7}d  {h.mean_ic:>+10.4f}  {h.std_ic:>8.4f}"
                f"  {h.ir:>8.2f}  {h.t_stat:>8.2f}  {h.hit_rate:>5.0%}"
                f"  {h.n_periods:>5}"
            )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class SignalDecayAnalyzer:
    """Analyze how signal predictive power decays across forward horizons.

    Args:
        config: Decay analysis configuration.
    """

    def __init__(self, config: DecayConfig | None = None) -> None:
        self._config = config or DecayConfig()

    def analyze(
        self,
        signal_scores: pd.DataFrame,
        asset_returns: pd.DataFrame,
        signal_name: str = "signal",
    ) -> DecayResult:
        """Run signal decay analysis.

        Args:
            signal_scores: Cross-sectional signal scores.
                DatetimeIndex × symbols.  Values in [-1, 1].
            asset_returns: Daily asset returns.
                DatetimeIndex × symbols.
            signal_name: Label for the signal.

        Returns:
            :class:`DecayResult` with IC at each horizon.

        Raises:
            ValueError: If inputs are empty or misaligned.
        """
        if signal_scores.empty or asset_returns.empty:
            raise ValueError("signal_scores and asset_returns must not be empty")

        # Align dates and symbols
        common_dates = signal_scores.index.intersection(asset_returns.index)
        common_syms = list(
            set(signal_scores.columns) & set(asset_returns.columns)
        )

        if len(common_dates) < self._config.min_periods:
            raise ValueError(
                f"Need at least {self._config.min_periods} common dates, "
                f"got {len(common_dates)}"
            )
        if len(common_syms) < self._config.min_assets:
            raise ValueError(
                f"Need at least {self._config.min_assets} common symbols, "
                f"got {len(common_syms)}"
            )

        scores = signal_scores.reindex(index=common_dates, columns=common_syms)
        returns = asset_returns.reindex(index=common_dates, columns=common_syms)

        # Compute IC at each horizon
        horizon_ics: list[HorizonIC] = []
        ic_series: dict[int, pd.Series] = {}

        for h in sorted(self._config.horizons):
            fwd = self._forward_returns(returns, h)
            ic_ts = self._cross_sectional_ic(scores, fwd)

            if len(ic_ts) < self._config.min_periods:
                continue

            mean_ic = float(ic_ts.mean())
            std_ic = float(ic_ts.std())
            ir = mean_ic / std_ic if std_ic > 1e-8 else 0.0
            n = len(ic_ts)
            t_stat = mean_ic / (std_ic / np.sqrt(n)) if std_ic > 1e-8 else 0.0
            hit_rate = float((ic_ts > 0).mean())

            horizon_ics.append(
                HorizonIC(
                    horizon=h,
                    mean_ic=mean_ic,
                    std_ic=std_ic,
                    ir=ir,
                    t_stat=t_stat,
                    hit_rate=hit_rate,
                    n_periods=n,
                )
            )
            ic_series[h] = ic_ts

        if not horizon_ics:
            return DecayResult(signal_name=signal_name)

        # Find peak and optimal horizon
        peak_idx = max(range(len(horizon_ics)), key=lambda i: horizon_ics[i].mean_ic)
        peak_ic = horizon_ics[peak_idx].mean_ic
        optimal_horizon = horizon_ics[peak_idx].horizon

        # Compute half-life
        half_life = self._compute_half_life(horizon_ics, peak_ic, peak_idx)

        return DecayResult(
            signal_name=signal_name,
            horizon_ics=horizon_ics,
            ic_series=ic_series,
            optimal_horizon=optimal_horizon,
            half_life=half_life,
            peak_ic=peak_ic,
        )

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _forward_returns(returns: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Compute forward cumulative returns at a given horizon."""
        return returns.rolling(window=horizon).sum().shift(-horizon)

    def _cross_sectional_ic(
        self,
        scores: pd.DataFrame,
        forward_returns: pd.DataFrame,
    ) -> pd.Series:
        """Compute per-date rank IC between scores and forward returns."""
        ics: list[float] = []
        dates: list = []

        for dt in scores.index:
            s = scores.loc[dt].dropna()
            f = forward_returns.loc[dt].dropna() if dt in forward_returns.index else pd.Series(dtype=float)

            common = s.index.intersection(f.index)
            if len(common) < self._config.min_assets:
                continue

            s_vals = s.reindex(common).values
            f_vals = f.reindex(common).values

            # Skip if no variation
            if np.std(s_vals) < 1e-10 or np.std(f_vals) < 1e-10:
                continue

            corr = _spearman_r(s_vals, f_vals)
            if not np.isnan(corr):
                ics.append(corr)
                dates.append(dt)

        return pd.Series(ics, index=dates, name="ic")

    @staticmethod
    def _compute_half_life(
        horizon_ics: list[HorizonIC], peak_ic: float, peak_idx: int
    ) -> int | None:
        """Find the horizon where IC drops below half of peak.

        Only looks at horizons beyond the peak.
        """
        if peak_ic <= 0:
            return None

        threshold = peak_ic / 2.0

        for i in range(peak_idx + 1, len(horizon_ics)):
            if horizon_ics[i].mean_ic < threshold:
                return horizon_ics[i].horizon

        return None


# ---------------------------------------------------------------------------
# Pure-numpy Spearman rank correlation (no scipy dependency)
# ---------------------------------------------------------------------------


def _spearman_r(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Spearman rank correlation between two 1-D arrays.

    Equivalent to ``scipy.stats.spearmanr(a, b)[0]`` but without the
    scipy dependency.
    """
    n = len(a)
    if n < 2:
        return float("nan")

    # Rank using argsort-of-argsort
    rank_a = np.empty(n, dtype=float)
    rank_b = np.empty(n, dtype=float)
    rank_a[np.argsort(a)] = np.arange(n, dtype=float)
    rank_b[np.argsort(b)] = np.arange(n, dtype=float)

    # Pearson on ranks
    ra = rank_a - rank_a.mean()
    rb = rank_b - rank_b.mean()
    denom = np.sqrt((ra @ ra) * (rb @ rb))
    if denom < 1e-12:
        return 0.0
    return float(ra @ rb / denom)
