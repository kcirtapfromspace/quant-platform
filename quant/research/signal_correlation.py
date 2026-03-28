"""Signal correlation monitoring for alpha crowding detection.

When multiple alpha signals become highly correlated, the portfolio
concentrates risk on a single implicit factor.  This monitor tracks
pairwise signal correlations over rolling windows, detects convergence
events, and computes a diversification metric across the signal library.

Key outputs:

  * **Pairwise correlation matrix** — current and rolling history.
  * **Average pairwise correlation** — single number summarising crowding.
  * **Effective signal count** — analogous to effective N from eigenvalues:
    how many independent signals the library contains.
  * **Convergence events** — periods where avg correlation exceeds threshold.
  * **Signal clustering** — groups of highly correlated signals.

Usage::

    from quant.research.signal_correlation import (
        SignalCorrelationMonitor,
        CorrelationConfig,
    )

    monitor = SignalCorrelationMonitor(CorrelationConfig(
        window=63,
        convergence_threshold=0.60,
    ))
    result = monitor.analyze(signal_scores)
    print(result.summary())
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CorrelationConfig:
    """Configuration for signal correlation monitoring.

    Attributes:
        window:                 Rolling window for correlation estimation.
        min_periods:            Minimum observations for valid correlation.
        convergence_threshold:  Average pairwise correlation above this
                                triggers a convergence event.
        cluster_threshold:      Pairwise correlation above this groups
                                signals into the same cluster.
    """

    window: int = 63
    min_periods: int = 30
    convergence_threshold: float = 0.60
    cluster_threshold: float = 0.70


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SignalPair:
    """Correlation between a pair of signals."""

    signal_a: str
    signal_b: str
    correlation: float
    is_high: bool


@dataclass(frozen=True, slots=True)
class ConvergenceEvent:
    """Period of elevated signal correlation."""

    start_date: object  # datetime-like
    end_date: object
    duration_days: int
    peak_avg_correlation: float


@dataclass(frozen=True, slots=True)
class SignalCluster:
    """Group of highly correlated signals."""

    cluster_id: int
    signals: tuple[str, ...]
    avg_internal_corr: float


@dataclass
class SignalCorrelationResult:
    """Complete signal correlation analysis.

    Attributes:
        pairs:                  All pairwise correlations.
        correlation_matrix:     Current correlation matrix as dict.
        rolling_avg_corr:       Time series of avg pairwise correlation.
        current_avg_corr:       Most recent avg pairwise correlation.
        effective_signal_count: Number of independent signal dimensions.
        convergence_events:     Periods of elevated correlation.
        clusters:               Groups of correlated signals.
        is_converged:           True if currently in convergence.
        n_signals:              Number of signals.
        n_dates:                Number of observation dates.
    """

    pairs: list[SignalPair] = field(default_factory=list)
    correlation_matrix: dict[tuple[str, str], float] = field(
        repr=False, default_factory=dict,
    )
    rolling_avg_corr: pd.Series = field(repr=False, default_factory=lambda: pd.Series(dtype=float))
    current_avg_corr: float = 0.0
    effective_signal_count: float = 0.0
    convergence_events: list[ConvergenceEvent] = field(default_factory=list)
    clusters: list[SignalCluster] = field(default_factory=list)
    is_converged: bool = False
    n_signals: int = 0
    n_dates: int = 0

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"Signal Correlation Monitor ({self.n_signals} signals, {self.n_dates} dates)",
            "=" * 60,
            "",
            f"Current avg correlation  : {self.current_avg_corr:.4f}",
            f"Effective signal count   : {self.effective_signal_count:.1f}",
            f"Currently converged      : {'YES' if self.is_converged else 'No'}",
            f"Convergence events       : {len(self.convergence_events)}",
            f"Signal clusters          : {len(self.clusters)}",
        ]

        if self.pairs:
            lines.extend(["", "Top Pairwise Correlations:"])
            top = sorted(self.pairs, key=lambda p: -abs(p.correlation))[:10]
            for p in top:
                flag = "*" if p.is_high else ""
                lines.append(
                    f"  {p.signal_a:<15s} × {p.signal_b:<15s} "
                    f"{p.correlation:>+.4f} {flag}"
                )

        if self.clusters:
            lines.extend(["", "Clusters:"])
            for cl in self.clusters:
                lines.append(
                    f"  #{cl.cluster_id}: {', '.join(cl.signals)} "
                    f"(avg={cl.avg_internal_corr:.3f})"
                )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------


class SignalCorrelationMonitor:
    """Monitors pairwise correlation across a library of alpha signals.

    Args:
        config: Monitoring configuration.
    """

    def __init__(self, config: CorrelationConfig | None = None) -> None:
        self._config = config or CorrelationConfig()

    @property
    def config(self) -> CorrelationConfig:
        return self._config

    def analyze(
        self,
        signals: dict[str, pd.DataFrame] | pd.DataFrame,
    ) -> SignalCorrelationResult:
        """Analyze signal-level correlations.

        Args:
            signals: Either a ``{name: DataFrame}`` mapping of signal scores
                     where each DataFrame is (dates × symbols), or a single
                     DataFrame where columns are signal names and rows are
                     dates (cross-sectional average scores).

        Returns:
            :class:`SignalCorrelationResult` with pairwise, rolling, and
            cluster analysis.

        Raises:
            ValueError: If fewer than 2 signals provided.
        """
        cfg = self._config

        # Normalise input to a DataFrame of (dates × signals)
        if isinstance(signals, dict):
            # Each signal is a cross-section (dates × symbols).
            # Reduce each to a time series of cross-sectional average score.
            series_map: dict[str, pd.Series] = {}
            for name, df in signals.items():
                series_map[name] = df.mean(axis=1)
            score_df = pd.DataFrame(series_map).dropna()
        else:
            score_df = signals.dropna()

        signal_names = sorted(score_df.columns.tolist())
        n_signals = len(signal_names)

        if n_signals < 2:
            raise ValueError("Need at least 2 signals for correlation analysis")

        score_df = score_df[signal_names]
        n_dates = len(score_df)

        # Current pairwise correlations (full-sample)
        corr_matrix = score_df.corr(method="spearman")
        pairs: list[SignalPair] = []
        corr_dict: dict[tuple[str, str], float] = {}

        for i in range(n_signals):
            for j in range(i + 1, n_signals):
                si, sj = signal_names[i], signal_names[j]
                c = float(corr_matrix.iloc[i, j])
                if np.isnan(c):
                    c = 0.0
                corr_dict[(si, sj)] = c
                pairs.append(SignalPair(
                    signal_a=si, signal_b=sj,
                    correlation=c,
                    is_high=abs(c) >= cfg.cluster_threshold,
                ))

        # Current average pairwise correlation
        corr_values = [p.correlation for p in pairs]
        current_avg = float(np.mean(corr_values)) if corr_values else 0.0

        # Rolling average pairwise correlation
        rolling_avg = self._rolling_avg_correlation(score_df, cfg)

        # Effective signal count from eigenvalues
        eff_count = self._effective_count(corr_matrix)

        # Convergence events
        events = self._detect_convergence(rolling_avg, cfg.convergence_threshold)

        # Is currently converged?
        is_converged = (
            len(rolling_avg) > 0
            and float(rolling_avg.iloc[-1]) >= cfg.convergence_threshold
        )

        # Signal clustering (greedy union-find on cluster_threshold)
        clusters = self._cluster_signals(signal_names, corr_dict, cfg.cluster_threshold)

        return SignalCorrelationResult(
            pairs=pairs,
            correlation_matrix=corr_dict,
            rolling_avg_corr=rolling_avg,
            current_avg_corr=current_avg,
            effective_signal_count=eff_count,
            convergence_events=events,
            clusters=clusters,
            is_converged=is_converged,
            n_signals=n_signals,
            n_dates=n_dates,
        )

    @staticmethod
    def _rolling_avg_correlation(
        score_df: pd.DataFrame,
        cfg: CorrelationConfig,
    ) -> pd.Series:
        """Compute rolling average pairwise Spearman correlation."""
        n = len(score_df.columns)
        if n < 2:
            return pd.Series(dtype=float)

        window = cfg.window
        min_per = cfg.min_periods
        dates = score_df.index
        avg_corrs: list[float] = []
        valid_dates: list = []

        for end in range(min_per, len(dates) + 1):
            start = max(0, end - window)
            chunk = score_df.iloc[start:end]
            if len(chunk) < min_per:
                continue
            corr = chunk.corr(method="spearman")
            # Upper triangle mean
            vals = []
            for i in range(n):
                for j in range(i + 1, n):
                    v = corr.iloc[i, j]
                    if not np.isnan(v):
                        vals.append(v)
            if vals:
                avg_corrs.append(float(np.mean(vals)))
                valid_dates.append(dates[end - 1])

        return pd.Series(avg_corrs, index=valid_dates, name="avg_pairwise_corr")

    @staticmethod
    def _effective_count(corr_matrix: pd.DataFrame) -> float:
        """Effective number of independent signals from eigenvalue decomposition.

        Uses the entropy-based formula:
            N_eff = exp(-sum(p_i * log(p_i)))
        where p_i = λ_i / sum(λ).
        """
        eigenvalues = np.linalg.eigvalsh(corr_matrix.values)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        total = eigenvalues.sum()
        if total < 1e-10:
            return float(len(corr_matrix))
        proportions = eigenvalues / total
        entropy = -np.sum(proportions * np.log(proportions))
        return float(np.exp(entropy))

    @staticmethod
    def _detect_convergence(
        rolling_avg: pd.Series,
        threshold: float,
    ) -> list[ConvergenceEvent]:
        """Detect periods where avg correlation exceeds threshold."""
        if len(rolling_avg) == 0:
            return []

        events: list[ConvergenceEvent] = []
        in_event = False
        start_date = None
        peak = 0.0

        for date, val in rolling_avg.items():
            if val >= threshold:
                if not in_event:
                    in_event = True
                    start_date = date
                    peak = val
                else:
                    peak = max(peak, val)
            elif in_event:
                # Event ended
                end_date = date
                duration = (end_date - start_date).days if hasattr(end_date - start_date, "days") else 1
                events.append(ConvergenceEvent(
                    start_date=start_date,
                    end_date=end_date,
                    duration_days=max(duration, 1),
                    peak_avg_correlation=peak,
                ))
                in_event = False

        # If still in event at end of series
        if in_event and start_date is not None:
            end_date = rolling_avg.index[-1]
            duration = (end_date - start_date).days if hasattr(end_date - start_date, "days") else 1
            events.append(ConvergenceEvent(
                start_date=start_date,
                end_date=end_date,
                duration_days=max(duration, 1),
                peak_avg_correlation=peak,
            ))

        return events

    @staticmethod
    def _cluster_signals(
        names: list[str],
        corr_dict: dict[tuple[str, str], float],
        threshold: float,
    ) -> list[SignalCluster]:
        """Greedy single-linkage clustering based on correlation threshold."""
        n = len(names)
        parent = list(range(n))
        name_idx = {name: i for i, name in enumerate(names)}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for (si, sj), c in corr_dict.items():
            if abs(c) >= threshold:
                union(name_idx[si], name_idx[sj])

        # Build clusters (only non-singleton)
        groups: dict[int, list[str]] = {}
        for i, name in enumerate(names):
            root = find(i)
            groups.setdefault(root, []).append(name)

        clusters: list[SignalCluster] = []
        cid = 0
        for members in groups.values():
            if len(members) < 2:
                continue
            # Average internal correlation
            internal_corrs: list[float] = []
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    key = (members[a], members[b])
                    alt_key = (members[b], members[a])
                    c = corr_dict.get(key, corr_dict.get(alt_key, 0.0))
                    internal_corrs.append(c)
            avg_int = float(np.mean(internal_corrs)) if internal_corrs else 0.0
            clusters.append(SignalCluster(
                cluster_id=cid,
                signals=tuple(sorted(members)),
                avg_internal_corr=avg_int,
            ))
            cid += 1

        return clusters
