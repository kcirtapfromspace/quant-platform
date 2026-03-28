"""Signal decay analysis for alpha research.

Measures how quickly a trading signal's predictive power decays over time.
This is essential for determining optimal holding periods, rebalance
frequency, and signal blending weights.

Key metrics:

  * **Information Coefficient (IC) curve**: rank correlation between signal
    scores and forward returns at each lag (1-day, 2-day, ..., N-day).
  * **IC half-life**: the lag at which IC drops to half its peak value.
  * **Cumulative IC**: area under the IC curve — the total information
    content available if positions are held over the full window.
  * **Optimal holding period**: the lag that maximises cumulative IC per
    unit of turnover cost.
  * **Turnover-adjusted IC (ICIR)**: IC normalised by its standard
    deviation (analogous to information ratio for signals).

Usage::

    from quant.research.signal_decay import SignalDecayAnalyzer, DecayConfig

    analyzer = SignalDecayAnalyzer(DecayConfig(max_lag=20))
    result = analyzer.analyze(
        signal_scores=scores_df,     # DatetimeIndex × symbols
        forward_returns=returns_df,  # DatetimeIndex × symbols
    )
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
class DecayConfig:
    """Configuration for signal decay analysis.

    Attributes:
        max_lag:            Maximum forward-return lag to evaluate (trading days).
        min_observations:   Minimum cross-sectional observations per date.
        min_dates:          Minimum number of dates with valid IC.
        turnover_cost_bps:  Round-trip turnover cost in bps for optimal
                            holding period estimation.
    """

    max_lag: int = 20
    min_observations: int = 10
    min_dates: int = 20
    turnover_cost_bps: float = 20.0


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LagMetric:
    """IC statistics at a single lag.

    Attributes:
        lag:        Forward-return lag in trading days.
        mean_ic:    Average cross-sectional rank IC.
        std_ic:     Standard deviation of IC across dates.
        icir:       IC information ratio (mean / std).
        hit_rate:   Fraction of dates with positive IC.
        n_dates:    Number of valid date observations.
    """

    lag: int
    mean_ic: float
    std_ic: float
    icir: float
    hit_rate: float
    n_dates: int


@dataclass
class DecayResult:
    """Complete signal decay analysis.

    Attributes:
        lag_metrics:            Per-lag IC statistics.
        peak_ic:                Maximum mean IC across all lags.
        peak_lag:               Lag at which peak IC occurs.
        half_life:              Lag where IC drops to half of peak (None if
                                IC never halves within max_lag).
        cumulative_ic:          Cumulative IC curve (sum of IC from lag 1 to k).
        optimal_holding_period: Lag that maximises cumulative IC net of cost.
        optimal_net_ic:         Net IC at optimal holding period.
        n_symbols:              Number of symbols in cross-section.
        n_dates:                Total date observations in the dataset.
    """

    lag_metrics: list[LagMetric]
    peak_ic: float
    peak_lag: int
    half_life: int | None
    cumulative_ic: list[float]
    optimal_holding_period: int
    optimal_net_ic: float
    n_symbols: int
    n_dates: int
    ic_series: dict[int, pd.Series] = field(repr=False, default_factory=dict)

    def summary(self) -> str:
        """Return a human-readable summary."""
        hl_str = f"{self.half_life} days" if self.half_life is not None else "N/A"
        lines = [
            f"Signal Decay Analysis ({self.n_symbols} symbols, {self.n_dates} dates)",
            "=" * 60,
            "",
            f"Peak IC                 : {self.peak_ic:.4f} (lag {self.peak_lag})",
            f"IC half-life            : {hl_str}",
            f"Optimal holding period  : {self.optimal_holding_period} days",
            f"Optimal net IC          : {self.optimal_net_ic:.4f}",
            "",
            f"{'Lag':>5s} {'IC':>8s} {'Std':>8s} {'ICIR':>8s} "
            f"{'Hit%':>7s} {'Cum IC':>8s} {'N':>5s}",
            "-" * 55,
        ]
        for i, lm in enumerate(self.lag_metrics):
            cum = self.cumulative_ic[i] if i < len(self.cumulative_ic) else 0.0
            lines.append(
                f"{lm.lag:>5d} {lm.mean_ic:>+8.4f} {lm.std_ic:>8.4f} "
                f"{lm.icir:>+8.3f} {lm.hit_rate:>6.1%} "
                f"{cum:>+8.4f} {lm.n_dates:>5d}"
            )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class SignalDecayAnalyzer:
    """Analyzes how quickly a signal's predictive power decays.

    Args:
        config: Analysis configuration.
    """

    def __init__(self, config: DecayConfig | None = None) -> None:
        self._config = config or DecayConfig()

    @property
    def config(self) -> DecayConfig:
        return self._config

    def analyze(
        self,
        signal_scores: pd.DataFrame,
        forward_returns: pd.DataFrame,
    ) -> DecayResult:
        """Analyze signal decay across lags.

        Args:
            signal_scores:   DataFrame (DatetimeIndex × symbols) of signal
                             scores.  Higher score = more bullish.
            forward_returns: DataFrame (DatetimeIndex × symbols) of asset
                             returns.  The analyzer computes cumulative
                             forward returns internally for each lag.

        Returns:
            :class:`DecayResult` with IC curve, half-life, and optimal
            holding period.

        Raises:
            ValueError: If inputs have no common dates or symbols, or if
                        there are fewer than ``min_dates`` valid observations.
        """
        cfg = self._config

        # Align dates and symbols
        common_dates = signal_scores.index.intersection(forward_returns.index)
        common_symbols = sorted(
            set(signal_scores.columns) & set(forward_returns.columns)
        )

        if len(common_dates) < cfg.min_dates:
            raise ValueError(
                f"Need at least {cfg.min_dates} common dates, "
                f"got {len(common_dates)}"
            )
        if len(common_symbols) < cfg.min_observations:
            raise ValueError(
                f"Need at least {cfg.min_observations} common symbols, "
                f"got {len(common_symbols)}"
            )

        scores = signal_scores.loc[common_dates, common_symbols].copy()
        returns = forward_returns.loc[common_dates, common_symbols].copy()

        n_symbols = len(common_symbols)
        n_dates = len(common_dates)

        # Compute IC at each lag
        lag_metrics: list[LagMetric] = []
        ic_series_dict: dict[int, pd.Series] = {}

        for lag in range(1, cfg.max_lag + 1):
            # Cumulative forward return over `lag` days
            fwd = returns.rolling(lag).sum().shift(-lag)

            # Cross-sectional rank IC for each date
            ics: list[float] = []
            ic_dates: list = []
            for date in scores.index:
                s = scores.loc[date].dropna()
                f = fwd.loc[date].dropna() if date in fwd.index else pd.Series(dtype=float)
                common = s.index.intersection(f.index)
                if len(common) < cfg.min_observations:
                    continue
                # Spearman rank correlation
                ic = float(s[common].rank().corr(f[common].rank()))
                if not np.isnan(ic):
                    ics.append(ic)
                    ic_dates.append(date)

            if len(ics) < cfg.min_dates:
                lag_metrics.append(
                    LagMetric(lag=lag, mean_ic=0.0, std_ic=0.0,
                              icir=0.0, hit_rate=0.0, n_dates=0)
                )
                continue

            ic_arr = np.array(ics)
            mean_ic = float(ic_arr.mean())
            std_ic = float(ic_arr.std(ddof=1)) if len(ic_arr) > 1 else 0.0
            icir = mean_ic / std_ic if std_ic > 1e-10 else 0.0
            hit_rate = float((ic_arr > 0).mean())

            lag_metrics.append(
                LagMetric(
                    lag=lag, mean_ic=mean_ic, std_ic=std_ic,
                    icir=icir, hit_rate=hit_rate, n_dates=len(ics),
                )
            )
            ic_series_dict[lag] = pd.Series(ics, index=ic_dates, name=f"IC_lag{lag}")

        # Peak IC
        if lag_metrics:
            peak_lm = max(lag_metrics, key=lambda x: x.mean_ic)
            peak_ic = peak_lm.mean_ic
            peak_lag = peak_lm.lag
        else:
            peak_ic = 0.0
            peak_lag = 1

        # Half-life: first lag after peak where IC drops below peak/2
        half_life = self._compute_half_life(lag_metrics, peak_ic, peak_lag)

        # Cumulative IC
        cumulative: list[float] = []
        running = 0.0
        for lm in lag_metrics:
            running += lm.mean_ic
            cumulative.append(running)

        # Optimal holding period: maximise cumulative_IC / lag - cost_per_day
        # Cost per day = turnover_cost_bps / (lag * 10000)
        # Net value at lag k = cumulative_IC[k] - turnover_cost / k
        cost_frac = cfg.turnover_cost_bps / 10_000
        best_lag = 1
        best_net = -float("inf")
        for i, lm in enumerate(lag_metrics):
            lag = lm.lag
            # Average IC per day * lag - cost
            cum_ic = cumulative[i]
            net = cum_ic - cost_frac / lag if lag > 0 else cum_ic
            if net > best_net:
                best_net = net
                best_lag = lag

        return DecayResult(
            lag_metrics=lag_metrics,
            peak_ic=peak_ic,
            peak_lag=peak_lag,
            half_life=half_life,
            cumulative_ic=cumulative,
            optimal_holding_period=best_lag,
            optimal_net_ic=best_net,
            n_symbols=n_symbols,
            n_dates=n_dates,
            ic_series=ic_series_dict,
        )

    @staticmethod
    def _compute_half_life(
        metrics: list[LagMetric], peak_ic: float, peak_lag: int,
    ) -> int | None:
        """Find the lag where IC first drops below peak/2 after the peak."""
        if peak_ic <= 0:
            return None
        threshold = peak_ic / 2
        for lm in metrics:
            if lm.lag > peak_lag and lm.mean_ic < threshold:
                return lm.lag
        return None
