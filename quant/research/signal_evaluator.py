"""Cross-sectional signal quality evaluator.

Comprehensive evaluation of a trading signal's ability to rank assets by
forward returns.  Distinct from :mod:`signal_decay` (which measures how IC
changes with holding period), this module evaluates signal quality at a
single forward horizon through multiple lenses.

Metrics computed:

  * **IC statistics** — mean IC, IC standard deviation, ICIR (IC / σ_IC).
  * **Quantile returns** — mean return of each signal-sorted quantile,
    long-short spread, monotonicity score.
  * **Hit rate** — fraction of correctly-predicted return directions.
  * **Signal turnover** — average rank-change between consecutive dates.

Usage::

    from quant.research.signal_evaluator import (
        SignalEvaluator,
        EvaluatorConfig,
    )

    evaluator = SignalEvaluator(EvaluatorConfig(n_quantiles=5))
    result = evaluator.evaluate(signal_df, returns_df)
    print(result.summary())
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class EvaluatorConfig:
    """Configuration for signal evaluation.

    Attributes:
        n_quantiles:      Number of quantiles for return bucketing (e.g. 5).
        min_observations: Minimum cross-sectional observations per date.
        min_dates:        Minimum number of valid dates for meaningful stats.
        forward_period:   Forward return horizon in trading days.
    """

    n_quantiles: int = 5
    min_observations: int = 20
    min_dates: int = 30
    forward_period: int = 1


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class QuantileStats:
    """Return statistics for a single signal quantile.

    Attributes:
        quantile:       Quantile number (1 = lowest signal, N = highest).
        mean_return:    Mean daily return for assets in this quantile.
        std_return:     Standard deviation of daily returns.
        n_observations: Total asset-date observations in this quantile.
    """

    quantile: int
    mean_return: float
    std_return: float
    n_observations: int


@dataclass
class EvaluationResult:
    """Comprehensive signal evaluation result.

    Attributes:
        mean_ic:            Average cross-sectional Spearman rank IC.
        ic_std:             Standard deviation of IC time series.
        ic_ir:              Information ratio of IC (mean / std).
        ic_series:          IC at each date (pd.Series).
        quantile_returns:   Per-quantile return statistics.
        long_short_return:  Annualised return of top minus bottom quantile.
        monotonicity:       Spearman correlation of quantile mean returns
                            with quantile number (1.0 = perfectly monotonic).
        hit_rate:           Fraction of correct directional predictions.
        turnover:           Average signal turnover (0 = stable, 1 = random).
        n_dates:            Number of valid evaluation dates.
        n_assets_avg:       Average cross-sectional breadth per date.
    """

    mean_ic: float = 0.0
    ic_std: float = 0.0
    ic_ir: float = 0.0
    ic_series: pd.Series = field(
        default_factory=lambda: pd.Series(dtype=float), repr=False,
    )
    quantile_returns: list[QuantileStats] = field(default_factory=list)
    long_short_return: float = 0.0
    monotonicity: float = 0.0
    hit_rate: float = 0.0
    turnover: float = 0.0
    n_dates: int = 0
    n_assets_avg: float = 0.0

    def summary(self) -> str:
        """Return a human-readable evaluation summary."""
        lines = [
            f"Signal Evaluation ({self.n_dates} dates, "
            f"{self.n_assets_avg:.0f} avg assets)",
            "=" * 60,
            "",
            "IC statistics:",
            f"  Mean IC      : {self.mean_ic:+.4f}",
            f"  IC Std       : {self.ic_std:.4f}",
            f"  ICIR         : {self.ic_ir:+.2f}",
            "",
            "Signal quality:",
            f"  Hit rate     : {self.hit_rate:.1%}",
            f"  Turnover     : {self.turnover:.2%}",
            f"  Monotonicity : {self.monotonicity:+.2f}",
            "",
            "Quantile returns (annualised):",
        ]

        for qs in self.quantile_returns:
            ann_ret = qs.mean_return * TRADING_DAYS
            lines.append(
                f"  Q{qs.quantile:<2d}: {ann_ret:+.2%}  "
                f"(n={qs.n_observations})",
            )

        lines.append("")
        ls_ann = self.long_short_return
        lines.append(f"  Long-short   : {ls_ann:+.2%} annualised")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class SignalEvaluator:
    """Cross-sectional signal quality evaluator.

    Args:
        config: Evaluation configuration.
    """

    def __init__(self, config: EvaluatorConfig | None = None) -> None:
        self._config = config or EvaluatorConfig()

    @property
    def config(self) -> EvaluatorConfig:
        return self._config

    def evaluate(
        self,
        signal: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> EvaluationResult:
        """Evaluate signal quality against realised returns.

        Args:
            signal:  T x N DataFrame of signal scores (rows = dates,
                     columns = assets).  Higher signal ⇒ expected higher
                     return.
            returns: T x N DataFrame of asset returns (same shape/index
                     convention as signal).

        Returns:
            :class:`EvaluationResult` with comprehensive metrics.

        Raises:
            ValueError: If fewer than ``min_dates`` valid evaluation dates.
        """
        cfg = self._config

        # Align columns
        common = signal.columns.intersection(returns.columns)
        if len(common) < cfg.min_observations:
            msg = (
                f"Need at least {cfg.min_observations} common assets, "
                f"got {len(common)}"
            )
            raise ValueError(msg)

        sig = signal[common]
        ret = returns[common]

        # Build forward returns (shift returns back by forward_period)
        fwd = ret.shift(-cfg.forward_period)

        # Identify valid dates (enough non-NaN cross-sectional observations)
        valid_dates = []
        for dt in sig.index:
            if dt not in fwd.index:
                continue
            s_row = sig.loc[dt]
            r_row = fwd.loc[dt]
            mask = s_row.notna() & r_row.notna()
            if mask.sum() >= cfg.min_observations:
                valid_dates.append(dt)

        if len(valid_dates) < cfg.min_dates:
            msg = (
                f"Need at least {cfg.min_dates} valid dates, "
                f"got {len(valid_dates)}"
            )
            raise ValueError(msg)

        # ── IC time series ────────────────────────────────────────
        ic_values = {}
        breadths = []
        for dt in valid_dates:
            s_row = sig.loc[dt]
            r_row = fwd.loc[dt]
            mask = s_row.notna() & r_row.notna()
            s_valid = s_row[mask].values
            r_valid = r_row[mask].values
            breadths.append(int(mask.sum()))

            if len(s_valid) >= 3:
                corr, _ = sp_stats.spearmanr(s_valid, r_valid)
                ic_values[dt] = float(corr) if np.isfinite(corr) else 0.0
            else:
                ic_values[dt] = 0.0

        ic_series = pd.Series(ic_values, name="ic")
        mean_ic = float(ic_series.mean())
        ic_std = float(ic_series.std(ddof=1)) if len(ic_series) > 1 else 0.0
        ic_ir = mean_ic / ic_std if ic_std > 1e-15 else 0.0

        # ── Quantile returns ──────────────────────────────────────
        quantile_returns = self._compute_quantile_returns(
            sig, fwd, valid_dates, cfg.n_quantiles, cfg.min_observations,
        )

        # Long-short spread (annualised)
        if len(quantile_returns) >= 2:
            top_ret = quantile_returns[-1].mean_return
            bot_ret = quantile_returns[0].mean_return
            long_short = (top_ret - bot_ret) * TRADING_DAYS
        else:
            long_short = 0.0

        # Monotonicity: Spearman of quantile number vs mean return
        monotonicity = self._compute_monotonicity(quantile_returns)

        # ── Hit rate ──────────────────────────────────────────────
        hit_rate = self._compute_hit_rate(
            sig, fwd, valid_dates, cfg.min_observations,
        )

        # ── Signal turnover ───────────────────────────────────────
        turnover = self._compute_turnover(sig, valid_dates)

        return EvaluationResult(
            mean_ic=mean_ic,
            ic_std=ic_std,
            ic_ir=ic_ir,
            ic_series=ic_series,
            quantile_returns=quantile_returns,
            long_short_return=long_short,
            monotonicity=monotonicity,
            hit_rate=hit_rate,
            turnover=turnover,
            n_dates=len(valid_dates),
            n_assets_avg=float(np.mean(breadths)),
        )

    # ── Internal methods ──────────────────────────────────────────

    @staticmethod
    def _compute_quantile_returns(
        sig: pd.DataFrame,
        fwd: pd.DataFrame,
        valid_dates: list,
        n_quantiles: int,
        min_obs: int,
    ) -> list[QuantileStats]:
        """Compute return statistics per signal quantile."""
        # Collect returns by quantile across all dates
        buckets: dict[int, list[float]] = {
            q: [] for q in range(1, n_quantiles + 1)
        }

        for dt in valid_dates:
            s_row = sig.loc[dt]
            r_row = fwd.loc[dt]
            mask = s_row.notna() & r_row.notna()
            s_valid = s_row[mask]
            r_valid = r_row[mask]

            if len(s_valid) < min_obs:
                continue

            # Assign quantiles (1 = lowest signal, n = highest)
            try:
                q_labels = pd.qcut(
                    s_valid.rank(method="first"), n_quantiles, labels=False,
                ) + 1
            except ValueError:
                continue

            for q in range(1, n_quantiles + 1):
                q_mask = q_labels == q
                if q_mask.any():
                    buckets[q].extend(r_valid[q_mask].tolist())

        result = []
        for q in range(1, n_quantiles + 1):
            rets = np.array(buckets[q])
            if len(rets) > 0:
                result.append(QuantileStats(
                    quantile=q,
                    mean_return=float(rets.mean()),
                    std_return=float(rets.std(ddof=1)) if len(rets) > 1 else 0.0,
                    n_observations=len(rets),
                ))
            else:
                result.append(QuantileStats(
                    quantile=q, mean_return=0.0, std_return=0.0, n_observations=0,
                ))
        return result

    @staticmethod
    def _compute_monotonicity(quantile_returns: list[QuantileStats]) -> float:
        """Spearman rank correlation of quantile number vs mean return."""
        if len(quantile_returns) < 3:
            return 0.0
        qs = np.array([q.quantile for q in quantile_returns])
        means = np.array([q.mean_return for q in quantile_returns])
        if np.std(means) < 1e-15:
            return 0.0
        corr, _ = sp_stats.spearmanr(qs, means)
        return float(corr) if np.isfinite(corr) else 0.0

    @staticmethod
    def _compute_hit_rate(
        sig: pd.DataFrame,
        fwd: pd.DataFrame,
        valid_dates: list,
        min_obs: int,
    ) -> float:
        """Fraction of (date, asset) pairs where sign(signal) == sign(return)."""
        hits = 0
        total = 0
        for dt in valid_dates:
            s_row = sig.loc[dt]
            r_row = fwd.loc[dt]
            mask = s_row.notna() & r_row.notna()
            s_valid = s_row[mask].values
            r_valid = r_row[mask].values

            if len(s_valid) < min_obs:
                continue

            # Count hits: same sign (exclude zeros)
            nonzero = (s_valid != 0) & (r_valid != 0)
            if nonzero.any():
                s_nz = s_valid[nonzero]
                r_nz = r_valid[nonzero]
                hits += int(np.sum(np.sign(s_nz) == np.sign(r_nz)))
                total += int(nonzero.sum())

        return hits / total if total > 0 else 0.5

    @staticmethod
    def _compute_turnover(
        sig: pd.DataFrame,
        valid_dates: list,
    ) -> float:
        """Average signal turnover: 1 − rank_correlation(signal_t, signal_{t-1})."""
        if len(valid_dates) < 2:
            return 0.0

        rank_corrs = []
        for i in range(1, len(valid_dates)):
            prev_dt = valid_dates[i - 1]
            curr_dt = valid_dates[i]
            s_prev = sig.loc[prev_dt]
            s_curr = sig.loc[curr_dt]
            mask = s_prev.notna() & s_curr.notna()

            if mask.sum() >= 3:
                corr, _ = sp_stats.spearmanr(
                    s_prev[mask].values, s_curr[mask].values,
                )
                if np.isfinite(corr):
                    rank_corrs.append(corr)

        if not rank_corrs:
            return 0.0

        return 1.0 - float(np.mean(rank_corrs))
