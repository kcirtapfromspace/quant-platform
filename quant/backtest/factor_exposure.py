"""Factor exposure analysis for strategy and portfolio returns.

Decomposes strategy returns into systematic factor exposures using
rolling OLS regression.  Helps PMs understand where returns come from,
detect unintended factor bets, and monitor exposure drift over time.

Built-in factors (constructed from the asset return universe):

  * **Market (beta)**: equal-weighted cross-sectional average return.
  * **Momentum**: long top-quintile / short bottom-quintile trailing return.
  * **Volatility**: long low-vol / short high-vol quintile returns.
  * **Size proxy**: long bottom-half / short top-half average absolute return
    (assets with smaller typical moves vs larger).

Usage::

    from quant.backtest.factor_exposure import FactorExposureAnalyzer

    analyzer = FactorExposureAnalyzer()
    result = analyzer.analyze(strategy_returns, asset_returns)
    print(result.summary())
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class FactorExposureConfig:
    """Configuration for factor exposure analysis.

    Attributes:
        rolling_window:     Days for rolling factor regression.
        min_periods:        Minimum days for a valid regression window.
        quintile_frac:      Fraction of assets per quintile (default 0.2).
        momentum_lookback:  Days of trailing return for momentum sort.
        vol_lookback:       Days for volatility estimation.
    """

    rolling_window: int = 63
    min_periods: int = 30
    quintile_frac: float = 0.2
    momentum_lookback: int = 21
    vol_lookback: int = 21


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FactorBeta:
    """Exposure to a single factor over the full sample."""

    factor: str
    beta: float
    t_stat: float
    r_squared_contribution: float


@dataclass
class FactorExposureResult:
    """Complete factor exposure analysis results."""

    # Full-sample regression
    factor_betas: list[FactorBeta]
    alpha: float  # annualised alpha (intercept)
    r_squared: float  # total R-squared
    n_days: int

    # Rolling exposures (DatetimeIndex × factor names)
    rolling_betas: pd.DataFrame = field(repr=False)

    # Factor return series (DatetimeIndex × factor names)
    factor_returns: pd.DataFrame = field(repr=False)

    # Residual returns (unexplained by factors)
    residual_returns: pd.Series = field(repr=False)

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"Factor Exposure Analysis ({self.n_days} days)",
            "=" * 60,
            "",
            f"R-squared          : {self.r_squared:.3f}",
            f"Alpha (ann.)       : {self.alpha:+.2%}",
            "",
            f"{'Factor':<15s} {'Beta':>8s} {'t-stat':>8s} {'R2 contr':>10s}",
            "-" * 45,
        ]
        for fb in sorted(self.factor_betas, key=lambda x: abs(x.beta), reverse=True):
            lines.append(
                f"{fb.factor:<15s} {fb.beta:>+7.3f} {fb.t_stat:>8.2f} "
                f"{fb.r_squared_contribution:>9.3f}"
            )

        # Rolling exposure summary
        if not self.rolling_betas.empty:
            lines.extend(["", "Rolling Beta (latest):", "-" * 45])
            latest = self.rolling_betas.iloc[-1]
            for col in self.rolling_betas.columns:
                val = latest[col]
                if math.isfinite(val):
                    lines.append(f"  {col:<15s}: {val:+.3f}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class FactorExposureAnalyzer:
    """Factor exposure analyzer using rolling OLS regression."""

    def __init__(self, config: FactorExposureConfig | None = None) -> None:
        self._config = config or FactorExposureConfig()

    def analyze(
        self,
        strategy_returns: pd.Series,
        asset_returns: pd.DataFrame,
    ) -> FactorExposureResult:
        """Analyze factor exposures of a return stream.

        Args:
            strategy_returns: Daily strategy returns (DatetimeIndex).
            asset_returns:    Daily asset returns (DatetimeIndex × symbols).

        Returns:
            :class:`FactorExposureResult` with betas and rolling exposures.

        Raises:
            ValueError: If inputs have fewer than ``min_periods`` overlapping days.
        """
        cfg = self._config

        # Align indices
        common = strategy_returns.index.intersection(asset_returns.index)
        if len(common) < cfg.min_periods:
            raise ValueError(
                f"Need at least {cfg.min_periods} overlapping days, "
                f"got {len(common)}"
            )

        y = strategy_returns.loc[common].values.astype(float)
        asset_df = asset_returns.loc[common]
        n_days = len(common)

        # Build factor returns
        factor_df = self._build_factors(asset_df)
        factor_names = list(factor_df.columns)
        design = factor_df.values.astype(float)

        # Full-sample OLS: y = alpha + design @ betas + epsilon
        betas, alpha, residuals, r_squared, r2_contribs, t_stats = self._ols(
            y, design, factor_names
        )

        factor_betas = [
            FactorBeta(
                factor=factor_names[i],
                beta=betas[i],
                t_stat=t_stats[i],
                r_squared_contribution=r2_contribs[i],
            )
            for i in range(len(factor_names))
        ]

        # Rolling betas
        rolling = self._rolling_ols(y, design, factor_names, common, cfg.rolling_window, cfg.min_periods)

        resid_series = pd.Series(residuals, index=common, name="residual")

        ann_alpha = alpha * TRADING_DAYS_PER_YEAR

        return FactorExposureResult(
            factor_betas=factor_betas,
            alpha=ann_alpha,
            r_squared=r_squared,
            n_days=n_days,
            rolling_betas=rolling,
            factor_returns=factor_df,
            residual_returns=resid_series,
        )

    # ── Factor construction ────────────────────────────────────────────

    def _build_factors(self, asset_df: pd.DataFrame) -> pd.DataFrame:
        """Construct factor return series from the asset universe."""
        cfg = self._config
        n = len(asset_df)
        n_assets = asset_df.shape[1]
        q = max(1, int(n_assets * cfg.quintile_frac))

        factors: dict[str, pd.Series] = {}

        # Market factor: equal-weighted average return
        factors["market"] = asset_df.mean(axis=1)

        # Momentum factor: long top-q / short bottom-q by trailing return
        if n > cfg.momentum_lookback and n_assets >= 3:
            trailing = asset_df.rolling(cfg.momentum_lookback, min_periods=cfg.momentum_lookback).sum()
            mom_factor = []
            for i in range(n):
                row = trailing.iloc[i].dropna()
                if len(row) < max(2 * q, 3):
                    mom_factor.append(0.0)
                    continue
                sorted_syms = row.sort_values()
                short_syms = sorted_syms.index[:q]
                long_syms = sorted_syms.index[-q:]
                day_ret = asset_df.iloc[i]
                long_ret = day_ret[long_syms].mean() if len(long_syms) > 0 else 0.0
                short_ret = day_ret[short_syms].mean() if len(short_syms) > 0 else 0.0
                mom_factor.append(long_ret - short_ret)
            factors["momentum"] = pd.Series(mom_factor, index=asset_df.index)

        # Volatility factor: long low-vol / short high-vol
        if n > cfg.vol_lookback and n_assets >= 3:
            rolling_vol = asset_df.rolling(cfg.vol_lookback, min_periods=cfg.vol_lookback).std()
            vol_factor = []
            for i in range(n):
                row = rolling_vol.iloc[i].dropna()
                if len(row) < max(2 * q, 3):
                    vol_factor.append(0.0)
                    continue
                sorted_syms = row.sort_values()
                low_vol_syms = sorted_syms.index[:q]
                high_vol_syms = sorted_syms.index[-q:]
                day_ret = asset_df.iloc[i]
                low_ret = day_ret[low_vol_syms].mean() if len(low_vol_syms) > 0 else 0.0
                high_ret = day_ret[high_vol_syms].mean() if len(high_vol_syms) > 0 else 0.0
                vol_factor.append(low_ret - high_ret)
            factors["volatility"] = pd.Series(vol_factor, index=asset_df.index)

        # Size proxy: long small-move / short big-move assets
        if n > cfg.vol_lookback and n_assets >= 3:
            avg_abs = asset_df.abs().rolling(cfg.vol_lookback, min_periods=cfg.vol_lookback).mean()
            size_factor = []
            for i in range(n):
                row = avg_abs.iloc[i].dropna()
                if len(row) < max(2 * q, 3):
                    size_factor.append(0.0)
                    continue
                sorted_syms = row.sort_values()
                small_syms = sorted_syms.index[:q]
                big_syms = sorted_syms.index[-q:]
                day_ret = asset_df.iloc[i]
                small_ret = day_ret[small_syms].mean() if len(small_syms) > 0 else 0.0
                big_ret = day_ret[big_syms].mean() if len(big_syms) > 0 else 0.0
                size_factor.append(small_ret - big_ret)
            factors["size"] = pd.Series(size_factor, index=asset_df.index)

        return pd.DataFrame(factors).fillna(0.0)

    # ── OLS regression ─────────────────────────────────────────────────

    @staticmethod
    def _ols(
        y: np.ndarray,
        design: np.ndarray,
        factor_names: list[str],
    ) -> tuple[np.ndarray, float, np.ndarray, float, np.ndarray, np.ndarray]:
        """Ordinary least squares with intercept.

        Returns (betas, alpha, residuals, r_squared, r2_contributions, t_stats).
        """
        n = len(y)
        k = design.shape[1]

        # Add intercept column
        dm = np.column_stack([np.ones(n), design])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(dm, y, rcond=None)
        except np.linalg.LinAlgError:
            zeros = np.zeros(k)
            return zeros, 0.0, y.copy(), 0.0, zeros, zeros

        alpha = coeffs[0]
        betas = coeffs[1:]
        fitted = dm @ coeffs
        residuals = y - fitted

        # R-squared
        ss_res = float((residuals**2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r_squared = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        r_squared = max(0.0, r_squared)

        # Per-factor R-squared contribution (sequential, approximate)
        r2_contribs = np.zeros(k)
        for i in range(k):
            dm_partial = np.column_stack([np.ones(n), design[:, i : i + 1]])
            try:
                c_partial, _, _, _ = np.linalg.lstsq(dm_partial, y, rcond=None)
            except np.linalg.LinAlgError:
                continue
            fitted_p = dm_partial @ c_partial
            ss_res_p = float(((y - fitted_p) ** 2).sum())
            r2_contribs[i] = max(0.0, 1 - ss_res_p / ss_tot) if ss_tot > 1e-12 else 0.0

        # t-statistics
        t_stats = np.zeros(k)
        if n > k + 1:
            mse = ss_res / (n - k - 1)
            try:
                cov_matrix = mse * np.linalg.inv(dm.T @ dm)
                for i in range(k):
                    se = math.sqrt(max(0.0, cov_matrix[i + 1, i + 1]))
                    t_stats[i] = betas[i] / se if se > 1e-12 else 0.0
            except np.linalg.LinAlgError:
                pass

        return betas, alpha, residuals, r_squared, r2_contribs, t_stats

    @staticmethod
    def _rolling_ols(
        y: np.ndarray,
        design: np.ndarray,
        factor_names: list[str],
        index: pd.DatetimeIndex,
        window: int,
        min_periods: int,
    ) -> pd.DataFrame:
        """Compute rolling betas over a sliding window."""
        n = len(y)
        k = design.shape[1]
        results = np.full((n, k), np.nan)

        for i in range(min_periods - 1, n):
            start = max(0, i - window + 1)
            y_w = y[start : i + 1]
            dw = design[start : i + 1]

            if len(y_w) < min_periods:
                continue

            dm = np.column_stack([np.ones(len(y_w)), dw])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(dm, y_w, rcond=None)
                results[i] = coeffs[1:]
            except np.linalg.LinAlgError:
                continue

        return pd.DataFrame(results, index=index, columns=factor_names)
