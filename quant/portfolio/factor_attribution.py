"""Factor-based return attribution via multi-factor regression.

Decomposes portfolio returns into factor contributions and residual
alpha using ordinary least squares (OLS) regression.  This answers the
critical CIO question: *is our alpha genuine, or just disguised factor
beta?*

Supported decomposition:

  * **Factor loadings (betas)**: portfolio exposure to each factor.
  * **Factor contributions**: beta × factor return for each period.
  * **Residual alpha**: return not explained by factor exposures.
  * **R-squared**: fraction of variance explained by factors.
  * **Factor risk attribution**: fraction of portfolio variance from
    each factor vs. idiosyncratic risk.

The module supports both:
  - **External factors**: user-supplied factor return series (e.g. Fama-French).
  - **Built-in factors**: market, size, value, momentum, low-vol constructed
    from cross-sectional asset data.

All computation uses numpy — no statsmodels or scipy dependency.

Usage::

    from quant.portfolio.factor_attribution import FactorAttributor

    attributor = FactorAttributor()
    report = attributor.attribute(
        portfolio_returns=daily_returns,
        factor_returns=factor_df,  # DatetimeIndex × factor_names
    )
    print(report.alpha, report.r_squared)
    for fc in report.factor_contributions:
        print(fc.factor_name, fc.beta, fc.contribution)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FactorContribution:
    """Attribution of return to a single factor.

    Attributes:
        factor_name:  Name of the factor (e.g. "market", "momentum").
        beta:         Portfolio loading on this factor (regression coeff).
        t_stat:       t-statistic for the beta estimate.
        contribution: Cumulative return attributed to this factor
                      (beta × cumulative factor return).
        avg_daily_contribution: Average daily return from this factor.
        risk_contribution: Fraction of portfolio variance explained by
                          this factor (0–1).
    """

    factor_name: str
    beta: float
    t_stat: float
    contribution: float
    avg_daily_contribution: float
    risk_contribution: float


@dataclass
class FactorAttributionReport:
    """Complete factor-based attribution report.

    Attributes:
        alpha:              Annualised alpha (intercept × 252).
        alpha_daily:        Daily alpha (regression intercept).
        alpha_t_stat:       t-statistic for alpha.
        r_squared:          Fraction of return variance explained by factors.
        adjusted_r_squared: R² adjusted for number of factors.
        total_return:       Portfolio cumulative return over period.
        factor_return:      Cumulative return explained by factors.
        residual_return:    total_return − factor_return (≈ alpha contribution).
        factor_contributions: Per-factor decomposition.
        residual_vol:       Annualised volatility of residual (idiosyncratic risk).
        factor_vol:         Annualised volatility of factor component.
        n_observations:     Number of daily observations used.
        factor_exposures:   Dict ``{factor_name: beta}`` for easy integration
                            with existing AttributionReport.
    """

    alpha: float
    alpha_daily: float
    alpha_t_stat: float
    r_squared: float
    adjusted_r_squared: float
    total_return: float
    factor_return: float
    residual_return: float
    factor_contributions: list[FactorContribution] = field(default_factory=list)
    residual_vol: float = 0.0
    factor_vol: float = 0.0
    n_observations: int = 0
    factor_exposures: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# OLS helpers (pure numpy, no statsmodels)
# ---------------------------------------------------------------------------


def _ols(y: np.ndarray, x_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Ordinary least squares via normal equations.

    Args:
        y: Dependent variable, shape ``(n,)``.
        x_mat: Design matrix with intercept column, shape ``(n, k)``.

    Returns:
        ``(coefficients, standard_errors, r_squared)``
        where coefficients and standard_errors have shape ``(k,)``.
    """
    n, k = x_mat.shape

    # Normal equations: beta = (X'X)^{-1} X'y
    xtx = x_mat.T @ x_mat
    xty = x_mat.T @ y

    # Regularise for numerical stability
    xtx += np.eye(k) * 1e-10

    betas = np.linalg.solve(xtx, xty)

    # Residuals and R-squared
    residuals = y - x_mat @ betas
    ss_res = float(residuals @ residuals)
    ss_tot = float(((y - y.mean()) ** 2).sum()) if y.std() > 1e-12 else 1e-12
    r_sq = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    # Standard errors
    if n > k:
        mse = ss_res / (n - k)
        try:
            var_beta = np.diag(np.linalg.inv(xtx)) * mse
            se = np.sqrt(np.maximum(var_beta, 0.0))
        except np.linalg.LinAlgError:
            se = np.full(k, np.nan)
    else:
        se = np.full(k, np.nan)

    return betas, se, r_sq


# ---------------------------------------------------------------------------
# Built-in factor construction
# ---------------------------------------------------------------------------


def construct_factors(
    asset_returns: pd.DataFrame,
    weights: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Construct standard equity factors from cross-sectional asset data.

    Builds long-short factor-mimicking portfolios:
      - **market**: equal-weight average of all assets.
      - **size**: small-cap minus large-cap (proxied by inverse vol).
      - **momentum**: winners minus losers (trailing 63-day return).
      - **low_vol**: low-volatility minus high-volatility.
      - **mean_reversion**: recent losers minus recent winners (5-day).

    Args:
        asset_returns: Daily returns, DatetimeIndex × symbols.
        weights: Portfolio weights (optional, not used for factor construction).

    Returns:
        DataFrame with DatetimeIndex × factor_names.
    """
    n_days, n_assets = asset_returns.shape
    if n_days < 10 or n_assets < 2:
        return pd.DataFrame(index=asset_returns.index)

    factors: dict[str, pd.Series] = {}

    # Market factor: equal-weight average
    factors["market"] = asset_returns.mean(axis=1)

    # Rolling statistics for factor construction
    lookback_mom = min(63, n_days - 1)
    lookback_vol = min(63, n_days - 1)
    lookback_mr = min(5, n_days - 1)

    # Momentum: cumulative return over lookback
    cum_ret = asset_returns.rolling(window=lookback_mom, min_periods=max(10, lookback_mom // 2)).sum()

    # Volatility: rolling standard deviation
    roll_vol = asset_returns.rolling(window=lookback_vol, min_periods=max(10, lookback_vol // 2)).std()

    # Mean reversion: short-term cumulative return
    short_ret = asset_returns.rolling(window=lookback_mr, min_periods=3).sum()

    # Build long-short factor portfolios day by day
    mom_returns = []
    vol_returns = []
    mr_returns = []

    for i in range(n_days):
        # Momentum factor: top half minus bottom half by trailing return
        if i >= lookback_mom and not cum_ret.iloc[i].isna().all():
            scores = cum_ret.iloc[i].dropna()
            if len(scores) >= 2:
                median = scores.median()
                long_syms = scores[scores >= median].index
                short_syms = scores[scores < median].index
                day_ret = asset_returns.iloc[i]
                long_ret = day_ret[long_syms].mean() if len(long_syms) > 0 else 0.0
                short_ret_val = day_ret[short_syms].mean() if len(short_syms) > 0 else 0.0
                mom_returns.append(long_ret - short_ret_val)
            else:
                mom_returns.append(0.0)
        else:
            mom_returns.append(0.0)

        # Low-vol factor: low vol minus high vol
        if i >= lookback_vol and not roll_vol.iloc[i].isna().all():
            vols = roll_vol.iloc[i].dropna()
            if len(vols) >= 2:
                median = vols.median()
                low_vol_syms = vols[vols <= median].index
                high_vol_syms = vols[vols > median].index
                day_ret = asset_returns.iloc[i]
                low_ret = day_ret[low_vol_syms].mean() if len(low_vol_syms) > 0 else 0.0
                high_ret = day_ret[high_vol_syms].mean() if len(high_vol_syms) > 0 else 0.0
                vol_returns.append(low_ret - high_ret)
            else:
                vol_returns.append(0.0)
        else:
            vol_returns.append(0.0)

        # Mean-reversion factor: recent losers minus recent winners
        if i >= lookback_mr and not short_ret.iloc[i].isna().all():
            scores = short_ret.iloc[i].dropna()
            if len(scores) >= 2:
                median = scores.median()
                losers = scores[scores <= median].index
                winners = scores[scores > median].index
                day_ret = asset_returns.iloc[i]
                loser_ret = day_ret[losers].mean() if len(losers) > 0 else 0.0
                winner_ret = day_ret[winners].mean() if len(winners) > 0 else 0.0
                mr_returns.append(loser_ret - winner_ret)
            else:
                mr_returns.append(0.0)
        else:
            mr_returns.append(0.0)

    factors["momentum"] = pd.Series(mom_returns, index=asset_returns.index)
    factors["low_vol"] = pd.Series(vol_returns, index=asset_returns.index)
    factors["mean_reversion"] = pd.Series(mr_returns, index=asset_returns.index)

    return pd.DataFrame(factors)


# ---------------------------------------------------------------------------
# Factor attributor
# ---------------------------------------------------------------------------


class FactorAttributor:
    """Decompose portfolio returns into factor contributions and alpha.

    Runs a multi-factor OLS regression of portfolio returns on factor
    returns.  Supports both user-supplied factor series and automatic
    factor construction from asset-level data.

    Args:
        min_observations: Minimum data points required for regression.
    """

    def __init__(self, min_observations: int = 30) -> None:
        self._min_obs = min_observations

    def attribute(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame | None = None,
        asset_returns: pd.DataFrame | None = None,
    ) -> FactorAttributionReport:
        """Run factor-based attribution.

        Args:
            portfolio_returns: Daily portfolio returns.
            factor_returns: Factor return series (DatetimeIndex × factors).
                If None, factors are constructed from *asset_returns*.
            asset_returns: Per-asset daily returns. Used for automatic
                factor construction if *factor_returns* is not provided.

        Returns:
            :class:`FactorAttributionReport` with full decomposition.

        Raises:
            ValueError: If neither factor_returns nor asset_returns is
                provided, or insufficient data.
        """
        if factor_returns is None:
            if asset_returns is None:
                raise ValueError(
                    "Must provide either factor_returns or asset_returns "
                    "for automatic factor construction"
                )
            factor_returns = construct_factors(asset_returns)

        if factor_returns.empty:
            return self._empty_report(portfolio_returns)

        # Align indices
        common = portfolio_returns.index.intersection(factor_returns.index)
        if len(common) < self._min_obs:
            return self._empty_report(portfolio_returns)

        y = portfolio_returns.reindex(common).fillna(0.0).values
        x_factors = factor_returns.reindex(common).fillna(0.0).values
        factor_names = list(factor_returns.columns)

        n = len(y)
        k = x_factors.shape[1]

        # Add intercept column (first column)
        x_mat = np.column_stack([np.ones(n), x_factors])

        # Run OLS
        betas, se, r_sq = _ols(y, x_mat)

        alpha_daily = float(betas[0])
        alpha_ann = alpha_daily * TRADING_DAYS_PER_YEAR
        alpha_t = float(betas[0] / se[0]) if se[0] > 1e-12 else 0.0

        # Adjusted R-squared
        adj_r_sq = (
            1.0 - (1.0 - r_sq) * (n - 1) / (n - k - 1) if n > k + 1 else r_sq
        )

        # Factor and residual components
        factor_component = x_factors @ betas[1:]
        residual = y - (alpha_daily + factor_component)

        # Cumulative returns
        total_ret = float((1 + pd.Series(y)).prod() - 1)
        factor_cum_ret = float((1 + pd.Series(factor_component)).prod() - 1)
        residual_ret = total_ret - factor_cum_ret

        # Volatilities
        residual_vol = float(np.std(residual, ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)) if n > 1 else 0.0
        factor_vol = float(np.std(factor_component, ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)) if n > 1 else 0.0

        # Risk attribution: variance decomposition
        total_var = float(np.var(y, ddof=1)) if n > 1 else 1e-12
        if total_var < 1e-12:
            total_var = 1e-12

        # Per-factor risk contribution via covariance
        factor_risk_contribs = self._factor_risk_decomposition(
            betas[1:], x_factors, total_var
        )

        # Build per-factor contributions
        contributions: list[FactorContribution] = []
        factor_exposures: dict[str, float] = {}

        for j, fname in enumerate(factor_names):
            beta_j = float(betas[j + 1])
            t_stat_j = float(betas[j + 1] / se[j + 1]) if se[j + 1] > 1e-12 else 0.0
            factor_j = x_factors[:, j]

            # Contribution: beta × cumulative factor return
            cum_factor_j = float((1 + pd.Series(factor_j)).prod() - 1)
            contribution_j = beta_j * cum_factor_j
            avg_daily_j = float(np.mean(beta_j * factor_j))

            risk_pct = factor_risk_contribs[j] if j < len(factor_risk_contribs) else 0.0

            contributions.append(
                FactorContribution(
                    factor_name=fname,
                    beta=beta_j,
                    t_stat=t_stat_j,
                    contribution=contribution_j,
                    avg_daily_contribution=avg_daily_j,
                    risk_contribution=risk_pct,
                )
            )
            factor_exposures[fname] = beta_j

        return FactorAttributionReport(
            alpha=alpha_ann,
            alpha_daily=alpha_daily,
            alpha_t_stat=alpha_t,
            r_squared=r_sq,
            adjusted_r_squared=adj_r_sq,
            total_return=total_ret,
            factor_return=factor_cum_ret,
            residual_return=residual_ret,
            factor_contributions=contributions,
            residual_vol=residual_vol,
            factor_vol=factor_vol,
            n_observations=n,
            factor_exposures=factor_exposures,
        )

    @staticmethod
    def _factor_risk_decomposition(
        betas: np.ndarray,
        factor_data: np.ndarray,
        total_var: float,
    ) -> list[float]:
        """Compute fraction of portfolio variance attributed to each factor.

        Uses the marginal contribution approach:
        ``risk_j = beta_j * sum_i(beta_i * cov(f_j, f_i)) / total_var``.
        """
        k = len(betas)
        if k == 0 or total_var < 1e-12:
            return []

        # Factor covariance matrix
        if factor_data.shape[0] < 2:
            return [0.0] * k

        cov_matrix = np.cov(factor_data, rowvar=False)
        if cov_matrix.ndim == 0:
            cov_matrix = np.array([[float(cov_matrix)]])

        # Marginal risk contribution: beta * Sigma * beta
        beta_cov = cov_matrix @ betas
        marginal = betas * beta_cov
        total_factor_var = float(marginal.sum())

        if total_factor_var < 1e-12:
            return [0.0] * k

        # Normalise so that factor contributions sum to factor's share of total variance
        factor_share = min(1.0, total_factor_var / total_var)
        return [float(m / total_factor_var * factor_share) for m in marginal]

    @staticmethod
    def _empty_report(portfolio_returns: pd.Series) -> FactorAttributionReport:
        """Return an empty report when attribution cannot be computed."""
        total_ret = float((1 + portfolio_returns.fillna(0.0)).prod() - 1)
        return FactorAttributionReport(
            alpha=0.0,
            alpha_daily=0.0,
            alpha_t_stat=0.0,
            r_squared=0.0,
            adjusted_r_squared=0.0,
            total_return=total_ret,
            factor_return=0.0,
            residual_return=total_ret,
            n_observations=len(portfolio_returns),
        )
