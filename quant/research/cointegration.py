"""Cointegration testing for pairs trading and statistical arbitrage.

Implements the Engle-Granger two-step procedure for identifying cointegrated
pairs:  regress one price series on another via OLS, then test the residual
(spread) for stationarity with an Augmented Dickey-Fuller test.

Key outputs:

  * **ADF statistic** and cointegration decision at a configurable significance.
  * **Hedge ratio** (OLS or Total Least Squares) and intercept.
  * **Half-life** of mean reversion (Ornstein-Uhlenbeck implied).
  * **Spread and z-score time series** for backtesting.
  * **Universe screen** that tests all pairs and ranks by cointegration strength.

Usage::

    from quant.research.cointegration import (
        CointegrationTester,
        CointegrationConfig,
    )

    tester = CointegrationTester(CointegrationConfig(significance=0.05))
    result = tester.test_pair(prices_y, prices_x)
    print(result.summary())

    # Screen a universe
    screen = tester.screen_universe(prices_df)
    for pair in screen.pairs[:10]:
        print(pair.asset_y, pair.asset_x, pair.adf_statistic)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import combinations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Engle-Granger critical values (asymptotic, 2-variable with constant)
# From MacKinnon (1991/2010) — more conservative than standard ADF because
# the residuals are constructed to minimise squared errors.
# ---------------------------------------------------------------------------
_EG_CRITICAL: dict[float, float] = {
    0.01: -3.90,
    0.05: -3.34,
    0.10: -3.04,
}

# For approximate p-value interpolation
_EG_GRID = sorted(_EG_CRITICAL.items())  # [(0.01, -3.90), (0.05, -3.34), ...]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class HedgeMethod(Enum):
    """Hedge ratio estimation method."""

    OLS = auto()
    TLS = auto()


@dataclass
class CointegrationConfig:
    """Configuration for cointegration testing.

    Attributes:
        significance:     Significance level for the Engle-Granger test.
        adf_max_lags:     Maximum ADF augmentation lags (selected by BIC).
        min_observations: Minimum price observations required.
        hedge_method:     Hedge ratio estimation method.
        z_score_window:   Rolling window for z-score computation.
    """

    significance: float = 0.05
    adf_max_lags: int = 10
    min_observations: int = 60
    hedge_method: HedgeMethod = HedgeMethod.OLS
    z_score_window: int = 63


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass
class PairResult:
    """Cointegration test result for a single pair.

    Attributes:
        asset_y:          Dependent asset (long leg).
        asset_x:          Independent asset (short leg).
        hedge_ratio:      Hedge ratio β (units of X per unit of Y).
        intercept:        Regression intercept α.
        adf_statistic:    ADF test statistic on the spread.
        adf_critical:     Critical value at the configured significance.
        is_cointegrated:  True if ``adf_statistic < adf_critical``.
        half_life:        Mean-reversion half-life in trading days.
        spread_mean:      Long-run spread mean.
        spread_std:       Spread standard deviation.
        spread_series:    Spread time series (y − β·x − α).
        z_score_series:   Rolling z-score of the spread.
        n_observations:   Number of price observations used.
    """

    asset_y: str
    asset_x: str
    hedge_ratio: float
    intercept: float
    adf_statistic: float
    adf_critical: float
    is_cointegrated: bool
    half_life: float
    spread_mean: float
    spread_std: float
    spread_series: pd.Series = field(repr=False)
    z_score_series: pd.Series = field(repr=False)
    n_observations: int = 0

    def summary(self) -> str:
        """Return a human-readable summary."""
        coint_str = "YES" if self.is_cointegrated else "NO"
        lines = [
            f"Cointegration: {self.asset_y} ~ {self.asset_x}",
            "=" * 60,
            "",
            f"Cointegrated   : {coint_str}",
            f"ADF statistic  : {self.adf_statistic:.3f}",
            f"Critical value : {self.adf_critical:.3f}",
            f"Hedge ratio    : {self.hedge_ratio:.4f}",
            f"Intercept      : {self.intercept:.4f}",
            f"Half-life      : {self.half_life:.1f} days",
            f"Spread mean    : {self.spread_mean:.4f}",
            f"Spread std     : {self.spread_std:.4f}",
            f"Observations   : {self.n_observations}",
        ]
        return "\n".join(lines)


@dataclass
class ScreenResult:
    """Result of screening a universe for cointegrated pairs.

    Attributes:
        pairs:            All tested pairs, sorted by ADF statistic.
        n_tested:         Total pairs tested.
        n_cointegrated:   Number of cointegrated pairs at configured significance.
    """

    pairs: list[PairResult] = field(default_factory=list)
    n_tested: int = 0
    n_cointegrated: int = 0

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"Cointegration Screen ({self.n_tested} pairs tested)",
            "=" * 60,
            "",
            f"Cointegrated   : {self.n_cointegrated} / {self.n_tested}",
            "",
        ]
        if self.pairs:
            lines.append("Top pairs (by ADF statistic):")
            for p in self.pairs[:10]:
                flag = "*" if p.is_cointegrated else " "
                lines.append(
                    f"  {flag} {p.asset_y:<8s} ~ {p.asset_x:<8s}  "
                    f"ADF={p.adf_statistic:+.3f}  "
                    f"HL={p.half_life:.0f}d",
                )
            if len(self.pairs) > 10:
                lines.append(f"  ... and {len(self.pairs) - 10} more")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tester
# ---------------------------------------------------------------------------


class CointegrationTester:
    """Engle-Granger cointegration tester.

    Args:
        config: Test configuration.
    """

    def __init__(self, config: CointegrationConfig | None = None) -> None:
        self._config = config or CointegrationConfig()

    @property
    def config(self) -> CointegrationConfig:
        return self._config

    def test_pair(
        self,
        y: pd.Series,
        x: pd.Series,
        *,
        asset_y: str | None = None,
        asset_x: str | None = None,
    ) -> PairResult:
        """Test a pair of price series for cointegration.

        Args:
            y:       Price series of the dependent asset.
            x:       Price series of the independent asset.
            asset_y: Label for asset Y (defaults to series name or "Y").
            asset_x: Label for asset X (defaults to series name or "X").

        Returns:
            :class:`PairResult` with test statistics and spread series.

        Raises:
            ValueError: If fewer than ``min_observations`` aligned prices.
        """
        cfg = self._config
        name_y = asset_y or getattr(y, "name", None) or "Y"
        name_x = asset_x or getattr(x, "name", None) or "X"

        # Align and drop NaN
        aligned = pd.concat(
            [y.rename("y"), x.rename("x")], axis=1, sort=True,
        ).dropna()
        n = len(aligned)

        if n < cfg.min_observations:
            msg = (
                f"Need at least {cfg.min_observations} aligned observations, "
                f"got {n}"
            )
            raise ValueError(msg)

        yv = aligned["y"].values.astype(np.float64)
        xv = aligned["x"].values.astype(np.float64)

        # Step 1: Estimate hedge ratio
        if cfg.hedge_method == HedgeMethod.OLS:
            beta, alpha = _ols_hedge(yv, xv)
        else:
            beta, alpha = _tls_hedge(yv, xv)

        # Step 2: Compute spread
        spread_vals = yv - beta * xv - alpha

        # Step 3: ADF test on spread
        adf_stat, n_lags = _adf_test(spread_vals, cfg.adf_max_lags)

        # Critical value
        crit = _EG_CRITICAL.get(cfg.significance)
        if crit is None:
            crit = _EG_CRITICAL[0.05]

        is_coint = adf_stat < crit

        # Half-life of mean reversion
        half_life = _half_life(spread_vals)

        # Spread statistics
        spread_mean = float(np.mean(spread_vals))
        spread_std = float(np.std(spread_vals, ddof=1)) if n > 1 else 0.0

        # Build series
        spread_series = pd.Series(
            spread_vals, index=aligned.index, name="spread",
        )

        # Rolling z-score
        roll_mean = spread_series.rolling(cfg.z_score_window, min_periods=1).mean()
        roll_std = spread_series.rolling(cfg.z_score_window, min_periods=1).std()
        z_score = (spread_series - roll_mean) / roll_std.replace(0, np.nan)
        z_score = z_score.fillna(0.0)
        z_score.name = "z_score"

        return PairResult(
            asset_y=str(name_y),
            asset_x=str(name_x),
            hedge_ratio=beta,
            intercept=alpha,
            adf_statistic=adf_stat,
            adf_critical=crit,
            is_cointegrated=is_coint,
            half_life=half_life,
            spread_mean=spread_mean,
            spread_std=spread_std,
            spread_series=spread_series,
            z_score_series=z_score,
            n_observations=n,
        )

    def screen_universe(self, prices: pd.DataFrame) -> ScreenResult:
        """Screen all pairs in a price DataFrame for cointegration.

        Tests both orderings (Y~X and X~Y) for each pair and keeps the
        direction with the lower (more negative) ADF statistic.

        Args:
            prices: T x N DataFrame of price series (columns = assets).

        Returns:
            :class:`ScreenResult` with all pairs sorted by ADF statistic.
        """
        symbols = list(prices.columns)
        results: list[PairResult] = []
        n_tested = 0

        for sym_a, sym_b in combinations(symbols, 2):
            try:
                r_ab = self.test_pair(
                    prices[sym_a], prices[sym_b],
                    asset_y=sym_a, asset_x=sym_b,
                )
                r_ba = self.test_pair(
                    prices[sym_b], prices[sym_a],
                    asset_y=sym_b, asset_x=sym_a,
                )
                best = r_ab if r_ab.adf_statistic < r_ba.adf_statistic else r_ba
                results.append(best)
                n_tested += 1
            except ValueError:
                n_tested += 1

        # Sort by ADF statistic (most negative first)
        results.sort(key=lambda r: r.adf_statistic)

        n_coint = sum(1 for r in results if r.is_cointegrated)

        return ScreenResult(
            pairs=results,
            n_tested=n_tested,
            n_cointegrated=n_coint,
        )


# ---------------------------------------------------------------------------
# Internal: hedge ratio estimation
# ---------------------------------------------------------------------------


def _ols_hedge(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """OLS regression: y = α + β·x + ε. Returns (β, α)."""
    n = len(y)
    design = np.column_stack([np.ones(n), x])
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coeffs[1]), float(coeffs[0])


def _tls_hedge(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """Total Least Squares hedge ratio via SVD. Returns (β, α)."""
    # Centre
    y_mean = y.mean()
    x_mean = x.mean()
    yc = y - y_mean
    xc = x - x_mean

    # SVD of [xc, yc]
    mat = np.column_stack([xc, yc])
    _u, _s, vt = np.linalg.svd(mat, full_matrices=False)

    # TLS solution: β = -v[0,1] / v[1,1]  (from the smallest singular vector)
    v_last = vt[-1]
    if abs(v_last[1]) < 1e-15:
        # Degenerate — fall back to OLS
        return _ols_hedge(y, x)

    beta = -v_last[0] / v_last[1]
    alpha = y_mean - beta * x_mean
    return float(beta), float(alpha)


# ---------------------------------------------------------------------------
# Internal: Augmented Dickey-Fuller test
# ---------------------------------------------------------------------------


def _adf_test(
    series: np.ndarray,
    max_lags: int,
) -> tuple[float, int]:
    """Augmented Dickey-Fuller test.

    Regresses Δy_t = α + β·y_{t-1} + Σ γ_i·Δy_{t-i} + ε_t and returns the
    t-statistic of β.

    Lag order is selected by BIC from 0..max_lags.

    Returns:
        (adf_statistic, selected_lags)
    """
    n = len(series)
    dy = np.diff(series)  # Δy_t, length n-1

    best_bic = np.inf
    best_stat = 0.0
    best_lags = 0

    for p in range(0, min(max_lags + 1, n - 2)):
        stat, bic = _adf_regression(series, dy, p)
        if bic < best_bic:
            best_bic = bic
            best_stat = stat
            best_lags = p

    return best_stat, best_lags


def _adf_regression(
    series: np.ndarray,
    dy: np.ndarray,
    n_lags: int,
) -> tuple[float, float]:
    """Run a single ADF regression with a given lag order.

    Returns:
        (t_statistic, BIC)
    """
    n_dy = len(dy)
    start = n_lags  # skip first n_lags observations of Δy

    if start >= n_dy - 2:
        return 0.0, np.inf

    # Dependent: Δy_t  (t = start..n_dy-1)
    y_dep = dy[start:]
    n_obs = len(y_dep)

    # Regressors: constant, y_{t-1}, [Δy_{t-1}, ..., Δy_{t-p}]
    cols = [np.ones(n_obs)]

    # y_{t-1}: for dy[t] = series[t] - series[t-1], the lagged level is series[t]
    # But standard ADF: y_dep = dy[start:], and the lagged level is series[start : n_dy]
    # (i.e., series at time t, where Δy_t = series[t+1] - series[t])
    # Actually: dy[t] = series[t+1] - series[t] for t=0..n-2
    # So y_{t-1} in terms of the original series is series[t] where dy index t maps
    # to series index t.  For the ADF regression on dy[start:], the lagged level
    # is series[start : start + n_obs] = series[start : n_dy].
    y_lag = series[start : start + n_obs]
    cols.append(y_lag)

    # Augmentation lags: Δy_{t-1}, ..., Δy_{t-p}
    for lag_i in range(1, n_lags + 1):
        lag_start = start - lag_i
        cols.append(dy[lag_start : lag_start + n_obs])

    design = np.column_stack(cols)

    # OLS
    coeffs, residuals, rank, _sv = np.linalg.lstsq(design, y_dep, rcond=None)

    if rank < design.shape[1]:
        return 0.0, np.inf

    # Residuals
    y_hat = design @ coeffs
    resid = y_dep - y_hat
    sse = float(np.sum(resid ** 2))
    k = design.shape[1]

    if n_obs <= k:
        return 0.0, np.inf

    sigma2 = sse / (n_obs - k)

    # Standard error of β (coefficient index 1 = y_{t-1})
    try:
        cov_matrix = sigma2 * np.linalg.inv(design.T @ design)
    except np.linalg.LinAlgError:
        return 0.0, np.inf

    se_beta = math.sqrt(max(cov_matrix[1, 1], 0.0))

    if se_beta < 1e-15:
        return 0.0, np.inf

    t_stat = float(coeffs[1]) / se_beta

    # BIC
    bic = n_obs * math.log(max(sse / n_obs, 1e-30)) + k * math.log(n_obs)

    return t_stat, bic


# ---------------------------------------------------------------------------
# Internal: half-life estimation
# ---------------------------------------------------------------------------


def _half_life(spread: np.ndarray) -> float:
    """Estimate half-life of mean reversion from the spread.

    Fits an AR(1) model:  s_t = φ·s_{t-1} + c + ε_t
    and computes  half_life = −ln(2) / ln(φ).
    """
    n = len(spread)
    if n < 3:
        return float("inf")

    s_lag = spread[:-1]
    s_curr = spread[1:]

    # OLS: s_curr = c + φ * s_lag
    design = np.column_stack([np.ones(n - 1), s_lag])
    coeffs, *_ = np.linalg.lstsq(design, s_curr, rcond=None)
    phi = float(coeffs[1])

    if phi >= 1.0 or phi <= 0.0:
        return float("inf")

    return -math.log(2) / math.log(phi)
