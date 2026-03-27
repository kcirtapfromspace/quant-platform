"""Walk-forward factor attribution — rolling out-of-sample factor decomposition.

Runs :class:`FactorAttributor` over sliding windows to reveal how factor
exposures, alpha, and model fit evolve through time.  This is the CIO's
primary tool for detecting:

  * **Alpha decay**: does residual alpha persist or fade?
  * **Beta drift**: are factor exposures stable or regime-dependent?
  * **Model adequacy**: does R-squared stay high, or do regimes break the model?

The output is a time-indexed panel of attribution snapshots, plus
convenience accessors for plotting beta paths and rolling alpha.

Usage::

    from quant.portfolio.walk_forward_attribution import (
        WalkForwardAttributor,
        WalkForwardAttributionConfig,
    )

    wfa = WalkForwardAttributor()
    result = wfa.run(
        portfolio_returns=daily_returns,
        factor_returns=factor_df,
        config=WalkForwardAttributionConfig(window=126, step=21),
    )
    print(result.summary())
    beta_df = result.beta_paths()      # DatetimeIndex x factor_name
    alpha_ts = result.rolling_alpha()   # DatetimeIndex -> annualised alpha
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from quant.portfolio.factor_attribution import (
    FactorAttributionReport,
    FactorAttributor,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardAttributionConfig:
    """Configuration for walk-forward attribution.

    Attributes:
        window:           Number of trading days per attribution window.
        step:             Number of trading days to advance between windows.
        min_observations: Minimum observations for OLS within a window.
                          Passed through to :class:`FactorAttributor`.
    """

    window: int = 126
    step: int = 21
    min_observations: int = 30


# ---------------------------------------------------------------------------
# Per-window snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WindowSnapshot:
    """Attribution result for a single rolling window.

    Attributes:
        window_start:  First date of the window.
        window_end:    Last date of the window.
        report:        Full :class:`FactorAttributionReport` for this window.
    """

    window_start: date
    window_end: date
    report: FactorAttributionReport


# ---------------------------------------------------------------------------
# Aggregate result
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardAttributionResult:
    """Aggregated walk-forward attribution results.

    Attributes:
        config:         Configuration used.
        snapshots:      Per-window attribution snapshots, ordered by time.
        factor_names:   Factor names present across windows.
    """

    config: WalkForwardAttributionConfig
    snapshots: list[WindowSnapshot] = field(default_factory=list)
    factor_names: list[str] = field(default_factory=list)

    @property
    def n_windows(self) -> int:
        return len(self.snapshots)

    # ── Convenience accessors ──────────────────────────────────────

    def beta_paths(self) -> pd.DataFrame:
        """Time series of factor betas.

        Returns:
            DataFrame with ``window_end`` as DatetimeIndex and one column
            per factor.
        """
        if not self.snapshots:
            return pd.DataFrame()

        rows: list[dict[str, float]] = []
        dates: list[date] = []
        for snap in self.snapshots:
            row = {
                fc.factor_name: fc.beta
                for fc in snap.report.factor_contributions
            }
            rows.append(row)
            dates.append(snap.window_end)

        return pd.DataFrame(rows, index=pd.DatetimeIndex(dates))

    def rolling_alpha(self) -> pd.Series:
        """Time series of annualised alpha estimates.

        Returns:
            Series indexed by ``window_end``.
        """
        if not self.snapshots:
            return pd.Series(dtype=float)

        dates = [s.window_end for s in self.snapshots]
        alphas = [s.report.alpha for s in self.snapshots]
        return pd.Series(alphas, index=pd.DatetimeIndex(dates), name="alpha")

    def rolling_r_squared(self) -> pd.Series:
        """Time series of R-squared values.

        Returns:
            Series indexed by ``window_end``.
        """
        if not self.snapshots:
            return pd.Series(dtype=float)

        dates = [s.window_end for s in self.snapshots]
        r2s = [s.report.r_squared for s in self.snapshots]
        return pd.Series(r2s, index=pd.DatetimeIndex(dates), name="r_squared")

    def rolling_residual_vol(self) -> pd.Series:
        """Time series of annualised residual (idiosyncratic) volatility.

        Returns:
            Series indexed by ``window_end``.
        """
        if not self.snapshots:
            return pd.Series(dtype=float)

        dates = [s.window_end for s in self.snapshots]
        vols = [s.report.residual_vol for s in self.snapshots]
        return pd.Series(vols, index=pd.DatetimeIndex(dates), name="residual_vol")

    def risk_contribution_paths(self) -> pd.DataFrame:
        """Time series of per-factor risk contributions.

        Returns:
            DataFrame with ``window_end`` as DatetimeIndex and one column
            per factor (values sum to factor's share of total variance).
        """
        if not self.snapshots:
            return pd.DataFrame()

        rows: list[dict[str, float]] = []
        dates: list[date] = []
        for snap in self.snapshots:
            row = {
                fc.factor_name: fc.risk_contribution
                for fc in snap.report.factor_contributions
            }
            rows.append(row)
            dates.append(snap.window_end)

        return pd.DataFrame(rows, index=pd.DatetimeIndex(dates))

    # ── Summary ────────────────────────────────────────────────────

    def summary(self) -> str:
        """Human-readable summary of walk-forward attribution."""
        if not self.snapshots:
            return "Walk-Forward Attribution: no windows computed"

        first = self.snapshots[0]
        last = self.snapshots[-1]

        alpha_ts = self.rolling_alpha()
        r2_ts = self.rolling_r_squared()
        beta_df = self.beta_paths()

        lines = [
            "Walk-Forward Factor Attribution",
            "=" * 55,
            f"  Windows       : {self.n_windows}",
            f"  Period        : {first.window_start} -> {last.window_end}",
            f"  Window size   : {self.config.window}d  step={self.config.step}d",
            "",
            "Rolling Alpha (annualised)",
            "-" * 55,
            f"  Mean          : {alpha_ts.mean():+.4f}",
            f"  Std           : {alpha_ts.std():.4f}",
            f"  Min           : {alpha_ts.min():+.4f}",
            f"  Max           : {alpha_ts.max():+.4f}",
            "",
            "Rolling R-squared",
            "-" * 55,
            f"  Mean          : {r2_ts.mean():.3f}",
            f"  Min           : {r2_ts.min():.3f}",
            f"  Max           : {r2_ts.max():.3f}",
        ]

        if not beta_df.empty:
            lines.append("")
            lines.append("Factor Beta Stability")
            lines.append("-" * 55)
            for col in beta_df.columns:
                series = beta_df[col]
                lines.append(
                    f"  {col:18s}  mean={series.mean():+.3f}  "
                    f"std={series.std():.3f}  "
                    f"range=[{series.min():+.3f}, {series.max():+.3f}]"
                )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class WalkForwardAttributor:
    """Run factor attribution in rolling windows.

    Args:
        min_observations: Minimum observations per window for OLS.
            Passed to each :class:`FactorAttributor` instance.
    """

    def __init__(self, min_observations: int | None = None) -> None:
        self._min_obs = min_observations

    def run(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame | None = None,
        asset_returns: pd.DataFrame | None = None,
        config: WalkForwardAttributionConfig | None = None,
    ) -> WalkForwardAttributionResult:
        """Run walk-forward attribution.

        Args:
            portfolio_returns: Daily portfolio return series.
            factor_returns: Factor return series (DatetimeIndex x factors).
                If None, factors are auto-constructed from *asset_returns*.
            asset_returns: Per-asset daily returns for auto-construction.
            config: Walk-forward configuration.

        Returns:
            :class:`WalkForwardAttributionResult` with rolling attribution.

        Raises:
            ValueError: If neither factor_returns nor asset_returns provided,
                or data is too short for at least one window.
        """
        if config is None:
            config = WalkForwardAttributionConfig()

        if factor_returns is None and asset_returns is None:
            raise ValueError(
                "Must provide either factor_returns or asset_returns"
            )

        min_obs = self._min_obs or config.min_observations
        attributor = FactorAttributor(min_observations=min_obs)

        # Ensure DatetimeIndex
        if not isinstance(portfolio_returns.index, pd.DatetimeIndex):
            portfolio_returns = portfolio_returns.copy()
            portfolio_returns.index = pd.to_datetime(portfolio_returns.index)

        n = len(portfolio_returns)
        if n < config.window:
            raise ValueError(
                f"Need at least {config.window} observations, got {n}"
            )

        snapshots: list[WindowSnapshot] = []
        factor_names_set: set[str] = set()

        pos = 0
        while pos + config.window <= n:
            window_ret = portfolio_returns.iloc[pos : pos + config.window]
            window_start = window_ret.index[0]
            window_end = window_ret.index[-1]

            # Slice factor or asset returns to same window
            w_factors: pd.DataFrame | None = None
            w_assets: pd.DataFrame | None = None

            if factor_returns is not None:
                mask = (factor_returns.index >= window_start) & (
                    factor_returns.index <= window_end
                )
                w_factors = factor_returns.loc[mask]
            if asset_returns is not None:
                mask = (asset_returns.index >= window_start) & (
                    asset_returns.index <= window_end
                )
                w_assets = asset_returns.loc[mask]

            report = attributor.attribute(
                portfolio_returns=window_ret,
                factor_returns=w_factors,
                asset_returns=w_assets,
            )

            snapshots.append(
                WindowSnapshot(
                    window_start=_to_date(window_start),
                    window_end=_to_date(window_end),
                    report=report,
                )
            )

            for fc in report.factor_contributions:
                factor_names_set.add(fc.factor_name)

            pos += config.step

        return WalkForwardAttributionResult(
            config=config,
            snapshots=snapshots,
            factor_names=sorted(factor_names_set),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_date(ts) -> date:
    """Convert a timestamp-like to a date."""
    if hasattr(ts, "date"):
        return ts.date()
    return ts
