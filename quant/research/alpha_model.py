"""Cross-sectional alpha model: signal scores → expected returns.

Converts raw, dimensionless signal scores into calibrated expected return
forecasts suitable for the portfolio optimizer.  Implements the fundamental
law of active management:

    E[r_i] = IC · σ_i · z_i

Where:
  * **IC** — information coefficient (signal's predictive power).
  * **σ_i** — asset-level volatility.
  * **z_i** — cross-sectionally standardised signal score.

Supported calibration methods:

  * **IC-vol scaling** — classical IC × vol × z-score.
  * **Rank scaling** — convert signals to percentile ranks, then scale by
    a target spread (top quintile return - bottom quintile return).
  * **Raw scaling** — treat signal scores directly as expected return
    magnitude, scaled by a user-defined multiplier.

The output is a ``pd.Series`` of expected returns keyed by symbol, ready
to pass as ``expected_returns`` to any optimizer.

Usage::

    from quant.research.alpha_model import AlphaModel, AlphaModelConfig

    model = AlphaModel(AlphaModelConfig(
        method="ic_vol",
        information_coefficient=0.05,
    ))
    expected_returns = model.forecast(signal_scores, asset_volatilities)
    # expected_returns is a pd.Series usable in optimizer.optimize()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

TRADING_DAYS = 252

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ForecastMethod(Enum):
    """Method for converting signals to expected returns."""

    IC_VOL = "ic_vol"
    RANK = "rank"
    RAW = "raw"


@dataclass
class AlphaModelConfig:
    """Configuration for the cross-sectional alpha model.

    Attributes:
        method:                 Forecast method.
        information_coefficient: Signal IC for IC_VOL method.
        target_spread:          Annualised return spread for RANK method
                                (top minus bottom quintile).
        raw_multiplier:         Multiplier for RAW method.
        winsorise_z:            Winsorise z-scores at ±this value.
        min_assets:             Minimum number of assets for cross-sectional ops.
        annualise:              Whether to annualise the output.
        neutralise:             If True, cross-sectionally demean forecasts
                                (dollar-neutral alpha).
    """

    method: ForecastMethod = ForecastMethod.IC_VOL
    information_coefficient: float = 0.05
    target_spread: float = 0.10
    raw_multiplier: float = 0.01
    winsorise_z: float = 3.0
    min_assets: int = 2
    annualise: bool = True
    neutralise: bool = True


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass
class AlphaModelResult:
    """Result of cross-sectional alpha model forecast.

    Attributes:
        expected_returns:   Calibrated expected returns per asset (pd.Series).
        z_scores:           Standardised signal scores (pd.Series).
        ranks:              Percentile ranks of signals (pd.Series, 0–1).
        n_assets:           Number of assets in the universe.
        method:             Forecast method used.
        cross_sectional_vol: Average asset volatility used in scaling.
        forecast_spread:    Spread between highest and lowest forecast.
    """

    expected_returns: pd.Series = field(repr=False)
    z_scores: pd.Series = field(repr=False)
    ranks: pd.Series = field(repr=False)
    n_assets: int = 0
    method: ForecastMethod = ForecastMethod.IC_VOL
    cross_sectional_vol: float = 0.0
    forecast_spread: float = 0.0

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"Alpha Model Forecast ({self.n_assets} assets, {self.method.value})",
            "=" * 60,
            "",
            f"Avg cross-sectional vol  : {self.cross_sectional_vol:.4f}",
            f"Forecast spread (hi-lo)  : {self.forecast_spread:+.4f}",
            "",
            f"{'Symbol':<10s} {'Signal z':>10s} {'Rank':>8s} {'E[r]':>10s}",
            "-" * 42,
        ]
        sorted_syms = self.expected_returns.sort_values(ascending=False).index
        for sym in sorted_syms[:10]:
            lines.append(
                f"{sym:<10s} "
                f"{self.z_scores.get(sym, 0):>+10.3f} "
                f"{self.ranks.get(sym, 0):>8.1%} "
                f"{self.expected_returns[sym]:>+10.4f}"
            )
        if len(sorted_syms) > 10:
            lines.append(f"  ... and {len(sorted_syms) - 10} more")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class AlphaModel:
    """Cross-sectional alpha model converting signals to expected returns.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: AlphaModelConfig | None = None) -> None:
        self._config = config or AlphaModelConfig()

    @property
    def config(self) -> AlphaModelConfig:
        return self._config

    def forecast(
        self,
        signal_scores: pd.Series,
        asset_volatilities: pd.Series | None = None,
    ) -> AlphaModelResult:
        """Produce expected return forecasts from signal scores.

        Args:
            signal_scores:      Raw signal scores per asset (pd.Series, symbol index).
            asset_volatilities: Annualised volatility per asset (pd.Series).
                                Required for IC_VOL method.  If ``None``, a
                                uniform volatility of 0.20 is assumed.

        Returns:
            :class:`AlphaModelResult` with calibrated expected returns.

        Raises:
            ValueError: If fewer than ``min_assets`` provided.
        """
        cfg = self._config
        scores = signal_scores.dropna()

        if len(scores) < cfg.min_assets:
            raise ValueError(
                f"Need at least {cfg.min_assets} assets, got {len(scores)}"
            )

        symbols = list(scores.index)
        n = len(symbols)

        # Default volatilities
        if asset_volatilities is not None:
            vols = asset_volatilities.reindex(symbols).fillna(0.20)
        else:
            vols = pd.Series(0.20, index=symbols)

        # Cross-sectional z-score
        z = self._z_score(scores, cfg.winsorise_z)

        # Percentile ranks (0 = lowest signal, 1 = highest)
        ranks = scores.rank(pct=True)

        # Compute expected returns by method
        if cfg.method == ForecastMethod.IC_VOL:
            er = self._ic_vol_forecast(z, vols, cfg)
        elif cfg.method == ForecastMethod.RANK:
            er = self._rank_forecast(ranks, cfg)
        else:
            er = self._raw_forecast(scores, cfg)

        # Neutralise (demean cross-sectionally)
        if cfg.neutralise:
            er = er - er.mean()

        spread = float(er.max() - er.min()) if len(er) > 1 else 0.0
        avg_vol = float(vols.mean())

        return AlphaModelResult(
            expected_returns=er,
            z_scores=z,
            ranks=ranks,
            n_assets=n,
            method=cfg.method,
            cross_sectional_vol=avg_vol,
            forecast_spread=spread,
        )

    @staticmethod
    def _z_score(scores: pd.Series, winsorise: float) -> pd.Series:
        """Cross-sectional z-score with winsorisation."""
        mean = scores.mean()
        std = scores.std(ddof=1)
        if std < 1e-12:
            return pd.Series(0.0, index=scores.index)
        z = (scores - mean) / std
        return z.clip(-winsorise, winsorise)

    @staticmethod
    def _ic_vol_forecast(
        z: pd.Series,
        vols: pd.Series,
        cfg: AlphaModelConfig,
    ) -> pd.Series:
        """IC-vol scaling: E[r_i] = IC · σ_i · z_i."""
        return cfg.information_coefficient * vols * z

    @staticmethod
    def _rank_forecast(
        ranks: pd.Series,
        cfg: AlphaModelConfig,
    ) -> pd.Series:
        """Rank-based scaling: map ranks to [-spread/2, +spread/2]."""
        # Center ranks at 0 (rank 0.5 → 0.0)
        centered = ranks - 0.5
        return centered * cfg.target_spread

    @staticmethod
    def _raw_forecast(
        scores: pd.Series,
        cfg: AlphaModelConfig,
    ) -> pd.Series:
        """Raw scaling: E[r_i] = multiplier · score_i."""
        return scores * cfg.raw_multiplier
