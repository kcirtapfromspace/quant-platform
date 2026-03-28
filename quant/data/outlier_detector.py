"""Statistical outlier detection for market data.

Catches anomalous price, return, and volume observations before they enter
the signal pipeline, where they would silently corrupt alpha scores,
covariance estimates, and position sizing.

Complements :mod:`quant.data.validation` (structural checks) with
distributional checks that require historical context.

Detection methods:
  1. **Return Z-score** — flags daily returns > k standard deviations
     from the rolling mean.
  2. **Hampel filter** — robust MAD-based outlier detection on log-prices,
     resistant to masking by other outliers.
  3. **Volume spike** — flags days where volume exceeds k× the rolling
     median (catches phantom liquidity and data errors).
  4. **Stale price** — flags sequences of N+ identical closing prices
     (indicative of stale/missing data from the provider).
  5. **Cross-sectional return** — flags returns that are extreme relative
     to the universe on the same day (catches single-stock data errors
     that pass time-series checks).

Usage::

    from quant.data.outlier_detector import OutlierDetector, OutlierConfig

    detector = OutlierDetector(OutlierConfig(return_z_threshold=4.0))
    report = detector.scan(prices_df, volumes_df)
    if report.has_outliers:
        print(report.summary())
        clean_prices = report.cleaned_prices(method="nan")
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from loguru import logger


class OutlierType(str, enum.Enum):
    RETURN_ZSCORE = "return_zscore"
    HAMPEL = "hampel"
    VOLUME_SPIKE = "volume_spike"
    STALE_PRICE = "stale_price"
    CROSS_SECTIONAL = "cross_sectional"


class CleanMethod(str, enum.Enum):
    NAN = "nan"
    FFILL = "ffill"
    MEDIAN = "median"


@dataclass(frozen=True, slots=True)
class Outlier:
    """A single detected outlier observation."""

    symbol: str
    date: pd.Timestamp
    outlier_type: OutlierType
    value: float
    threshold: float
    detail: str


@dataclass
class OutlierConfig:
    """Configuration for the outlier detector.

    Attributes:
        return_z_threshold: Number of rolling standard deviations for
            a return to be flagged (default 4.0, catches ~0.006% of
            normal distribution).
        return_z_window: Rolling window for Z-score computation (days).
        hampel_window: Window size for the Hampel filter (days).
        hampel_threshold: Number of MADs from the rolling median for
            a log-price to be flagged (default 3.5).
        volume_spike_multiple: Volume must exceed this multiple of the
            rolling median to be flagged (default 10.0).
        volume_window: Rolling window for volume median (days).
        stale_price_days: Number of consecutive identical closes to
            flag as stale (default 5 trading days).
        cross_sectional_z: Return Z-score relative to the universe
            cross-section on the same day (default 4.0).
        min_history: Minimum number of observations before outlier
            detection activates (default 30 days).
    """

    return_z_threshold: float = 4.0
    return_z_window: int = 63
    hampel_window: int = 21
    hampel_threshold: float = 3.5
    volume_spike_multiple: float = 10.0
    volume_window: int = 63
    stale_price_days: int = 5
    cross_sectional_z: float = 4.0
    min_history: int = 30


@dataclass
class OutlierReport:
    """Results of an outlier scan.

    Attributes:
        outliers: List of detected outlier observations.
        flagged_mask: Boolean DataFrame (dates × symbols) where True
            indicates at least one outlier was detected.
        n_observations: Total number of observations scanned.
    """

    outliers: list[Outlier] = field(default_factory=list)
    flagged_mask: pd.DataFrame | None = None
    n_observations: int = 0

    @property
    def has_outliers(self) -> bool:
        return len(self.outliers) > 0

    @property
    def n_outliers(self) -> int:
        return len(self.outliers)

    @property
    def outlier_rate(self) -> float:
        if self.n_observations == 0:
            return 0.0
        return self.n_outliers / self.n_observations

    def by_type(self) -> dict[OutlierType, list[Outlier]]:
        """Group outliers by detection method."""
        result: dict[OutlierType, list[Outlier]] = {}
        for o in self.outliers:
            result.setdefault(o.outlier_type, []).append(o)
        return result

    def by_symbol(self) -> dict[str, list[Outlier]]:
        """Group outliers by symbol."""
        result: dict[str, list[Outlier]] = {}
        for o in self.outliers:
            result.setdefault(o.symbol, []).append(o)
        return result

    def summary(self) -> str:
        """Human-readable summary of the outlier scan."""
        lines = [
            f"Outlier scan: {self.n_outliers} outliers in "
            f"{self.n_observations} observations "
            f"({self.outlier_rate:.4%} rate)",
        ]
        by_type = self.by_type()
        for otype in OutlierType:
            count = len(by_type.get(otype, []))
            if count > 0:
                lines.append(f"  {otype.value}: {count}")

        by_sym = self.by_symbol()
        top_symbols = sorted(by_sym.items(), key=lambda x: -len(x[1]))[:5]
        if top_symbols:
            lines.append("  Top affected symbols:")
            for sym, outliers in top_symbols:
                lines.append(f"    {sym}: {len(outliers)}")
        return "\n".join(lines)

    def cleaned_prices(
        self,
        prices: pd.DataFrame,
        method: CleanMethod | str = CleanMethod.NAN,
    ) -> pd.DataFrame:
        """Return a copy of *prices* with outlier observations replaced.

        Args:
            prices: Original prices DataFrame (dates × symbols).
            method: Replacement strategy — "nan" (default), "ffill"
                (forward-fill from last clean value), or "median"
                (rolling median from config window).

        Returns:
            Cleaned copy of prices.
        """
        if isinstance(method, str):
            method = CleanMethod(method)

        clean = prices.copy()
        if self.flagged_mask is None or not self.has_outliers:
            return clean

        mask = self.flagged_mask.reindex_like(clean).fillna(False)

        if method == CleanMethod.NAN:
            clean[mask] = np.nan
        elif method == CleanMethod.FFILL:
            clean[mask] = np.nan
            clean = clean.ffill()
        elif method == CleanMethod.MEDIAN:
            rolling_med = clean.rolling(21, min_periods=1).median()
            clean = clean.where(~mask, rolling_med)

        return clean


class OutlierDetector:
    """Statistical outlier detector for market data.

    Scans price and volume DataFrames for anomalous observations using
    multiple complementary detection methods.

    Args:
        config: Detection thresholds and window sizes.
    """

    def __init__(self, config: OutlierConfig | None = None) -> None:
        self._config = config or OutlierConfig()

    @property
    def config(self) -> OutlierConfig:
        return self._config

    def scan(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame | None = None,
    ) -> OutlierReport:
        """Run all outlier detection methods on the given data.

        Args:
            prices: DataFrame of adjusted close prices (dates × symbols).
                Index must be DatetimeIndex or convertible.
            volumes: Optional DataFrame of daily volumes (same shape).

        Returns:
            :class:`OutlierReport` with all detected outliers and a
            flagged mask for easy cleaning.
        """
        cfg = self._config
        outliers: list[Outlier] = []
        symbols = list(prices.columns)
        dates = prices.index

        n_obs = int(prices.notna().sum().sum())

        # Pre-compute returns for multiple checks
        returns = prices.pct_change()

        # Initialize combined flag mask
        flagged = pd.DataFrame(False, index=dates, columns=symbols)

        # 1. Return Z-score
        if len(dates) > cfg.min_history:
            zs_outliers, zs_mask = self._check_return_zscore(returns, symbols)
            outliers.extend(zs_outliers)
            flagged |= zs_mask

        # 2. Hampel filter on log-prices
        if len(dates) > cfg.min_history:
            hm_outliers, hm_mask = self._check_hampel(prices, symbols)
            outliers.extend(hm_outliers)
            flagged |= hm_mask

        # 3. Volume spikes
        if volumes is not None and len(dates) > cfg.min_history:
            vs_outliers, vs_mask = self._check_volume_spike(volumes, symbols)
            outliers.extend(vs_outliers)
            flagged |= vs_mask

        # 4. Stale prices
        sp_outliers, sp_mask = self._check_stale_prices(prices, symbols)
        outliers.extend(sp_outliers)
        flagged |= sp_mask

        # 5. Cross-sectional return outliers
        if len(symbols) >= 3 and len(dates) > cfg.min_history:
            cs_outliers, cs_mask = self._check_cross_sectional(returns, symbols)
            outliers.extend(cs_outliers)
            flagged |= cs_mask

        report = OutlierReport(
            outliers=outliers,
            flagged_mask=flagged,
            n_observations=n_obs,
        )

        if report.has_outliers:
            logger.info(
                "Outlier detector: {} outliers in {} observations ({:.4%})",
                report.n_outliers,
                report.n_observations,
                report.outlier_rate,
            )
        else:
            logger.debug(
                "Outlier detector: clean — {} observations scanned",
                report.n_observations,
            )

        return report

    # ── Detection methods ─────────────────────────────────────────────────

    def _check_return_zscore(
        self,
        returns: pd.DataFrame,
        symbols: list[str],
    ) -> tuple[list[Outlier], pd.DataFrame]:
        """Flag returns > k rolling standard deviations from rolling mean."""
        cfg = self._config
        outliers: list[Outlier] = []
        mask = pd.DataFrame(False, index=returns.index, columns=symbols)

        roll_mean = returns.rolling(cfg.return_z_window, min_periods=cfg.min_history).mean()
        roll_std = returns.rolling(cfg.return_z_window, min_periods=cfg.min_history).std()

        # Avoid division by zero
        roll_std = roll_std.replace(0.0, np.nan)

        z_scores = (returns - roll_mean) / roll_std

        for sym in symbols:
            z = z_scores[sym]
            flagged_idx = z.index[z.abs() > cfg.return_z_threshold]
            for dt in flagged_idx:
                z_val = float(z.loc[dt])
                ret_val = float(returns.loc[dt, sym])
                outliers.append(Outlier(
                    symbol=sym,
                    date=dt,
                    outlier_type=OutlierType.RETURN_ZSCORE,
                    value=ret_val,
                    threshold=cfg.return_z_threshold,
                    detail=f"return={ret_val:.4%}, z-score={z_val:.2f}",
                ))
                mask.loc[dt, sym] = True

        return outliers, mask

    def _check_hampel(
        self,
        prices: pd.DataFrame,
        symbols: list[str],
    ) -> tuple[list[Outlier], pd.DataFrame]:
        """Hampel filter: flag log-prices > k MADs from rolling median."""
        cfg = self._config
        outliers: list[Outlier] = []
        mask = pd.DataFrame(False, index=prices.index, columns=symbols)

        log_prices = np.log(prices.clip(lower=1e-10))
        half_w = cfg.hampel_window // 2
        # MAD scale factor for normal distribution
        mad_scale = 1.4826

        for sym in symbols:
            series = log_prices[sym].values
            n = len(series)
            for i in range(cfg.min_history, n):
                start = max(0, i - half_w)
                end = min(n, i + half_w + 1)
                window = series[start:end]
                valid = window[np.isfinite(window)]
                if len(valid) < 3:
                    continue

                med = np.median(valid)
                mad = np.median(np.abs(valid - med)) * mad_scale

                if mad < 1e-10:
                    continue

                deviation = abs(series[i] - med) / mad
                if deviation > cfg.hampel_threshold:
                    dt = prices.index[i]
                    outliers.append(Outlier(
                        symbol=sym,
                        date=dt,
                        outlier_type=OutlierType.HAMPEL,
                        value=float(prices.loc[dt, sym]),
                        threshold=cfg.hampel_threshold,
                        detail=f"price={prices.loc[dt, sym]:.4f}, "
                               f"MAD-deviation={deviation:.2f}",
                    ))
                    mask.iloc[i, mask.columns.get_loc(sym)] = True

        return outliers, mask

    def _check_volume_spike(
        self,
        volumes: pd.DataFrame,
        symbols: list[str],
    ) -> tuple[list[Outlier], pd.DataFrame]:
        """Flag days where volume exceeds k× the rolling median."""
        cfg = self._config
        outliers: list[Outlier] = []
        mask = pd.DataFrame(False, index=volumes.index, columns=symbols)

        roll_med = volumes.rolling(
            cfg.volume_window, min_periods=cfg.min_history
        ).median()

        # Avoid division by zero
        roll_med = roll_med.replace(0.0, np.nan)

        ratio = volumes / roll_med

        for sym in symbols:
            r = ratio[sym]
            flagged_idx = r.index[r > cfg.volume_spike_multiple]
            for dt in flagged_idx:
                vol = float(volumes.loc[dt, sym])
                med = float(roll_med.loc[dt, sym])
                outliers.append(Outlier(
                    symbol=sym,
                    date=dt,
                    outlier_type=OutlierType.VOLUME_SPIKE,
                    value=vol,
                    threshold=cfg.volume_spike_multiple * med,
                    detail=f"volume={vol:.0f}, median={med:.0f}, "
                           f"ratio={vol / med:.1f}x",
                ))
                mask.loc[dt, sym] = True

        return outliers, mask

    def _check_stale_prices(
        self,
        prices: pd.DataFrame,
        symbols: list[str],
    ) -> tuple[list[Outlier], pd.DataFrame]:
        """Flag sequences of N+ identical closing prices."""
        cfg = self._config
        outliers: list[Outlier] = []
        mask = pd.DataFrame(False, index=prices.index, columns=symbols)

        for sym in symbols:
            series = prices[sym].values
            n = len(series)
            i = 0
            while i < n:
                # Find run of identical values
                j = i + 1
                while j < n and series[j] == series[i] and np.isfinite(series[i]):
                    j += 1
                run_len = j - i
                if run_len >= cfg.stale_price_days:
                    for k in range(i, j):
                        dt = prices.index[k]
                        if k == i:
                            outliers.append(Outlier(
                                symbol=sym,
                                date=dt,
                                outlier_type=OutlierType.STALE_PRICE,
                                value=float(series[i]),
                                threshold=float(cfg.stale_price_days),
                                detail=f"price={series[i]:.4f} unchanged for "
                                       f"{run_len} consecutive days",
                            ))
                        mask.iloc[k, mask.columns.get_loc(sym)] = True
                i = j

        return outliers, mask

    def _check_cross_sectional(
        self,
        returns: pd.DataFrame,
        symbols: list[str],
    ) -> tuple[list[Outlier], pd.DataFrame]:
        """Flag returns that are extreme relative to the cross-section."""
        cfg = self._config
        outliers: list[Outlier] = []
        mask = pd.DataFrame(False, index=returns.index, columns=symbols)

        for dt in returns.index[cfg.min_history:]:
            row = returns.loc[dt, symbols]
            valid = row.dropna()
            if len(valid) < 3:
                continue

            mu = valid.mean()
            sigma = valid.std()
            if sigma < 1e-10:
                continue

            z_scores = (valid - mu) / sigma
            for sym in z_scores.index:
                z = float(z_scores[sym])
                if abs(z) > cfg.cross_sectional_z:
                    outliers.append(Outlier(
                        symbol=sym,
                        date=dt,
                        outlier_type=OutlierType.CROSS_SECTIONAL,
                        value=float(valid[sym]),
                        threshold=cfg.cross_sectional_z,
                        detail=f"return={valid[sym]:.4%}, "
                               f"cross-sectional z={z:.2f} "
                               f"(universe μ={mu:.4%}, σ={sigma:.4%})",
                    ))
                    mask.loc[dt, sym] = True

        return outliers, mask
