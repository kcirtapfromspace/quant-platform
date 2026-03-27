"""Market regime detection and adaptive capital allocation.

Identifies the current market regime along three orthogonal dimensions:

  1. **Volatility regime**: is realised volatility elevated or suppressed
     relative to its long-run level?
  2. **Trend regime**: are returns serially correlated (trending) or
     mean-reverting?
  3. **Correlation regime**: are cross-asset correlations elevated
     (systemic risk) or dispersed (stock-picking environment)?

These dimensions are combined into a composite :class:`MarketRegime`
label.  The :class:`RegimeWeightAdapter` then maps the detected regime
to capital-allocation adjustments that tilt toward strategies suited to
the environment (e.g. overweight trend-following in trending markets).

All computations are pure Python — no scipy, no external dependencies.

Usage::

    from quant.signals.regime import RegimeDetector, RegimeWeightAdapter

    detector = RegimeDetector()
    state = detector.detect(daily_returns_df)
    print(state.regime, state.confidence)

    adapter = RegimeWeightAdapter()
    adjusted = adapter.adapt(
        regime=state,
        base_weights={"momentum": 0.40, "mean_rev": 0.35, "trend": 0.25},
        strategy_types={"momentum": "momentum", "mean_rev": "mean_reversion", "trend": "trend"},
    )
"""
from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Regime types
# ---------------------------------------------------------------------------


class MarketRegime(enum.Enum):
    """Composite market regime label."""

    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    CRISIS = "crisis"
    NORMAL = "normal"


class VolRegime(enum.Enum):
    HIGH = "high"
    LOW = "low"
    NORMAL = "normal"


class TrendRegime(enum.Enum):
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    RANDOM = "random"


class CorrelationRegime(enum.Enum):
    HIGH = "high"
    LOW = "low"
    NORMAL = "normal"


# ---------------------------------------------------------------------------
# Regime state
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RegimeState:
    """Detected market regime with supporting diagnostics.

    Attributes:
        regime:       Composite regime label.
        confidence:   Overall confidence in the regime call (0–1).
        vol_regime:   Volatility sub-regime.
        trend_regime: Trend persistence sub-regime.
        corr_regime:  Correlation sub-regime.
        metrics:      Supporting indicator values.
    """

    regime: MarketRegime
    confidence: float
    vol_regime: VolRegime
    trend_regime: TrendRegime
    corr_regime: CorrelationRegime
    metrics: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Detector configuration
# ---------------------------------------------------------------------------


@dataclass
class RegimeConfig:
    """Configuration for the regime detector.

    Attributes:
        vol_short_window:  Rolling window (days) for recent volatility.
        vol_long_window:   Rolling window (days) for long-term volatility.
        vol_high_threshold: Ratio above which vol is classified as HIGH.
        vol_low_threshold:  Ratio below which vol is classified as LOW.
        trend_window:      Window for autocorrelation / trend persistence.
        trend_threshold:   Autocorrelation above which regime is TRENDING.
        mr_threshold:      Negative autocorrelation below which regime is
                           MEAN_REVERTING.
        corr_window:       Rolling window for average pairwise correlation.
        corr_high_threshold: Average correlation above which regime is HIGH.
        corr_low_threshold:  Average correlation below which regime is LOW.
        crisis_vol_threshold: Vol ratio above which regime is CRISIS.
    """

    vol_short_window: int = 21
    vol_long_window: int = 252
    vol_high_threshold: float = 1.25
    vol_low_threshold: float = 0.75
    trend_window: int = 63
    trend_threshold: float = 0.10
    mr_threshold: float = -0.10
    corr_window: int = 63
    corr_high_threshold: float = 0.60
    corr_low_threshold: float = 0.25
    crisis_vol_threshold: float = 2.0


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class RegimeDetector:
    """Detect the current market regime from historical returns.

    Uses three orthogonal indicators — volatility, trend, and
    correlation — to classify the market into one of six composite
    regimes.

    Args:
        config: Detector configuration.  Defaults are calibrated for
            US equity daily returns.
    """

    def __init__(self, config: RegimeConfig | None = None) -> None:
        self._config = config or RegimeConfig()

    @property
    def config(self) -> RegimeConfig:
        return self._config

    def detect(self, returns: list[list[float]] | None = None, *, returns_1d: list[float] | None = None, returns_2d: list[list[float]] | None = None) -> RegimeState:
        """Detect the current market regime.

        Accepts either a 2-D list of returns (multiple assets) or a 1-D
        list (single aggregate return series).

        Args:
            returns: 2-D list ``[n_days][n_assets]``.  Preferred form.
            returns_1d: 1-D list of aggregate daily returns (e.g. index).
                Used for vol and trend regimes.  Correlation requires 2-D.
            returns_2d: Alias for *returns* (for clarity).

        Returns:
            :class:`RegimeState` with composite label and sub-regimes.
        """
        # Resolve input
        data_2d = returns_2d or returns
        data_1d = returns_1d

        if data_2d is not None:
            n_days = len(data_2d)
            n_assets = len(data_2d[0]) if n_days > 0 else 0
            # Build aggregate (equal-weight) return series
            if data_1d is None:
                data_1d = [
                    sum(row) / len(row) if len(row) > 0 else 0.0
                    for row in data_2d
                ]
        elif data_1d is not None:
            n_days = len(data_1d)
            n_assets = 1
        else:
            raise ValueError("Must provide returns, returns_1d, or returns_2d")

        if n_days < max(self._config.vol_short_window, 10):
            return RegimeState(
                regime=MarketRegime.NORMAL,
                confidence=0.0,
                vol_regime=VolRegime.NORMAL,
                trend_regime=TrendRegime.RANDOM,
                corr_regime=CorrelationRegime.NORMAL,
            )

        # ── Sub-regime detection ────────────────────────────────────
        vol_regime, vol_ratio = self._detect_vol(data_1d)
        trend_regime, autocorr = self._detect_trend(data_1d)

        if data_2d is not None and n_assets >= 2:
            corr_regime, avg_corr = self._detect_correlation(data_2d)
        else:
            corr_regime = CorrelationRegime.NORMAL
            avg_corr = 0.0

        # ── Composite regime ────────────────────────────────────────
        regime, confidence = self._composite(
            vol_regime, vol_ratio, trend_regime, autocorr, corr_regime, avg_corr
        )

        return RegimeState(
            regime=regime,
            confidence=confidence,
            vol_regime=vol_regime,
            trend_regime=trend_regime,
            corr_regime=corr_regime,
            metrics={
                "vol_ratio": vol_ratio,
                "autocorrelation": autocorr,
                "avg_correlation": avg_corr,
            },
        )

    # ── Volatility regime ─────────────────────────────────────────

    def _detect_vol(self, returns: list[float]) -> tuple[VolRegime, float]:
        """Compare recent volatility to long-term volatility."""
        cfg = self._config
        n = len(returns)
        short_n = min(cfg.vol_short_window, n)
        long_n = min(cfg.vol_long_window, n)

        short_vol = _std(returns[-short_n:])
        long_vol = _std(returns[-long_n:])

        if long_vol < 1e-12:
            return VolRegime.NORMAL, 1.0

        ratio = short_vol / long_vol

        if ratio >= cfg.crisis_vol_threshold:
            return VolRegime.HIGH, ratio
        if ratio >= cfg.vol_high_threshold:
            return VolRegime.HIGH, ratio
        if ratio <= cfg.vol_low_threshold:
            return VolRegime.LOW, ratio
        return VolRegime.NORMAL, ratio

    # ── Trend regime ──────────────────────────────────────────────

    def _detect_trend(self, returns: list[float]) -> tuple[TrendRegime, float]:
        """Estimate trend persistence via lag-1 autocorrelation."""
        cfg = self._config
        n = len(returns)
        window = min(cfg.trend_window, n - 1)
        if window < 5:
            return TrendRegime.RANDOM, 0.0

        recent = returns[-window:]
        ac = _autocorrelation(recent, lag=1)

        if ac >= cfg.trend_threshold:
            return TrendRegime.TRENDING, ac
        if ac <= cfg.mr_threshold:
            return TrendRegime.MEAN_REVERTING, ac
        return TrendRegime.RANDOM, ac

    # ── Correlation regime ────────────────────────────────────────

    def _detect_correlation(
        self, returns_2d: list[list[float]]
    ) -> tuple[CorrelationRegime, float]:
        """Compute average pairwise correlation over recent window."""
        cfg = self._config
        n_days = len(returns_2d)
        window = min(cfg.corr_window, n_days)
        recent = returns_2d[-window:]

        n_assets = len(recent[0]) if recent else 0
        if n_assets < 2 or window < 5:
            return CorrelationRegime.NORMAL, 0.0

        # Extract per-asset return series
        asset_series: list[list[float]] = [
            [recent[t][a] for t in range(window)]
            for a in range(n_assets)
        ]

        # Average pairwise correlation
        pair_count = 0
        corr_sum = 0.0
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                c = _pearson(asset_series[i], asset_series[j])
                if math.isfinite(c):
                    corr_sum += c
                    pair_count += 1

        avg_corr = corr_sum / pair_count if pair_count > 0 else 0.0

        if avg_corr >= cfg.corr_high_threshold:
            return CorrelationRegime.HIGH, avg_corr
        if avg_corr <= cfg.corr_low_threshold:
            return CorrelationRegime.LOW, avg_corr
        return CorrelationRegime.NORMAL, avg_corr

    # ── Composite logic ───────────────────────────────────────────

    @staticmethod
    def _composite(
        vol_regime: VolRegime,
        vol_ratio: float,
        trend_regime: TrendRegime,
        autocorr: float,
        corr_regime: CorrelationRegime,
        avg_corr: float,
    ) -> tuple[MarketRegime, float]:
        """Combine sub-regimes into a composite label with confidence."""
        # Crisis: very high vol + high correlation
        if vol_ratio >= 2.0 and corr_regime == CorrelationRegime.HIGH:
            confidence = min(1.0, (vol_ratio - 1.5) / 1.5 * 0.5 + avg_corr * 0.5)
            return MarketRegime.CRISIS, confidence

        # Crisis: extremely high vol even without high correlation
        if vol_ratio >= 2.5:
            return MarketRegime.CRISIS, min(1.0, (vol_ratio - 2.0) / 1.0)

        # Trending
        if trend_regime == TrendRegime.TRENDING:
            confidence = min(1.0, abs(autocorr) / 0.3)
            return MarketRegime.TRENDING, confidence

        # Mean-reverting
        if trend_regime == TrendRegime.MEAN_REVERTING:
            confidence = min(1.0, abs(autocorr) / 0.3)
            return MarketRegime.MEAN_REVERTING, confidence

        # Risk-on: low vol + low correlation
        if vol_regime == VolRegime.LOW and corr_regime != CorrelationRegime.HIGH:
            return MarketRegime.RISK_ON, 0.6

        # Risk-off: high vol (but not crisis) + high correlation
        if vol_regime == VolRegime.HIGH and corr_regime == CorrelationRegime.HIGH:
            return MarketRegime.RISK_OFF, 0.6

        # Normal
        return MarketRegime.NORMAL, 0.3


# ---------------------------------------------------------------------------
# Adaptive weight allocation
# ---------------------------------------------------------------------------


# Strategy type → regime affinity
# Positive = strategy benefits from this regime
_REGIME_AFFINITY: dict[str, dict[MarketRegime, float]] = {
    "trend": {
        MarketRegime.TRENDING: 0.30,
        MarketRegime.MEAN_REVERTING: -0.30,
        MarketRegime.CRISIS: -0.20,
        MarketRegime.RISK_ON: 0.10,
        MarketRegime.RISK_OFF: -0.10,
        MarketRegime.NORMAL: 0.0,
    },
    "mean_reversion": {
        MarketRegime.TRENDING: -0.20,
        MarketRegime.MEAN_REVERTING: 0.30,
        MarketRegime.CRISIS: -0.30,
        MarketRegime.RISK_ON: 0.15,
        MarketRegime.RISK_OFF: -0.10,
        MarketRegime.NORMAL: 0.0,
    },
    "momentum": {
        MarketRegime.TRENDING: 0.20,
        MarketRegime.MEAN_REVERTING: -0.15,
        MarketRegime.CRISIS: -0.25,
        MarketRegime.RISK_ON: 0.15,
        MarketRegime.RISK_OFF: -0.15,
        MarketRegime.NORMAL: 0.0,
    },
    "volatility": {
        MarketRegime.TRENDING: 0.0,
        MarketRegime.MEAN_REVERTING: 0.10,
        MarketRegime.CRISIS: 0.20,
        MarketRegime.RISK_ON: -0.10,
        MarketRegime.RISK_OFF: 0.15,
        MarketRegime.NORMAL: 0.0,
    },
    "quality": {
        MarketRegime.TRENDING: 0.05,
        MarketRegime.MEAN_REVERTING: 0.05,
        MarketRegime.CRISIS: 0.25,
        MarketRegime.RISK_ON: -0.05,
        MarketRegime.RISK_OFF: 0.15,
        MarketRegime.NORMAL: 0.0,
    },
    "breakout": {
        MarketRegime.TRENDING: 0.25,
        MarketRegime.MEAN_REVERTING: -0.25,
        MarketRegime.CRISIS: -0.15,
        MarketRegime.RISK_ON: 0.10,
        MarketRegime.RISK_OFF: -0.10,
        MarketRegime.NORMAL: 0.0,
    },
}


class RegimeWeightAdapter:
    """Adjust strategy capital weights based on detected regime.

    Given base weights and a regime state, applies affinity-based tilts
    to overweight strategies suited to the current environment and
    underweight those that are not.

    The adapter guarantees:
      - All weights remain non-negative.
      - Weights sum to the same total as the base weights (scaled,
        not clipped to zero then un-normalised).

    Args:
        affinity_table: Custom ``{strategy_type: {regime: tilt}}`` table.
            Defaults to the built-in :data:`_REGIME_AFFINITY`.
        max_tilt: Maximum absolute tilt per strategy (caps extreme
            adjustments).
    """

    def __init__(
        self,
        affinity_table: dict[str, dict[MarketRegime, float]] | None = None,
        max_tilt: float = 0.30,
    ) -> None:
        self._affinity = affinity_table or _REGIME_AFFINITY
        self._max_tilt = max_tilt

    def adapt(
        self,
        regime: RegimeState,
        base_weights: dict[str, float],
        strategy_types: dict[str, str],
    ) -> dict[str, float]:
        """Compute regime-adjusted capital weights.

        Args:
            regime: Current regime state from :class:`RegimeDetector`.
            base_weights: ``{strategy_name: capital_weight}`` baseline
                allocation.
            strategy_types: ``{strategy_name: strategy_type}`` mapping
                each strategy to a type key in the affinity table
                (e.g. ``"trend"``, ``"momentum"``, ``"mean_reversion"``).

        Returns:
            ``{strategy_name: adjusted_weight}`` with the same sum as
            *base_weights* (or very close after clamping).
        """
        if not base_weights:
            return {}

        total_base = sum(base_weights.values())
        if total_base <= 0:
            return dict(base_weights)

        # Compute raw tilted weights
        raw: dict[str, float] = {}
        for name, base_w in base_weights.items():
            stype = strategy_types.get(name, "")
            affinity = self._affinity.get(stype, {})
            tilt = affinity.get(regime.regime, 0.0)

            # Scale tilt by regime confidence
            tilt *= regime.confidence

            # Cap the tilt
            tilt = max(-self._max_tilt, min(self._max_tilt, tilt))

            # Apply multiplicatively: weight * (1 + tilt)
            adjusted = base_w * (1 + tilt)
            raw[name] = max(0.0, adjusted)

        # Re-normalise to preserve total capital allocation
        raw_total = sum(raw.values())
        if raw_total <= 0:
            return dict(base_weights)

        scale = total_base / raw_total
        return {name: w * scale for name, w in raw.items()}


# ---------------------------------------------------------------------------
# Pure-Python math helpers (no numpy/scipy)
# ---------------------------------------------------------------------------


def _mean(xs: list[float]) -> float:
    n = len(xs)
    return sum(xs) / n if n > 0 else 0.0


def _std(xs: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mu = _mean(xs)
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    return math.sqrt(var) if var > 0 else 0.0


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation coefficient between two lists."""
    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0
    mx = _mean(xs[:n])
    my = _mean(ys[:n])
    cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / (n - 1)
    sx = _std(xs[:n])
    sy = _std(ys[:n])
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    return cov / (sx * sy)


def _autocorrelation(xs: list[float], lag: int = 1) -> float:
    """Lag-k autocorrelation of a list."""
    n = len(xs)
    if n < lag + 2:
        return 0.0
    return _pearson(xs[:-lag], xs[lag:])
