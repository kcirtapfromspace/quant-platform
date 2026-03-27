"""Cross-sectional ranking signals and scoring utilities.

Provides tools for ranking assets *relative to their peers* rather than
in isolation.  This is the foundation for quantile-based long-short
portfolio construction (e.g. "buy top decile momentum, short bottom
decile").

Key components:

  * **Scoring utilities** — ``percentile_rank``, ``z_score_normalize``,
    ``winsorize`` for transforming raw per-asset scores into
    cross-sectionally comparable values.
  * **QuantileSelector** — selects long / short legs from a scored
    universe based on configurable quantile cutoffs.
  * **CrossSectionalSignal** — abstract base class for signals that
    score the entire universe simultaneously (contrast with
    ``BaseSignal`` which scores one symbol at a time).
  * **Concrete signals** — ``CrossSectionalMomentum``,
    ``CrossSectionalMeanReversion``, ``CrossSectionalVolatility``
    implementing classic cross-sectional factors.

All computations are pure Python with ``pandas`` for time-series
convenience, matching the existing signal conventions.

Usage::

    from quant.signals.cross_sectional import (
        CrossSectionalMomentum,
        QuantileSelector,
    )

    signal = CrossSectionalMomentum(lookback=63)
    scored = signal.score_universe(universe_features, timestamp)

    selector = QuantileSelector(long_quantile=0.2, short_quantile=0.2)
    longs, shorts = selector.select(scored)
"""
from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from quant.signals.base import SignalOutput

# ---------------------------------------------------------------------------
# Scoring utilities
# ---------------------------------------------------------------------------


def percentile_rank(scores: dict[str, float]) -> dict[str, float]:
    """Map raw scores to percentile ranks in [0, 1].

    Uses average ranking for ties.  Higher raw score → higher percentile.
    With *N* assets, the k-th ranked asset (1-based) receives percentile
    ``(k - 0.5) / N``.

    Args:
        scores: ``{symbol: raw_score}``.

    Returns:
        ``{symbol: percentile}`` in [0, 1].
    """
    if not scores:
        return {}

    items = sorted(scores.items(), key=lambda kv: kv[1])
    n = len(items)

    # Handle ties via average rank
    ranks: dict[str, float] = {}
    i = 0
    while i < n:
        j = i
        while j < n and items[j][1] == items[i][1]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1  # 1-based average
        for k in range(i, j):
            ranks[items[k][0]] = (avg_rank - 0.5) / n
        i = j

    return ranks


def z_score_normalize(scores: dict[str, float]) -> dict[str, float]:
    """Standardise scores to zero mean and unit variance across the universe.

    Args:
        scores: ``{symbol: raw_score}``.

    Returns:
        ``{symbol: z_score}``.  Returns all zeros if variance is zero.
    """
    if len(scores) < 2:
        return dict.fromkeys(scores, 0.0)

    vals = list(scores.values())
    mu = sum(vals) / len(vals)
    var = sum((v - mu) ** 2 for v in vals) / len(vals)
    sigma = math.sqrt(var) if var > 0 else 0.0

    if sigma < 1e-12:
        return dict.fromkeys(scores, 0.0)

    return {sym: (v - mu) / sigma for sym, v in scores.items()}


def winsorize(
    scores: dict[str, float],
    lower: float = 0.05,
    upper: float = 0.95,
) -> dict[str, float]:
    """Clip extreme values at the given percentile thresholds.

    Values below the *lower* percentile are set to that threshold;
    values above the *upper* percentile are set to that threshold.

    Args:
        scores: ``{symbol: value}``.
        lower: Lower percentile cutoff (default 5th percentile).
        upper: Upper percentile cutoff (default 95th percentile).

    Returns:
        ``{symbol: clipped_value}``.
    """
    if not scores:
        return {}

    vals = sorted(scores.values())
    n = len(vals)
    lo_idx = max(0, int(math.floor(lower * n)))
    hi_idx = min(n - 1, int(math.ceil(upper * n)) - 1)
    lo_val = vals[lo_idx]
    hi_val = vals[hi_idx]

    return {sym: max(lo_val, min(hi_val, v)) for sym, v in scores.items()}


# ---------------------------------------------------------------------------
# Quantile selector
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class QuantileSelection:
    """Result of quantile-based universe filtering.

    Attributes:
        long_symbols: Symbols selected for the long leg.
        short_symbols: Symbols selected for the short leg.
        long_scores: Raw scores for long-leg symbols.
        short_scores: Raw scores for short-leg symbols.
        all_ranks: Percentile ranks for the entire universe.
    """

    long_symbols: list[str]
    short_symbols: list[str]
    long_scores: dict[str, float]
    short_scores: dict[str, float]
    all_ranks: dict[str, float]


class QuantileSelector:
    """Select long/short legs by quantile rank.

    After scoring the universe, the top ``long_quantile`` fraction of
    assets become the long leg and the bottom ``short_quantile``
    fraction become the short leg.

    Args:
        long_quantile: Fraction of universe for the long leg
            (default 0.2 = top quintile).
        short_quantile: Fraction of universe for the short leg
            (default 0.2 = bottom quintile).
        min_assets: Minimum number of assets per leg.  If the universe
            is too small, legs may have fewer.
    """

    def __init__(
        self,
        long_quantile: float = 0.2,
        short_quantile: float = 0.2,
        min_assets: int = 1,
    ) -> None:
        if not 0 < long_quantile <= 1:
            raise ValueError("long_quantile must be in (0, 1]")
        if not 0 < short_quantile <= 1:
            raise ValueError("short_quantile must be in (0, 1]")
        if long_quantile + short_quantile > 1:
            raise ValueError("long_quantile + short_quantile must be <= 1")
        self._long_q = long_quantile
        self._short_q = short_quantile
        self._min_assets = min_assets

    def select(self, scores: dict[str, float]) -> QuantileSelection:
        """Partition the universe into long, short, and neutral buckets.

        Args:
            scores: ``{symbol: raw_score}`` — higher is more bullish.

        Returns:
            :class:`QuantileSelection` with long and short legs.
        """
        if not scores:
            return QuantileSelection([], [], {}, {}, {})

        ranks = percentile_rank(scores)
        sorted_syms = sorted(scores.keys(), key=lambda s: scores[s])

        n = len(sorted_syms)
        n_short = max(self._min_assets, max(1, int(math.floor(n * self._short_q))))
        n_long = max(self._min_assets, max(1, int(math.floor(n * self._long_q))))

        # Clamp to available
        n_short = min(n_short, n)
        n_long = min(n_long, n - n_short)

        short_syms = sorted_syms[:n_short]
        long_syms = sorted_syms[-n_long:] if n_long > 0 else []

        return QuantileSelection(
            long_symbols=long_syms,
            short_symbols=short_syms,
            long_scores={s: scores[s] for s in long_syms},
            short_scores={s: scores[s] for s in short_syms},
            all_ranks=ranks,
        )


# ---------------------------------------------------------------------------
# Abstract cross-sectional signal
# ---------------------------------------------------------------------------


class CrossSectionalSignal(abc.ABC):
    """Abstract base for signals that score the entire universe at once.

    Unlike :class:`BaseSignal` which produces a score per symbol in
    isolation, a cross-sectional signal ranks all assets relative to
    each other.  This enables long-short quantile construction.

    Subclasses implement :meth:`_raw_scores` to produce unnormalised
    per-symbol scores from features.  The base class handles
    normalisation, ranking, and conversion to ``SignalOutput``.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique signal identifier."""

    @property
    @abc.abstractmethod
    def required_features(self) -> list[str]:
        """Feature names this signal depends on."""

    @abc.abstractmethod
    def _raw_scores(
        self,
        universe_features: dict[str, dict[str, pd.Series]],
        timestamp: datetime,
    ) -> dict[str, float]:
        """Compute unnormalised raw scores for every symbol.

        Args:
            universe_features: ``{symbol: {feature_name: pd.Series}}``.
            timestamp: Current bar timestamp.

        Returns:
            ``{symbol: raw_score}`` — higher means more bullish.
            Symbols for which the signal cannot be computed should be
            omitted.
        """

    def score_universe(
        self,
        universe_features: dict[str, dict[str, pd.Series]],
        timestamp: datetime,
        *,
        normalize: str = "percentile",
    ) -> dict[str, SignalOutput]:
        """Score every symbol cross-sectionally.

        Args:
            universe_features: ``{symbol: {feature_name: pd.Series}}``.
            timestamp: Current bar timestamp.
            normalize: Normalisation method — ``"percentile"`` (default),
                ``"zscore"``, or ``"raw"``.

        Returns:
            ``{symbol: SignalOutput}`` with cross-sectionally normalised
            scores in [-1, 1].
        """
        raw = self._raw_scores(universe_features, timestamp)
        if not raw:
            return {}

        if normalize == "percentile":
            normed = percentile_rank(raw)
            # Map [0, 1] percentile to [-1, 1] score
            normed = {sym: 2.0 * v - 1.0 for sym, v in normed.items()}
        elif normalize == "zscore":
            normed = z_score_normalize(raw)
            # Clamp z-scores to [-1, 1] using tanh-like scaling
            normed = {sym: max(-1.0, min(1.0, v / 3.0)) for sym, v in normed.items()}
        else:
            normed = dict(raw)

        n_scored = len(raw)
        confidence = min(1.0, n_scored / 10.0)  # More assets → higher confidence

        outputs: dict[str, SignalOutput] = {}
        for sym, score in normed.items():
            clamped = max(-1.0, min(1.0, score))
            outputs[sym] = SignalOutput(
                symbol=sym,
                timestamp=timestamp,
                score=clamped,
                confidence=confidence,
                target_position=max(-1.0, min(1.0, clamped * confidence)),
                metadata={
                    "signal_name": self.name,
                    "raw_score": raw[sym],
                    "normalize": normalize,
                    "universe_size": n_scored,
                },
            )

        return outputs


# ---------------------------------------------------------------------------
# Cross-sectional momentum
# ---------------------------------------------------------------------------


class CrossSectionalMomentum(CrossSectionalSignal):
    """Rank assets by total return over a lookback window.

    Classic Jegadeesh–Titman momentum: buy recent winners, sell recent
    losers.  Assets are ranked by cumulative return; top-ranked receive
    positive scores, bottom-ranked receive negative scores.

    Args:
        lookback: Number of trading days for return computation
            (default 63 ≈ 3 months).
        skip_recent: Number of recent days to skip (short-term reversal
            offset, default 5 ≈ 1 week).
    """

    def __init__(self, lookback: int = 63, skip_recent: int = 5) -> None:
        self._lookback = lookback
        self._skip_recent = skip_recent

    @property
    def name(self) -> str:
        return "xs_momentum"

    @property
    def required_features(self) -> list[str]:
        return ["returns"]

    def _raw_scores(
        self,
        universe_features: dict[str, dict[str, pd.Series]],
        timestamp: datetime,
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        min_len = self._lookback + self._skip_recent

        for sym, features in universe_features.items():
            returns = features.get("returns")
            if returns is None:
                continue
            returns = returns.dropna()
            if len(returns) < min_len:
                continue

            # Cumulative return over [lookback+skip, skip] window
            if self._skip_recent > 0:
                window = returns.iloc[-(self._lookback + self._skip_recent) : -self._skip_recent]
            else:
                window = returns.iloc[-self._lookback :]

            cum_return = float((1 + window).prod()) - 1.0
            scores[sym] = cum_return

        return scores


# ---------------------------------------------------------------------------
# Cross-sectional mean reversion
# ---------------------------------------------------------------------------


class CrossSectionalMeanReversion(CrossSectionalSignal):
    """Rank assets by deviation from their rolling mean — contrarian signal.

    Assets that have fallen the most relative to their rolling average
    receive the highest (most bullish) scores.  This is the cross-
    sectional analogue of a Bollinger Band reversion strategy.

    Args:
        lookback: Rolling window for mean computation (default 21).
        vol_lookback: Rolling window for volatility normalisation
            (default 63).
    """

    def __init__(self, lookback: int = 21, vol_lookback: int = 63) -> None:
        self._lookback = lookback
        self._vol_lookback = vol_lookback

    @property
    def name(self) -> str:
        return "xs_mean_reversion"

    @property
    def required_features(self) -> list[str]:
        return ["returns"]

    def _raw_scores(
        self,
        universe_features: dict[str, dict[str, pd.Series]],
        timestamp: datetime,
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        min_len = max(self._lookback, self._vol_lookback)

        for sym, features in universe_features.items():
            returns = features.get("returns")
            if returns is None:
                continue
            returns = returns.dropna()
            if len(returns) < min_len:
                continue

            # Reconstruct cumulative price from returns
            cum = (1 + returns).cumprod()

            current = float(cum.iloc[-1])
            rolling_mean = float(cum.iloc[-self._lookback :].mean())

            # Volatility for normalisation
            vol = float(returns.iloc[-self._vol_lookback :].std())
            if vol < 1e-12:
                continue

            # Deviation normalised by vol: negative = undervalued → bullish
            # Flip sign: more negative deviation → higher score (contrarian)
            deviation = (current - rolling_mean) / (rolling_mean * vol) if rolling_mean != 0 else 0.0
            scores[sym] = -deviation  # contrarian: buy losers, sell winners

        return scores


# ---------------------------------------------------------------------------
# Cross-sectional volatility (low-vol anomaly)
# ---------------------------------------------------------------------------


class CrossSectionalVolatility(CrossSectionalSignal):
    """Rank assets by realised volatility — low-vol anomaly factor.

    Lower-volatility assets receive higher scores.  The low-volatility
    anomaly is one of the most robust cross-sectional factors: stocks
    with lower realised volatility tend to deliver higher risk-adjusted
    returns.

    Args:
        lookback: Rolling window for volatility computation (default 63).
    """

    def __init__(self, lookback: int = 63) -> None:
        self._lookback = lookback

    @property
    def name(self) -> str:
        return "xs_volatility"

    @property
    def required_features(self) -> list[str]:
        return ["returns"]

    def _raw_scores(
        self,
        universe_features: dict[str, dict[str, pd.Series]],
        timestamp: datetime,
    ) -> dict[str, float]:
        scores: dict[str, float] = {}

        for sym, features in universe_features.items():
            returns = features.get("returns")
            if returns is None:
                continue
            returns = returns.dropna()
            if len(returns) < self._lookback:
                continue

            vol = float(returns.iloc[-self._lookback :].std()) * math.sqrt(252)
            if vol < 1e-12:
                continue

            # Negative vol → higher score (low-vol anomaly: favor low vol)
            scores[sym] = -vol

        return scores


# ---------------------------------------------------------------------------
# Scores-to-weights converter
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class QuantileWeights:
    """Target portfolio weights from quantile-based construction.

    Attributes:
        weights: ``{symbol: weight}`` — positive for longs, negative
            for shorts.  Sums to approximately 0 for a dollar-neutral
            portfolio, or to 1 for long-only.
        long_symbols: Symbols in the long leg.
        short_symbols: Symbols in the short leg.
        metadata: Diagnostics.
    """

    weights: dict[str, float]
    long_symbols: list[str]
    short_symbols: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


def scores_to_quantile_weights(
    scores: dict[str, float],
    *,
    long_quantile: float = 0.2,
    short_quantile: float = 0.2,
    dollar_neutral: bool = True,
    equal_weight: bool = True,
) -> QuantileWeights:
    """Convert cross-sectional scores to quantile-based portfolio weights.

    Selects the top ``long_quantile`` fraction as longs and bottom
    ``short_quantile`` fraction as shorts.

    Args:
        scores: ``{symbol: score}`` — higher is more bullish.
        long_quantile: Fraction for long leg (default top 20%).
        short_quantile: Fraction for short leg (default bottom 20%).
        dollar_neutral: If True, long and short legs have equal gross
            weight (default True).
        equal_weight: If True, equal-weight within each leg.  If False,
            weight proportional to rank distance from cutoff.

    Returns:
        :class:`QuantileWeights` with per-symbol weights.
    """
    selector = QuantileSelector(long_quantile, short_quantile)
    selection = selector.select(scores)

    weights: dict[str, float] = {}

    if not selection.long_symbols and not selection.short_symbols:
        return QuantileWeights({}, [], [], {"method": "quantile"})

    if equal_weight:
        # Equal weight within each leg
        n_long = len(selection.long_symbols)
        n_short = len(selection.short_symbols)

        if dollar_neutral:
            long_w = 0.5 / n_long if n_long > 0 else 0.0
            short_w = -0.5 / n_short if n_short > 0 else 0.0
        else:
            long_w = 1.0 / n_long if n_long > 0 else 0.0
            short_w = 0.0

        for sym in selection.long_symbols:
            weights[sym] = long_w
        for sym in selection.short_symbols:
            weights[sym] = short_w
    else:
        # Score-weighted within each leg
        if selection.long_scores:
            long_total = sum(abs(v) for v in selection.long_scores.values())
            scale = 0.5 if dollar_neutral else 1.0
            for sym, s in selection.long_scores.items():
                weights[sym] = (abs(s) / long_total * scale) if long_total > 0 else 0.0

        if selection.short_scores and dollar_neutral:
            short_total = sum(abs(v) for v in selection.short_scores.values())
            for sym, s in selection.short_scores.items():
                weights[sym] = -(abs(s) / short_total * 0.5) if short_total > 0 else 0.0

    return QuantileWeights(
        weights=weights,
        long_symbols=selection.long_symbols,
        short_symbols=selection.short_symbols,
        metadata={
            "method": "quantile",
            "long_quantile": long_quantile,
            "short_quantile": short_quantile,
            "dollar_neutral": dollar_neutral,
            "equal_weight": equal_weight,
            "universe_size": len(scores),
        },
    )
