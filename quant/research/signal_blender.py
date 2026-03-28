"""Regime-conditional signal blending.

Combines multiple alpha signals into a composite score with weights that
adapt to the current market regime.  In trending markets, momentum signals
receive higher weight; in mean-reverting regimes, reversal signals dominate.

Blending methods:

  * **Equal weight** — simple average of all signals.
  * **IC-weighted** — weight each signal by its recent information coefficient.
  * **Regime-conditional** — maintain separate weight vectors per regime and
    select based on the current regime state.

Usage::

    from quant.research.signal_blender import (
        SignalBlender,
        BlenderConfig,
        RegimeWeights,
    )

    blender = SignalBlender(BlenderConfig(
        method="regime_conditional",
        regime_weights={
            "high_vol": RegimeWeights({"momentum": 0.2, "reversal": 0.5, "quality": 0.3}),
            "low_vol":  RegimeWeights({"momentum": 0.5, "reversal": 0.2, "quality": 0.3}),
        },
    ))
    composite = blender.blend(signal_scores, regime="high_vol")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class BlendMethod(Enum):
    """Signal blending method."""

    EQUAL_WEIGHT = "equal_weight"
    IC_WEIGHTED = "ic_weighted"
    REGIME_CONDITIONAL = "regime_conditional"


@dataclass
class RegimeWeights:
    """Signal weights for a specific regime.

    Attributes:
        weights: ``{signal_name: weight}``.  Weights are normalised to sum to 1.
    """

    weights: dict[str, float]

    def normalised(self) -> dict[str, float]:
        total = sum(abs(w) for w in self.weights.values())
        if total < 1e-10:
            n = len(self.weights)
            return dict.fromkeys(self.weights, 1.0 / n) if n > 0 else {}
        return {k: w / total for k, w in self.weights.items()}


@dataclass
class BlenderConfig:
    """Configuration for signal blending.

    Attributes:
        method:           Blending method.
        regime_weights:   ``{regime_name: RegimeWeights}`` for regime-conditional
                          blending.  Ignored for other methods.
        ic_lookback:      Rolling window for IC estimation (IC-weighted method).
        ic_min_periods:   Minimum observations for valid IC estimate.
        default_regime:   Fallback regime if current regime not in weights map.
        z_score_signals:  If True, z-score each signal cross-sectionally before
                          blending.
    """

    method: BlendMethod = BlendMethod.EQUAL_WEIGHT
    regime_weights: dict[str, RegimeWeights] = field(default_factory=dict)
    ic_lookback: int = 63
    ic_min_periods: int = 20
    default_regime: str = "normal"
    z_score_signals: bool = True


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SignalWeight:
    """Weight assigned to a single signal in the blend."""

    signal_name: str
    weight: float
    recent_ic: float  # Recent IC (used for IC-weighted)


@dataclass
class BlendResult:
    """Output of signal blending.

    Attributes:
        composite_scores:  Combined signal scores (DatetimeIndex × symbols).
        signal_weights:    Per-signal weight used in the blend.
        method:            Blending method used.
        regime:            Current regime (if regime-conditional).
        n_signals:         Number of signals blended.
        n_dates:           Number of dates in output.
    """

    composite_scores: pd.DataFrame = field(repr=False)
    signal_weights: list[SignalWeight] = field(default_factory=list)
    method: BlendMethod = BlendMethod.EQUAL_WEIGHT
    regime: str | None = None
    n_signals: int = 0
    n_dates: int = 0

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"Signal Blend ({self.n_signals} signals, {self.n_dates} dates)",
            "=" * 55,
            "",
            f"Method  : {self.method.value}",
        ]
        if self.regime:
            lines.append(f"Regime  : {self.regime}")
        lines.extend([
            "",
            f"{'Signal':<20s} {'Weight':>8s} {'IC':>8s}",
            "-" * 40,
        ])
        for sw in sorted(self.signal_weights, key=lambda x: -abs(x.weight)):
            lines.append(
                f"{sw.signal_name:<20s} {sw.weight:>+8.4f} {sw.recent_ic:>+8.4f}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Blender
# ---------------------------------------------------------------------------


class SignalBlender:
    """Regime-conditional signal blender.

    Args:
        config: Blending configuration.
    """

    def __init__(self, config: BlenderConfig | None = None) -> None:
        self._config = config or BlenderConfig()

    @property
    def config(self) -> BlenderConfig:
        return self._config

    def blend(
        self,
        signals: dict[str, pd.DataFrame],
        forward_returns: pd.DataFrame | None = None,
        regime: str | None = None,
    ) -> BlendResult:
        """Blend multiple signals into a composite score.

        Args:
            signals:          ``{signal_name: DataFrame}`` where each DataFrame
                              has shape (DatetimeIndex × symbols).
            forward_returns:  Asset forward returns for IC estimation.
                              Required for ``IC_WEIGHTED`` method.
            regime:           Current regime label for ``REGIME_CONDITIONAL``.

        Returns:
            :class:`BlendResult` with composite scores and weight audit.

        Raises:
            ValueError: If no signals provided or regime weights not configured.
        """
        cfg = self._config

        if not signals:
            raise ValueError("Need at least 1 signal")

        signal_names = sorted(signals.keys())
        n_signals = len(signal_names)

        # Align dates and symbols across all signals
        common_dates = signals[signal_names[0]].index
        common_symbols = set(signals[signal_names[0]].columns)
        for name in signal_names[1:]:
            common_dates = common_dates.intersection(signals[name].index)
            common_symbols &= set(signals[name].columns)
        common_symbols_list = sorted(common_symbols)

        if len(common_dates) == 0 or len(common_symbols_list) == 0:
            raise ValueError("No common dates or symbols across signals")

        # Build aligned signal DataFrames
        aligned: dict[str, pd.DataFrame] = {}
        for name in signal_names:
            df = signals[name].loc[common_dates, common_symbols_list].copy()
            if cfg.z_score_signals:
                df = self._z_score_cross_section(df)
            aligned[name] = df

        # Compute weights based on method
        if cfg.method == BlendMethod.EQUAL_WEIGHT:
            raw_weights = dict.fromkeys(signal_names, 1.0 / n_signals)
            ics = dict.fromkeys(signal_names, 0.0)

        elif cfg.method == BlendMethod.IC_WEIGHTED:
            if forward_returns is None:
                raise ValueError("forward_returns required for IC_WEIGHTED method")
            raw_weights, ics = self._ic_weights(
                aligned, forward_returns, common_dates, common_symbols_list, cfg,
            )

        elif cfg.method == BlendMethod.REGIME_CONDITIONAL:
            effective_regime = regime or cfg.default_regime
            if effective_regime not in cfg.regime_weights:
                if cfg.default_regime in cfg.regime_weights:
                    effective_regime = cfg.default_regime
                else:
                    # Fall back to equal weight
                    raw_weights = dict.fromkeys(signal_names, 1.0 / n_signals)
                    ics = dict.fromkeys(signal_names, 0.0)
                    effective_regime = regime or "unknown"
                    return self._build_result(
                        aligned, raw_weights, ics, signal_names,
                        common_dates, cfg.method, effective_regime,
                    )

            rw = cfg.regime_weights[effective_regime].normalised()
            # Fill missing signals with 0
            raw_weights = {name: rw.get(name, 0.0) for name in signal_names}
            ics = dict.fromkeys(signal_names, 0.0)
            regime = effective_regime
        else:
            raw_weights = dict.fromkeys(signal_names, 1.0 / n_signals)
            ics = dict.fromkeys(signal_names, 0.0)

        return self._build_result(
            aligned, raw_weights, ics, signal_names,
            common_dates, cfg.method, regime,
        )

    def _build_result(
        self,
        aligned: dict[str, pd.DataFrame],
        weights: dict[str, float],
        ics: dict[str, float],
        signal_names: list[str],
        common_dates: pd.DatetimeIndex,
        method: BlendMethod,
        regime: str | None,
    ) -> BlendResult:
        """Compute composite scores and build result."""
        # Weighted sum
        first = aligned[signal_names[0]]
        composite = pd.DataFrame(0.0, index=first.index, columns=first.columns)
        for name in signal_names:
            composite += aligned[name] * weights.get(name, 0.0)

        signal_weight_list = [
            SignalWeight(
                signal_name=name,
                weight=weights.get(name, 0.0),
                recent_ic=ics.get(name, 0.0),
            )
            for name in signal_names
        ]

        return BlendResult(
            composite_scores=composite,
            signal_weights=signal_weight_list,
            method=method,
            regime=regime,
            n_signals=len(signal_names),
            n_dates=len(common_dates),
        )

    @staticmethod
    def _z_score_cross_section(df: pd.DataFrame) -> pd.DataFrame:
        """Z-score each row (cross-section) independently."""
        mean = df.mean(axis=1)
        std = df.std(axis=1)
        std = std.replace(0.0, 1.0)
        return df.sub(mean, axis=0).div(std, axis=0)

    @staticmethod
    def _ic_weights(
        aligned: dict[str, pd.DataFrame],
        forward_returns: pd.DataFrame,
        common_dates: pd.DatetimeIndex,
        common_symbols: list[str],
        cfg: BlenderConfig,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Compute IC-based weights from recent signal performance."""
        fwd = forward_returns.reindex(index=common_dates, columns=common_symbols).fillna(0.0)

        ics: dict[str, float] = {}
        for name, sig_df in aligned.items():
            # Compute rolling IC: rank corr per date
            ic_values: list[float] = []
            lookback_dates = common_dates[-cfg.ic_lookback:]
            for date in lookback_dates:
                if date not in sig_df.index or date not in fwd.index:
                    continue
                s = sig_df.loc[date].dropna()
                f = fwd.loc[date].dropna()
                common_syms = s.index.intersection(f.index)
                if len(common_syms) < 5:
                    continue
                ic = float(s[common_syms].rank().corr(f[common_syms].rank()))
                if not np.isnan(ic):
                    ic_values.append(ic)

            ics[name] = float(np.mean(ic_values)) if len(ic_values) >= cfg.ic_min_periods else 0.0

        # Weight by positive IC; negative IC gets 0
        positive_ics = {k: max(v, 0.0) for k, v in ics.items()}
        total = sum(positive_ics.values())
        if total < 1e-10:
            n = len(aligned)
            weights = dict.fromkeys(aligned, 1.0 / n)
        else:
            weights = {k: v / total for k, v in positive_ics.items()}

        return weights, ics
