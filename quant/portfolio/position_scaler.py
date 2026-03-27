"""Conviction-weighted position scaling — transform raw alphas into risk-adjusted signals.

Bridges the gap between alpha combination and portfolio optimisation by
scaling raw alpha scores using conviction strength, per-asset volatility,
and Kelly-inspired sizing.  This layer answers: *given my alpha view and
its reliability, how aggressively should I bet on each position?*

Three scaling methods:

  * **Conviction**: scale alpha by signal confidence (``alpha × confidence``).
  * **Volatility-adjusted**: normalise so each asset contributes equal risk
    (``alpha / vol``), optionally targeting a portfolio volatility.
  * **Kelly**: size proportional to edge/variance — ``alpha × confidence / vol²``.

All methods preserve the sign and relative ordering of alphas while
adjusting magnitudes.  Output is a dict of scaled alphas suitable for
feeding directly into :class:`PortfolioEngine.construct`.

Usage::

    from quant.portfolio.position_scaler import PositionScaler, ScalingConfig

    scaler = PositionScaler(ScalingConfig(method="vol_adjusted"))
    scaled = scaler.scale(alpha_scores, returns_history)
    # Feed scaled alphas into portfolio construction
"""
from __future__ import annotations

import enum
import math
from dataclasses import dataclass

import pandas as pd

from quant.portfolio.alpha import AlphaScore

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ScalingMethod(enum.Enum):
    """Position scaling method."""

    CONVICTION = "conviction"
    VOL_ADJUSTED = "vol_adjusted"
    KELLY = "kelly"
    NONE = "none"


@dataclass
class ScalingConfig:
    """Configuration for position scaling.

    Attributes:
        method:            Scaling method to use.
        vol_lookback:      Lookback days for volatility estimation.
        vol_target:        Target annualised portfolio volatility (for
                           vol_adjusted method).  If None, just normalises
                           by per-asset vol without targeting.
        kelly_fraction:    Fractional Kelly (0–1).  Full Kelly = 1.0.
                           Half-Kelly (0.5) is standard for robustness.
        min_confidence:    Minimum confidence to keep a position.  Alphas
                           below this are zeroed out.
        max_leverage:      Cap on total absolute scaled alpha to prevent
                           excessive leverage.
    """

    method: ScalingMethod = ScalingMethod.CONVICTION
    vol_lookback: int = 63
    vol_target: float | None = None
    kelly_fraction: float = 0.5
    min_confidence: float = 0.0
    max_leverage: float = 3.0


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScaledPosition:
    """Scaled position signal for one asset.

    Attributes:
        symbol:         Asset symbol.
        raw_alpha:      Original alpha score ([-1, 1]).
        confidence:     Signal confidence ([0, 1]).
        scaled_alpha:   Risk-adjusted alpha after scaling.
        vol_estimate:   Annualised volatility estimate (0 if unavailable).
    """

    symbol: str
    raw_alpha: float
    confidence: float
    scaled_alpha: float
    vol_estimate: float


# ---------------------------------------------------------------------------
# Scaler
# ---------------------------------------------------------------------------


class PositionScaler:
    """Scale alpha signals by conviction, volatility, or Kelly sizing.

    Args:
        config: Scaling configuration.
    """

    def __init__(self, config: ScalingConfig | None = None) -> None:
        self._config = config or ScalingConfig()

    @property
    def config(self) -> ScalingConfig:
        return self._config

    def scale(
        self,
        alpha_scores: dict[str, AlphaScore],
        returns_history: pd.DataFrame | None = None,
    ) -> dict[str, ScaledPosition]:
        """Scale alpha scores into risk-adjusted position signals.

        Args:
            alpha_scores: {symbol: AlphaScore} from alpha combination.
            returns_history: Historical returns for volatility estimation.
                Required for vol_adjusted and kelly methods.

        Returns:
            {symbol: ScaledPosition} with scaled alphas.
        """
        cfg = self._config

        # Estimate per-asset volatility
        vols = self._estimate_vols(alpha_scores, returns_history)

        # Apply scaling method
        positions: dict[str, ScaledPosition] = {}
        for sym, alpha in alpha_scores.items():
            # Filter low-confidence signals
            if alpha.confidence < cfg.min_confidence:
                positions[sym] = ScaledPosition(
                    symbol=sym,
                    raw_alpha=alpha.score,
                    confidence=alpha.confidence,
                    scaled_alpha=0.0,
                    vol_estimate=vols.get(sym, 0.0),
                )
                continue

            vol = vols.get(sym, 0.0)
            scaled = self._apply_method(alpha.score, alpha.confidence, vol)
            positions[sym] = ScaledPosition(
                symbol=sym,
                raw_alpha=alpha.score,
                confidence=alpha.confidence,
                scaled_alpha=scaled,
                vol_estimate=vol,
            )

        # Apply leverage cap
        positions = self._cap_leverage(positions)

        return positions

    def scale_to_alpha_dict(
        self,
        alpha_scores: dict[str, AlphaScore],
        returns_history: pd.DataFrame | None = None,
    ) -> dict[str, AlphaScore]:
        """Scale and return as AlphaScore dict (drop-in for PortfolioEngine).

        Same as :meth:`scale` but returns AlphaScore objects with the
        score replaced by the scaled alpha.  This makes the output
        compatible with ``PortfolioEngine.construct(alpha_scores=...)``.
        """
        positions = self.scale(alpha_scores, returns_history)
        result: dict[str, AlphaScore] = {}
        for sym, pos in positions.items():
            original = alpha_scores[sym]
            # Clamp to [-1, 1] for AlphaScore validation
            clamped = max(-1.0, min(1.0, pos.scaled_alpha))
            result[sym] = AlphaScore(
                symbol=original.symbol,
                timestamp=original.timestamp,
                score=clamped,
                confidence=original.confidence,
                signal_contributions=original.signal_contributions,
            )
        return result

    # ── Scaling methods ────────────────────────────────────────────

    def _apply_method(
        self, alpha: float, confidence: float, vol: float
    ) -> float:
        """Dispatch to the configured scaling method."""
        method = self._config.method

        if method == ScalingMethod.NONE:
            return alpha

        if method == ScalingMethod.CONVICTION:
            return alpha * confidence

        if method == ScalingMethod.VOL_ADJUSTED:
            return self._vol_adjusted(alpha, confidence, vol)

        if method == ScalingMethod.KELLY:
            return self._kelly(alpha, confidence, vol)

        return alpha

    def _vol_adjusted(
        self, alpha: float, confidence: float, vol: float
    ) -> float:
        """Scale alpha inversely to volatility.

        ``scaled = alpha * confidence / vol``  (normalised to vol)

        If vol_target is set, scales further so the position's vol
        contribution approximates the target.
        """
        if vol <= 1e-8:
            return alpha * confidence

        scaled = alpha * confidence / vol

        if self._config.vol_target is not None:
            # Scale so that abs(scaled) * vol ≈ vol_target
            scaled = alpha * confidence * (self._config.vol_target / vol)

        return scaled

    def _kelly(
        self, alpha: float, confidence: float, vol: float
    ) -> float:
        """Kelly-inspired sizing: edge / variance.

        ``f* = fraction * confidence * alpha / vol²``

        This sizes positions proportional to expected edge relative to
        the variance of returns — the classic Kelly formula adapted for
        continuous returns.
        """
        if vol <= 1e-8:
            return alpha * confidence * self._config.kelly_fraction

        # Kelly: edge/variance, where edge ~ alpha * confidence
        vol_daily = vol / math.sqrt(TRADING_DAYS_PER_YEAR)
        variance = vol_daily ** 2
        edge = alpha * confidence

        return self._config.kelly_fraction * edge / variance

    # ── Volatility estimation ──────────────────────────────────────

    def _estimate_vols(
        self,
        alpha_scores: dict[str, AlphaScore],
        returns_history: pd.DataFrame | None,
    ) -> dict[str, float]:
        """Estimate annualised volatility per asset."""
        symbols = list(alpha_scores.keys())

        if returns_history is None or returns_history.empty:
            return dict.fromkeys(symbols, 0.0)

        lookback = self._config.vol_lookback
        vols: dict[str, float] = {}

        for sym in symbols:
            if sym in returns_history.columns:
                series = returns_history[sym].dropna()
                if len(series) >= min(10, lookback):
                    tail = series.tail(lookback)
                    vols[sym] = float(tail.std() * math.sqrt(TRADING_DAYS_PER_YEAR))
                else:
                    vols[sym] = 0.0
            else:
                vols[sym] = 0.0

        return vols

    # ── Leverage cap ───────────────────────────────────────────────

    def _cap_leverage(
        self, positions: dict[str, ScaledPosition]
    ) -> dict[str, ScaledPosition]:
        """Cap total absolute scaled alpha to prevent excessive leverage."""
        total_abs = sum(abs(p.scaled_alpha) for p in positions.values())

        if total_abs <= self._config.max_leverage or total_abs < 1e-12:
            return positions

        scale_factor = self._config.max_leverage / total_abs

        return {
            sym: ScaledPosition(
                symbol=p.symbol,
                raw_alpha=p.raw_alpha,
                confidence=p.confidence,
                scaled_alpha=p.scaled_alpha * scale_factor,
                vol_estimate=p.vol_estimate,
            )
            for sym, p in positions.items()
        }
