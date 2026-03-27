"""Position sizing models: Kelly criterion, fixed-fraction, volatility-targeting.

Hot-path computations are delegated to ``quant_rs.risk`` Rust kernels.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass

import quant_rs as _qrs


class SizingMethod(enum.Enum):
    KELLY = "kelly"
    FIXED_FRACTION = "fixed_fraction"
    VOLATILITY_TARGET = "volatility_target"


@dataclass
class KellyParams:
    win_probability: float
    win_loss_ratio: float
    fraction: float = 1.0

    def __post_init__(self) -> None:
        if not (0.0 < self.win_probability < 1.0):
            raise ValueError("win_probability must be in (0, 1)")
        if self.win_loss_ratio <= 0:
            raise ValueError("win_loss_ratio must be positive")
        if not (0.0 < self.fraction <= 1.0):
            raise ValueError("fraction must be in (0, 1]")


@dataclass
class FixedFractionParams:
    fraction: float

    def __post_init__(self) -> None:
        if not (0.0 < self.fraction <= 1.0):
            raise ValueError("fraction must be in (0, 1]")


@dataclass
class VolatilityTargetParams:
    target_annual_volatility: float
    asset_annual_volatility: float
    price: float

    def __post_init__(self) -> None:
        if self.target_annual_volatility <= 0:
            raise ValueError("target_annual_volatility must be positive")
        if self.asset_annual_volatility <= 0:
            raise ValueError("asset_annual_volatility must be positive")
        if self.price <= 0:
            raise ValueError("price must be positive")


class PositionSizer:
    """Computes the desired position size (as a fraction of capital) using
    the configured sizing method.

    All methods return a *capital fraction* in [0, 1].  The caller is
    responsible for converting that fraction to shares/contracts.
    """

    def kelly(self, params: KellyParams) -> float:
        """Full or fractional Kelly position size.

        f* = (b*p - q) / b  where b=win_loss_ratio, p=win_prob, q=1-p.

        Returns the fraction clamped to [0, 1].  A negative Kelly
        (negative expected value) returns 0.0.
        """
        full_kelly = _qrs.risk.kelly_fraction(params.win_probability, params.win_loss_ratio)
        return min(1.0, full_kelly * params.fraction)

    def fixed_fraction(self, params: FixedFractionParams) -> float:
        """Risk a fixed fraction of capital per trade."""
        return params.fraction

    def volatility_target(
        self, params: VolatilityTargetParams, capital: float
    ) -> float:
        """Size a position so the portfolio volatility contribution equals the
        target annual volatility.

        dollar_size = (target_vol * capital) / asset_vol
        fraction    = dollar_size / capital = target_vol / asset_vol

        Returns the fraction clamped to [0, 1].
        """
        if capital <= 0:
            return 0.0
        fraction = params.target_annual_volatility / params.asset_annual_volatility
        return min(1.0, fraction)

    def compute(
        self,
        method: SizingMethod,
        capital: float,
        *,
        kelly_params: KellyParams | None = None,
        fixed_fraction_params: FixedFractionParams | None = None,
        volatility_target_params: VolatilityTargetParams | None = None,
    ) -> float:
        """Dispatch to the chosen sizing method.

        Returns:
            Capital fraction in [0, 1] representing the desired position size.

        Raises:
            ValueError: If required params for the chosen method are missing,
                or if capital is zero or negative.
        """
        if capital <= 0:
            return 0.0

        if method == SizingMethod.KELLY:
            if kelly_params is None:
                raise ValueError("kelly_params required for KELLY method")
            return self.kelly(kelly_params)

        if method == SizingMethod.FIXED_FRACTION:
            if fixed_fraction_params is None:
                raise ValueError("fixed_fraction_params required for FIXED_FRACTION")
            return self.fixed_fraction(fixed_fraction_params)

        if method == SizingMethod.VOLATILITY_TARGET:
            if volatility_target_params is None:
                raise ValueError(
                    "volatility_target_params required for VOLATILITY_TARGET"
                )
            return self.volatility_target(volatility_target_params, capital)

        raise ValueError(f"Unknown sizing method: {method}")
