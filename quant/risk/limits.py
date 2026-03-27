"""Portfolio-level exposure limits: per-symbol, per-sector, gross/net."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExposureLimits:
    """Configurable exposure limits applied to the portfolio.

    All limits are expressed as fractions of total capital (e.g., 0.20 = 20%).

    Attributes:
        max_position_fraction: Maximum exposure to a single symbol
            as a fraction of capital (default 0.20).
        max_sector_fraction: Maximum gross exposure to a single sector
            as a fraction of capital (default 0.40).
        max_gross_exposure: Maximum sum of absolute position values
            as a fraction of capital (default 1.50 = 150%).
        max_net_exposure: Maximum absolute net exposure (longs minus
            shorts) as a fraction of capital (default 1.00 = 100%).
        max_order_fraction: Maximum single-order size as a fraction
            of capital (default 0.10).
    """

    max_position_fraction: float = 0.20
    max_sector_fraction: float = 0.40
    max_gross_exposure: float = 1.50
    max_net_exposure: float = 1.00
    max_order_fraction: float = 0.10

    def __post_init__(self) -> None:
        for attr, val in [
            ("max_position_fraction", self.max_position_fraction),
            ("max_sector_fraction", self.max_sector_fraction),
            ("max_gross_exposure", self.max_gross_exposure),
            ("max_net_exposure", self.max_net_exposure),
            ("max_order_fraction", self.max_order_fraction),
        ]:
            if val <= 0:
                raise ValueError(f"{attr} must be positive")

    def check_position(
        self,
        symbol: str,
        proposed_dollar_value: float,
        capital: float,
    ) -> tuple[bool, str]:
        """Check if a position size for a single symbol is within limit.

        Args:
            symbol: Asset identifier (for reporting).
            proposed_dollar_value: Absolute dollar value of the resulting
                position after the proposed order.
            capital: Current total portfolio capital.

        Returns:
            (approved, reason) tuple.  approved=True if within limits.
        """
        if capital <= 0:
            return False, "Capital is zero or negative"
        fraction = abs(proposed_dollar_value) / capital
        if fraction > self.max_position_fraction:
            return (
                False,
                f"{symbol} position {fraction:.1%} exceeds per-symbol limit "
                f"{self.max_position_fraction:.1%}",
            )
        return True, ""

    def check_order_size(
        self,
        symbol: str,
        order_dollar_value: float,
        capital: float,
    ) -> tuple[bool, str]:
        """Check if a single order size is within the max order limit."""
        if capital <= 0:
            return False, "Capital is zero or negative"
        fraction = abs(order_dollar_value) / capital
        if fraction > self.max_order_fraction:
            return (
                False,
                f"{symbol} order size {fraction:.1%} exceeds max order limit "
                f"{self.max_order_fraction:.1%}",
            )
        return True, ""

    def check_sector(
        self,
        sector: str,
        proposed_sector_gross_value: float,
        capital: float,
    ) -> tuple[bool, str]:
        """Check if gross sector exposure remains within limit."""
        if capital <= 0:
            return False, "Capital is zero or negative"
        fraction = abs(proposed_sector_gross_value) / capital
        if fraction > self.max_sector_fraction:
            return (
                False,
                f"Sector '{sector}' exposure {fraction:.1%} exceeds limit "
                f"{self.max_sector_fraction:.1%}",
            )
        return True, ""

    def check_gross_exposure(
        self,
        proposed_gross_exposure: float,
        capital: float,
    ) -> tuple[bool, str]:
        """Check if total gross exposure (sum of abs positions) is within limit."""
        if capital <= 0:
            return False, "Capital is zero or negative"
        fraction = proposed_gross_exposure / capital
        if fraction > self.max_gross_exposure:
            return (
                False,
                f"Gross exposure {fraction:.1%} exceeds limit "
                f"{self.max_gross_exposure:.1%}",
            )
        return True, ""

    def check_net_exposure(
        self,
        proposed_net_exposure: float,
        capital: float,
    ) -> tuple[bool, str]:
        """Check if net exposure (longs - shorts) is within limit."""
        if capital <= 0:
            return False, "Capital is zero or negative"
        fraction = abs(proposed_net_exposure) / capital
        if fraction > self.max_net_exposure:
            return (
                False,
                f"Net exposure {fraction:.1%} exceeds limit "
                f"{self.max_net_exposure:.1%}",
            )
        return True, ""
