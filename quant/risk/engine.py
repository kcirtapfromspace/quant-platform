"""Main risk engine: coordinates all pre-execution checks synchronously.

All risk checks run synchronously and must pass before an order is
submitted to the OMS.  This ensures no order bypasses risk controls.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger

from quant.risk.circuit_breaker import DrawdownCircuitBreaker
from quant.risk.limits import ExposureLimits
from quant.risk.sizing import (
    FixedFractionParams,
    KellyParams,
    PositionSizer,
    SizingMethod,
    VolatilityTargetParams,
)


@dataclass
class Order:
    """A proposed order to be validated by the risk engine.

    Attributes:
        symbol: Asset identifier (e.g. "AAPL").
        quantity: Signed quantity — positive for buy, negative for sell.
        price: Estimated fill price per unit.
        sector: Optional sector label for sector-level exposure checks.
    """

    symbol: str
    quantity: float
    price: float
    sector: str | None = None

    @property
    def dollar_value(self) -> float:
        return self.quantity * self.price

    @property
    def abs_dollar_value(self) -> float:
        return abs(self.dollar_value)


@dataclass
class PortfolioState:
    """Snapshot of current portfolio used for risk calculations.

    Attributes:
        capital: Total portfolio value (cash + market value of positions).
        positions: Dict mapping symbol → current signed dollar value.
        sector_exposures: Dict mapping sector → current gross dollar exposure.
        peak_portfolio_value: All-time peak portfolio value (for drawdown).
    """

    capital: float
    positions: dict[str, float] = field(default_factory=dict)
    sector_exposures: dict[str, float] = field(default_factory=dict)
    peak_portfolio_value: float = 0.0

    def gross_exposure(self) -> float:
        return sum(abs(v) for v in self.positions.values())

    def net_exposure(self) -> float:
        return sum(self.positions.values())


@dataclass
class RiskCheckResult:
    """Result of a risk engine validation.

    Attributes:
        approved: True if the order passed all checks.
        adjusted_quantity: Approved order quantity (may be reduced from
            the original to satisfy limits; 0 if rejected).
        reason: Human-readable description of why the order was rejected
            or adjusted.  Empty string if fully approved.
        checks_passed: List of check names that passed.
        checks_failed: List of check names that failed.
    """

    approved: bool
    adjusted_quantity: float
    reason: str
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)


@dataclass
class RiskConfig:
    """Configuration for the RiskEngine.

    Attributes:
        limits: Exposure limit configuration.
        circuit_breaker: Drawdown circuit breaker configuration.
        sizing_method: Default position sizing method.
        kelly_params: Required when sizing_method=KELLY.
        fixed_fraction_params: Required when sizing_method=FIXED_FRACTION.
        volatility_target_params: Required when sizing_method=VOLATILITY_TARGET.
    """

    limits: ExposureLimits = field(default_factory=ExposureLimits)
    circuit_breaker: DrawdownCircuitBreaker = field(
        default_factory=DrawdownCircuitBreaker
    )
    sizing_method: SizingMethod = SizingMethod.FIXED_FRACTION
    kelly_params: KellyParams | None = None
    fixed_fraction_params: FixedFractionParams | None = field(
        default_factory=lambda: FixedFractionParams(fraction=0.02)
    )
    volatility_target_params: VolatilityTargetParams | None = None


class RiskEngine:
    """Synchronous pre-execution risk engine.

    Runs all configured risk checks in sequence before an order is forwarded
    to the OMS.  Returns a :class:`RiskCheckResult` — the OMS must honour
    the ``approved`` flag and use ``adjusted_quantity`` rather than the
    original order quantity.

    Usage::

        config = RiskConfig(
            limits=ExposureLimits(max_position_fraction=0.15),
            circuit_breaker=DrawdownCircuitBreaker(max_drawdown_threshold=0.08),
        )
        engine = RiskEngine(config)

        result = engine.validate(order, portfolio)
        if result.approved:
            oms.submit(order.symbol, result.adjusted_quantity, order.price)
    """

    def __init__(self, config: RiskConfig | None = None) -> None:
        self._config = config or RiskConfig()
        self._sizer = PositionSizer()

    # ── Public API ────────────────────────────────────────────────────────

    def validate(self, order: Order, portfolio: PortfolioState) -> RiskCheckResult:
        """Run all risk checks synchronously for *order* given *portfolio*.

        Checks are run in this order:
          1. Drawdown circuit breaker
          2. Max order size
          3. Per-symbol position limit
          4. Per-sector exposure limit
          5. Gross exposure limit
          6. Net exposure limit

        If any check fails the order is rejected immediately (further checks
        are skipped to avoid redundant logging).

        Args:
            order: Proposed order to validate.
            portfolio: Current portfolio state snapshot.

        Returns:
            :class:`RiskCheckResult` with approved flag and adjusted quantity.
        """
        checks_passed: list[str] = []
        checks_failed: list[str] = []

        cfg = self._config

        # ── 1. Circuit breaker ────────────────────────────────────────────
        ok, reason = cfg.circuit_breaker.check(portfolio.capital)
        if not ok:
            logger.warning("Risk: circuit breaker tripped — {}", reason)
            checks_failed.append("circuit_breaker")
            return RiskCheckResult(
                approved=False,
                adjusted_quantity=0.0,
                reason=reason,
                checks_passed=checks_passed,
                checks_failed=checks_failed,
            )
        checks_passed.append("circuit_breaker")

        # ── 2. Max order size ─────────────────────────────────────────────
        ok, reason = cfg.limits.check_order_size(
            order.symbol, order.dollar_value, portfolio.capital
        )
        if not ok:
            logger.warning("Risk: order size rejected — {}", reason)
            checks_failed.append("max_order_size")
            return RiskCheckResult(
                approved=False,
                adjusted_quantity=0.0,
                reason=reason,
                checks_passed=checks_passed,
                checks_failed=checks_failed,
            )
        checks_passed.append("max_order_size")

        # ── 3. Per-symbol position limit ──────────────────────────────────
        current_position = portfolio.positions.get(order.symbol, 0.0)
        resulting_position = current_position + order.dollar_value
        ok, reason = cfg.limits.check_position(
            order.symbol, resulting_position, portfolio.capital
        )
        if not ok:
            logger.warning("Risk: position limit breached — {}", reason)
            checks_failed.append("position_limit")
            return RiskCheckResult(
                approved=False,
                adjusted_quantity=0.0,
                reason=reason,
                checks_passed=checks_passed,
                checks_failed=checks_failed,
            )
        checks_passed.append("position_limit")

        # ── 4. Sector exposure limit ──────────────────────────────────────
        if order.sector is not None:
            current_sector = portfolio.sector_exposures.get(order.sector, 0.0)
            resulting_sector = current_sector + order.abs_dollar_value
            ok, reason = cfg.limits.check_sector(
                order.sector, resulting_sector, portfolio.capital
            )
            if not ok:
                logger.warning("Risk: sector limit breached — {}", reason)
                checks_failed.append("sector_limit")
                return RiskCheckResult(
                    approved=False,
                    adjusted_quantity=0.0,
                    reason=reason,
                    checks_passed=checks_passed,
                    checks_failed=checks_failed,
                )
        checks_passed.append("sector_limit")

        # ── 5. Gross exposure limit ───────────────────────────────────────
        resulting_gross = portfolio.gross_exposure() + order.abs_dollar_value
        ok, reason = cfg.limits.check_gross_exposure(
            resulting_gross, portfolio.capital
        )
        if not ok:
            logger.warning("Risk: gross exposure limit breached — {}", reason)
            checks_failed.append("gross_exposure")
            return RiskCheckResult(
                approved=False,
                adjusted_quantity=0.0,
                reason=reason,
                checks_passed=checks_passed,
                checks_failed=checks_failed,
            )
        checks_passed.append("gross_exposure")

        # ── 6. Net exposure limit ─────────────────────────────────────────
        resulting_net = portfolio.net_exposure() + order.dollar_value
        ok, reason = cfg.limits.check_net_exposure(resulting_net, portfolio.capital)
        if not ok:
            logger.warning("Risk: net exposure limit breached — {}", reason)
            checks_failed.append("net_exposure")
            return RiskCheckResult(
                approved=False,
                adjusted_quantity=0.0,
                reason=reason,
                checks_passed=checks_passed,
                checks_failed=checks_failed,
            )
        checks_passed.append("net_exposure")

        # ── All checks passed ─────────────────────────────────────────────
        logger.debug(
            "Risk: order approved — {} {} @ {:.2f} (${:.0f})",
            order.symbol,
            order.quantity,
            order.price,
            order.abs_dollar_value,
        )
        return RiskCheckResult(
            approved=True,
            adjusted_quantity=order.quantity,
            reason="",
            checks_passed=checks_passed,
            checks_failed=checks_failed,
        )

    def compute_position_size(
        self, capital: float, price: float
    ) -> float:
        """Compute the number of units to trade using the configured sizing method.

        Args:
            capital: Available capital.
            price: Current asset price per unit.

        Returns:
            Number of units (rounded down to integer for equities).
        """
        if capital <= 0 or price <= 0:
            return 0.0

        cfg = self._config
        fraction = self._sizer.compute(
            cfg.sizing_method,
            capital,
            kelly_params=cfg.kelly_params,
            fixed_fraction_params=cfg.fixed_fraction_params,
            volatility_target_params=cfg.volatility_target_params,
        )
        dollar_size = capital * fraction
        return dollar_size / price
