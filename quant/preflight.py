"""Pre-flight safety checks for the strategy runner.

Validates broker connectivity, account state, and market hours before
each execution cycle.  Designed to be called by
:class:`~quant.service.StrategyService` before invoking ``run_once()``.

Usage::

    from quant.preflight import PreflightChecker, PreflightConfig

    checker = PreflightChecker(PreflightConfig(min_cash=10_000))
    result = checker.run(oms=oms)
    if not result.passed:
        for failure in result.failures:
            print(f"FAIL: {failure.check} — {failure.reason}")
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime, time, timezone

from loguru import logger

from quant.oms.system import OrderManagementSystem


class PreflightCheck(enum.Enum):
    BROKER_CONNECTED = "broker_connected"
    SUFFICIENT_CASH = "sufficient_cash"
    MARKET_OPEN = "market_open"


@dataclass(frozen=True, slots=True)
class PreflightFailure:
    """A single pre-flight check that did not pass."""

    check: PreflightCheck
    reason: str


@dataclass(frozen=True, slots=True)
class PreflightResult:
    """Aggregated result of all pre-flight checks.

    Attributes:
        passed:   True if all checks passed.
        failures: List of checks that failed (empty when passed=True).
    """

    passed: bool
    failures: list[PreflightFailure] = field(default_factory=list)


@dataclass
class PreflightConfig:
    """Configuration for pre-flight checks.

    Attributes:
        min_cash:            Minimum cash balance required to proceed.
        check_market_hours:  If True, verify that the current time falls within
                             US equity market hours (09:30–16:00 ET, Mon–Fri).
                             Set to False for crypto or paper-only strategies.
        market_open:         Market open time in ET (default 09:30).
        market_close:        Market close time in ET (default 16:00).
        utc_offset_hours:    Offset from UTC to market timezone.
                             US Eastern = -4 (EDT) or -5 (EST).
                             Default -4 for EDT.
    """

    min_cash: float = 1_000.0
    check_market_hours: bool = False
    market_open: time = time(9, 30)
    market_close: time = time(16, 0)
    utc_offset_hours: int = -4


class PreflightChecker:
    """Runs pre-flight safety checks before a strategy execution cycle.

    Args:
        config: Pre-flight configuration.
    """

    def __init__(self, config: PreflightConfig | None = None) -> None:
        self._config = config or PreflightConfig()

    def run(
        self,
        oms: OrderManagementSystem,
        now: datetime | None = None,
    ) -> PreflightResult:
        """Execute all pre-flight checks.

        Args:
            oms: The OMS instance to check connectivity and cash.
            now: Override current time (for testing). Defaults to utcnow.

        Returns:
            PreflightResult with pass/fail status and any failure details.
        """
        now = now or datetime.now(timezone.utc)
        failures: list[PreflightFailure] = []

        failures.extend(self._check_broker(oms))
        failures.extend(self._check_cash(oms))
        if self._config.check_market_hours:
            failures.extend(self._check_market_hours(now))

        result = PreflightResult(passed=len(failures) == 0, failures=failures)

        if result.passed:
            logger.info("Preflight: all checks passed")
        else:
            for f in failures:
                logger.warning("Preflight FAIL: {} — {}", f.check.value, f.reason)

        return result

    def _check_broker(self, oms: OrderManagementSystem) -> list[PreflightFailure]:
        try:
            if not oms._broker.is_connected:
                return [
                    PreflightFailure(
                        check=PreflightCheck.BROKER_CONNECTED,
                        reason="Broker adapter is not connected",
                    )
                ]
        except Exception as exc:
            return [
                PreflightFailure(
                    check=PreflightCheck.BROKER_CONNECTED,
                    reason=f"Broker connectivity check raised: {exc}",
                )
            ]
        return []

    def _check_cash(self, oms: OrderManagementSystem) -> list[PreflightFailure]:
        try:
            cash = oms.get_account_cash()
            if cash < self._config.min_cash:
                return [
                    PreflightFailure(
                        check=PreflightCheck.SUFFICIENT_CASH,
                        reason=(
                            f"Cash ${cash:,.2f} below minimum ${self._config.min_cash:,.2f}"
                        ),
                    )
                ]
        except Exception as exc:
            return [
                PreflightFailure(
                    check=PreflightCheck.SUFFICIENT_CASH,
                    reason=f"Cash check raised: {exc}",
                )
            ]
        return []

    def _check_market_hours(
        self, now: datetime
    ) -> list[PreflightFailure]:
        # Convert UTC to market-local time
        utc_hour = now.hour
        utc_minute = now.minute
        local_minutes = (
            (utc_hour * 60 + utc_minute) + self._config.utc_offset_hours * 60
        ) % (24 * 60)

        open_minutes = self._config.market_open.hour * 60 + self._config.market_open.minute
        close_minutes = self._config.market_close.hour * 60 + self._config.market_close.minute

        # Check weekday (0=Monday, 6=Sunday)
        weekday = now.weekday()
        if weekday >= 5:
            return [
                PreflightFailure(
                    check=PreflightCheck.MARKET_OPEN,
                    reason=f"Weekend (day={weekday})",
                )
            ]

        if not (open_minutes <= local_minutes < close_minutes):
            local_hour, local_min = divmod(local_minutes, 60)
            return [
                PreflightFailure(
                    check=PreflightCheck.MARKET_OPEN,
                    reason=(
                        f"Outside market hours: local time {local_hour:02d}:{local_min:02d}, "
                        f"market {self._config.market_open.strftime('%H:%M')}"
                        f"–{self._config.market_close.strftime('%H:%M')}"
                    ),
                )
            ]

        return []
