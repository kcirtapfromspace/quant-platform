"""Strategy Runner Service — production-ready execution daemon.

Wraps :class:`~quant.runner.StrategyRunner` with:
  - **Scheduling**: APScheduler-driven cron execution (e.g. daily at market close).
  - **Pre-flight checks**: broker connectivity, cash, market hours.
  - **Instrumentation**: Prometheus metrics for every run cycle.
  - **Graceful lifecycle**: start / stop / run-once entry points.

Usage::

    from quant.service import StrategyService, ServiceConfig

    service = StrategyService(
        runner=runner,
        oms=oms,
        config=ServiceConfig(schedule_hour=16, schedule_minute=5),
    )
    service.start()  # blocks — runs on schedule

Single-shot mode::

    service.run_once()  # execute one cycle with preflight + metrics
"""
from __future__ import annotations

import time as _time
from dataclasses import dataclass, field

from loguru import logger

from quant.monitoring.metrics import (
    RUNNER_LAST_RUN_TIMESTAMP,
    RUNNER_PORTFOLIO_VALUE,
    RUNNER_PORTFOLIO_VOLATILITY,
    RUNNER_PREFLIGHT_FAILURES,
    RUNNER_RUN_DURATION,
    RUNNER_RUNS_TOTAL,
    RUNNER_TRADES_REJECTED,
    RUNNER_TRADES_SUBMITTED,
    RUNNER_TURNOVER,
)
from quant.oms.system import OrderManagementSystem
from quant.preflight import PreflightChecker, PreflightConfig, PreflightResult
from quant.runner import RunnerState, RunResult, StrategyRunner


@dataclass
class ServiceConfig:
    """Configuration for the StrategyService.

    Attributes:
        schedule_hour:    Hour (0–23) to trigger the daily run (local system time).
        schedule_minute:  Minute (0–59) to trigger the daily run.
        preflight_config: Pre-flight check configuration.
        skip_preflight:   If True, skip pre-flight checks entirely.
        metrics_port:     Prometheus scrape endpoint port.  Set to 0 to disable.
    """

    schedule_hour: int = 16
    schedule_minute: int = 5
    preflight_config: PreflightConfig = field(default_factory=PreflightConfig)
    skip_preflight: bool = False
    metrics_port: int = 8000


class StrategyService:
    """Production service wrapper for the strategy runner.

    Adds scheduling, pre-flight safety checks, and Prometheus
    instrumentation around each :meth:`StrategyRunner.run_once` call.

    Args:
        runner:  An initialised StrategyRunner instance.
        oms:     The OrderManagementSystem (used for pre-flight checks).
        config:  Service-level configuration.
    """

    def __init__(
        self,
        runner: StrategyRunner,
        oms: OrderManagementSystem,
        config: ServiceConfig | None = None,
    ) -> None:
        self._runner = runner
        self._oms = oms
        self._config = config or ServiceConfig()
        self._preflight = PreflightChecker(self._config.preflight_config)
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def run_once(self) -> RunResult | None:
        """Execute a single strategy cycle: preflight -> run -> instrument.

        Returns:
            RunResult from the runner, or None if pre-flight checks failed.
        """
        # Pre-flight
        if not self._config.skip_preflight:
            pf_result = self._preflight.run(self._oms)
            if not pf_result.passed:
                self._record_preflight_failures(pf_result)
                return None

        # Run with timing
        t0 = _time.monotonic()
        result = self._runner.run_once()
        elapsed = _time.monotonic() - t0

        # Instrument
        self._record_metrics(result, elapsed)
        return result

    def start(self) -> None:
        """Start the scheduled execution loop (blocks until interrupted).

        Launches a Prometheus metrics server (if configured), then enters
        the APScheduler blocking loop.
        """
        if self._config.metrics_port > 0:
            from quant.monitoring.metrics import start_metrics_server

            start_metrics_server(self._config.metrics_port)

        try:
            from apscheduler.schedulers.blocking import BlockingScheduler
            from apscheduler.triggers.cron import CronTrigger
        except ImportError:
            logger.error(
                "apscheduler not installed. Install with: pip install apscheduler"
            )
            raise

        self._running = True
        scheduler = BlockingScheduler()
        scheduler.add_job(
            self._scheduled_run,
            trigger=CronTrigger(
                hour=self._config.schedule_hour,
                minute=self._config.schedule_minute,
            ),
            id="strategy_runner",
            name="Strategy runner daily cycle",
        )

        logger.info(
            "StrategyService: scheduled daily at {:02d}:{:02d}. Press Ctrl-C to stop.",
            self._config.schedule_hour,
            self._config.schedule_minute,
        )

        try:
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("StrategyService: shutting down")
        finally:
            self._running = False

    def stop(self) -> None:
        """Signal the service to stop."""
        self._running = False

    # ── Private ──────────────────────────────────────────────────────────

    def _scheduled_run(self) -> None:
        """Wrapper called by the scheduler — catches all exceptions."""
        logger.info("StrategyService: scheduled cycle starting")
        try:
            result = self.run_once()
            if result is None:
                logger.warning("StrategyService: cycle skipped (preflight failed)")
            elif result.state == RunnerState.ERROR:
                logger.error("StrategyService: cycle completed with error: {}", result.error)
            else:
                logger.info(
                    "StrategyService: cycle complete | portfolio=${:,.0f} | "
                    "submitted={} rejected={}",
                    result.portfolio_value,
                    result.n_submitted,
                    result.n_rejected,
                )
        except Exception:
            logger.exception("StrategyService: unhandled exception in scheduled cycle")

    def _record_metrics(self, result: RunResult, elapsed: float) -> None:
        """Update Prometheus metrics after a run cycle."""
        status_label = result.state.value
        RUNNER_RUNS_TOTAL.labels(status=status_label).inc()
        RUNNER_RUN_DURATION.observe(elapsed)
        RUNNER_LAST_RUN_TIMESTAMP.set(result.timestamp.timestamp())
        RUNNER_PORTFOLIO_VALUE.set(result.portfolio_value)
        RUNNER_TRADES_SUBMITTED.inc(result.n_submitted)
        RUNNER_TRADES_REJECTED.inc(result.n_rejected)

        if result.construction and result.construction.rebalance_triggered:
            RUNNER_PORTFOLIO_VOLATILITY.set(result.construction.optimization.risk)
            RUNNER_TURNOVER.set(result.construction.rebalance.turnover)

    @staticmethod
    def _record_preflight_failures(pf_result: PreflightResult) -> None:
        """Increment preflight failure counters."""
        for failure in pf_result.failures:
            RUNNER_PREFLIGHT_FAILURES.labels(check=failure.check.value).inc()
