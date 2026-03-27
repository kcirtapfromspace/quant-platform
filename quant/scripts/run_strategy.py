"""CLI entry point for the strategy runner service.

Examples
--------
# Single execution cycle (paper trading):
quant-run once --mode paper

# Run as daemon (daily at 16:05 local time):
quant-run schedule --mode paper --time 16:05

# Pre-flight checks only (no trading):
quant-run preflight --mode paper
"""
from __future__ import annotations

import sys

import click
from loguru import logger

from quant.config import load_universe
from quant.execution.paper import PaperBrokerAdapter
from quant.oms.system import OrderManagementSystem
from quant.portfolio.constraints import PortfolioConstraints
from quant.portfolio.engine import PortfolioConfig
from quant.portfolio.optimizers import OptimizationMethod
from quant.preflight import PreflightChecker, PreflightConfig
from quant.risk.engine import RiskConfig
from quant.risk.limits import ExposureLimits
from quant.runner import RunnerConfig, RunnerState, StrategyRunner
from quant.service import ServiceConfig, StrategyService


def _setup_logging(verbose: bool) -> None:
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=level, format="{time:HH:mm:ss} | {level:<8} | {message}")


def _make_oms(mode: str, initial_cash: float) -> OrderManagementSystem:
    """Create an OMS with the appropriate broker adapter."""
    if mode == "paper":
        broker = PaperBrokerAdapter(initial_cash=initial_cash, default_fill_price=150.0)
    else:
        click.echo(
            "Live broker adapters require explicit configuration. "
            "Set up AlpacaAdapter or IBAdapter in a custom script."
        )
        sys.exit(1)

    oms = OrderManagementSystem(broker=broker)
    oms.start()
    return oms


def _make_runner(
    oms: OrderManagementSystem,
    universe: list[str],
    signals: list | None = None,
    optimization_method: str = "risk_parity",
) -> StrategyRunner:
    """Build a StrategyRunner with default configuration."""
    method_map = {
        "risk_parity": OptimizationMethod.RISK_PARITY,
        "mean_variance": OptimizationMethod.MEAN_VARIANCE,
        "minimum_variance": OptimizationMethod.MINIMUM_VARIANCE,
        "max_diversification": OptimizationMethod.MAX_DIVERSIFICATION,
    }
    opt_method = method_map.get(optimization_method, OptimizationMethod.RISK_PARITY)

    config = RunnerConfig(
        universe=universe,
        signals=signals or [],
        portfolio_config=PortfolioConfig(
            optimization_method=opt_method,
            constraints=PortfolioConstraints(
                long_only=True, max_weight=0.25, max_gross_exposure=1.0
            ),
        ),
        risk_config=RiskConfig(
            limits=ExposureLimits(
                max_position_fraction=0.25,
                max_order_fraction=0.25,
                max_gross_exposure=1.50,
            ),
        ),
    )
    return StrategyRunner(config=config, oms=oms)


@click.group()
@click.option("--verbose", is_flag=True, default=False)
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """Quant Infrastructure — strategy runner CLI."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    _setup_logging(verbose)


@main.command()
@click.option(
    "--mode",
    default="paper",
    type=click.Choice(["paper", "live"]),
    show_default=True,
    help="Execution mode",
)
@click.option("--cash", default=1_000_000.0, show_default=True, help="Initial cash balance")
@click.option(
    "--optimizer",
    default="risk_parity",
    type=click.Choice(["risk_parity", "mean_variance", "minimum_variance", "max_diversification"]),
    show_default=True,
)
@click.option("--symbols", default=None, type=str, help="Comma-separated symbols (overrides universe file)")
@click.option("--skip-preflight", is_flag=True, default=False, help="Skip pre-flight checks")
@click.pass_context
def once(
    ctx: click.Context,
    mode: str,
    cash: float,
    optimizer: str,
    symbols: str | None,
    skip_preflight: bool,
) -> None:
    """Execute a single strategy cycle."""
    universe = [s.strip().upper() for s in symbols.split(",")] if symbols else load_universe()
    oms = _make_oms(mode, cash)

    runner = _make_runner(oms, universe, optimization_method=optimizer)
    service = StrategyService(
        runner=runner,
        oms=oms,
        config=ServiceConfig(skip_preflight=skip_preflight, metrics_port=0),
    )

    result = service.run_once()
    if result is None:
        click.echo("Pre-flight checks failed — run aborted.")
        sys.exit(1)

    if result.state == RunnerState.ERROR:
        click.echo(f"Run failed: {result.error}")
        sys.exit(1)

    click.echo(
        f"Done: portfolio=${result.portfolio_value:,.0f} | "
        f"submitted={result.n_submitted} rejected={result.n_rejected}"
    )
    if result.construction and result.construction.rebalance_triggered:
        click.echo(
            f"  vol={result.construction.optimization.risk:.1%} "
            f"turnover={result.construction.rebalance.turnover:.1%}"
        )


@main.command("schedule")
@click.option(
    "--mode",
    default="paper",
    type=click.Choice(["paper", "live"]),
    show_default=True,
)
@click.option("--cash", default=1_000_000.0, show_default=True, help="Initial cash balance")
@click.option(
    "--time",
    "run_time",
    default="16:05",
    show_default=True,
    help="Daily run time in HH:MM (local system time)",
)
@click.option(
    "--optimizer",
    default="risk_parity",
    type=click.Choice(["risk_parity", "mean_variance", "minimum_variance", "max_diversification"]),
    show_default=True,
)
@click.option("--symbols", default=None, type=str, help="Comma-separated symbols")
@click.option("--metrics-port", default=8000, show_default=True, help="Prometheus port (0 to disable)")
@click.option("--skip-preflight", is_flag=True, default=False)
@click.pass_context
def schedule_cmd(
    ctx: click.Context,
    mode: str,
    cash: float,
    run_time: str,
    optimizer: str,
    symbols: str | None,
    metrics_port: int,
    skip_preflight: bool,
) -> None:
    """Run as a daemon, executing the strategy daily at --time."""
    universe = [s.strip().upper() for s in symbols.split(",")] if symbols else load_universe()
    hour, minute = (int(x) for x in run_time.split(":"))

    oms = _make_oms(mode, cash)
    runner = _make_runner(oms, universe, optimization_method=optimizer)
    service = StrategyService(
        runner=runner,
        oms=oms,
        config=ServiceConfig(
            schedule_hour=hour,
            schedule_minute=minute,
            skip_preflight=skip_preflight,
            metrics_port=metrics_port,
        ),
    )

    service.start()  # blocks


@main.command()
@click.option(
    "--mode",
    default="paper",
    type=click.Choice(["paper", "live"]),
    show_default=True,
)
@click.option("--cash", default=1_000_000.0, show_default=True)
@click.pass_context
def preflight(ctx: click.Context, mode: str, cash: float) -> None:
    """Run pre-flight checks only (no trading)."""
    oms = _make_oms(mode, cash)
    checker = PreflightChecker(PreflightConfig(min_cash=1_000.0))
    result = checker.run(oms)

    if result.passed:
        click.echo("All pre-flight checks passed.")
    else:
        click.echo("Pre-flight checks FAILED:")
        for failure in result.failures:
            click.echo(f"  {failure.check.value}: {failure.reason}")
        sys.exit(1)


if __name__ == "__main__":
    main()
