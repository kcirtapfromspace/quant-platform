"""CLI entry point for end-of-day market data ingestion.

Examples
--------
# Incremental update (daily cron job):
quant-ingest run --db /data/market.duckdb

# Full history fetch for a new symbol list:
quant-ingest run --db /data/market.duckdb --mode full --start 2020-01-01

# Check stored coverage:
quant-ingest status --db /data/market.duckdb

# Run as daemon (blocks, runs daily at 18:00 ET):
quant-ingest schedule --db /data/market.duckdb
"""
from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

import click
from loguru import logger

from quant.config import load_universe
from quant.data.ingest.yahoo import YahooFinanceSource
from quant.data.pipeline import IngestionPipeline
from quant.data.storage.duckdb import MarketDataStore


def _setup_logging(verbose: bool) -> None:
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=level, format="{time:HH:mm:ss} | {level:<8} | {message}")


@click.group()
@click.option("--verbose", is_flag=True, default=False)
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """Quant Infrastructure — market data ingestion CLI."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    _setup_logging(verbose)


@main.command()
@click.option("--db", required=True, type=click.Path(), help="Path to DuckDB file")
@click.option(
    "--mode",
    default="incremental",
    type=click.Choice(["incremental", "full"]),
    show_default=True,
)
@click.option("--start", default=None, type=str, help="Start date YYYY-MM-DD (full mode)")
@click.option("--end", default=None, type=str, help="End date YYYY-MM-DD (default: today)")
@click.option(
    "--symbols",
    default=None,
    type=str,
    help="Comma-separated symbols (overrides universe file)",
)
@click.pass_context
def run(
    ctx: click.Context,
    db: str,
    mode: str,
    start: str | None,
    end: str | None,
    symbols: str | None,
) -> None:
    """Fetch and store market data."""
    if symbols:
        universe = [s.strip().upper() for s in symbols.split(",")]
    else:
        universe = load_universe()

    start_date = date.fromisoformat(start) if start else None
    end_date = date.fromisoformat(end) if end else None

    with MarketDataStore(db) as store:
        pipeline = IngestionPipeline(store=store, source=YahooFinanceSource())
        result = pipeline.run(
            symbols=universe,
            mode=mode,  # type: ignore[arg-type]
            start=start_date,
            end=end_date,
        )

    click.echo(f"Done: {result.summary()}")
    if result.gaps_detected:
        click.echo(f"Gaps detected in {len(result.gaps_detected)} symbol(s):")
        for sym, gaps in sorted(result.gaps_detected.items()):
            click.echo(f"  {sym}: {gaps[:5]}{'...' if len(gaps) > 5 else ''}")

    sys.exit(0 if result.records_invalid == 0 else 1)


@main.command()
@click.option("--db", required=True, type=click.Path(), help="Path to DuckDB file")
@click.pass_context
def status(ctx: click.Context, db: str) -> None:
    """Show coverage summary for the database."""
    if not Path(db).exists():
        click.echo(f"Database not found: {db}")
        sys.exit(1)

    with MarketDataStore(db) as store:
        syms = store.symbols()
        if not syms:
            click.echo("No data in database.")
            return

        click.echo(f"Symbols: {len(syms)}")
        for sym in syms:
            total = store.count(sym)
            latest = store.latest_date(sym)
            click.echo(f"  {sym:<8} rows={total:<6} latest={latest}")


@main.command("schedule")
@click.option("--db", required=True, type=click.Path(), help="Path to DuckDB file")
@click.option(
    "--time",
    "run_time",
    default="18:00",
    show_default=True,
    help="Daily run time in HH:MM (local system time)",
)
@click.pass_context
def schedule_cmd(ctx: click.Context, db: str, run_time: str) -> None:
    """Run as a daemon, triggering an incremental ingest daily at --time."""
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        click.echo("apscheduler not installed. Run: pip install apscheduler")
        sys.exit(1)

    hour, minute = (int(x) for x in run_time.split(":"))
    universe = load_universe()

    def _job() -> None:
        logger.info("Scheduled ingest starting at {}", datetime.now().isoformat())
        with MarketDataStore(db) as store:
            pipeline = IngestionPipeline(store=store, source=YahooFinanceSource())
            result = pipeline.run(symbols=universe, mode="incremental")
        logger.info("Scheduled ingest complete: {}", result.summary())

    scheduler = BlockingScheduler()
    scheduler.add_job(
        _job,
        trigger=CronTrigger(hour=hour, minute=minute),
        id="eod_ingest",
        name="EOD market data ingest",
    )
    click.echo(f"Scheduler running. Daily ingest at {run_time}. Press Ctrl-C to stop.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        click.echo("Scheduler stopped.")


if __name__ == "__main__":
    main()
