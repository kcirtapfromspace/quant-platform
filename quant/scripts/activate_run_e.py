#!/usr/bin/env python3
"""runE activation script — Alpaca paper trading (QUA-56).

Activates the runE multi-sleeve momentum strategy on the Alpaca paper
trading account:

  - Multi-sleeve portfolio: momentum (40%), trend (35%), adaptive combined (25%)
  - RegimeWeightAdapter (max_tilt=0.30)
  - Rebalance frequency: 63 trading days
  - 50-symbol universe (top 50 from sp500_universe.txt)
  - $1M notional ($QUANT_PAPER_NOTIONAL)

CRO halt triggers (Gate 1 conditions):
  - Paper MaxDD >= 22% at any time → halt immediately, escalate to CRO
  - Paper Sharpe < 0.40 (annualized) after 30 trading days → halt, escalate

Modes::

    # Initial activation — enter positions, start daemon:
    python -m quant.scripts.activate_run_e

    # Single-shot boot only (no daemon):
    python -m quant.scripts.activate_run_e --once

    # Status check (print NAV, Sharpe, drawdown, halt state):
    python -m quant.scripts.activate_run_e --status
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import click
import pandas as pd
from loguru import logger

from quant.config import get_db_path
from quant.data.storage.duckdb import MarketDataStore
from quant.execution.alpaca import AlpacaAdapter
from quant.features import DEFAULT_REGISTRY, FeatureEngine, InMemoryFeatureCache
from quant.oms.system import OrderManagementSystem
from quant.orchestrator import OrchestratorConfig, StrategyOrchestrator, StrategySleeve
from quant.portfolio.alpha import CombinationMethod
from quant.portfolio.constraints import PortfolioConstraints
from quant.portfolio.engine import PortfolioConfig
from quant.portfolio.optimizers import OptimizationMethod
from quant.risk.circuit_breaker import DrawdownCircuitBreaker
from quant.risk.engine import RiskConfig
from quant.risk.limits import ExposureLimits
from quant.signals.adaptive_combiner import AdaptiveCombinerConfig
from quant.signals.base import BaseSignal
from quant.signals.regime import RegimeConfig, RegimeDetector, RegimeWeightAdapter
from quant.signals.strategies import MomentumSignal, TrendFollowingSignal

# ── Constants ─────────────────────────────────────────────────────────────────

UNIVERSE_FILE = Path(__file__).parent.parent.parent / "data" / "sp500_universe.txt"
STATE_FILE = Path(__file__).parent.parent.parent / "data" / "run_e_state.json"
UNIVERSE_SIZE = 50
INITIAL_NOTIONAL = float(os.environ.get("QUANT_PAPER_NOTIONAL", 1_000_000))

# CRO halt thresholds (Gate 1)
SHARPE_HALT_THRESHOLD = 0.40
SHARPE_LOOKBACK_DAYS = 30      # trading days before Sharpe check activates
MAXDD_HALT_THRESHOLD = 0.22    # immediate halt

# Rebalance every 63 trading days (~quarterly)
REBALANCE_FREQ_DAYS = 63

# Feature lookback for computing indicators (calendar days)
FEATURE_LOOKBACK_CAL = 400


# ── Universe ──────────────────────────────────────────────────────────────────

def load_run_e_universe() -> list[str]:
    """Load top-50 symbols from sp500_universe.txt."""
    if not UNIVERSE_FILE.exists():
        raise FileNotFoundError(f"Universe file not found: {UNIVERSE_FILE}")
    symbols: list[str] = []
    for line in UNIVERSE_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            symbols.append(line.upper())
        if len(symbols) == UNIVERSE_SIZE:
            break
    if len(symbols) < UNIVERSE_SIZE:
        raise ValueError(
            f"Universe file has only {len(symbols)} symbols (need {UNIVERSE_SIZE})"
        )
    return symbols


# ── State management ──────────────────────────────────────────────────────────

@dataclass
class RunEState:
    """Persistent state for the runE strategy daemon.

    Tracks rebalance schedule, NAV history, and halt conditions.
    """

    activated_at: str = ""                   # ISO datetime
    last_rebalance_date: str = ""            # ISO date
    trading_days_elapsed: int = 0
    daily_nav: list[dict] = field(default_factory=list)  # [{date, nav}]
    halted: bool = False
    halt_reason: str = ""
    last_allocation: dict[str, float] = field(default_factory=dict)  # {symbol: weight}


def _load_state() -> RunEState:
    if STATE_FILE.exists():
        raw = json.loads(STATE_FILE.read_text())
        s = RunEState()
        for k, v in raw.items():
            if hasattr(s, k):
                setattr(s, k, v)
        return s
    return RunEState()


def _save_state(state: RunEState) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(asdict(state), indent=2))


# ── Providers ─────────────────────────────────────────────────────────────────

def make_feature_provider(store: MarketDataStore) -> "FeatureProvider":
    """Build a feature provider callable for the orchestrator.

    Pulls OHLCV data from DuckDB and computes required signal features.
    """
    engine = FeatureEngine(store, DEFAULT_REGISTRY, cache=InMemoryFeatureCache())

    def feature_provider(symbol: str, signal: BaseSignal) -> dict[str, pd.Series]:
        end = date.today()
        start = end - timedelta(days=FEATURE_LOOKBACK_CAL)
        required = list(signal.required_features)

        try:
            results = engine.compute(
                symbols=[symbol],
                features=required,
                start=start,
                end=end,
            )
            return results.get(symbol, {})
        except Exception:
            logger.warning("feature_provider: failed for {} — returning empty", symbol)
            return {}

    return feature_provider


def make_returns_provider(store: MarketDataStore) -> "ReturnsProvider":
    """Build a returns DataFrame provider for covariance estimation."""

    def returns_provider(symbols: list[str], lookback_days: int) -> pd.DataFrame:
        end = date.today()
        start = end - timedelta(days=max(lookback_days * 2, 400))
        try:
            ohlcv = store.query_multi(symbols, start=start, end=end)
            if ohlcv.empty:
                return pd.DataFrame(columns=symbols)

            price_wide = (
                ohlcv[ohlcv["symbol"].isin(symbols)][["symbol", "date", "adj_close"]]
                .pivot(index="date", columns="symbol", values="adj_close")
                .sort_index()
            )
            returns = price_wide.pct_change().dropna(how="all")
            # Keep only the last lookback_days rows
            if len(returns) > lookback_days:
                returns = returns.iloc[-lookback_days:]
            return returns.reindex(columns=symbols)
        except Exception:
            logger.warning("returns_provider: failed — returning empty DataFrame")
            return pd.DataFrame(columns=symbols)

    return returns_provider


# ── Orchestrator factory ──────────────────────────────────────────────────────

def build_orchestrator(
    universe: list[str],
    oms: OrderManagementSystem,
    store: MarketDataStore,
    circuit_breaker: DrawdownCircuitBreaker,
) -> StrategyOrchestrator:
    """Build the runE multi-sleeve orchestrator."""

    # ── Signals ──────────────────────────────────────────────────────────────
    momentum_signal = MomentumSignal(rsi_period=14, lookback=5, return_scale=0.05)
    trend_signal = TrendFollowingSignal(fast_ma=20, slow_ma=50)

    adaptive_cfg = AdaptiveCombinerConfig(
        ic_lookback=126,
        min_ic_periods=20,
        min_ic=0.0,
        ic_halflife=21,
        shrinkage=0.3,
        min_assets=3,
    )

    # ── Portfolio config (MVO, long-only, max 5% per position) ───────────────
    def _pc(risk_aversion: float = 5.0) -> PortfolioConfig:
        return PortfolioConfig(
            optimization_method=OptimizationMethod.MEAN_VARIANCE,
            constraints=PortfolioConstraints(long_only=True, max_weight=0.05),
            rebalance_threshold=0.01,
            cov_lookback_days=252,
            optimizer_kwargs={"risk_aversion": risk_aversion},
        )

    # ── Sleeves ───────────────────────────────────────────────────────────────
    sleeves = [
        StrategySleeve(
            name="momentum_us_equity",
            signals=[MomentumSignal(rsi_period=14, lookback=5, return_scale=0.05)],
            capital_weight=0.40,
            strategy_type="momentum",
            portfolio_config=_pc(),
            combination_method=CombinationMethod.EQUAL_WEIGHT,
            lookback_days=252,
        ),
        StrategySleeve(
            name="trend_following_us_equity",
            signals=[TrendFollowingSignal(fast_ma=20, slow_ma=50)],
            capital_weight=0.35,
            strategy_type="trend",
            portfolio_config=_pc(),
            combination_method=CombinationMethod.EQUAL_WEIGHT,
            lookback_days=252,
        ),
        StrategySleeve(
            name="adaptive_combined",
            signals=[
                MomentumSignal(rsi_period=14, lookback=5, return_scale=0.05),
                TrendFollowingSignal(fast_ma=20, slow_ma=50),
            ],
            capital_weight=0.25,
            strategy_type="momentum",
            portfolio_config=_pc(),
            combination_method=CombinationMethod.RANK_WEIGHTED,
            adaptive_combiner_config=adaptive_cfg,
            lookback_days=252,
        ),
    ]

    # ── Top-level risk limits ─────────────────────────────────────────────────
    top_risk = RiskConfig(
        limits=ExposureLimits(
            max_position_fraction=0.05,
            max_order_fraction=0.05,
            max_gross_exposure=1.0,
        )
    )

    # ── Regime detection ──────────────────────────────────────────────────────
    regime_detector = RegimeDetector(RegimeConfig(
        vol_short_window=21,
        vol_long_window=252,
        vol_high_threshold=1.25,
        vol_low_threshold=0.75,
        trend_window=63,
        trend_threshold=0.10,
        mr_threshold=-0.10,
        corr_window=63,
        corr_high_threshold=0.60,
        corr_low_threshold=0.25,
        crisis_vol_threshold=2.0,
    ))
    regime_adapter = RegimeWeightAdapter(max_tilt=0.30)

    config = OrchestratorConfig(
        universe=universe,
        risk_config=top_risk,
        min_order_value=100.0,
        net_conflicting=True,
        regime_detector=regime_detector,
        regime_adapter=regime_adapter,
        regime_lookback_days=252,
        circuit_breaker=circuit_breaker,
    )

    feature_provider = make_feature_provider(store)
    returns_provider = make_returns_provider(store)

    return StrategyOrchestrator(
        config=config,
        sleeves=sleeves,
        oms=oms,
        feature_provider=feature_provider,
        returns_provider=returns_provider,
    )


# ── Halt condition checks ─────────────────────────────────────────────────────

def _annualized_sharpe(daily_nav: list[dict]) -> float | None:
    """Compute annualized Sharpe from daily NAV history."""
    if len(daily_nav) < 2:
        return None
    navs = [entry["nav"] for entry in daily_nav]
    returns = [
        (navs[i] / navs[i - 1]) - 1.0
        for i in range(1, len(navs))
    ]
    if len(returns) < 2:
        return None
    n = len(returns)
    mean_r = sum(returns) / n
    variance = sum((r - mean_r) ** 2 for r in returns) / (n - 1)
    std_r = math.sqrt(variance)
    if std_r == 0:
        return None
    return (mean_r / std_r) * math.sqrt(252)


def _current_drawdown(daily_nav: list[dict]) -> float:
    """Compute current drawdown from peak NAV in history."""
    if not daily_nav:
        return 0.0
    navs = [entry["nav"] for entry in daily_nav]
    peak = max(navs)
    current = navs[-1]
    if peak <= 0:
        return 0.0
    return (peak - current) / peak


def check_halt_conditions(state: RunEState, current_nav: float) -> tuple[bool, str]:
    """Evaluate CRO halt conditions.

    Returns:
        (should_halt, reason) — reason is empty if no halt.
    """
    # Update NAV history with today if not already present
    today_str = date.today().isoformat()
    if not state.daily_nav or state.daily_nav[-1]["date"] != today_str:
        state.daily_nav.append({"date": today_str, "nav": current_nav})

    # 1. MaxDD halt (immediate, any time)
    dd = _current_drawdown(state.daily_nav)
    if dd >= MAXDD_HALT_THRESHOLD:
        return True, (
            f"MaxDD halt: current drawdown {dd:.1%} >= {MAXDD_HALT_THRESHOLD:.1%} threshold. "
            "Immediate halt — escalate to CRO."
        )

    # 2. Sharpe halt (only after 30 trading days)
    if state.trading_days_elapsed >= SHARPE_LOOKBACK_DAYS:
        sharpe = _annualized_sharpe(state.daily_nav)
        if sharpe is not None and sharpe < SHARPE_HALT_THRESHOLD:
            return True, (
                f"Sharpe halt: annualized Sharpe {sharpe:.3f} < {SHARPE_HALT_THRESHOLD} "
                f"after {state.trading_days_elapsed} trading days. Halt — escalate to CRO."
            )

    return False, ""


# ── NAV computation ───────────────────────────────────────────────────────────

def get_portfolio_nav(oms: OrderManagementSystem) -> float:
    """Compute total portfolio NAV: cash + position market values."""
    cash = oms.get_account_cash()
    positions = oms.get_all_positions()
    pos_value = sum(p.market_value for p in positions.values())
    return cash + pos_value


def get_allocation_summary(oms: OrderManagementSystem, nav: float) -> dict[str, float]:
    """Return current positions as weight fractions."""
    if nav <= 0:
        return {}
    positions = oms.get_all_positions()
    return {
        sym: pos.market_value / nav
        for sym, pos in positions.items()
        if abs(pos.market_value) > 1.0
    }


# ── Rebalance schedule ────────────────────────────────────────────────────────

def _should_rebalance(state: RunEState) -> bool:
    """Return True if 63+ trading days have elapsed since last rebalance."""
    if not state.last_rebalance_date:
        return True  # First run — always rebalance
    last = date.fromisoformat(state.last_rebalance_date)
    today = date.today()
    # Use calendar days / 1.33 ≈ trading days (conservative)
    calendar_days = (today - last).days
    approx_trading_days = int(calendar_days / 1.33)
    return approx_trading_days >= REBALANCE_FREQ_DAYS


# ── Boot cycle ────────────────────────────────────────────────────────────────

def run_boot_cycle(
    orchestrator: StrategyOrchestrator,
    oms: OrderManagementSystem,
    state: RunEState,
) -> dict:
    """Run the initial rebalance to enter positions.

    Returns summary dict for CRO notification.
    """
    logger.info("runE boot: running initial position entry cycle ...")
    result = orchestrator.run_once()

    nav = get_portfolio_nav(oms)
    allocation = get_allocation_summary(oms, nav)

    now_str = datetime.now(timezone.utc).isoformat()
    today_str = date.today().isoformat()

    state.activated_at = now_str
    state.last_rebalance_date = today_str
    state.trading_days_elapsed = 0
    state.daily_nav = [{"date": today_str, "nav": nav}]
    state.halted = False
    state.halt_reason = ""
    state.last_allocation = allocation
    _save_state(state)

    summary = {
        "activated_at": now_str,
        "starting_nav": nav,
        "n_positions": len(allocation),
        "n_submitted": result.n_submitted,
        "n_rejected": result.n_rejected,
        "circuit_breaker_tripped": result.circuit_breaker_tripped,
        "top_10_positions": dict(
            sorted(allocation.items(), key=lambda kv: -abs(kv[1]))[:10]
        ),
    }

    logger.info(
        "runE boot complete | NAV={:,.0f} | positions={} | submitted={} rejected={}",
        nav,
        len(allocation),
        result.n_submitted,
        result.n_rejected,
    )
    return summary


# ── Daily cycle ───────────────────────────────────────────────────────────────

def run_daily_cycle(
    orchestrator: StrategyOrchestrator,
    oms: OrderManagementSystem,
    circuit_breaker: DrawdownCircuitBreaker,
    state: RunEState,
) -> None:
    """Daily monitoring and conditional rebalance.

    1. Compute current NAV.
    2. Check halt conditions (MaxDD, Sharpe).
    3. If due: run rebalance (63-day frequency).
    4. Update and persist state.
    """
    today_str = date.today().isoformat()
    logger.info("runE daily cycle: {}", today_str)

    # Compute NAV
    nav = get_portfolio_nav(oms)
    circuit_breaker.update(nav)

    # Check halt conditions
    should_halt, halt_reason = check_halt_conditions(state, nav)
    if should_halt:
        state.halted = True
        state.halt_reason = halt_reason
        _save_state(state)
        logger.error("runE HALTED: {}", halt_reason)
        return

    # Increment trading day counter
    state.trading_days_elapsed += 1

    # Rebalance if due
    if _should_rebalance(state):
        logger.info(
            "runE: rebalance triggered ({} days since last on {})",
            (date.today() - date.fromisoformat(state.last_rebalance_date)).days
            if state.last_rebalance_date
            else "first run",
            state.last_rebalance_date or "—",
        )
        result = orchestrator.run_once()
        state.last_rebalance_date = today_str
        state.last_allocation = get_allocation_summary(oms, nav)
        logger.info(
            "runE rebalance complete | NAV={:,.0f} | submitted={} rejected={}",
            nav,
            result.n_submitted,
            result.n_rejected,
        )
    else:
        logger.info("runE: no rebalance (holding — next due in ~{} trading days)", (
            REBALANCE_FREQ_DAYS
            - int(
                (date.today() - date.fromisoformat(state.last_rebalance_date)).days / 1.33
            )
            if state.last_rebalance_date
            else 0
        ))

    _save_state(state)

    # Emit Prometheus metrics
    _push_metrics(nav, state)


def _push_metrics(nav: float, state: RunEState) -> None:
    """Push current NAV and Sharpe to Prometheus pushgateway (best-effort)."""
    try:
        import os as _os
        from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
        pgw = _os.environ.get("PUSHGATEWAY_URL", "")
        if not pgw:
            return
        registry = CollectorRegistry()
        g_nav = Gauge("run_e_nav", "runE portfolio NAV", registry=registry)
        g_nav.set(nav)
        dd = _current_drawdown(state.daily_nav)
        g_dd = Gauge("run_e_drawdown", "runE current drawdown from peak", registry=registry)
        g_dd.set(dd)
        sharpe = _annualized_sharpe(state.daily_nav) or 0.0
        g_sharpe = Gauge("run_e_sharpe_annualized", "runE annualized Sharpe", registry=registry)
        g_sharpe.set(sharpe)
        push_to_gateway(pgw, job="run_e", registry=registry)
        logger.debug("runE: pushed metrics to {}", pgw)
    except Exception as exc:
        logger.debug("runE: metrics push failed (non-critical) — {}", exc)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _setup_logging(verbose: bool) -> None:
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=level, format="{time:HH:mm:ss} | {level:<8} | {message}")


def _build_oms_and_store() -> tuple[OrderManagementSystem, MarketDataStore]:
    adapter = AlpacaAdapter(paper=True, stream_fills=True)
    adapter.connect()
    oms = OrderManagementSystem(broker=adapter)
    oms.start()
    store = MarketDataStore(get_db_path())
    return oms, store


@click.command()
@click.option("--once", is_flag=True, default=False, help="Boot only: enter initial positions then exit.")
@click.option("--status", is_flag=True, default=False, help="Print current state and exit.")
@click.option("--run-hour", default=16, show_default=True, help="Daily run hour (local time).")
@click.option("--run-minute", default=5, show_default=True, help="Daily run minute.")
@click.option("--metrics-port", default=8000, show_default=True, help="Prometheus port (0=disabled).")
@click.option("--verbose", is_flag=True, default=False)
def main(
    once: bool,
    status: bool,
    run_hour: int,
    run_minute: int,
    metrics_port: int,
    verbose: bool,
) -> None:
    """Activate and run the runE strategy on Alpaca paper trading."""
    _setup_logging(verbose)

    # ── Status mode ──────────────────────────────────────────────────────────
    if status:
        state = _load_state()
        if not state.activated_at:
            click.echo("runE not yet activated.")
            return
        nav = state.daily_nav[-1]["nav"] if state.daily_nav else 0.0
        dd = _current_drawdown(state.daily_nav)
        sharpe = _annualized_sharpe(state.daily_nav)
        click.echo(f"runE Status — {date.today().isoformat()}")
        click.echo(f"  Activated:           {state.activated_at}")
        click.echo(f"  Last rebalance:      {state.last_rebalance_date}")
        click.echo(f"  Trading days:        {state.trading_days_elapsed}")
        click.echo(f"  Last NAV:            ${nav:,.0f}")
        click.echo(f"  Current drawdown:    {dd:.2%}")
        click.echo(f"  Annualized Sharpe:   {sharpe:.3f}" if sharpe else "  Annualized Sharpe:   (insufficient data)")
        click.echo(f"  Halted:              {state.halted}")
        if state.halt_reason:
            click.echo(f"  Halt reason:         {state.halt_reason}")
        return

    # ── Build infrastructure ──────────────────────────────────────────────────
    universe = load_run_e_universe()
    logger.info("runE: universe loaded — {} symbols", len(universe))

    oms, store = _build_oms_and_store()

    circuit_breaker = DrawdownCircuitBreaker(
        max_drawdown_threshold=MAXDD_HALT_THRESHOLD,
        reset_on_new_peak=False,  # per CRO: halt stays until manually reviewed
    )

    orchestrator = build_orchestrator(universe, oms, store, circuit_breaker)

    state = _load_state()

    # ── Check if already halted ───────────────────────────────────────────────
    if state.halted:
        click.echo(f"ERROR: runE is halted — {state.halt_reason}")
        click.echo("Resolve the halt condition and delete data/run_e_state.json to restart.")
        sys.exit(1)

    # ── Boot / initial activation ─────────────────────────────────────────────
    if not state.activated_at or once:
        summary = run_boot_cycle(orchestrator, oms, state)
        click.echo("\n=== runE Activated ===")
        click.echo(f"  Start date:      {date.today().isoformat()}")
        click.echo(f"  Starting NAV:    ${summary['starting_nav']:,.0f}")
        click.echo(f"  Positions:       {summary['n_positions']}")
        click.echo(f"  Orders submitted: {summary['n_submitted']}")
        click.echo(f"  Orders rejected:  {summary['n_rejected']}")
        click.echo("\n  Top 10 initial allocation:")
        for sym, w in summary["top_10_positions"].items():
            click.echo(f"    {sym:<8} {w:.2%}")

        if once:
            return

    # ── Scheduled daemon mode ─────────────────────────────────────────────────
    if metrics_port > 0:
        try:
            from prometheus_client import start_http_server
            start_http_server(metrics_port)
            logger.info("runE: Prometheus metrics on port {}", metrics_port)
        except Exception as exc:
            logger.warning("runE: failed to start metrics server — {}", exc)

    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        logger.error("apscheduler not installed. Run: pip install apscheduler")
        sys.exit(1)

    def _daily_job() -> None:
        # Reload state in case it was modified externally
        current_state = _load_state()
        if current_state.halted:
            logger.error("runE: strategy is halted — skipping daily cycle")
            return
        run_daily_cycle(orchestrator, oms, circuit_breaker, current_state)

    scheduler = BlockingScheduler()
    scheduler.add_job(
        _daily_job,
        trigger=CronTrigger(hour=run_hour, minute=run_minute),
        id="run_e_daily",
        name="runE daily cycle",
        misfire_grace_time=3600,
    )

    logger.info(
        "runE: scheduled daily at {:02d}:{:02d} local time. "
        "Rebalance every {} trading days. Press Ctrl-C to stop.",
        run_hour,
        run_minute,
        REBALANCE_FREQ_DAYS,
    )

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("runE: shutting down")
    finally:
        oms.stop()
        store.close()


if __name__ == "__main__":
    main()
