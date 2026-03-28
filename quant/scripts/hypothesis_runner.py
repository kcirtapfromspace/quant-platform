"""Hypothesis validation runner — executes a single alpha hypothesis backtest on the cluster.

Entry point for K8s batch Jobs (see k8s/hypothesis-validation/backtest-job-template.yaml).

Environment variables (all required unless noted):
    HYPOTHESIS_NAME         one of: regime_adaptive_momentum, ic_weighted_ensemble,
                            statistical_arbitrage_pairs, hmm_regime_detection, volatility_crush
    RUN_PARAMS_JSON         JSON with hypothesis-specific param overrides (optional, default "{}")
    START_DATE              data start date YYYY-MM-DD (default 2020-01-01)
    END_DATE                data end date YYYY-MM-DD (default 2025-12-31)
    IS_WINDOW               in-sample window in trading days (default 252)
    OOS_WINDOW              out-of-sample window in trading days (default 63)
    STEP_SIZE               step size in trading days (default 63)
    RESULTS_PATH            directory to write results JSON (default /results)
    DB_PATH                 DuckDB market data file (default ~/.quant/market.duckdb)
    JOB_NAME                K8s job name for result tagging (optional)
    PROMETHEUS_PUSHGATEWAY  push gateway URL for metrics export (optional)
    CRO_SHARPE_MIN          float (default 0.60)
    CRO_PROFIT_FACTOR_MIN   float (default 1.10)
    CRO_MAX_DRAWDOWN_MAX    float (default 0.20)
    CRO_WFE_MIN             float (default 0.20)

Exit codes:
    0   backtest ran and passed all CRO gates
    1   backtest ran but failed one or more CRO gates
    2   fatal error (data missing, config invalid, etc.)
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from quant.backtest.multi_strategy import MultiStrategyConfig, SleeveConfig
from quant.backtest.multi_strategy_walk_forward import (
    MultiStrategyWalkForwardAnalyzer,
    MultiStrategyWalkForwardConfig,
    MultiStrategyWalkForwardResult,
)
from quant.data.storage.duckdb import MarketDataStore
from quant.portfolio.alpha import CombinationMethod
from quant.portfolio.engine import PortfolioConfig, PortfolioConstraints
from quant.portfolio.optimizers import OptimizationMethod
from quant.signals.adaptive_combiner import AdaptiveCombinerConfig
from quant.signals.factors import VolatilitySignal
from quant.signals.regime import RegimeConfig, RegimeWeightAdapter
from quant.signals.strategies import MeanReversionSignal, MomentumSignal, TrendFollowingSignal


# ---------------------------------------------------------------------------
# CRO gate thresholds (read from env, with defaults)
# ---------------------------------------------------------------------------

_CRO = {
    "sharpe_min": float(os.environ.get("CRO_SHARPE_MIN", "0.60")),
    "profit_factor_min": float(os.environ.get("CRO_PROFIT_FACTOR_MIN", "1.10")),
    "max_drawdown_max": float(os.environ.get("CRO_MAX_DRAWDOWN_MAX", "0.20")),
    "wfe_min": float(os.environ.get("CRO_WFE_MIN", "0.20")),
}


# ---------------------------------------------------------------------------
# Hypothesis strategy builders
# ---------------------------------------------------------------------------

def _portfolio_config(max_weight: float = 0.25) -> PortfolioConfig:
    return PortfolioConfig(
        optimization_method=OptimizationMethod.RISK_PARITY,
        constraints=PortfolioConstraints(
            long_only=True, max_weight=max_weight, max_gross_exposure=1.0
        ),
    )


def _regime_config(lookback: int = 252) -> RegimeConfig:
    return RegimeConfig(lookback=lookback)


def build_regime_adaptive_momentum(params: dict[str, Any]) -> MultiStrategyConfig:
    """Hypothesis 1: Regime-Adaptive Cross-Sectional Momentum.

    Combines cross-sectional momentum with regime detection to tilt
    allocation toward trending regimes and away from choppy/crash regimes.
    """
    lookback = int(params.get("lookback_days", 42))
    regime_lookback = int(params.get("regime_lookback", 252))
    max_tilt = float(params.get("max_tilt", 0.30))

    return MultiStrategyConfig(
        sleeves=[
            SleeveConfig(
                name="cs_momentum",
                signals=[MomentumSignal(rsi_period=14, lookback=lookback)],
                capital_weight=0.60,
                strategy_type="momentum",
                portfolio_config=_portfolio_config(),
                combination_method=CombinationMethod.EQUAL_WEIGHT,
            ),
            SleeveConfig(
                name="low_vol_factor",
                signals=[VolatilitySignal(period=lookback)],
                capital_weight=0.40,
                strategy_type="momentum",
                portfolio_config=_portfolio_config(),
                combination_method=CombinationMethod.EQUAL_WEIGHT,
            ),
        ],
        rebalance_frequency=21,
        commission_bps=10.0,
        initial_capital=1_000_000.0,
        regime_config=_regime_config(lookback=regime_lookback),
        regime_adapter=RegimeWeightAdapter(max_tilt=max_tilt),
        regime_lookback_days=regime_lookback,
        name="regime_adaptive_momentum",
    )


def build_ic_weighted_ensemble(params: dict[str, Any]) -> MultiStrategyConfig:
    """Hypothesis 2: IC-Weighted Signal Ensemble.

    Combines momentum, mean-reversion, and trend-following signals with
    adaptive IC-based weighting to dynamically favour the best-performing
    signal in recent history.
    """
    ic_lookback = int(params.get("ic_lookback_days", 42))
    decay_halflife = int(params.get("decay_halflife", 21))

    adaptive_cfg = AdaptiveCombinerConfig(
        ic_lookback=ic_lookback,
        ic_halflife=decay_halflife,
    )

    return MultiStrategyConfig(
        sleeves=[
            SleeveConfig(
                name="ic_ensemble",
                signals=[
                    MomentumSignal(rsi_period=14, lookback=21),
                    MeanReversionSignal(),
                    TrendFollowingSignal(fast_ma=20, slow_ma=50),
                ],
                capital_weight=1.00,
                strategy_type="momentum",
                portfolio_config=_portfolio_config(),
                combination_method=CombinationMethod.RANK_WEIGHTED,
                adaptive_combiner_config=adaptive_cfg,
            ),
        ],
        rebalance_frequency=21,
        commission_bps=10.0,
        initial_capital=1_000_000.0,
        name="ic_weighted_ensemble",
    )


def build_statistical_arbitrage_pairs(params: dict[str, Any]) -> MultiStrategyConfig:
    """Hypothesis 3: Statistical Arbitrage Pairs.

    Uses mean-reversion across correlated pairs identified by co-movement.
    Proxied here as a cross-sectional mean-reversion strategy — a full pairs
    engine (cointegration testing, spread z-scoring) is a future enhancement.
    """
    return MultiStrategyConfig(
        sleeves=[
            SleeveConfig(
                name="mean_reversion_cs",
                signals=[
                    MeanReversionSignal(bb_period=20, num_std=float(params.get("entry_zscore", 2.0))),
                    MomentumSignal(rsi_period=14, lookback=5),
                ],
                capital_weight=1.00,
                strategy_type="mean_reversion",
                portfolio_config=_portfolio_config(max_weight=0.10),
                combination_method=CombinationMethod.EQUAL_WEIGHT,
            ),
        ],
        rebalance_frequency=5,
        commission_bps=10.0,
        initial_capital=1_000_000.0,
        name="statistical_arbitrage_pairs",
    )


def build_hmm_regime_detection(params: dict[str, Any]) -> MultiStrategyConfig:
    """Hypothesis 4: HMM Regime Detection.

    Uses the platform's regime detector to identify bull/bear/crash/choppy
    regimes and applies regime-conditional signal weighting.
    Uses the GPU-enabled image on Jetson nodes for ML inference.
    """
    hmm_lookback = int(params.get("hmm_lookback", 252))
    max_tilt = 0.50  # stronger regime tilt for dedicated HMM hypothesis

    return MultiStrategyConfig(
        sleeves=[
            SleeveConfig(
                name="regime_momentum",
                signals=[MomentumSignal(rsi_period=14, lookback=21)],
                capital_weight=0.50,
                strategy_type="momentum",
                portfolio_config=_portfolio_config(),
                combination_method=CombinationMethod.EQUAL_WEIGHT,
            ),
            SleeveConfig(
                name="regime_mean_rev",
                signals=[MeanReversionSignal()],
                capital_weight=0.50,
                strategy_type="mean_reversion",
                portfolio_config=_portfolio_config(),
                combination_method=CombinationMethod.EQUAL_WEIGHT,
            ),
        ],
        rebalance_frequency=21,
        commission_bps=10.0,
        initial_capital=1_000_000.0,
        regime_config=_regime_config(lookback=hmm_lookback),
        regime_adapter=RegimeWeightAdapter(max_tilt=max_tilt),
        regime_lookback_days=hmm_lookback,
        name="hmm_regime_detection",
    )


def build_volatility_crush(params: dict[str, Any]) -> MultiStrategyConfig:
    """Hypothesis 5: Volatility Crush.

    Buys assets after a volatility spike (volatility mean-reversion).
    Proxied using the VolatilitySignal in inverse mode: score high-vol
    assets after spike as potential mean-reversion candidates.
    """
    vol_lookback = int(params.get("vol_lookback", 21))

    return MultiStrategyConfig(
        sleeves=[
            SleeveConfig(
                name="vol_crush",
                signals=[
                    VolatilitySignal(
                        period=vol_lookback,
                        low_vol=float(params.get("crush_threshold_pct", 0.25)),
                        high_vol=0.60,
                    ),
                    MeanReversionSignal(),
                ],
                capital_weight=1.00,
                strategy_type="mean_reversion",
                portfolio_config=_portfolio_config(max_weight=0.15),
                combination_method=CombinationMethod.EQUAL_WEIGHT,
            ),
        ],
        rebalance_frequency=int(params.get("holding_days", 5)),
        commission_bps=10.0,
        initial_capital=1_000_000.0,
        name="volatility_crush",
    )


_HYPOTHESIS_BUILDERS = {
    "regime_adaptive_momentum": build_regime_adaptive_momentum,
    "ic_weighted_ensemble": build_ic_weighted_ensemble,
    "statistical_arbitrage_pairs": build_statistical_arbitrage_pairs,
    "hmm_regime_detection": build_hmm_regime_detection,
    "volatility_crush": build_volatility_crush,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_returns(db_path: str, start: date, end: date) -> pd.DataFrame:
    """Load daily close-to-close returns from DuckDB into a wide DataFrame."""
    store = MarketDataStore.open(db_path)
    symbols = store.list_symbols()
    frames = {}
    for sym in symbols:
        records = store.query(sym, start, end)
        if len(records) < 2:
            continue
        prices = pd.Series(
            [r.adj_close for r in records],
            index=pd.DatetimeIndex([r.date for r in records]),
            name=sym,
        )
        frames[sym] = prices.pct_change().dropna()

    if not frames:
        raise RuntimeError(f"No data loaded from {db_path} for {start}–{end}")

    df = pd.DataFrame(frames).dropna(how="all")
    logger.info("Loaded returns: {} symbols × {} days", df.shape[1], df.shape[0])
    return df


# ---------------------------------------------------------------------------
# CRO gate validation
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    hypothesis: str
    params: dict[str, Any]
    job_name: str
    run_timestamp: str
    n_folds: int
    oos_sharpe: float
    profit_factor: float
    max_drawdown: float
    wfe: float
    passes: bool
    failures: list[str]


def check_cro_gates(
    result: MultiStrategyWalkForwardResult,
    hypothesis: str,
    params: dict[str, Any],
    job_name: str,
) -> GateResult:
    failures: list[str] = []

    oos_returns = result.oos_returns.dropna()
    gains = oos_returns[oos_returns > 0].sum()
    losses = abs(oos_returns[oos_returns < 0].sum())
    profit_factor = float(gains / losses) if losses > 1e-12 else float("inf")

    max_dd = abs(result.oos_max_drawdown)
    sharpe = result.oos_sharpe
    wfe = result.mean_wfe

    if sharpe < _CRO["sharpe_min"]:
        failures.append(f"Sharpe {sharpe:.3f} < {_CRO['sharpe_min']}")
    if profit_factor < _CRO["profit_factor_min"]:
        failures.append(f"PF {profit_factor:.3f} < {_CRO['profit_factor_min']}")
    if max_dd >= _CRO["max_drawdown_max"]:
        failures.append(f"MaxDD {max_dd:.2%} >= {_CRO['max_drawdown_max']:.0%}")
    if wfe < _CRO["wfe_min"]:
        failures.append(f"WFE {wfe:.3f} < {_CRO['wfe_min']}")

    return GateResult(
        hypothesis=hypothesis,
        params=params,
        job_name=job_name,
        run_timestamp=datetime.now(timezone.utc).isoformat(),
        n_folds=result.n_folds,
        oos_sharpe=round(sharpe, 4),
        profit_factor=round(profit_factor, 4) if not (profit_factor == float("inf")) else 9999.0,
        max_drawdown=round(max_dd, 4),
        wfe=round(wfe, 4),
        passes=len(failures) == 0,
        failures=failures,
    )


# ---------------------------------------------------------------------------
# Prometheus push gateway export
# ---------------------------------------------------------------------------

def push_metrics(gate: GateResult, pushgateway_url: str) -> None:
    """Push backtest metrics to Prometheus Pushgateway in text exposition format."""
    labels = f'hypothesis="{gate.hypothesis}",job="{gate.job_name}"'
    lines = [
        f'# HELP backtest_oos_sharpe OOS Sharpe ratio',
        f'# TYPE backtest_oos_sharpe gauge',
        f'backtest_oos_sharpe{{{labels}}} {gate.oos_sharpe}',
        f'# HELP backtest_profit_factor OOS Profit Factor',
        f'# TYPE backtest_profit_factor gauge',
        f'backtest_profit_factor{{{labels}}} {gate.profit_factor}',
        f'# HELP backtest_max_drawdown OOS Max Drawdown',
        f'# TYPE backtest_max_drawdown gauge',
        f'backtest_max_drawdown{{{labels}}} {gate.max_drawdown}',
        f'# HELP backtest_wfe Walk-Forward Efficiency',
        f'# TYPE backtest_wfe gauge',
        f'backtest_wfe{{{labels}}} {gate.wfe}',
        f'# HELP backtest_passes_gates 1 if all CRO gates passed',
        f'# TYPE backtest_passes_gates gauge',
        f'backtest_passes_gates{{{labels}}} {1 if gate.passes else 0}',
        f'# HELP backtest_n_folds Number of walk-forward folds',
        f'# TYPE backtest_n_folds gauge',
        f'backtest_n_folds{{{labels}}} {gate.n_folds}',
        '',
    ]
    payload = '\n'.join(lines).encode('utf-8')
    url = f"{pushgateway_url.rstrip('/')}/metrics/job/hypothesis_backtest/hypothesis/{gate.hypothesis}"
    try:
        req = urllib.request.Request(url, data=payload, method='POST')
        req.add_header('Content-Type', 'text/plain; version=0.0.4')
        with urllib.request.urlopen(req, timeout=10) as resp:
            logger.info("Metrics pushed to Pushgateway: HTTP {}", resp.status)
    except Exception as exc:
        logger.warning("Failed to push metrics to {}: {}", url, exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<8} | {message}")

    hypothesis = os.environ.get("HYPOTHESIS_NAME", "").strip()
    if not hypothesis:
        logger.error("HYPOTHESIS_NAME env var is required")
        return 2
    if hypothesis not in _HYPOTHESIS_BUILDERS:
        logger.error("Unknown hypothesis '{}'. Valid: {}", hypothesis, list(_HYPOTHESIS_BUILDERS))
        return 2

    raw_params = os.environ.get("RUN_PARAMS_JSON", "{}").strip()
    try:
        params: dict[str, Any] = json.loads(raw_params) if raw_params else {}
    except json.JSONDecodeError as exc:
        logger.error("Invalid RUN_PARAMS_JSON: {}", exc)
        return 2

    start = date.fromisoformat(os.environ.get("START_DATE", "2020-01-01"))
    end = date.fromisoformat(os.environ.get("END_DATE", "2025-12-31"))
    is_window = int(os.environ.get("IS_WINDOW", "252"))
    oos_window = int(os.environ.get("OOS_WINDOW", "63"))
    step_size = int(os.environ.get("STEP_SIZE", "63"))
    results_path = Path(os.environ.get("RESULTS_PATH", "/results"))
    db_path = os.environ.get("DB_PATH", str(Path.home() / ".quant" / "market.duckdb"))
    job_name = os.environ.get("JOB_NAME", f"{hypothesis}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}")
    pushgateway = os.environ.get("PROMETHEUS_PUSHGATEWAY", "")

    logger.info("Hypothesis: {}", hypothesis)
    logger.info("Params: {}", params)
    logger.info("Window: IS={} OOS={} step={}", is_window, oos_window, step_size)
    logger.info("Data: {} → {} from {}", start, end, db_path)

    # Load returns
    try:
        returns = load_returns(db_path, start, end)
    except Exception as exc:
        logger.error("Failed to load returns: {}", exc)
        return 2

    # Build strategy config
    try:
        ms_config = _HYPOTHESIS_BUILDERS[hypothesis](params)
    except Exception as exc:
        logger.error("Failed to build hypothesis config: {}", exc)
        return 2

    wf_config = MultiStrategyWalkForwardConfig(
        multi_strategy_config=ms_config,
        is_window=is_window,
        oos_window=oos_window,
        step_size=step_size,
        name=hypothesis,
    )

    # Run walk-forward analysis
    logger.info("Running walk-forward backtest ({} × {} day folds)...", "expanding" if wf_config.expanding else "rolling", is_window)
    try:
        analyzer = MultiStrategyWalkForwardAnalyzer()
        result = analyzer.run(returns, wf_config)
    except Exception as exc:
        logger.error("Walk-forward analysis failed: {}", exc)
        return 2

    logger.info(
        "Done: {} folds | OOS Sharpe={:.3f} | MaxDD={:.2%} | WFE={:.3f}",
        result.n_folds,
        result.oos_sharpe,
        abs(result.oos_max_drawdown),
        result.mean_wfe,
    )

    # CRO gate check
    gate = check_cro_gates(result, hypothesis, params, job_name)

    if gate.passes:
        logger.info("CRO gates: ALL PASSED")
    else:
        logger.warning("CRO gates: FAILED — {}", "; ".join(gate.failures))

    # Save results
    results_path.mkdir(parents=True, exist_ok=True)
    out_file = results_path / f"{job_name}.json"
    gate_dict = asdict(gate)
    out_file.write_text(json.dumps(gate_dict, indent=2, default=str))
    logger.info("Results written to {}", out_file)

    # Also write a summary to stdout for K8s log capture
    summary = {
        "hypothesis": gate.hypothesis,
        "job_name": gate.job_name,
        "oos_sharpe": gate.oos_sharpe,
        "profit_factor": gate.profit_factor,
        "max_drawdown": gate.max_drawdown,
        "wfe": gate.wfe,
        "passes": gate.passes,
        "failures": gate.failures,
    }
    print(json.dumps(summary))

    # Push metrics to Prometheus Pushgateway
    if pushgateway:
        push_metrics(gate, pushgateway)

    return 0 if gate.passes else 1


if __name__ == "__main__":
    sys.exit(main())
