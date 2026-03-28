#!/usr/bin/env python3
"""QUA-71b: Portfolio-level A/B test — EMA-IC vs Bayesian NormalGamma combiner.

Supersedes QUA-69 (single-symbol signal test).  Uses the same portfolio-level
walk-forward framework as QUA-58 / run_mvp_backtest.py to evaluate signal
combiners at the portfolio level against CRO-approved gates.

Config (matches QUA-58 runE_baseline_90_30):
  - Universe: same 50-symbol representative subset as QUA-49/58
  - WF: 90-day IS min / 30-day OOS / expanding windows / 64 folds
  - Commission: 10 bps one-way

Variants:
  runA_baseline_ema_ic  — AdaptiveSignalCombiner (EMA-IC, approved baseline)
  runB_bayesian_ng      — BayesianAdaptiveSignalCombiner (NormalGamma posterior IC)

The ONLY difference between runA and runB is the IC estimator used in the
adaptive_combined sleeve.  All signals, portfolio optimizer, regime config,
capital weights, and walk-forward parameters are identical.

CRO gates (CEO-approved, QUA-22):
  OOS Sharpe >= 0.60 | PF >= 1.10 | MaxDD < 20% | WFE >= 0.20

CRO acceptance criteria (from plans/2026-03-28-CRO-response-QUA71-correction.md):
  Bayesian must meet all four gates independently (not just beat baseline).
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import quant_rs as _qrs
from loguru import logger

from quant.backtest.multi_strategy import MultiStrategyConfig, SleeveConfig
from quant.backtest.multi_strategy_walk_forward import (
    MultiStrategyWalkForwardAnalyzer,
    MultiStrategyWalkForwardConfig,
    MultiStrategyWalkForwardResult,
)
from quant.portfolio.alpha import CombinationMethod
from quant.portfolio.engine import PortfolioConfig, PortfolioConstraints
from quant.portfolio.optimizers import OptimizationMethod
from quant.signals.adaptive_combiner import (
    AdaptiveCombinerConfig,
    BayesianAdaptiveCombinerConfig,
)
from quant.signals.base import BaseSignal, SignalOutput
from quant.signals.regime import RegimeConfig, RegimeWeightAdapter

# ── Constants ────────────────────────────────────────────────────────────────

DATA_START = date(2020, 1, 1)
DATA_END = date(2025, 12, 31)

# Same DB as run_mvp_backtest.py
DB_PATH = str(Path.home() / ".quant" / "mvp_backtest.duckdb")

RESULTS_DIR = (
    Path(__file__).parent.parent.parent
    / "backtest-results"
    / "qua71b-bayesian-portfolio"
)

# Same 50-symbol representative subset as QUA-49/58
_BACKTEST_SYMS: list[str] = [
    # Information Technology (8)
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "ORCL", "AMD",
    # Consumer Discretionary (5)
    "AMZN", "TSLA", "HD", "MCD", "NKE",
    # Consumer Staples (4)
    "WMT", "PG", "KO", "PEP",
    # Financials (6)
    "JPM", "BAC", "GS", "V", "MA", "BLK",
    # Health Care (6)
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "TMO",
    # Industrials (5)
    "HON", "GE", "CAT", "DE", "UPS",
    # Energy (4)
    "XOM", "CVX", "COP", "EOG",
    # Utilities (3)
    "NEE", "DUK", "SO",
    # Real Estate (3)
    "AMT", "PLD", "CCI",
    # Materials (3)
    "LIN", "SHW", "APD",
    # Communication Services (3)
    "NFLX", "DIS", "TMUS",
]

# ── Signal classes (identical to run_mvp_backtest.py) ────────────────────────


class MomentumSignalFromReturns(BaseSignal):
    """MomentumSignal: RSI-based momentum from cumulative price index."""

    def __init__(
        self,
        rsi_period: int = 14,
        lookback: int = 5,
        return_scale: float = 0.05,
    ) -> None:
        self._rsi_period = rsi_period
        self._lookback = lookback
        self._return_scale = return_scale

    @property
    def name(self) -> str:
        return f"momentum_rsi{self._rsi_period}"

    @property
    def required_features(self) -> list[str]:
        return ["returns"]

    def compute(
        self,
        symbol: str,
        features: dict[str, pd.Series],
        timestamp: datetime,
    ) -> SignalOutput:
        returns = features["returns"].fillna(0.0)
        if len(returns) < self._rsi_period + self._lookback + 5:
            return SignalOutput(
                symbol=symbol, timestamp=timestamp,
                score=0.0, confidence=0.0, target_position=0.0,
                metadata={"reason": "insufficient_data"},
            )
        prices = ((1.0 + returns).cumprod() * 100.0).tolist()
        rsi_values = _qrs.features.rsi(prices, self._rsi_period)
        returns_list = [float(r) for r in returns]
        score, confidence, target_position = _qrs.signals.momentum_signal(
            rsi_values, returns_list, self._lookback, self._return_scale
        )
        return SignalOutput(
            symbol=symbol, timestamp=timestamp,
            score=score, confidence=confidence,
            target_position=target_position,
            metadata={},
        )


class TrendFollowingSignalFromReturns(BaseSignal):
    """TrendFollowingSignal: MACD + SMA from cumulative price index."""

    def __init__(self, fast_ma: int = 20, slow_ma: int = 50) -> None:
        if fast_ma >= slow_ma:
            raise ValueError("fast_ma must be < slow_ma")
        self._fast_ma = fast_ma
        self._slow_ma = slow_ma

    @property
    def name(self) -> str:
        return f"trend_ma{self._fast_ma}_{self._slow_ma}"

    @property
    def required_features(self) -> list[str]:
        return ["returns"]

    def compute(
        self,
        symbol: str,
        features: dict[str, pd.Series],
        timestamp: datetime,
    ) -> SignalOutput:
        returns = features["returns"].fillna(0.0)
        min_len = self._slow_ma + 30
        if len(returns) < min_len:
            return SignalOutput(
                symbol=symbol, timestamp=timestamp,
                score=0.0, confidence=0.0, target_position=0.0,
                metadata={"reason": "insufficient_data"},
            )
        prices = ((1.0 + returns).cumprod() * 100.0).tolist()
        macd_hist = _qrs.features.macd_histogram(prices, 12, 26, 9)
        fast_ma = _qrs.features.rolling_mean(prices, self._fast_ma)
        slow_ma = _qrs.features.rolling_mean(prices, self._slow_ma)
        score, confidence, target_position = _qrs.signals.trend_following_signal(
            macd_hist, fast_ma, slow_ma
        )
        return SignalOutput(
            symbol=symbol, timestamp=timestamp,
            score=score, confidence=confidence,
            target_position=target_position,
            metadata={},
        )


# ── Config helpers ────────────────────────────────────────────────────────────


def _portfolio_config() -> PortfolioConfig:
    return PortfolioConfig(
        optimization_method=OptimizationMethod.MEAN_VARIANCE,
        constraints=PortfolioConstraints(
            long_only=True,
            max_weight=0.05,
            max_gross_exposure=0.6,
        ),
        rebalance_threshold=0.01,
        cov_lookback_days=252,
        optimizer_kwargs={"risk_aversion": 5.0},
    )


def _regime_config() -> RegimeConfig:
    return RegimeConfig(
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
    )


def _adaptive_config_ema() -> AdaptiveCombinerConfig:
    """Standard EMA-IC combiner config (approved baseline)."""
    return AdaptiveCombinerConfig(
        ic_lookback=126,
        min_ic_periods=20,
        min_ic=0.0,
        ic_halflife=21,
        shrinkage=0.3,
        min_assets=3,
    )


def _adaptive_config_bayesian() -> BayesianAdaptiveCombinerConfig:
    """Bayesian NormalGamma IC combiner config (QUA-71b test variant)."""
    return BayesianAdaptiveCombinerConfig(
        ic_lookback=126,
        min_ic_periods=20,
        min_ic=0.0,
        ic_halflife=21,   # retained for fallback EWM path in parent; NG ignores it
        shrinkage=0.3,
        min_assets=3,
    )


def _make_wf_config(
    ms_config: MultiStrategyConfig,
    name: str,
) -> MultiStrategyWalkForwardConfig:
    """Expanding walk-forward: IS=252 min, OOS=63, step=63.

    Matches QUA-58 runE_baseline_90_30 expanding config.
    """
    return MultiStrategyWalkForwardConfig(
        multi_strategy_config=ms_config,
        is_window=252,
        oos_window=63,
        step_size=63,
        expanding=True,
        name=name,
    )


# ── Data loading ──────────────────────────────────────────────────────────────


def load_returns() -> pd.DataFrame:
    """Load returns from the existing MVP backtest DuckDB (no re-ingest)."""
    if not Path(DB_PATH).exists():
        raise FileNotFoundError(
            f"MVP backtest DB not found at {DB_PATH}. "
            "Run run_mvp_backtest.py first to ingest data."
        )

    conn = duckdb.connect(DB_PATH, read_only=True)
    try:
        syms_upper = [s.upper() for s in _BACKTEST_SYMS]
        placeholders = ", ".join(f"'{s}'" for s in syms_upper)
        long_df = conn.execute(
            f"""
            SELECT symbol, date, adj_close
            FROM ohlcv
            WHERE symbol IN ({placeholders})
              AND date >= ?
              AND date <= ?
            ORDER BY date, symbol
            """,
            [DATA_START, DATA_END],
        ).df()
    finally:
        conn.close()

    if long_df.empty:
        raise RuntimeError("No price data loaded from database")

    long_df["date"] = pd.to_datetime(long_df["date"])
    prices = long_df.pivot(index="date", columns="symbol", values="adj_close")
    prices = prices.sort_index()

    coverage_counts = prices.notna().sum()
    prices = prices[coverage_counts[coverage_counts >= 252].index]

    returns = prices.pct_change().dropna(how="all")
    coverage = returns.notna().mean()
    returns = returns[coverage[coverage >= 0.80].index]

    avail = [s for s in _BACKTEST_SYMS if s in returns.columns]
    if len(avail) < 20:
        avail = list(returns.columns)
    returns = returns[avail]

    logger.info(
        "Loaded returns: {} symbols, {} trading days ({} to {})",
        len(returns.columns),
        len(returns),
        returns.index[0].date(),
        returns.index[-1].date(),
    )
    return returns


# ── Backtest runs ─────────────────────────────────────────────────────────────


def _build_ensemble_config(
    name: str,
    combiner_config: AdaptiveCombinerConfig,
) -> MultiStrategyConfig:
    """Build the full ensemble config, varying only the combiner.

    Mirrors run2_full_ensemble from run_mvp_backtest.py:
    3 sleeves (momentum 40%, trend 35%, adaptive_combined 25%),
    regime tilt 30%, commission 10 bps, min_history 100.
    """
    return MultiStrategyConfig(
        sleeves=[
            SleeveConfig(
                name="momentum_us_equity",
                signals=[MomentumSignalFromReturns(rsi_period=14, lookback=5, return_scale=0.05)],
                capital_weight=0.40,
                strategy_type="momentum",
                portfolio_config=_portfolio_config(),
                combination_method=CombinationMethod.EQUAL_WEIGHT,
            ),
            SleeveConfig(
                name="trend_following_us_equity",
                signals=[TrendFollowingSignalFromReturns(fast_ma=20, slow_ma=50)],
                capital_weight=0.35,
                strategy_type="trend",
                portfolio_config=_portfolio_config(),
                combination_method=CombinationMethod.EQUAL_WEIGHT,
            ),
            SleeveConfig(
                name="adaptive_combined",
                signals=[
                    MomentumSignalFromReturns(rsi_period=14, lookback=5, return_scale=0.05),
                    TrendFollowingSignalFromReturns(fast_ma=20, slow_ma=50),
                ],
                capital_weight=0.25,
                strategy_type="momentum",
                portfolio_config=_portfolio_config(),
                combination_method=CombinationMethod.RANK_WEIGHTED,
                adaptive_combiner_config=combiner_config,
            ),
        ],
        rebalance_frequency=21,
        commission_bps=10.0,
        initial_capital=1_000_000.0,
        regime_config=_regime_config(),
        regime_adapter=RegimeWeightAdapter(max_tilt=0.30),
        regime_lookback_days=252,
        min_history=100,
        name=name,
    )


# ── Validation ────────────────────────────────────────────────────────────────


@dataclass
class ValidationResult:
    run_name: str
    oos_sharpe: float
    profit_factor: float
    wf_efficiency: float
    max_drawdown: float
    n_folds: int
    passes: bool
    failures: list[str]


def validate_result(result: MultiStrategyWalkForwardResult, run_name: str) -> ValidationResult:
    failures = []

    oos_sharpe = result.oos_sharpe
    max_drawdown = abs(result.oos_max_drawdown)

    oos_returns = result.oos_returns.dropna()
    gains = oos_returns[oos_returns > 0].sum()
    losses = abs(oos_returns[oos_returns < 0].sum())
    profit_factor = gains / losses if losses > 1e-12 else float("inf")

    wf_efficiency = result.mean_wfe

    if oos_sharpe < 0.60:
        failures.append(f"OOS Sharpe {oos_sharpe:.3f} < 0.60")
    if profit_factor < 1.10:
        failures.append(f"Profit factor {profit_factor:.3f} < 1.10")
    if wf_efficiency < 0.20:
        failures.append(f"WF efficiency {wf_efficiency:.3f} < 0.20")
    if max_drawdown >= 0.20:
        failures.append(f"Max drawdown {max_drawdown:.2%} >= 20%")

    return ValidationResult(
        run_name=run_name,
        oos_sharpe=oos_sharpe,
        profit_factor=profit_factor,
        wf_efficiency=wf_efficiency,
        max_drawdown=max_drawdown,
        n_folds=result.n_folds,
        passes=len(failures) == 0,
        failures=failures,
    )


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<8} | {message}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    logger.info("QUA-71b run_id={}", run_id)

    # ── Load returns ─────────────────────────────────────────────────
    logger.info("Loading returns from {}", DB_PATH)
    try:
        returns = load_returns()
    except Exception as e:
        logger.error("Failed to load returns: {}", e)
        return 1

    logger.info(
        "Universe: {} symbols, {} trading days", len(returns.columns), len(returns)
    )

    results: dict[str, MultiStrategyWalkForwardResult] = {}

    # ── runA: Baseline EMA-IC (approved, mirrors run2_full_ensemble) ──
    logger.info("=== runA_baseline_ema_ic — EMA-IC AdaptiveSignalCombiner ===")
    try:
        ms_config_a = _build_ensemble_config("runA_baseline_ema_ic", _adaptive_config_ema())
        analyzer_a = MultiStrategyWalkForwardAnalyzer()
        results["runA_baseline_ema_ic"] = analyzer_a.run(
            returns, _make_wf_config(ms_config_a, "runA_baseline_ema_ic")
        )
        logger.info("runA complete: {} folds", results["runA_baseline_ema_ic"].n_folds)
    except Exception as e:
        logger.error("runA failed: {}", e)
        return 1

    # Checkpoint
    _checkpoint(run_id, "runA", results["runA_baseline_ema_ic"])

    # ── runB: Bayesian NormalGamma IC ─────────────────────────────────
    logger.info("=== runB_bayesian_ng — BayesianAdaptiveSignalCombiner (NormalGamma) ===")
    try:
        ms_config_b = _build_ensemble_config("runB_bayesian_ng", _adaptive_config_bayesian())
        analyzer_b = MultiStrategyWalkForwardAnalyzer()
        results["runB_bayesian_ng"] = analyzer_b.run(
            returns, _make_wf_config(ms_config_b, "runB_bayesian_ng")
        )
        logger.info("runB complete: {} folds", results["runB_bayesian_ng"].n_folds)
    except Exception as e:
        logger.error("runB failed: {}", e)
        return 1

    # ── Validate ──────────────────────────────────────────────────────
    validations = [
        validate_result(results[name], name) for name in ("runA_baseline_ema_ic", "runB_bayesian_ng")
    ]

    # ── Report ────────────────────────────────────────────────────────
    _print_report(validations)

    # ── Save results ──────────────────────────────────────────────────
    output_path = RESULTS_DIR / f"results_qua71b_{run_id}.json"
    output = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "framework": "QUA-71b portfolio A/B test",
        "wf_config": {
            "is_window": 252,
            "oos_window": 63,
            "step_size": 63,
            "expanding": True,
        },
        "cro_gates": {
            "oos_sharpe_min": 0.60,
            "profit_factor_min": 1.10,
            "max_drawdown_max": 0.20,
            "wfe_min": 0.20,
        },
        "runs": {
            v.run_name: {
                "oos_sharpe": round(v.oos_sharpe, 4),
                "profit_factor": round(v.profit_factor, 4),
                "wf_efficiency": round(v.wf_efficiency, 4),
                "max_drawdown": round(v.max_drawdown, 4),
                "n_folds": v.n_folds,
                "passes": v.passes,
                "failures": v.failures,
            }
            for v in validations
        },
        "delta_bayesian_vs_baseline": _compute_delta(validations),
    }
    output_path.write_text(json.dumps(output, indent=2))
    logger.info("Results saved: {}", output_path)

    # Exit 0 if Bayesian passes all gates, 1 otherwise
    bayesian_val = next(v for v in validations if "bayesian" in v.run_name)
    if bayesian_val.passes:
        logger.info("QUA-71b PASS — Bayesian combiner meets all CRO gates")
        return 0
    else:
        logger.warning("QUA-71b FAIL — Bayesian combiner does not meet all CRO gates")
        logger.warning("Failures: {}", bayesian_val.failures)
        return 2


def _checkpoint(run_id: str, label: str, result: MultiStrategyWalkForwardResult) -> None:
    try:
        p = RESULTS_DIR / f"partial_{label}_{run_id}.json"
        p.write_text(json.dumps({
            "run_id": run_id,
            "label": label,
            "oos_sharpe": result.oos_sharpe,
            "n_folds": result.n_folds,
        }))
        logger.info("Checkpoint saved: {}", p)
    except Exception as exc:
        logger.warning("Could not save checkpoint: {}", exc)


def _compute_delta(validations: list[ValidationResult]) -> dict[str, float]:
    if len(validations) < 2:
        return {}
    base = next((v for v in validations if "baseline" in v.run_name), None)
    bayes = next((v for v in validations if "bayesian" in v.run_name), None)
    if base is None or bayes is None:
        return {}
    return {
        "sharpe_delta": round(bayes.oos_sharpe - base.oos_sharpe, 4),
        "pf_delta": round(bayes.profit_factor - base.profit_factor, 4),
        "max_drawdown_delta": round(bayes.max_drawdown - base.max_drawdown, 4),
        "wfe_delta": round(bayes.wf_efficiency - base.wf_efficiency, 4),
    }


def _print_report(validations: list[ValidationResult]) -> None:
    logger.info("")
    logger.info("=" * 72)
    logger.info("QUA-71b RESULTS — EMA-IC vs Bayesian NormalGamma (portfolio level)")
    logger.info("=" * 72)
    logger.info(
        "{:<30} {:>8} {:>8} {:>8} {:>8}  {}",
        "Run", "Sharpe", "PF", "WFE", "MaxDD", "Status",
    )
    logger.info("-" * 72)
    for v in validations:
        status = "PASS" if v.passes else "FAIL"
        logger.info(
            "{:<30} {:>8.3f} {:>8.3f} {:>8.3f} {:>7.2%}  {} [{}]",
            v.run_name, v.oos_sharpe, v.profit_factor,
            v.wf_efficiency, v.max_drawdown, status,
            ", ".join(v.failures) if v.failures else "all gates met",
        )
    logger.info("=" * 72)

    delta = _compute_delta(validations)
    if delta:
        logger.info("Delta (Bayesian − Baseline):")
        logger.info(
            "  Sharpe {:+.3f}  PF {:+.3f}  MaxDD {:+.2%}  WFE {:+.3f}",
            delta["sharpe_delta"], delta["pf_delta"],
            delta["max_drawdown_delta"], delta["wfe_delta"],
        )
    logger.info("")


if __name__ == "__main__":
    sys.exit(main())
