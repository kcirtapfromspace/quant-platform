#!/usr/bin/env python3
"""Signal expansion backtest: add MeanReversion + VolatilitySignal sleeves.

Implements QUA-85 — Phase 2 signal expansion approved in QUA-76.
Adds orthogonal alpha sources to close the PF gap (1.23 -> 1.30+ target).

Sleeves:
  1. momentum_us_equity (35%) — RSI momentum
  2. trend_following_us_equity (30%) — MACD + SMA trend
  3. mean_reversion_us_equity (15%) — Bollinger Band z-score (NEW)
  4. adaptive_combined (20%) — Momentum + Trend + Volatility + ReturnQuality

WF: IS=90, OOS=30, expanding=True, step_size=30
Commission: 10 bps one-way
MVO optimizer, max_gross_exposure=0.6
min_history=100
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
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
from quant.signals.adaptive_combiner import AdaptiveCombinerConfig
from quant.signals.base import BaseSignal, SignalOutput
from quant.signals.factors import ReturnQualitySignal, VolatilitySignal
from quant.signals.regime import RegimeConfig, RegimeWeightAdapter

# ── Constants ────────────────────────────────────────────────────────────────

DATA_START = date(2018, 1, 1)
DATA_END = date(2025, 12, 31)
DB_PATH = str(Path.home() / ".quant" / "universe_v2.duckdb")
RESULTS_DIR = Path.home() / ".quant" / "backtest-results" / "signal-expansion"

# Representative 50-symbol backtest universe (same as QUA-49/QUA-58)
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

_SECTOR_MAP: dict[str, str] = {
    **{s: "Information Technology" for s in ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "ORCL", "AMD"]},
    **{s: "Consumer Discretionary" for s in ["AMZN", "TSLA", "HD", "MCD", "NKE"]},
    **{s: "Consumer Staples" for s in ["WMT", "PG", "KO", "PEP"]},
    **{s: "Financials" for s in ["JPM", "BAC", "GS", "V", "MA", "BLK"]},
    **{s: "Health Care" for s in ["UNH", "JNJ", "LLY", "ABBV", "MRK", "TMO"]},
    **{s: "Industrials" for s in ["HON", "GE", "CAT", "DE", "UPS"]},
    **{s: "Energy" for s in ["XOM", "CVX", "COP", "EOG"]},
    **{s: "Utilities" for s in ["NEE", "DUK", "SO"]},
    **{s: "Real Estate" for s in ["AMT", "PLD", "CCI"]},
    **{s: "Materials" for s in ["LIN", "SHW", "APD"]},
    **{s: "Communication Services" for s in ["NFLX", "DIS", "TMUS"]},
}


# ── Signal wrappers (compute features from returns only) ─────────────────────


class MomentumSignalFromReturns(BaseSignal):
    """MomentumSignal compatible with the backtest engine (receives only 'returns')."""

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
            target_position=target_position, metadata={},
        )


class TrendFollowingSignalFromReturns(BaseSignal):
    """TrendFollowingSignal compatible with the backtest engine (receives only 'returns')."""

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
            target_position=target_position, metadata={},
        )


class MeanReversionSignalFromReturns(BaseSignal):
    """MeanReversionSignal that computes Bollinger Bands from returns.

    Reconstructs a price index via (1+r).cumprod(), then computes BB mid/upper/lower
    using the Rust feature kernels, and calls the mean_reversion_signal Rust kernel.
    """

    def __init__(self, bb_period: int = 20, num_std: float = 2.0) -> None:
        self._bb_period = bb_period
        self._num_std = num_std

    @property
    def name(self) -> str:
        return f"mean_reversion_bb{self._bb_period}"

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
        min_len = self._bb_period + 10
        if len(returns) < min_len:
            return SignalOutput(
                symbol=symbol, timestamp=timestamp,
                score=0.0, confidence=0.0, target_position=0.0,
                metadata={"reason": "insufficient_data"},
            )

        prices = ((1.0 + returns).cumprod() * 100.0).tolist()
        bb_mid = _qrs.features.bb_mid(prices, self._bb_period)
        bb_upper = _qrs.features.bb_upper(prices, self._bb_period, self._num_std)
        bb_lower = _qrs.features.bb_lower(prices, self._bb_period, self._num_std)
        returns_list = [float(r) for r in returns]

        score, confidence, target_position = _qrs.signals.mean_reversion_signal(
            bb_mid, bb_upper, bb_lower, returns_list, self._num_std
        )
        return SignalOutput(
            symbol=symbol, timestamp=timestamp,
            score=score, confidence=confidence,
            target_position=target_position, metadata={},
        )


# ── Config helpers ───────────────────────────────────────────────────────────


def _portfolio_config(risk_aversion: float = 5.0) -> PortfolioConfig:
    return PortfolioConfig(
        optimization_method=OptimizationMethod.MEAN_VARIANCE,
        constraints=PortfolioConstraints(
            long_only=True,
            max_weight=0.05,
            max_gross_exposure=0.6,
        ),
        rebalance_threshold=0.01,
        cov_lookback_days=252,
        optimizer_kwargs={"risk_aversion": risk_aversion},
    )


def _adaptive_config() -> AdaptiveCombinerConfig:
    return AdaptiveCombinerConfig(
        ic_lookback=126,
        min_ic_periods=20,
        min_ic=0.0,
        ic_halflife=21,
        shrinkage=0.3,
        min_assets=3,
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


# ── Data loading ─────────────────────────────────────────────────────────────


def load_returns(db_path: str, symbols: list[str]) -> pd.DataFrame:
    """Load adj_close prices from DuckDB and compute daily returns."""
    logger.info("Loading price data for {} symbols", len(symbols))
    conn = duckdb.connect(db_path, read_only=True)
    try:
        placeholders = ", ".join(f"'{s}'" for s in symbols)
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

    returns = prices.pct_change()
    returns = returns.dropna(how="all")

    coverage = returns.notna().mean()
    good_cols = coverage[coverage >= 0.80].index
    returns = returns[good_cols]

    logger.info(
        "Loaded returns: {} symbols, {} trading days ({} to {})",
        len(returns.columns), len(returns),
        returns.index[0].date(), returns.index[-1].date(),
    )
    return returns


# ── Validation ───────────────────────────────────────────────────────────────


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


def validate_result(
    result: MultiStrategyWalkForwardResult,
    run_name: str,
) -> ValidationResult:
    """Check CRO validation gates (QUA-22 thresholds)."""
    failures = []

    oos_sharpe = result.oos_sharpe
    max_drawdown = abs(result.oos_max_drawdown)

    oos_returns = result.oos_returns.dropna()
    gains = oos_returns[oos_returns > 0].sum()
    losses = abs(oos_returns[oos_returns < 0].sum())
    profit_factor = gains / losses if losses > 1e-12 else float("inf")

    wf_efficiency = result.mean_wfe

    if oos_sharpe < 0.60:
        failures.append(f"OOS Sharpe {oos_sharpe:.2f} < 0.60")
    if profit_factor < 1.10:
        failures.append(f"Profit factor {profit_factor:.2f} < 1.10")
    if wf_efficiency < 0.20:
        failures.append(f"WF efficiency {wf_efficiency:.2f} < 0.20")
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


def format_validation_table(validations: list[ValidationResult]) -> str:
    lines = [
        "| Run | Sharpe | PF | WFE | Max DD | Folds | Status |",
        "|-----|--------|-----|-----|--------|-------|--------|",
    ]
    for v in validations:
        status = "PASS" if v.passes else "FAIL"
        lines.append(
            f"| {v.run_name:<40} | {v.oos_sharpe:>6.3f} | {v.profit_factor:>5.2f} | "
            f"{v.wf_efficiency:>5.3f} | {v.max_drawdown:>6.2%} | {v.n_folds:>5} | {status} |"
        )
    return "\n".join(lines)


# ── Backtest runs ────────────────────────────────────────────────────────────


def run_baseline(
    returns: pd.DataFrame,
    sector_map: dict[str, str],
) -> MultiStrategyWalkForwardResult:
    """Baseline: original 3-sleeve MVP ensemble (Run 2 from QUA-49) for comparison."""
    logger.info("=== Baseline: MVP 3-sleeve ensemble (momentum/trend/adaptive) ===")
    ms_config = MultiStrategyConfig(
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
                adaptive_combiner_config=_adaptive_config(),
            ),
        ],
        rebalance_frequency=21,
        commission_bps=10.0,
        initial_capital=1_000_000.0,
        sector_map=sector_map,
        regime_config=_regime_config(),
        regime_adapter=RegimeWeightAdapter(max_tilt=0.30),
        regime_lookback_days=252,
        min_history=100,
        name="baseline_mvp_ensemble",
    )
    analyzer = MultiStrategyWalkForwardAnalyzer()
    wf_config = MultiStrategyWalkForwardConfig(
        multi_strategy_config=ms_config,
        is_window=90,
        oos_window=30,
        step_size=30,
        expanding=True,
        name="baseline_mvp_wf",
    )
    return analyzer.run(returns, wf_config)


def run_signal_expansion(
    returns: pd.DataFrame,
    sector_map: dict[str, str],
) -> MultiStrategyWalkForwardResult:
    """Signal expansion: 4-sleeve ensemble with MeanReversion + enhanced adaptive."""
    logger.info("=== Signal Expansion: 4-sleeve ensemble ===")
    ms_config = MultiStrategyConfig(
        sleeves=[
            SleeveConfig(
                name="momentum_us_equity",
                signals=[MomentumSignalFromReturns(rsi_period=14, lookback=5, return_scale=0.05)],
                capital_weight=0.35,
                strategy_type="momentum",
                portfolio_config=_portfolio_config(),
                combination_method=CombinationMethod.EQUAL_WEIGHT,
            ),
            SleeveConfig(
                name="trend_following_us_equity",
                signals=[TrendFollowingSignalFromReturns(fast_ma=20, slow_ma=50)],
                capital_weight=0.30,
                strategy_type="trend",
                portfolio_config=_portfolio_config(),
                combination_method=CombinationMethod.EQUAL_WEIGHT,
            ),
            SleeveConfig(
                name="mean_reversion_us_equity",
                signals=[MeanReversionSignalFromReturns(bb_period=20, num_std=2.0)],
                capital_weight=0.15,
                strategy_type="mean_reversion",
                portfolio_config=_portfolio_config(),
                combination_method=CombinationMethod.EQUAL_WEIGHT,
            ),
            SleeveConfig(
                name="adaptive_combined",
                signals=[
                    MomentumSignalFromReturns(rsi_period=14, lookback=5, return_scale=0.05),
                    TrendFollowingSignalFromReturns(fast_ma=20, slow_ma=50),
                    VolatilitySignal(period=20, annualise=True, low_vol=0.12, high_vol=0.40),
                    ReturnQualitySignal(period=60, sharpe_cap=3.0),
                ],
                capital_weight=0.20,
                strategy_type="momentum",
                portfolio_config=_portfolio_config(),
                combination_method=CombinationMethod.RANK_WEIGHTED,
                adaptive_combiner_config=_adaptive_config(),
            ),
        ],
        rebalance_frequency=21,
        commission_bps=10.0,
        initial_capital=1_000_000.0,
        sector_map=sector_map,
        regime_config=_regime_config(),
        regime_adapter=RegimeWeightAdapter(max_tilt=0.30),
        regime_lookback_days=252,
        min_history=100,
        name="signal_expansion_ensemble",
    )
    analyzer = MultiStrategyWalkForwardAnalyzer()
    wf_config = MultiStrategyWalkForwardConfig(
        multi_strategy_config=ms_config,
        is_window=90,
        oos_window=30,
        step_size=30,
        expanding=True,
        name="signal_expansion_wf",
    )
    return analyzer.run(returns, wf_config)


def run_mean_reversion_standalone(
    returns: pd.DataFrame,
    sector_map: dict[str, str],
) -> MultiStrategyWalkForwardResult:
    """MeanReversion standalone — isolate signal contribution."""
    logger.info("=== MeanReversion standalone ===")
    ms_config = MultiStrategyConfig(
        sleeves=[
            SleeveConfig(
                name="mean_reversion_us_equity",
                signals=[MeanReversionSignalFromReturns(bb_period=20, num_std=2.0)],
                capital_weight=1.0,
                strategy_type="mean_reversion",
                portfolio_config=_portfolio_config(),
                combination_method=CombinationMethod.EQUAL_WEIGHT,
            ),
        ],
        rebalance_frequency=21,
        commission_bps=10.0,
        initial_capital=1_000_000.0,
        sector_map=sector_map,
        min_history=100,
        name="mean_reversion_standalone",
    )
    analyzer = MultiStrategyWalkForwardAnalyzer()
    wf_config = MultiStrategyWalkForwardConfig(
        multi_strategy_config=ms_config,
        is_window=90,
        oos_window=30,
        step_size=30,
        expanding=True,
        name="mean_reversion_standalone_wf",
    )
    return analyzer.run(returns, wf_config)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<8} | {message}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────
    if not Path(DB_PATH).exists():
        logger.error("Database not found: {}", DB_PATH)
        return 1

    all_returns = load_returns(DB_PATH, _BACKTEST_SYMS)
    avail = [s for s in _BACKTEST_SYMS if s in all_returns.columns]
    if len(avail) < 20:
        avail = list(all_returns.columns)
    returns = all_returns[avail]
    logger.info("Backtest universe: {} symbols", len(returns.columns))

    sector_map = {s: _SECTOR_MAP.get(s, "Unknown") for s in returns.columns}

    validations: list[ValidationResult] = []
    all_results: dict[str, dict] = {}

    # ── Run A: MeanReversion standalone ────────────────────────────────
    logger.info("Step 1/3: MeanReversion standalone")
    try:
        mr_result = run_mean_reversion_standalone(returns, sector_map)
        v_mr = validate_result(mr_result, "mean_reversion_standalone")
        validations.append(v_mr)
        all_results["mean_reversion_standalone"] = {
            "oos_sharpe": v_mr.oos_sharpe,
            "profit_factor": v_mr.profit_factor,
            "wf_efficiency": v_mr.wf_efficiency,
            "max_drawdown": v_mr.max_drawdown,
            "n_folds": v_mr.n_folds,
            "passes": v_mr.passes,
        }
        logger.info(
            "MeanReversion standalone: Sharpe={:.3f} PF={:.2f} WFE={:.3f} DD={:.2%}",
            v_mr.oos_sharpe, v_mr.profit_factor, v_mr.wf_efficiency, v_mr.max_drawdown,
        )
    except Exception as e:
        logger.error("MeanReversion standalone failed: {}", e)
        all_results["mean_reversion_standalone"] = {"error": str(e)}

    # ── Run B: Baseline (original 3-sleeve) ───────────────────────────
    logger.info("Step 2/3: Baseline 3-sleeve ensemble")
    try:
        baseline_result = run_baseline(returns, sector_map)
        v_base = validate_result(baseline_result, "baseline_mvp_ensemble")
        validations.append(v_base)
        all_results["baseline_mvp_ensemble"] = {
            "oos_sharpe": v_base.oos_sharpe,
            "profit_factor": v_base.profit_factor,
            "wf_efficiency": v_base.wf_efficiency,
            "max_drawdown": v_base.max_drawdown,
            "n_folds": v_base.n_folds,
            "passes": v_base.passes,
        }
        logger.info(
            "Baseline: Sharpe={:.3f} PF={:.2f} WFE={:.3f} DD={:.2%}",
            v_base.oos_sharpe, v_base.profit_factor, v_base.wf_efficiency, v_base.max_drawdown,
        )
    except Exception as e:
        logger.error("Baseline failed: {}", e)
        all_results["baseline_mvp_ensemble"] = {"error": str(e)}

    # ── Run C: Signal expansion (4-sleeve) ────────────────────────────
    logger.info("Step 3/3: Signal expansion 4-sleeve ensemble")
    try:
        expansion_result = run_signal_expansion(returns, sector_map)
        v_exp = validate_result(expansion_result, "signal_expansion_ensemble")
        validations.append(v_exp)
        all_results["signal_expansion_ensemble"] = {
            "oos_sharpe": v_exp.oos_sharpe,
            "profit_factor": v_exp.profit_factor,
            "wf_efficiency": v_exp.wf_efficiency,
            "max_drawdown": v_exp.max_drawdown,
            "n_folds": v_exp.n_folds,
            "passes": v_exp.passes,
        }
        logger.info(
            "Signal expansion: Sharpe={:.3f} PF={:.2f} WFE={:.3f} DD={:.2%}",
            v_exp.oos_sharpe, v_exp.profit_factor, v_exp.wf_efficiency, v_exp.max_drawdown,
        )
    except Exception as e:
        logger.error("Signal expansion failed: {}", e)
        all_results["signal_expansion_ensemble"] = {"error": str(e)}

    # ── Report ────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("QUA-85 Signal Expansion Backtest Results")
    logger.info("=" * 80)

    if validations:
        table = format_validation_table(validations)
        logger.info("\n{}", table)

    # Save results to JSON
    results_file = RESULTS_DIR / "results.json"
    results_file.write_text(json.dumps(all_results, indent=2, default=str))
    logger.info("Results saved to {}", results_file)

    # Save validation table as markdown
    if validations:
        md_file = RESULTS_DIR / "validation_table.md"
        md_file.write_text(
            f"# QUA-85 Signal Expansion Results\n\n"
            f"Date: {datetime.now(timezone.utc).isoformat()}\n\n"
            f"## CRO Gate Metrics\n\n"
            f"{format_validation_table(validations)}\n\n"
            f"## CRO Thresholds\n\n"
            f"| Gate | Target |\n|------|--------|\n"
            f"| Sharpe | >= 0.60 |\n"
            f"| PF | >= 1.10 (aspiration: 1.30) |\n"
            f"| MaxDD | < 20% |\n"
            f"| WFE | >= 0.20 |\n"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
