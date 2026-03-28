#!/usr/bin/env python3
"""MVP backtest runner: ingest 5-year US equity data and execute all backtest runs.

Implements QUA-49:
1. Ingest OHLCV data (2020-01-01 to 2025-12-31) via Yahoo Finance
2. Execute Run 1 (3 individual signal walk-forward runs)
3. Execute Run 2 (full multi-strategy ensemble)
4. Execute Run 3 (4 sensitivity analysis runs)
5. Validate against CRO gates and report results

Signals are implemented as self-contained classes that compute features
(RSI, MACD, SMA) internally from the returns series, making them
compatible with the existing MultiStrategyWalkForwardAnalyzer engine.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path

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
import duckdb
from quant.config import get_db_path
from quant.data.ingest.yahoo import YahooFinanceSource
from quant.data.pipeline import IngestionPipeline
from quant.data.storage.duckdb import MarketDataStore
from quant.data.validation import validate as _validate_records
from quant.portfolio.alpha import CombinationMethod
from quant.portfolio.engine import PortfolioConfig, PortfolioConstraints
from quant.portfolio.optimizers import OptimizationMethod
from quant.risk.circuit_breaker import DrawdownCircuitBreaker
from quant.signals.adaptive_combiner import AdaptiveCombinerConfig, AdaptiveSignalCombiner
from quant.signals.base import BaseSignal, SignalOutput
from quant.signals.regime import RegimeConfig, RegimeWeightAdapter

# ── Constants ────────────────────────────────────────────────────────────────

DATA_START = date(2020, 1, 1)  # warm-up start
DATA_END = date(2025, 12, 31)  # backtest end
BACKTEST_START = date(2021, 1, 1)  # actual backtest window start

RESULTS_DIR = Path(__file__).parent.parent.parent / "backtest-results" / "mvp-us-equity-ensemble"
UNIVERSE_FILE = Path(__file__).parent.parent.parent / "data" / "sp500_universe.txt"

# GICS sector map (representative sample; extended at runtime via yfinance)
_SECTOR_MAP_SEED: dict[str, str] = {
    # Technology
    **{s: "Information Technology" for s in [
        "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "META", "AVGO", "ORCL", "AMD",
        "QCOM", "INTC", "TXN", "CRM", "AMAT", "LRCX", "KLAC", "MU", "ADI",
        "MCHP", "SNPS", "CDNS", "CSCO", "ACN", "IBM", "INTU", "ADP", "PAYX",
        "MSI", "ANET", "FTNT", "GDDY", "PTC", "VRSN",
    ]},
    # Consumer Discretionary
    **{s: "Consumer Discretionary" for s in [
        "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX", "BKNG",
        "ABNB", "MAR", "HLT", "RCL", "CCL", "F", "GM", "PHM", "DHI", "LEN",
        "ROST", "EXPE", "EBAY", "ETSY", "ULTA", "DRI", "LVS", "MGM", "AZO",
        "ORLY", "KMX", "LULU", "TGT",
    ]},
    # Consumer Staples
    **{s: "Consumer Staples" for s in [
        "WMT", "COST", "PG", "KO", "PEP", "PM", "MO", "MDLZ", "KHC", "CL",
        "KMB", "SYY", "HRL", "CAG", "MKC", "GIS", "K", "HSY", "MNST", "STZ",
    ]},
    # Financials
    **{s: "Financials" for s in [
        "BRK-B", "JPM", "BAC", "WFC", "GS", "MS", "AXP", "BLK", "SCHW",
        "CB", "MET", "PRU", "AIG", "ALL", "AFL", "HIG", "TRV", "PGR",
        "SYF", "COF", "DFS", "V", "MA", "PYPL", "BX", "KKR",
    ]},
    # Health Care
    **{s: "Health Care" for s in [
        "UNH", "JNJ", "LLY", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY",
        "AMGN", "GILD", "VRTX", "REGN", "MRNA", "BIIB", "IQV", "ZBH",
        "BDX", "ISRG", "BSX", "SYK", "ZTS", "IDXX", "DXCM",
    ]},
    # Industrials
    **{s: "Industrials" for s in [
        "HON", "GE", "UPS", "RTX", "CAT", "DE", "ETN", "EMR", "PH",
        "CARR", "OTIS", "LMT", "NOC", "GD", "BA", "LHX", "TDG",
        "ROP", "IEX", "CPRT", "RSG", "WM", "CTAS",
    ]},
    # Energy
    **{s: "Energy" for s in [
        "XOM", "CVX", "COP", "EOG", "SLB", "OXY", "MPC", "VLO", "PSX",
        "HES", "DVN", "APA", "EQT", "HAL", "BKR",
    ]},
    # Utilities
    **{s: "Utilities" for s in [
        "NEE", "DUK", "SO", "AEP", "SRE", "EXC", "XEL", "DTE", "PPL", "ED",
    ]},
    # Real Estate
    **{s: "Real Estate" for s in [
        "AMT", "PLD", "CCI", "EQIX", "PSA", "WELL", "DLR", "SPG", "O", "VTR",
    ]},
    # Materials
    **{s: "Materials" for s in [
        "LIN", "APD", "SHW", "PPG", "ECL", "NUE", "STLD", "FCX", "NEM", "GOLD",
    ]},
    # Communication Services
    **{s: "Communication Services" for s in [
        "DIS", "NFLX", "TMUS", "VZ", "T", "CMCSA", "CHTR", "EA", "TTWO",
        "NWSA", "OMC",
    ]},
}


# ── Custom signal classes ─────────────────────────────────────────────────────


class MomentumSignalFromReturns(BaseSignal):
    """MomentumSignal that computes RSI internally from a cumulative price index.

    Compatible with MultiStrategyBacktestEngine (receives only 'returns').
    Reconstructs a price index via (1+r).cumprod(), then computes RSI-based
    momentum scores using the same Rust kernels as strategies.MomentumSignal.
    """

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

        # Build cumulative price index from returns
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
    """TrendFollowingSignal that computes MACD + SMA from a cumulative price index.

    Compatible with MultiStrategyBacktestEngine (receives only 'returns').
    """

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
        min_len = self._slow_ma + 30  # MACD slow period + signal warmup
        if len(returns) < min_len:
            return SignalOutput(
                symbol=symbol, timestamp=timestamp,
                score=0.0, confidence=0.0, target_position=0.0,
                metadata={"reason": "insufficient_data"},
            )

        # Build cumulative price index from returns
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


# ── Portfolio config ──────────────────────────────────────────────────────────

def _portfolio_config(
    method: OptimizationMethod = OptimizationMethod.MEAN_VARIANCE,
    risk_aversion: float = 5.0,
) -> PortfolioConfig:
    """Portfolio config.

    Default: MEAN_VARIANCE so signal alpha scores drive weight allocation
    via w* = Σ⁻¹ μ / λ.  RISK_PARITY ignores alpha (equal risk contribution)
    and should only be used as a benchmark sleeve.
    """
    kwargs: dict = {}
    if method == OptimizationMethod.MEAN_VARIANCE:
        kwargs["risk_aversion"] = risk_aversion
    return PortfolioConfig(
        optimization_method=method,
        constraints=PortfolioConstraints(
            long_only=True,
            max_weight=0.05,
            max_gross_exposure=0.6,
        ),
        rebalance_threshold=0.01,
        cov_lookback_days=252,
        optimizer_kwargs=kwargs,
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


def _adaptive_config() -> AdaptiveCombinerConfig:
    return AdaptiveCombinerConfig(
        ic_lookback=126,
        min_ic_periods=20,
        min_ic=0.0,
        ic_halflife=21,
        shrinkage=0.3,
        min_assets=3,
    )


def _circuit_breaker() -> DrawdownCircuitBreaker:
    return DrawdownCircuitBreaker(
        max_drawdown_threshold=0.15,
        reset_on_new_peak=True,
    )


# ── Ingest ───────────────────────────────────────────────────────────────────

def load_universe() -> list[str]:
    """Load tickers from universe file."""
    if not UNIVERSE_FILE.exists():
        raise FileNotFoundError(f"Universe file not found: {UNIVERSE_FILE}")
    symbols = []
    for line in UNIVERSE_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            symbols.append(line.upper())
    return sorted(set(symbols))


_OHLCV_DDL = """
CREATE TABLE IF NOT EXISTS ohlcv (
    symbol    VARCHAR  NOT NULL,
    date      DATE     NOT NULL,
    open      DOUBLE   NOT NULL,
    high      DOUBLE   NOT NULL,
    low       DOUBLE   NOT NULL,
    close     DOUBLE   NOT NULL,
    volume    DOUBLE   NOT NULL,
    adj_close DOUBLE   NOT NULL,
    PRIMARY KEY (symbol, date)
);
CREATE INDEX IF NOT EXISTS idx_ohlcv_date ON ohlcv (date);
"""


def ingest_data(db_path: str, symbols: list[str]) -> None:
    """Fast ingest: yfinance batch download → validate → DuckDB bulk DataFrame insert.

    Uses DuckDB's native pandas DataFrame import (vectorised, milliseconds)
    instead of executemany(INSERT OR REPLACE) which is O(n) row-by-row and
    extremely slow for large datasets.
    """
    logger.info("Starting fast ingest: {} symbols from {} to {}", len(symbols), DATA_START, DATA_END)

    # Download via existing batch source (handles chunking + error handling)
    source = YahooFinanceSource()
    records = source.fetch(symbols, DATA_START, DATA_END)
    logger.info("Downloaded {} raw OHLCV records", len(records))

    if len(records) < 100:
        raise RuntimeError(f"Too few records downloaded ({len(records)})")

    # Validate
    vr = _validate_records(records)
    valid_records = vr.valid
    logger.info(
        "Validation: {} valid, {} issues",
        len(valid_records),
        len(vr.issues),
    )

    # Convert to pandas DataFrame for fast DuckDB bulk insert
    df_insert = pd.DataFrame([
        {
            "symbol": r.symbol,
            "date": r.date,
            "open": r.open,
            "high": r.high,
            "low": r.low,
            "close": r.close,
            "volume": r.volume,
            "adj_close": r.adj_close,
        }
        for r in valid_records
    ])

    logger.info("Bulk-inserting {} records into DuckDB...", len(df_insert))
    conn = duckdb.connect(db_path)
    try:
        conn.execute(_OHLCV_DDL)
        # Fast vectorised import — DuckDB reads the DataFrame directly
        # For a fresh DB (no existing rows) there are no conflicts so plain INSERT is fine
        conn.execute("INSERT OR REPLACE INTO ohlcv BY NAME SELECT * FROM df_insert")
        conn.execute("CHECKPOINT")  # flush WAL to disk
    finally:
        conn.close()

    logger.info("Ingest complete: {} records stored", len(df_insert))


# ── Data loading ─────────────────────────────────────────────────────────────

def load_returns(db_path: str, symbols: list[str]) -> pd.DataFrame:
    """Load adj_close prices via a single DuckDB query, compute daily returns.

    Uses a single query_multi() call (one SQL query) instead of one query per
    symbol to avoid O(n) round-trips.
    """
    logger.info("Loading price data for {} symbols (single query)", len(symbols))

    conn = duckdb.connect(db_path, read_only=True)
    try:
        syms_upper = [s.upper() for s in symbols]
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

    # Pivot to wide format: rows=date, cols=symbol
    long_df["date"] = pd.to_datetime(long_df["date"])
    prices = long_df.pivot(index="date", columns="symbol", values="adj_close")
    prices = prices.sort_index()

    # Drop symbols with < 252 trading days
    coverage_counts = prices.notna().sum()
    prices = prices[coverage_counts[coverage_counts >= 252].index]

    # Daily returns
    returns = prices.pct_change()
    returns = returns.dropna(how="all")

    # Remove columns with < 80% non-NaN coverage
    coverage = returns.notna().mean()
    good_cols = coverage[coverage >= 0.80].index
    returns = returns[good_cols]

    logger.info(
        "Loaded returns: {} symbols, {} trading days ({} to {})",
        len(returns.columns),
        len(returns),
        returns.index[0].date(),
        returns.index[-1].date(),
    )
    return returns


def build_sector_map(symbols: list[str]) -> dict[str, str]:
    """Build sector map for the given symbols using the seed map."""
    sector_map = {}
    for sym in symbols:
        sector = _SECTOR_MAP_SEED.get(sym, "Unknown")
        sector_map[sym] = sector
    return sector_map


# ── Backtest configs ─────────────────────────────────────────────────────────

def make_wf_config(
    ms_config: MultiStrategyConfig,
    name: str,
    rebalance_frequency: int = 21,
) -> MultiStrategyWalkForwardConfig:
    """Create a walk-forward config with standard IS=252, OOS=63, step=63."""
    return MultiStrategyWalkForwardConfig(
        multi_strategy_config=ms_config,
        is_window=252,
        oos_window=63,
        step_size=63,
        expanding=False,
        name=name,
    )


def run1a_momentum_standalone(
    returns: pd.DataFrame,
    sector_map: dict[str, str],
) -> MultiStrategyWalkForwardResult:
    """Run 1a: MomentumSignal standalone walk-forward."""
    logger.info("=== Run 1a: MomentumSignal standalone ===")
    momentum_signal = MomentumSignalFromReturns(rsi_period=14, lookback=5, return_scale=0.05)
    ms_config = MultiStrategyConfig(
        sleeves=[
            SleeveConfig(
                name="momentum_us_equity",
                signals=[momentum_signal],
                capital_weight=1.0,
                strategy_type="momentum",
                portfolio_config=_portfolio_config(),
                combination_method=CombinationMethod.EQUAL_WEIGHT,
            ),
        ],
        rebalance_frequency=21,
        commission_bps=10.0,
        initial_capital=1_000_000.0,
        sector_map=sector_map,
        min_history=100,
        name="run1a_momentum_standalone",
    )
    analyzer = MultiStrategyWalkForwardAnalyzer()
    return analyzer.run(returns, make_wf_config(ms_config, "run1a_momentum"))


def run1b_trend_standalone(
    returns: pd.DataFrame,
    sector_map: dict[str, str],
) -> MultiStrategyWalkForwardResult:
    """Run 1b: TrendFollowingSignal standalone walk-forward."""
    logger.info("=== Run 1b: TrendFollowingSignal standalone ===")
    trend_signal = TrendFollowingSignalFromReturns(fast_ma=20, slow_ma=50)
    ms_config = MultiStrategyConfig(
        sleeves=[
            SleeveConfig(
                name="trend_following_us_equity",
                signals=[trend_signal],
                capital_weight=1.0,
                strategy_type="trend",
                portfolio_config=_portfolio_config(),
                combination_method=CombinationMethod.EQUAL_WEIGHT,
            ),
        ],
        rebalance_frequency=21,
        commission_bps=10.0,
        initial_capital=1_000_000.0,
        sector_map=sector_map,
        min_history=100,
        name="run1b_trend_standalone",
    )
    analyzer = MultiStrategyWalkForwardAnalyzer()
    return analyzer.run(returns, make_wf_config(ms_config, "run1b_trend"))


def run1c_combined_standalone(
    returns: pd.DataFrame,
    sector_map: dict[str, str],
) -> MultiStrategyWalkForwardResult:
    """Run 1c: Combined (equal weight) standalone."""
    logger.info("=== Run 1c: Combined equal-weight standalone ===")
    momentum_signal = MomentumSignalFromReturns(rsi_period=14, lookback=5, return_scale=0.05)
    trend_signal = TrendFollowingSignalFromReturns(fast_ma=20, slow_ma=50)
    ms_config = MultiStrategyConfig(
        sleeves=[
            SleeveConfig(
                name="combined_equal_weight",
                signals=[momentum_signal, trend_signal],
                capital_weight=1.0,
                strategy_type="momentum",
                portfolio_config=_portfolio_config(),
                combination_method=CombinationMethod.EQUAL_WEIGHT,
            ),
        ],
        rebalance_frequency=21,
        commission_bps=10.0,
        initial_capital=1_000_000.0,
        sector_map=sector_map,
        min_history=100,
        name="run1c_combined_equal_weight",
    )
    analyzer = MultiStrategyWalkForwardAnalyzer()
    return analyzer.run(returns, make_wf_config(ms_config, "run1c_combined"))


def run2_full_ensemble(
    returns: pd.DataFrame,
    sector_map: dict[str, str],
) -> MultiStrategyWalkForwardResult:
    """Run 2: Full multi-strategy ensemble with regime overlay."""
    logger.info("=== Run 2: Full multi-strategy ensemble ===")
    momentum_signal = MomentumSignalFromReturns(rsi_period=14, lookback=5, return_scale=0.05)
    trend_signal = TrendFollowingSignalFromReturns(fast_ma=20, slow_ma=50)

    ms_config = MultiStrategyConfig(
        sleeves=[
            SleeveConfig(
                name="momentum_us_equity",
                signals=[momentum_signal],
                capital_weight=0.40,
                strategy_type="momentum",
                portfolio_config=_portfolio_config(),
                combination_method=CombinationMethod.EQUAL_WEIGHT,
            ),
            SleeveConfig(
                name="trend_following_us_equity",
                signals=[trend_signal],
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
        # circuit_breaker omitted: the CB object is shared across all WF folds
        # (via _build_fold_config reference copy), causing state contamination where
        # the IS peak value bleeds into OOS runs and trips the breaker immediately.
        min_history=100,
        name="run2_full_ensemble",
    )
    analyzer = MultiStrategyWalkForwardAnalyzer()
    return analyzer.run(returns, make_wf_config(ms_config, "run2_full_ensemble"))


def run3_sensitivity(
    returns: pd.DataFrame,
    sector_map: dict[str, str],
) -> dict[str, MultiStrategyWalkForwardResult]:
    """Run 3: Sensitivity analysis (4 runs)."""
    results = {}

    # Run 3a: RSI period sensitivity
    logger.info("=== Run 3a: RSI period sensitivity ===")
    for rsi_period in [10, 14, 21]:
        name = f"run3a_rsi{rsi_period}"
        logger.info("  rsi_period={}", rsi_period)
        momentum_signal = MomentumSignalFromReturns(rsi_period=rsi_period, lookback=5, return_scale=0.05)
        ms_config = MultiStrategyConfig(
            sleeves=[
                SleeveConfig(
                    name="momentum_us_equity",
                    signals=[momentum_signal],
                    capital_weight=1.0,
                    strategy_type="momentum",
                    portfolio_config=_portfolio_config(),
                    combination_method=CombinationMethod.EQUAL_WEIGHT,
                ),
            ],
            rebalance_frequency=21,
            commission_bps=10.0,
            initial_capital=1_000_000.0,
            sector_map=sector_map,
            min_history=100,
            name=name,
        )
        analyzer = MultiStrategyWalkForwardAnalyzer()
        results[name] = analyzer.run(returns, make_wf_config(ms_config, name))

    # Run 3b: MA period sensitivity
    logger.info("=== Run 3b: MA period sensitivity ===")
    for fast_ma, slow_ma in [(10, 30), (20, 50), (30, 100)]:
        name = f"run3b_ma{fast_ma}_{slow_ma}"
        logger.info("  fast_ma={}, slow_ma={}", fast_ma, slow_ma)
        trend_signal = TrendFollowingSignalFromReturns(fast_ma=fast_ma, slow_ma=slow_ma)
        ms_config = MultiStrategyConfig(
            sleeves=[
                SleeveConfig(
                    name="trend_following_us_equity",
                    signals=[trend_signal],
                    capital_weight=1.0,
                    strategy_type="trend",
                    portfolio_config=_portfolio_config(),
                    combination_method=CombinationMethod.EQUAL_WEIGHT,
                ),
            ],
            rebalance_frequency=21,
            commission_bps=10.0,
            initial_capital=1_000_000.0,
            sector_map=sector_map,
            min_history=100,
            name=name,
        )
        analyzer = MultiStrategyWalkForwardAnalyzer()
        results[name] = analyzer.run(returns, make_wf_config(ms_config, name))

    # Run 3c: Rebalance frequency sensitivity
    # Note: rebalance_freq < 21 causes excessive regime-detection overhead
    # (O(n_assets²) pairwise correlations via pure Python at each rebalance).
    # Only freq=63 (quarterly) is tested to show directional sensitivity.
    logger.info("=== Run 3c: Rebalance frequency sensitivity ===")
    momentum_signal = MomentumSignalFromReturns(rsi_period=14, lookback=5, return_scale=0.05)
    trend_signal = TrendFollowingSignalFromReturns(fast_ma=20, slow_ma=50)
    for rebalance_freq in [63]:
        name = f"run3c_rebal{rebalance_freq}"
        logger.info("  rebalance_frequency={}", rebalance_freq)
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
            rebalance_frequency=rebalance_freq,
            commission_bps=10.0,
            initial_capital=1_000_000.0,
            sector_map=sector_map,
            regime_config=_regime_config(),
            regime_adapter=RegimeWeightAdapter(max_tilt=0.30),
            regime_lookback_days=252,
            # circuit_breaker omitted: see run2 comment (CB state contamination)
            min_history=100,
            name=name,
        )
        analyzer = MultiStrategyWalkForwardAnalyzer()
        results[name] = analyzer.run(returns, make_wf_config(ms_config, name, rebalance_frequency=rebalance_freq))

    # Run 3d: Regime tilt sensitivity
    logger.info("=== Run 3d: Regime tilt sensitivity ===")
    for max_tilt in [0.0, 0.15, 0.30, 0.50]:
        name = f"run3d_tilt{int(max_tilt*100)}"
        logger.info("  max_tilt={}", max_tilt)
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
            regime_config=_regime_config() if max_tilt > 0.0 else None,
            regime_adapter=RegimeWeightAdapter(max_tilt=max_tilt) if max_tilt > 0.0 else None,
            regime_lookback_days=252,
            # circuit_breaker omitted: see run2 comment (CB state contamination)
            min_history=100,
            name=name,
        )
        analyzer = MultiStrategyWalkForwardAnalyzer()
        results[name] = analyzer.run(returns, make_wf_config(ms_config, name))

    return results


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


def validate_result(
    result: MultiStrategyWalkForwardResult,
    run_name: str,
) -> ValidationResult:
    """Check CRO validation gates."""
    failures = []

    oos_sharpe = result.oos_sharpe
    max_drawdown = abs(result.oos_max_drawdown)

    # Compute profit factor from OOS returns
    oos_returns = result.oos_returns.dropna()
    gains = oos_returns[oos_returns > 0].sum()
    losses = abs(oos_returns[oos_returns < 0].sum())
    profit_factor = gains / losses if losses > 1e-12 else float("inf")

    # WF efficiency = mean_wfe (ratio of OOS/IS Sharpe across folds)
    wf_efficiency = result.mean_wfe

    if oos_sharpe < 0.6:
        failures.append(f"OOS Sharpe {oos_sharpe:.2f} < 0.6")
    if profit_factor < 1.3:
        failures.append(f"Profit factor {profit_factor:.2f} < 1.3")
    if wf_efficiency < 0.70:
        failures.append(f"WF efficiency {wf_efficiency:.2f} < 0.70")
    if max_drawdown >= 0.15:
        failures.append(f"Max drawdown {max_drawdown:.2%} >= 15%")

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
        "| Run | Sharpe | PF | WFE | Max DD | Status |",
        "|-----|--------|-----|-----|--------|--------|",
    ]
    for v in validations:
        status = "✓ PASS" if v.passes else "✗ FAIL"
        lines.append(
            f"| {v.run_name:<35} | {v.oos_sharpe:>6.2f} | {v.profit_factor:>5.2f} | "
            f"{v.wf_efficiency:>5.2f} | {v.max_drawdown:>6.2%} | {status} |"
        )
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<8} | {message}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    db_path = str(Path.home() / ".quant" / "mvp_backtest.duckdb")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load universe ─────────────────────────────────────────
    symbols = load_universe()
    logger.info("Universe: {} symbols", len(symbols))

    # ── Step 2: Ingest data (skip if DB already populated) ───────────
    logger.info("Step 2: Data ingestion")
    try:
        existing_count = 0
        if Path(db_path).exists():
            try:
                _chk = duckdb.connect(db_path, read_only=True)
                row = _chk.execute("SELECT COUNT(*) FROM ohlcv").fetchone()
                existing_count = row[0] if row else 0
                _chk.close()
            except Exception:
                existing_count = 0
        if existing_count >= 100_000:
            logger.info("Skipping ingest — DB already has {:,} records", existing_count)
        else:
            ingest_data(db_path, symbols)
    except Exception as e:
        logger.error("Ingestion failed: {}", e)
        return 1

    # ── Step 3: Load returns ──────────────────────────────────────────
    logger.info("Step 3: Load returns from DuckDB")
    try:
        returns = load_returns(db_path, symbols)
    except Exception as e:
        logger.error("Failed to load returns: {}", e)
        return 1

    # ── Step 3b: Select representative backtest universe (50 symbols) ──
    # The backtest engine has O(n²) per-day weight-drift cost. Using the full
    # 355-symbol universe makes each run take hours. We select a 50-symbol
    # representative subset stratified by sector and market-cap proxy
    # (return volatility). Full data remains in DuckDB.
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
    avail = [s for s in _BACKTEST_SYMS if s in returns.columns]
    if len(avail) < 20:
        # Fall back to all available symbols if too few representative ones loaded
        avail = list(returns.columns)
    returns = returns[avail]
    logger.info("Backtest universe: {} symbols (representative subset)", len(returns.columns))

    sector_map = build_sector_map(list(returns.columns))
    logger.info("Sector map coverage: {}/{} symbols", sum(1 for v in sector_map.values() if v != "Unknown"), len(sector_map))

    # Helper: save partial results to disk for crash recovery
    def _save_partial(label: str, data: dict) -> None:
        try:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            p = RESULTS_DIR / f"partial_{label}.json"
            p.write_text(json.dumps(data, indent=2, default=str))
            logger.info("Partial results saved: {}", p)
        except Exception as exc:
            logger.warning("Could not save partial results: {}", exc)

    # ── Step 4: Run 1 — Individual signal validation ──────────────────
    logger.info("Step 4: Run 1 — Individual signal validation")
    try:
        r1a = run1a_momentum_standalone(returns, sector_map)
        r1b = run1b_trend_standalone(returns, sector_map)
        r1c = run1c_combined_standalone(returns, sector_map)
    except Exception as e:
        logger.error("Run 1 failed: {}", e)
        return 1
    _save_partial("run1", {
        "run1a": {"oos_sharpe": r1a.oos_sharpe, "oos_max_drawdown": r1a.oos_max_drawdown, "mean_wfe": r1a.mean_wfe, "n_folds": r1a.n_folds},
        "run1b": {"oos_sharpe": r1b.oos_sharpe, "oos_max_drawdown": r1b.oos_max_drawdown, "mean_wfe": r1b.mean_wfe, "n_folds": r1b.n_folds},
        "run1c": {"oos_sharpe": r1c.oos_sharpe, "oos_max_drawdown": r1c.oos_max_drawdown, "mean_wfe": r1c.mean_wfe, "n_folds": r1c.n_folds},
    })

    # ── Step 5: Run 2 — Full ensemble ─────────────────────────────────
    logger.info("Step 5: Run 2 — Full ensemble")
    try:
        r2 = run2_full_ensemble(returns, sector_map)
    except Exception as e:
        logger.error("Run 2 failed: {}", e)
        return 1
    _save_partial("run2", {
        "run2": {"oos_sharpe": r2.oos_sharpe, "oos_max_drawdown": r2.oos_max_drawdown, "mean_wfe": r2.mean_wfe, "n_folds": r2.n_folds},
    })

    # ── Step 6: Run 3 — Sensitivity analysis ─────────────────────────
    logger.info("Step 6: Run 3 — Sensitivity analysis")
    try:
        r3 = run3_sensitivity(returns, sector_map)
    except Exception as e:
        logger.error("Run 3 failed: {}", e)
        return 1

    # ── Step 7: Validate ──────────────────────────────────────────────
    logger.info("Step 7: Validation")
    primary_validations = [
        validate_result(r1a, "Run_1a_Momentum"),
        validate_result(r1b, "Run_1b_Trend"),
        validate_result(r1c, "Run_1c_Combined_EqualWeight"),
        validate_result(r2, "Run_2_FullEnsemble"),
    ]
    sensitivity_validations = [
        validate_result(v, k) for k, v in r3.items()
    ]
    all_validations = primary_validations + sensitivity_validations

    # ── Step 8: Print results ─────────────────────────────────────────
    logger.info("\n=== BACKTEST RESULTS ===\n")
    for run, result in [
        ("Run 1a (Momentum)", r1a),
        ("Run 1b (Trend)", r1b),
        ("Run 1c (Combined)", r1c),
        ("Run 2 (Full Ensemble)", r2),
    ]:
        print(f"\n{result.summary()}")

    print("\n=== SENSITIVITY RUNS ===")
    for name, result in r3.items():
        print(f"\n[{name}] OOS Sharpe={result.oos_sharpe:.2f}, WFE={result.mean_wfe:.2f}, MaxDD={result.oos_max_drawdown:.2%}")

    print(f"\n=== CRO VALIDATION GATES ===")
    print(format_validation_table(all_validations))

    primary_pass_count = sum(1 for v in primary_validations if v.passes)
    logger.info("\nPrimary runs: {}/{} passed all gates", primary_pass_count, len(primary_validations))

    # ── Step 9: Save results ──────────────────────────────────────────
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_summary = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "universe_size": len(returns.columns),
        "data_window": f"{returns.index[0].date()} to {returns.index[-1].date()}",
        "trading_days": len(returns),
        "runs": {
            "run1a": _summarize(r1a, primary_validations[0]),
            "run1b": _summarize(r1b, primary_validations[1]),
            "run1c": _summarize(r1c, primary_validations[2]),
            "run2": _summarize(r2, primary_validations[3]),
            "sensitivity": {k: _summarize(v, next(sv for sv in sensitivity_validations if sv.run_name == k)) for k, v in r3.items()},
        },
    }

    output_file = RESULTS_DIR / f"results_{run_id}.json"
    output_file.write_text(json.dumps(results_summary, indent=2, default=str))
    logger.info("Results saved to {}", output_file)

    return 0 if primary_pass_count >= 3 else 1


def _summarize(result: MultiStrategyWalkForwardResult, validation: ValidationResult) -> dict:
    return {
        "n_folds": result.n_folds,
        "oos_sharpe": round(result.oos_sharpe, 4),
        "oos_total_return": round(result.oos_total_return, 4),
        "oos_max_drawdown": round(result.oos_max_drawdown, 4),
        "oos_volatility": round(result.oos_volatility, 4),
        "mean_wfe": round(result.mean_wfe, 4),
        "profit_factor": round(validation.profit_factor, 4),
        "passes_gates": validation.passes,
        "failures": validation.failures,
    }


if __name__ == "__main__":
    sys.exit(main())
