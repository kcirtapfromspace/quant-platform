#!/usr/bin/env python3
"""Per-asset HMM regime detection A/B backtest — QUA-88.

Compares three configurations on the QUA-85 4-sleeve ensemble:
  Run A — Global regime detection (QUA-85 baseline)
  Run B — Per-asset HMM signal filtering (no global regime)
  Run C — Combined: global regime + per-asset HMM filtering

Per-asset HMM wraps each signal so that individual assets in adverse
(high-vol) regimes get dampened scores, reducing losing trades.

WF: IS=90, OOS=30, expanding=True, step_size=30
Commission: 10 bps one-way
MVO optimizer, max_gross_exposure=0.6, min_history=100
"""
from __future__ import annotations

import json
import math
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
RESULTS_DIR = Path.home() / ".quant" / "backtest-results" / "per-asset-hmm"

_BACKTEST_SYMS: list[str] = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "ORCL", "AMD",
    "AMZN", "TSLA", "HD", "MCD", "NKE",
    "WMT", "PG", "KO", "PEP",
    "JPM", "BAC", "GS", "V", "MA", "BLK",
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "TMO",
    "HON", "GE", "CAT", "DE", "UPS",
    "XOM", "CVX", "COP", "EOG",
    "NEE", "DUK", "SO",
    "AMT", "PLD", "CCI",
    "LIN", "SHW", "APD",
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


# ── Per-asset HMM ────────────────────────────────────────────────────────────


class HmmRegimeDetector:
    """2-state HMM (LowVol / HighVol) with Baum-Welch EM + online forward update.

    Ported from quant-bayes/src/hmm.rs. State 0 = LowVol, state 1 = HighVol.
    """

    def __init__(self) -> None:
        self.means = [0.0, 0.0]
        self.stds = [1.0, 1.0]
        self.trans = [[0.95, 0.05], [0.05, 0.95]]
        self.state_probs = [0.5, 0.5]
        self.pi = [0.5, 0.5]

    def _emit(self, obs: float, s: int) -> float:
        std = max(self.stds[s], 1e-10)
        z = (obs - self.means[s]) / std
        return math.exp(-0.5 * z * z) / (std * 2.506_628_274_631_001)

    def _forward(self, obs: list[float]) -> tuple[list[list[float]], list[float]]:
        n = len(obs)
        alpha = [[0.0, 0.0] for _ in range(n)]
        c = [1.0] * n
        alpha[0][0] = self.pi[0] * self._emit(obs[0], 0)
        alpha[0][1] = self.pi[1] * self._emit(obs[0], 1)
        c[0] = alpha[0][0] + alpha[0][1]
        if c[0] > 1e-300:
            alpha[0][0] /= c[0]
            alpha[0][1] /= c[0]
        else:
            alpha[0] = [0.5, 0.5]
            c[0] = 1.0
        for t in range(1, n):
            alpha[t][0] = (
                alpha[t - 1][0] * self.trans[0][0]
                + alpha[t - 1][1] * self.trans[1][0]
            ) * self._emit(obs[t], 0)
            alpha[t][1] = (
                alpha[t - 1][0] * self.trans[0][1]
                + alpha[t - 1][1] * self.trans[1][1]
            ) * self._emit(obs[t], 1)
            c[t] = alpha[t][0] + alpha[t][1]
            if c[t] > 1e-300:
                alpha[t][0] /= c[t]
                alpha[t][1] /= c[t]
            else:
                alpha[t] = [0.5, 0.5]
                c[t] = 1.0
        return alpha, c

    def _backward(self, obs: list[float], c: list[float]) -> list[list[float]]:
        n = len(obs)
        beta = [[1.0, 1.0] for _ in range(n)]
        for t in range(n - 2, -1, -1):
            scale = c[t + 1]
            e0 = self._emit(obs[t + 1], 0)
            e1 = self._emit(obs[t + 1], 1)
            beta[t][0] = (
                self.trans[0][0] * e0 * beta[t + 1][0]
                + self.trans[0][1] * e1 * beta[t + 1][1]
            )
            beta[t][1] = (
                self.trans[1][0] * e0 * beta[t + 1][0]
                + self.trans[1][1] * e1 * beta[t + 1][1]
            )
            if scale > 1e-300:
                beta[t][0] /= scale
                beta[t][1] /= scale
        return beta

    def fit(self, obs: list[float]) -> None:
        if len(obs) < 2:
            return
        mean_all = sum(obs) / len(obs)
        sorted_idx = sorted(range(len(obs)), key=lambda i: abs(obs[i] - mean_all))
        half = max(len(obs) // 2, 1)
        low_vals = [obs[i] for i in sorted_idx[:half]]
        high_vals = [obs[i] for i in sorted_idx[half:]] or [mean_all]
        std_all = max((sum((v - mean_all) ** 2 for v in obs) / len(obs)) ** 0.5, 1e-10)
        m0 = sum(low_vals) / len(low_vals)
        m1 = sum(high_vals) / len(high_vals)
        s0 = (max((sum((v - m0) ** 2 for v in low_vals) / max(len(low_vals) - 1, 1)) ** 0.5, 1e-10)
              if len(low_vals) >= 2 else std_all * 0.5)
        s1 = (max((sum((v - m1) ** 2 for v in high_vals) / max(len(high_vals) - 1, 1)) ** 0.5, 1e-10)
              if len(high_vals) >= 2 else std_all * 1.5)
        self.means = [m0, m1]
        self.stds = [s0, s1]
        self.trans = [[0.95, 0.05], [0.05, 0.95]]
        self.pi = [0.5, 0.5]
        self.state_probs = [0.5, 0.5]

        prev_ll = float("-inf")
        for _ in range(50):
            alpha, c = self._forward(obs)
            beta = self._backward(obs, c)
            ll = sum(math.log(max(v, 1e-300)) for v in c)
            n = len(obs)
            gamma = []
            for t in range(n):
                g0 = alpha[t][0] * beta[t][0]
                g1 = alpha[t][1] * beta[t][1]
                s = g0 + g1
                if s > 1e-300:
                    gamma.append([g0 / s, g1 / s])
                else:
                    gamma.append([0.5, 0.5])
            xi_list = []
            for t in range(n - 1):
                xi = [[0.0, 0.0], [0.0, 0.0]]
                total = 0.0
                for i in range(2):
                    for j in range(2):
                        xi[i][j] = (alpha[t][i] * self.trans[i][j]
                                    * self._emit(obs[t + 1], j) * beta[t + 1][j])
                        total += xi[i][j]
                if total > 1e-300:
                    for row in xi:
                        row[0] /= total
                        row[1] /= total
                xi_list.append(xi)
            self.pi = gamma[0]
            for i in range(2):
                denom = sum(gamma[t][i] for t in range(n - 1))
                for j in range(2):
                    numer = sum(x[i][j] for x in xi_list)
                    self.trans[i][j] = numer / denom if denom > 1e-300 else 0.5
                row_sum = self.trans[i][0] + self.trans[i][1]
                if row_sum > 1e-300:
                    self.trans[i][0] /= row_sum
                    self.trans[i][1] /= row_sum
            for s in range(2):
                denom = sum(gamma[t][s] for t in range(n))
                if denom < 1e-300:
                    continue
                mean = sum(gamma[t][s] * obs[t] for t in range(n)) / denom
                var = sum(gamma[t][s] * (obs[t] - mean) ** 2 for t in range(n)) / denom
                self.means[s] = mean
                self.stds[s] = max(var ** 0.5, 1e-8)
            if abs(ll - prev_ll) < 1e-6:
                break
            prev_ll = ll
        alpha, _ = self._forward(obs)
        self.state_probs = list(alpha[-1])

    def update(self, obs: float) -> None:
        p0 = (self.state_probs[0] * self.trans[0][0]
              + self.state_probs[1] * self.trans[1][0])
        p1 = (self.state_probs[0] * self.trans[0][1]
              + self.state_probs[1] * self.trans[1][1])
        u0 = p0 * self._emit(obs, 0)
        u1 = p1 * self._emit(obs, 1)
        norm = u0 + u1
        if norm > 1e-300:
            self.state_probs = [u0 / norm, u1 / norm]

    @property
    def p_high_vol(self) -> float:
        return self.state_probs[1]


class PerAssetHmmManager:
    """Manages one HMM per asset. Fits on rolling window, online-updates daily."""

    def __init__(self, fit_lookback: int = 252, refit_interval: int = 63) -> None:
        self._fit_lookback = fit_lookback
        self._refit_interval = refit_interval
        self._hmms: dict[str, HmmRegimeDetector] = {}
        self._bars_since_fit: dict[str, int] = {}

    def get_regime_scale(
        self,
        symbol: str,
        returns_history: list[float],
        strategy_type: str,
        low_vol_weight: float = 1.0,
        high_vol_weight: float = 0.3,
    ) -> float:
        """Return a [0,1] scaling factor based on HMM regime for this asset.

        For momentum/trend: dampens in high-vol (low_vol_weight=1.0, high_vol_weight=0.3).
        For mean_reversion: keeps or boosts in high-vol (low_vol_weight=0.7, high_vol_weight=1.0).
        """
        if len(returns_history) < 60:
            return 1.0  # Not enough data, no filter

        hmm = self._hmms.get(symbol)
        bars = self._bars_since_fit.get(symbol, self._refit_interval)

        if hmm is None or bars >= self._refit_interval:
            hmm = HmmRegimeDetector()
            window = returns_history[-self._fit_lookback:]
            hmm.fit(window)
            self._hmms[symbol] = hmm
            self._bars_since_fit[symbol] = 0
        else:
            hmm.update(returns_history[-1])
            self._bars_since_fit[symbol] = bars + 1

        p_hv = hmm.p_high_vol

        if strategy_type == "mean_reversion":
            return 0.7 * (1.0 - p_hv) + 1.0 * p_hv
        else:
            return low_vol_weight * (1.0 - p_hv) + high_vol_weight * p_hv


# Global per-asset HMM manager (shared across all signal wrappers in a run)
_hmm_manager: PerAssetHmmManager | None = None


def _get_hmm_manager() -> PerAssetHmmManager:
    global _hmm_manager
    if _hmm_manager is None:
        _hmm_manager = PerAssetHmmManager(fit_lookback=252, refit_interval=63)
    return _hmm_manager


def _reset_hmm_manager() -> None:
    global _hmm_manager
    _hmm_manager = None


class HmmFilteredSignal(BaseSignal):
    """Wraps a BaseSignal, scaling its output by per-asset HMM regime weight.

    Assets in high-vol HMM regime get dampened scores (for momentum/trend)
    or maintained/boosted scores (for mean-reversion).
    """

    def __init__(self, inner: BaseSignal, strategy_type: str = "momentum") -> None:
        self._inner = inner
        self._strategy_type = strategy_type

    @property
    def name(self) -> str:
        return f"hmm_{self._inner.name}"

    @property
    def required_features(self) -> list[str]:
        return self._inner.required_features

    def compute(
        self,
        symbol: str,
        features: dict[str, pd.Series],
        timestamp: datetime,
    ) -> SignalOutput:
        base = self._inner.compute(symbol, features, timestamp)

        returns = features.get("returns")
        if returns is None or len(returns) < 60:
            return base

        returns_list = [float(r) for r in returns.fillna(0.0)]
        mgr = _get_hmm_manager()
        scale = mgr.get_regime_scale(
            symbol=symbol,
            returns_history=returns_list,
            strategy_type=self._strategy_type,
        )

        return SignalOutput(
            symbol=base.symbol,
            timestamp=base.timestamp,
            score=base.score * scale,
            confidence=base.confidence * scale,
            target_position=base.target_position * scale,
            metadata={**base.metadata, "hmm_scale": round(scale, 4)},
        )


# ── Signal wrappers (from QUA-85) ───────────────────────────────────────────


class MomentumSignalFromReturns(BaseSignal):
    def __init__(self, rsi_period: int = 14, lookback: int = 5, return_scale: float = 0.05) -> None:
        self._rsi_period = rsi_period
        self._lookback = lookback
        self._return_scale = return_scale

    @property
    def name(self) -> str:
        return f"momentum_rsi{self._rsi_period}"

    @property
    def required_features(self) -> list[str]:
        return ["returns"]

    def compute(self, symbol: str, features: dict[str, pd.Series], timestamp: datetime) -> SignalOutput:
        returns = features["returns"].fillna(0.0)
        if len(returns) < self._rsi_period + self._lookback + 5:
            return SignalOutput(symbol=symbol, timestamp=timestamp, score=0.0, confidence=0.0, target_position=0.0, metadata={"reason": "insufficient_data"})
        prices = ((1.0 + returns).cumprod() * 100.0).tolist()
        rsi_values = _qrs.features.rsi(prices, self._rsi_period)
        returns_list = [float(r) for r in returns]
        score, confidence, target_position = _qrs.signals.momentum_signal(rsi_values, returns_list, self._lookback, self._return_scale)
        return SignalOutput(symbol=symbol, timestamp=timestamp, score=score, confidence=confidence, target_position=target_position, metadata={})


class TrendFollowingSignalFromReturns(BaseSignal):
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

    def compute(self, symbol: str, features: dict[str, pd.Series], timestamp: datetime) -> SignalOutput:
        returns = features["returns"].fillna(0.0)
        min_len = self._slow_ma + 30
        if len(returns) < min_len:
            return SignalOutput(symbol=symbol, timestamp=timestamp, score=0.0, confidence=0.0, target_position=0.0, metadata={"reason": "insufficient_data"})
        prices = ((1.0 + returns).cumprod() * 100.0).tolist()
        macd_hist = _qrs.features.macd_histogram(prices, 12, 26, 9)
        fast_ma = _qrs.features.rolling_mean(prices, self._fast_ma)
        slow_ma = _qrs.features.rolling_mean(prices, self._slow_ma)
        score, confidence, target_position = _qrs.signals.trend_following_signal(macd_hist, fast_ma, slow_ma)
        return SignalOutput(symbol=symbol, timestamp=timestamp, score=score, confidence=confidence, target_position=target_position, metadata={})


class MeanReversionSignalFromReturns(BaseSignal):
    def __init__(self, bb_period: int = 20, num_std: float = 2.0) -> None:
        self._bb_period = bb_period
        self._num_std = num_std

    @property
    def name(self) -> str:
        return f"mean_reversion_bb{self._bb_period}"

    @property
    def required_features(self) -> list[str]:
        return ["returns"]

    def compute(self, symbol: str, features: dict[str, pd.Series], timestamp: datetime) -> SignalOutput:
        returns = features["returns"].fillna(0.0)
        min_len = self._bb_period + 10
        if len(returns) < min_len:
            return SignalOutput(symbol=symbol, timestamp=timestamp, score=0.0, confidence=0.0, target_position=0.0, metadata={"reason": "insufficient_data"})
        prices = ((1.0 + returns).cumprod() * 100.0).tolist()
        bb_mid = _qrs.features.bb_mid(prices, self._bb_period)
        bb_upper = _qrs.features.bb_upper(prices, self._bb_period, self._num_std)
        bb_lower = _qrs.features.bb_lower(prices, self._bb_period, self._num_std)
        returns_list = [float(r) for r in returns]
        score, confidence, target_position = _qrs.signals.mean_reversion_signal(bb_mid, bb_upper, bb_lower, returns_list, self._num_std)
        return SignalOutput(symbol=symbol, timestamp=timestamp, score=score, confidence=confidence, target_position=target_position, metadata={})


# ── Config helpers ───────────────────────────────────────────────────────────


def _portfolio_config(risk_aversion: float = 5.0) -> PortfolioConfig:
    return PortfolioConfig(
        optimization_method=OptimizationMethod.MEAN_VARIANCE,
        constraints=PortfolioConstraints(long_only=True, max_weight=0.05, max_gross_exposure=0.6),
        rebalance_threshold=0.01,
        cov_lookback_days=252,
        optimizer_kwargs={"risk_aversion": risk_aversion},
    )


def _adaptive_config() -> AdaptiveCombinerConfig:
    return AdaptiveCombinerConfig(
        ic_lookback=126, min_ic_periods=20, min_ic=0.0, ic_halflife=21, shrinkage=0.3, min_assets=3,
    )


def _regime_config() -> RegimeConfig:
    return RegimeConfig(
        vol_short_window=21, vol_long_window=252, vol_high_threshold=1.25, vol_low_threshold=0.75,
        trend_window=63, trend_threshold=0.10, mr_threshold=-0.10,
        corr_window=63, corr_high_threshold=0.60, corr_low_threshold=0.25, crisis_vol_threshold=2.0,
    )


# ── Data loading ─────────────────────────────────────────────────────────────


def load_returns(db_path: str, symbols: list[str]) -> pd.DataFrame:
    logger.info("Loading price data for {} symbols", len(symbols))
    conn = duckdb.connect(db_path, read_only=True)
    try:
        placeholders = ", ".join(f"'{s}'" for s in symbols)
        long_df = conn.execute(
            f"""
            SELECT symbol, date, adj_close
            FROM ohlcv
            WHERE symbol IN ({placeholders})
              AND date >= ? AND date <= ?
            ORDER BY date, symbol
            """,
            [DATA_START, DATA_END],
        ).df()
    finally:
        conn.close()
    if long_df.empty:
        raise RuntimeError("No price data loaded from database")
    long_df["date"] = pd.to_datetime(long_df["date"])
    prices = long_df.pivot(index="date", columns="symbol", values="adj_close").sort_index()
    coverage_counts = prices.notna().sum()
    prices = prices[coverage_counts[coverage_counts >= 252].index]
    returns = prices.pct_change().dropna(how="all")
    coverage = returns.notna().mean()
    returns = returns[coverage[coverage >= 0.80].index]
    logger.info("Loaded returns: {} symbols, {} trading days ({} to {})",
                len(returns.columns), len(returns), returns.index[0].date(), returns.index[-1].date())
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
        failures.append(f"OOS Sharpe {oos_sharpe:.2f} < 0.60")
    if profit_factor < 1.10:
        failures.append(f"Profit factor {profit_factor:.2f} < 1.10")
    if wf_efficiency < 0.20:
        failures.append(f"WF efficiency {wf_efficiency:.2f} < 0.20")
    if max_drawdown >= 0.20:
        failures.append(f"Max drawdown {max_drawdown:.2%} >= 20%")
    return ValidationResult(
        run_name=run_name, oos_sharpe=oos_sharpe, profit_factor=profit_factor,
        wf_efficiency=wf_efficiency, max_drawdown=max_drawdown,
        n_folds=result.n_folds, passes=len(failures) == 0, failures=failures,
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


# ── Sleeve builders ──────────────────────────────────────────────────────────


def _base_sleeves() -> list[SleeveConfig]:
    """QUA-85 4-sleeve config with base signals (no HMM wrapping)."""
    return [
        SleeveConfig(
            name="momentum_us_equity",
            signals=[MomentumSignalFromReturns(rsi_period=14, lookback=5, return_scale=0.05)],
            capital_weight=0.35, strategy_type="momentum",
            portfolio_config=_portfolio_config(), combination_method=CombinationMethod.EQUAL_WEIGHT,
        ),
        SleeveConfig(
            name="trend_following_us_equity",
            signals=[TrendFollowingSignalFromReturns(fast_ma=20, slow_ma=50)],
            capital_weight=0.30, strategy_type="trend",
            portfolio_config=_portfolio_config(), combination_method=CombinationMethod.EQUAL_WEIGHT,
        ),
        SleeveConfig(
            name="mean_reversion_us_equity",
            signals=[MeanReversionSignalFromReturns(bb_period=20, num_std=2.0)],
            capital_weight=0.15, strategy_type="mean_reversion",
            portfolio_config=_portfolio_config(), combination_method=CombinationMethod.EQUAL_WEIGHT,
        ),
        SleeveConfig(
            name="adaptive_combined",
            signals=[
                MomentumSignalFromReturns(rsi_period=14, lookback=5, return_scale=0.05),
                TrendFollowingSignalFromReturns(fast_ma=20, slow_ma=50),
                VolatilitySignal(period=20, annualise=True, low_vol=0.12, high_vol=0.40),
                ReturnQualitySignal(period=60, sharpe_cap=3.0),
            ],
            capital_weight=0.20, strategy_type="momentum",
            portfolio_config=_portfolio_config(), combination_method=CombinationMethod.RANK_WEIGHTED,
            adaptive_combiner_config=_adaptive_config(),
        ),
    ]


def _hmm_filtered_sleeves() -> list[SleeveConfig]:
    """QUA-85 4-sleeve config with per-asset HMM signal wrapping."""
    return [
        SleeveConfig(
            name="momentum_us_equity",
            signals=[HmmFilteredSignal(MomentumSignalFromReturns(rsi_period=14, lookback=5, return_scale=0.05), strategy_type="momentum")],
            capital_weight=0.35, strategy_type="momentum",
            portfolio_config=_portfolio_config(), combination_method=CombinationMethod.EQUAL_WEIGHT,
        ),
        SleeveConfig(
            name="trend_following_us_equity",
            signals=[HmmFilteredSignal(TrendFollowingSignalFromReturns(fast_ma=20, slow_ma=50), strategy_type="trend")],
            capital_weight=0.30, strategy_type="trend",
            portfolio_config=_portfolio_config(), combination_method=CombinationMethod.EQUAL_WEIGHT,
        ),
        SleeveConfig(
            name="mean_reversion_us_equity",
            signals=[HmmFilteredSignal(MeanReversionSignalFromReturns(bb_period=20, num_std=2.0), strategy_type="mean_reversion")],
            capital_weight=0.15, strategy_type="mean_reversion",
            portfolio_config=_portfolio_config(), combination_method=CombinationMethod.EQUAL_WEIGHT,
        ),
        SleeveConfig(
            name="adaptive_combined",
            signals=[
                HmmFilteredSignal(MomentumSignalFromReturns(rsi_period=14, lookback=5, return_scale=0.05), strategy_type="momentum"),
                HmmFilteredSignal(TrendFollowingSignalFromReturns(fast_ma=20, slow_ma=50), strategy_type="trend"),
                HmmFilteredSignal(VolatilitySignal(period=20, annualise=True, low_vol=0.12, high_vol=0.40), strategy_type="volatility"),
                HmmFilteredSignal(ReturnQualitySignal(period=60, sharpe_cap=3.0), strategy_type="quality"),
            ],
            capital_weight=0.20, strategy_type="momentum",
            portfolio_config=_portfolio_config(), combination_method=CombinationMethod.RANK_WEIGHTED,
            adaptive_combiner_config=_adaptive_config(),
        ),
    ]


# ── Backtest runs ────────────────────────────────────────────────────────────


def _run_wf(ms_config: MultiStrategyConfig, wf_name: str, returns: pd.DataFrame) -> MultiStrategyWalkForwardResult:
    analyzer = MultiStrategyWalkForwardAnalyzer()
    wf_config = MultiStrategyWalkForwardConfig(
        multi_strategy_config=ms_config,
        is_window=90, oos_window=30, step_size=30, expanding=True, name=wf_name,
    )
    return analyzer.run(returns, wf_config)


def run_a_global_regime(returns: pd.DataFrame, sector_map: dict[str, str]) -> MultiStrategyWalkForwardResult:
    """Run A — QUA-85 baseline: 4-sleeve with global regime detection."""
    logger.info("=== Run A: Global regime detection (QUA-85 baseline) ===")
    ms_config = MultiStrategyConfig(
        sleeves=_base_sleeves(),
        rebalance_frequency=21, commission_bps=10.0, initial_capital=1_000_000.0,
        sector_map=sector_map,
        regime_config=_regime_config(), regime_adapter=RegimeWeightAdapter(max_tilt=0.30),
        regime_lookback_days=252, min_history=100,
        name="global_regime_baseline",
    )
    return _run_wf(ms_config, "run_a_global_regime_wf", returns)


def run_b_per_asset_hmm(returns: pd.DataFrame, sector_map: dict[str, str]) -> MultiStrategyWalkForwardResult:
    """Run B — Per-asset HMM signal filtering, no global regime detection."""
    logger.info("=== Run B: Per-asset HMM signal filtering ===")
    _reset_hmm_manager()
    ms_config = MultiStrategyConfig(
        sleeves=_hmm_filtered_sleeves(),
        rebalance_frequency=21, commission_bps=10.0, initial_capital=1_000_000.0,
        sector_map=sector_map,
        regime_config=None, regime_adapter=None,
        min_history=100,
        name="per_asset_hmm",
    )
    return _run_wf(ms_config, "run_b_per_asset_hmm_wf", returns)


def run_c_combined(returns: pd.DataFrame, sector_map: dict[str, str]) -> MultiStrategyWalkForwardResult:
    """Run C — Combined: global regime + per-asset HMM signal filtering."""
    logger.info("=== Run C: Global regime + per-asset HMM ===")
    _reset_hmm_manager()
    ms_config = MultiStrategyConfig(
        sleeves=_hmm_filtered_sleeves(),
        rebalance_frequency=21, commission_bps=10.0, initial_capital=1_000_000.0,
        sector_map=sector_map,
        regime_config=_regime_config(), regime_adapter=RegimeWeightAdapter(max_tilt=0.30),
        regime_lookback_days=252, min_history=100,
        name="combined_global_plus_hmm",
    )
    return _run_wf(ms_config, "run_c_combined_wf", returns)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<8} | {message}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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

    # ── Run A: Global regime (QUA-85 baseline) ────────────────────────
    logger.info("Step 1/3: Run A — Global regime detection baseline")
    try:
        result_a = run_a_global_regime(returns, sector_map)
        v_a = validate_result(result_a, "A: global_regime (baseline)")
        validations.append(v_a)
        all_results["run_a_global_regime"] = {
            "oos_sharpe": v_a.oos_sharpe, "profit_factor": v_a.profit_factor,
            "wf_efficiency": v_a.wf_efficiency, "max_drawdown": v_a.max_drawdown,
            "n_folds": v_a.n_folds, "passes": v_a.passes, "failures": v_a.failures,
        }
        logger.info("Run A: Sharpe={:.3f} PF={:.2f} WFE={:.3f} DD={:.2%}",
                     v_a.oos_sharpe, v_a.profit_factor, v_a.wf_efficiency, v_a.max_drawdown)
    except Exception as e:
        logger.error("Run A failed: {}", e)
        all_results["run_a_global_regime"] = {"error": str(e)}

    # ── Run B: Per-asset HMM ──────────────────────────────────────────
    logger.info("Step 2/3: Run B — Per-asset HMM signal filtering")
    try:
        result_b = run_b_per_asset_hmm(returns, sector_map)
        v_b = validate_result(result_b, "B: per_asset_hmm")
        validations.append(v_b)
        all_results["run_b_per_asset_hmm"] = {
            "oos_sharpe": v_b.oos_sharpe, "profit_factor": v_b.profit_factor,
            "wf_efficiency": v_b.wf_efficiency, "max_drawdown": v_b.max_drawdown,
            "n_folds": v_b.n_folds, "passes": v_b.passes, "failures": v_b.failures,
        }
        logger.info("Run B: Sharpe={:.3f} PF={:.2f} WFE={:.3f} DD={:.2%}",
                     v_b.oos_sharpe, v_b.profit_factor, v_b.wf_efficiency, v_b.max_drawdown)
    except Exception as e:
        logger.error("Run B failed: {}", e)
        all_results["run_b_per_asset_hmm"] = {"error": str(e)}

    # ── Run C: Combined ───────────────────────────────────────────────
    logger.info("Step 3/3: Run C — Combined global regime + per-asset HMM")
    try:
        result_c = run_c_combined(returns, sector_map)
        v_c = validate_result(result_c, "C: combined")
        validations.append(v_c)
        all_results["run_c_combined"] = {
            "oos_sharpe": v_c.oos_sharpe, "profit_factor": v_c.profit_factor,
            "wf_efficiency": v_c.wf_efficiency, "max_drawdown": v_c.max_drawdown,
            "n_folds": v_c.n_folds, "passes": v_c.passes, "failures": v_c.failures,
        }
        logger.info("Run C: Sharpe={:.3f} PF={:.2f} WFE={:.3f} DD={:.2%}",
                     v_c.oos_sharpe, v_c.profit_factor, v_c.wf_efficiency, v_c.max_drawdown)
    except Exception as e:
        logger.error("Run C failed: {}", e)
        all_results["run_c_combined"] = {"error": str(e)}

    # ── Delta analysis ────────────────────────────────────────────────
    if "run_a_global_regime" in all_results and "run_b_per_asset_hmm" in all_results:
        a = all_results["run_a_global_regime"]
        b = all_results["run_b_per_asset_hmm"]
        if "error" not in a and "error" not in b:
            all_results["delta_b_vs_a"] = {
                "sharpe_delta": round(b["oos_sharpe"] - a["oos_sharpe"], 4),
                "pf_delta": round(b["profit_factor"] - a["profit_factor"], 4),
                "wfe_delta": round(b["wf_efficiency"] - a["wf_efficiency"], 4),
                "dd_delta": round(b["max_drawdown"] - a["max_drawdown"], 4),
            }
    if "run_a_global_regime" in all_results and "run_c_combined" in all_results:
        a = all_results["run_a_global_regime"]
        c = all_results["run_c_combined"]
        if "error" not in a and "error" not in c:
            all_results["delta_c_vs_a"] = {
                "sharpe_delta": round(c["oos_sharpe"] - a["oos_sharpe"], 4),
                "pf_delta": round(c["profit_factor"] - a["profit_factor"], 4),
                "wfe_delta": round(c["wf_efficiency"] - a["wf_efficiency"], 4),
                "dd_delta": round(c["max_drawdown"] - a["max_drawdown"], 4),
            }

    # ── Report ────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("QUA-88: Per-Asset HMM Regime Detection A/B Results")
    logger.info("=" * 80)

    if validations:
        table = format_validation_table(validations)
        logger.info("\n{}", table)

    results_file = RESULTS_DIR / "results.json"
    results_file.write_text(json.dumps(all_results, indent=2, default=str))
    logger.info("Results saved to {}", results_file)

    if validations:
        md_file = RESULTS_DIR / "validation_table.md"
        md_lines = [
            "# QUA-88: Per-Asset HMM A/B Results",
            "",
            f"Date: {datetime.now(timezone.utc).isoformat()}",
            "",
            "## Configuration",
            "",
            "- **Run A:** QUA-85 4-sleeve with global RegimeDetector (sleeve capital weight tilts)",
            "- **Run B:** QUA-85 4-sleeve with per-asset HMM signal filtering (no global regime)",
            "- **Run C:** QUA-85 4-sleeve with both global regime + per-asset HMM filtering",
            "",
            "## HMM Parameters",
            "",
            "- 2-state (LowVol/HighVol) Baum-Welch EM",
            "- Fit lookback: 252 days, refit interval: 63 days",
            "- Momentum/trend: LowVol=1.0, HighVol=0.3 scaling",
            "- Mean reversion: LowVol=0.7, HighVol=1.0 scaling",
            "",
            "## CRO Gate Metrics",
            "",
            format_validation_table(validations),
            "",
        ]
        if "delta_b_vs_a" in all_results:
            d = all_results["delta_b_vs_a"]
            md_lines.extend([
                "## Delta: Per-Asset HMM (B) vs Global Regime (A)",
                "",
                f"- Sharpe: {d['sharpe_delta']:+.4f}",
                f"- PF: {d['pf_delta']:+.4f}",
                f"- WFE: {d['wfe_delta']:+.4f}",
                f"- MaxDD: {d['dd_delta']:+.4f}",
                "",
            ])
        if "delta_c_vs_a" in all_results:
            d = all_results["delta_c_vs_a"]
            md_lines.extend([
                "## Delta: Combined (C) vs Global Regime (A)",
                "",
                f"- Sharpe: {d['sharpe_delta']:+.4f}",
                f"- PF: {d['pf_delta']:+.4f}",
                f"- WFE: {d['wfe_delta']:+.4f}",
                f"- MaxDD: {d['dd_delta']:+.4f}",
                "",
            ])
        md_file.write_text("\n".join(md_lines))
        logger.info("Validation table saved to {}", md_file)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
