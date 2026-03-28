#!/usr/bin/env python3
"""QUA-69: Bayesian vs EMA benchmark on real historical data (2018-2025 DuckDB).

Re-runs the QUA-68 OOS walk-forward benchmark using real OHLCV data from
~/.quant/universe_v2.duckdb instead of synthetic regime-switching GBM.

Config matches QUA-58 / QUA-53 runE:
  - Universe: first 50 symbols (alphabetical) from universe_v2.duckdb
  - WF: 90-day IS / 30-day OOS / expanding windows / 64 folds
  - Commission: 10 bps one-way
  - Variants: Baseline (EMA-IC + vol-threshold) vs Bayesian (NormalGamma IC + HMM)

CRO Gate 2 threshold: Bayesian PF >= 1.25 on real data.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import duckdb
import quant_rs

# ── quant_rs submodule aliases ────────────────────────────────────────────────

qr_f = quant_rs.features
qr_s = quant_rs.signals
qr_b = quant_rs.backtest

# ── Constants ──────────────────────────────────────────────────────────────────

DB_PATH = Path.home() / ".quant" / "universe_v2.duckdb"
RESULTS_DIR = Path.home() / ".quant" / "backtest-results" / "qua69-real-benchmark"

N_SYMBOLS = 50
N_FOLDS = 64
TRAIN_WINDOW = 90     # IS bars (expanding: 0..oos_start)
OOS_WINDOW = 30       # OOS bars per fold
COMMISSION = 0.001    # 10 bps one-way
SIG_THRESHOLD = 0.20
MIN_WARMUP = 50

# ── Bayesian building blocks (Python ports of Rust quant-bayes structs) ───────


class NormalGammaTracker:
    """O(1) Normal-Gamma conjugate posterior for online IC estimation.

    Hyperpriors: mu0=0, kappa0=1, alpha0=2, beta0=1 (weakly informative,
    centred on zero IC).  Posterior mean converges to sample mean as n→∞.
    """

    __slots__ = ("mu_n", "kappa_n", "alpha_n", "beta_n", "n")

    def __init__(self) -> None:
        self.mu_n = 0.0
        self.kappa_n = 1.0
        self.alpha_n = 2.0
        self.beta_n = 1.0
        self.n = 0

    def update(self, x: float) -> None:
        kappa_prev = self.kappa_n
        mu_prev = self.mu_n
        self.kappa_n = kappa_prev + 1.0
        self.mu_n = (kappa_prev * mu_prev + x) / self.kappa_n
        self.alpha_n += 0.5
        residual = x - mu_prev
        self.beta_n += kappa_prev * residual * residual / (2.0 * self.kappa_n)
        self.n += 1

    @property
    def posterior_mean(self) -> float:
        return self.mu_n


class HmmRegimeDetector:
    """2-state HMM (LowVol / HighVol) with Baum-Welch fit + online forward update.

    Ported from quant-bayes/src/hmm.rs.  State 0 = LowVol, state 1 = HighVol.
    """

    def __init__(self) -> None:
        self.means = [0.0, 0.0]
        self.stds = [1.0, 1.0]
        # Row-stochastic 2×2 transition matrix (sticky priors)
        self.trans = [[0.95, 0.05], [0.05, 0.95]]
        self.state_probs = [0.5, 0.5]  # filtered belief [P(LowVol), P(HighVol)]
        self.pi = [0.5, 0.5]

    # ── Emission ─────────────────────────────────────────────────────────────

    def _emit(self, obs: float, s: int) -> float:
        std = max(self.stds[s], 1e-10)
        z = (obs - self.means[s]) / std
        return math.exp(-0.5 * z * z) / (std * 2.506_628_274_631_001)

    # ── Forward / Backward ───────────────────────────────────────────────────

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
        # Initialise: sort by |obs - mean|; lower half → LowVol, upper → HighVol
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

        # Baum-Welch EM (max 50 iters, tol=1e-6)
        prev_ll = float("-inf")
        for _ in range(50):
            alpha, c = self._forward(obs)
            beta = self._backward(obs, c)
            ll = sum(math.log(max(v, 1e-300)) for v in c)
            n = len(obs)
            # Gamma
            gamma = []
            for t in range(n):
                g0 = alpha[t][0] * beta[t][0]
                g1 = alpha[t][1] * beta[t][1]
                s = g0 + g1
                if s > 1e-300:
                    gamma.append([g0 / s, g1 / s])
                else:
                    gamma.append([0.5, 0.5])
            # Xi
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
            # M-step
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
        # Set filtered state to final alpha
        alpha, _ = self._forward(obs)
        last = alpha[-1]
        self.state_probs = list(last)

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
    def regime_probs(self) -> list[float]:
        return self.state_probs


# ── IC combiners ──────────────────────────────────────────────────────────────


class NormalGammaCombiner:
    """Bayesian IC combiner: per-signal NormalGammaTracker posterior mean weights."""

    def __init__(self, n_signals: int) -> None:
        self._trackers = [NormalGammaTracker() for _ in range(n_signals)]

    def update_ic(self, idx: int, ic_obs: float) -> None:
        self._trackers[idx].update(ic_obs)

    def combine(self, signals: list[float]) -> float:
        weights = [max(t.posterior_mean, 0.0) for t in self._trackers]
        total = sum(weights)
        if total > 1e-15:
            return sum(w * s for w, s in zip(weights, signals)) / total
        return sum(signals) / len(signals)


class EmaIcCombiner:
    """Baseline IC combiner: EMA-smoothed IC with decay λ."""

    def __init__(self, n_signals: int, decay: float = 0.94) -> None:
        self._ema_ics = [0.0] * n_signals
        self._decay = decay

    def update(self, idx: int, ic_obs: float) -> None:
        d = self._decay
        self._ema_ics[idx] = d * self._ema_ics[idx] + (1.0 - d) * ic_obs

    def combine(self, signals: list[float]) -> float:
        weights = [max(v, 0.0) for v in self._ema_ics]
        total = sum(weights)
        if total > 1e-15:
            return sum(w * s for w, s in zip(weights, signals)) / total
        return sum(signals) / len(signals)


# ── Regime filter (baseline) ──────────────────────────────────────────────────


def threshold_low_vol(rolling_vols: list[float], bar: int) -> bool:
    """Return True (trade-active) when rolling vol is below μ + 1.5σ threshold."""
    if bar < 80:
        return True
    cur_vol = rolling_vols[bar]
    window = rolling_vols[max(0, bar - 60):bar]
    if len(window) < 2:
        return True
    mean_v = sum(window) / len(window)
    var_v = sum((v - mean_v) ** 2 for v in window) / len(window)
    return cur_vol <= mean_v + 1.5 * var_v ** 0.5


# ── Signal helpers ────────────────────────────────────────────────────────────


def _clamp11(v: float) -> float:
    return max(-1.0, min(1.0, v))


def quantise(combined: float, active: bool, threshold: float) -> float:
    if not active:
        return 0.0
    if combined > threshold:
        return 1.0
    if combined < -threshold:
        return -1.0
    return 0.0


# ── Per-symbol walk-forward ───────────────────────────────────────────────────


class SymbolWfResult(NamedTuple):
    bayes_oos_rets: list[float]
    base_oos_rets: list[float]
    bayes_trade_rets: list[float]
    base_trade_rets: list[float]
    bayes_is_sharpes: list[float]
    base_is_sharpes: list[float]
    bayes_oos_sharpes: list[float]
    base_oos_sharpes: list[float]


def _equity_curve_to_rets(curve: list[tuple]) -> list[float]:
    """Convert equity_curve [(portfolio_val, dd), ...] to daily net returns."""
    rets = []
    for i in range(1, len(curve)):
        prev = curve[i - 1][0]
        curr = curve[i][0]
        if prev > 0:
            rets.append((curr - prev) / prev)
        else:
            rets.append(0.0)
    return rets


def run_wf_symbol(
    prices: list[float],
    n_folds: int,
    train_window: int,
    oos_window: int,
    commission: float,
) -> SymbolWfResult:
    n = len(prices)

    # ── Precompute feature arrays ─────────────────────────────────────────────
    rets = qr_f.returns(prices)
    rsi_vals = qr_f.rsi(prices, 14)
    bb_mid = qr_f.bb_mid(prices, 20)
    bb_upper = qr_f.bb_upper(prices, 20, 2.0)
    bb_lower = qr_f.bb_lower(prices, 20, 2.0)
    macd_hist = qr_f.macd_histogram(prices, 12, 26, 9)
    fast_ma = qr_f.ema(prices, 12)
    slow_ma = qr_f.ema(prices, 26)

    # 20-bar rolling vol for baseline threshold
    rolling_vols = [0.01] * n
    for b in range(21, n):
        w = rets[b - 20:b]
        valid = [r for r in w if math.isfinite(r)]
        if len(valid) >= 2:
            m = sum(valid) / len(valid)
            var = sum((r - m) ** 2 for r in valid) / (len(valid) - 1)
            rolling_vols[b] = max(var ** 0.5, 1e-8)

    # ── Initialise models — carry state across full series ────────────────────
    bayes_comb = NormalGammaCombiner(3)
    regime_det = HmmRegimeDetector()
    ema_comb = EmaIcCombiner(3, decay=0.94)

    # Initial Baum-Welch fit on first train_window returns
    fit_rets = [r for r in rets[1:min(train_window, n - 1)] if math.isfinite(r)]
    if fit_rets:
        regime_det.fit(fit_rets)

    # ── Streaming signal generation (causal, no lookahead) ───────────────────
    # Pass small tail windows to signal functions (O(1) per bar).
    _WIN = 30  # covers all lookback requirements (max 26-bar indicator)

    bayes_sigs = [0.0] * n
    base_sigs = [0.0] * n

    for bar in range(n - 1):
        next_ret = (prices[bar + 1] - prices[bar]) / prices[bar]

        if bar < MIN_WARMUP:
            regime_det.update(next_ret)
            continue

        lo = max(0, bar + 1 - _WIN)
        mom, _, _ = qr_s.momentum_signal(
            rsi_vals[lo:bar + 1], rets[lo:bar + 1], 20, 0.02
        )
        mr, _, _ = qr_s.mean_reversion_signal(
            bb_mid[lo:bar + 1], bb_upper[lo:bar + 1],
            bb_lower[lo:bar + 1], rets[lo:bar + 1], 2.0,
        )
        tf, _, _ = qr_s.trend_following_signal(
            macd_hist[lo:bar + 1], fast_ma[lo:bar + 1], slow_ma[lo:bar + 1]
        )

        # Bayesian variant — HMM P(LowVol) > 0.5 gate
        bayes_active = regime_det.regime_probs[0] > 0.50
        cb_bayes = bayes_comb.combine([mom, mr, tf])
        bayes_sigs[bar] = quantise(cb_bayes, bayes_active, SIG_THRESHOLD)

        # Baseline variant — rolling-vol threshold gate
        base_active = threshold_low_vol(rolling_vols, bar)
        cb_base = ema_comb.combine([mom, mr, tf])
        base_sigs[bar] = quantise(cb_base, base_active, SIG_THRESHOLD)

        # Update IC trackers with realised IC (signal × next_ret)
        ic_mom = mom * next_ret
        ic_mr = mr * next_ret
        ic_tf = tf * next_ret
        bayes_comb.update_ic(0, ic_mom)
        bayes_comb.update_ic(1, ic_mr)
        bayes_comb.update_ic(2, ic_tf)
        ema_comb.update(0, ic_mom)
        ema_comb.update(1, ic_mr)
        ema_comb.update(2, ic_tf)
        regime_det.update(next_ret)

    # ── Walk-forward fold evaluation ──────────────────────────────────────────
    bayes_oos_rets: list[float] = []
    base_oos_rets: list[float] = []
    bayes_trade_rets: list[float] = []
    base_trade_rets: list[float] = []
    bayes_is_sharpes: list[float] = []
    base_is_sharpes: list[float] = []
    bayes_oos_sharpes: list[float] = []
    base_oos_sharpes: list[float] = []

    for fold in range(n_folds):
        oos_start = train_window + fold * oos_window
        oos_end = oos_start + oos_window
        if oos_end >= n:
            break

        # IS window: expanding (0 → oos_start), matching QUA-58 config
        is_start = 0

        # IS Sharpe via run_backtest (same signal/price slices)
        is_prices = prices[is_start:oos_start + 1]
        is_b_sigs = bayes_sigs[is_start:oos_start] + [0.0]
        is_s_sigs = base_sigs[is_start:oos_start] + [0.0]
        is_br = qr_b.run_backtest(is_prices, is_b_sigs, commission)
        is_sr = qr_b.run_backtest(is_prices, is_s_sigs, commission)
        bayes_is_sharpes.append(is_br["sharpe_ratio"])
        base_is_sharpes.append(is_sr["sharpe_ratio"])

        # OOS
        oos_prices = prices[oos_start:oos_end + 1]
        oos_b_sigs = bayes_sigs[oos_start:oos_end] + [0.0]
        oos_s_sigs = base_sigs[oos_start:oos_end] + [0.0]
        br = qr_b.run_backtest(oos_prices, oos_b_sigs, commission)
        sr = qr_b.run_backtest(oos_prices, oos_s_sigs, commission)

        bayes_oos_sharpes.append(br["sharpe_ratio"])
        base_oos_sharpes.append(sr["sharpe_ratio"])
        bayes_trade_rets.extend(t[3] for t in br["trades"])
        base_trade_rets.extend(t[3] for t in sr["trades"])
        bayes_oos_rets.extend(_equity_curve_to_rets(br["equity_curve"]))
        base_oos_rets.extend(_equity_curve_to_rets(sr["equity_curve"]))

    return SymbolWfResult(
        bayes_oos_rets=bayes_oos_rets,
        base_oos_rets=base_oos_rets,
        bayes_trade_rets=bayes_trade_rets,
        base_trade_rets=base_trade_rets,
        bayes_is_sharpes=bayes_is_sharpes,
        base_is_sharpes=base_is_sharpes,
        bayes_oos_sharpes=bayes_oos_sharpes,
        base_oos_sharpes=base_oos_sharpes,
    )


# ── Aggregate metrics ─────────────────────────────────────────────────────────


def sharpe_ratio(rets: list[float]) -> float:
    """Annualised Sharpe (252 trading days) from daily net returns."""
    valid = [r for r in rets if math.isfinite(r)]
    if len(valid) < 2:
        return 0.0
    mean = sum(valid) / len(valid)
    var = sum((r - mean) ** 2 for r in valid) / (len(valid) - 1)
    std = var ** 0.5
    if std < 1e-10:
        return 0.0
    return (mean / std) * (252 ** 0.5)


def profit_factor(trade_rets: list[float]) -> float:
    """Trade-level profit factor: sum(wins) / |sum(losses)|."""
    wins = sum(r for r in trade_rets if r > 0)
    losses = abs(sum(r for r in trade_rets if r < 0))
    if losses < 1e-10:
        return float("inf")
    return wins / losses


def max_drawdown(rets: list[float]) -> float:
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in rets:
        equity *= 1.0 + r
        if equity > peak:
            peak = equity
        if peak > 0:
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd
    return max_dd


def mean_wfe(is_sharpes: list[float], oos_sharpes: list[float]) -> float:
    """Walk-Forward Efficiency: mean(OOS Sharpe) / mean(IS Sharpe)."""
    if not is_sharpes:
        return 0.0
    m_is = sum(is_sharpes) / len(is_sharpes)
    m_oos = sum(oos_sharpes) / len(oos_sharpes)
    if abs(m_is) < 1e-6:
        return 0.0
    return max(-3.0, min(3.0, m_oos / m_is))


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load 50 symbols from universe_v2.duckdb ───────────────────────────────
    con = duckdb.connect(str(DB_PATH), read_only=True)
    all_symbols: list[str] = [
        r[0] for r in con.execute(
            "SELECT DISTINCT symbol FROM ohlcv ORDER BY symbol"
        ).fetchall()
    ]
    symbols = all_symbols[:N_SYMBOLS]
    print(f"QUA-69: Bayesian vs EMA — Real-Data OOS Walk-Forward Benchmark")
    print(f"  Database : {DB_PATH}")
    print(f"  Universe : {len(symbols)} symbols (first {N_SYMBOLS} alphabetical)")
    print(f"  Config   : {N_FOLDS} folds | {TRAIN_WINDOW}d IS (expanding) | {OOS_WINDOW}d OOS")
    print(f"  Commission: {int(COMMISSION * 10_000)} bps one-way")
    print(f"  Symbols  : {', '.join(symbols[:10])} ...")
    print()

    # ── Load adj_close for all symbols ───────────────────────────────────────
    rows = con.execute(
        """
        SELECT symbol, date, adj_close
        FROM ohlcv
        WHERE symbol IN ({})
        ORDER BY symbol, date
        """.format(",".join(f"'{s}'" for s in symbols))
    ).fetchall()
    con.close()

    # Group into per-symbol price lists
    from collections import defaultdict
    sym_prices: dict[str, list[float]] = defaultdict(list)
    for sym, _date, price in rows:
        if price is not None and price > 0:
            sym_prices[sym].append(float(price))

    # Warmup (MIN_WARMUP bars) is consumed inside the training window, not in addition.
    # Minimum: last OOS fold must fit within the series.
    min_bars = TRAIN_WINDOW + N_FOLDS * OOS_WINDOW + 1
    valid_symbols = [s for s in symbols if len(sym_prices.get(s, [])) >= min_bars]
    print(f"  Valid symbols with >= {min_bars} bars: {len(valid_symbols)}")
    print()

    # ── Aggregate containers ──────────────────────────────────────────────────
    all_bayes_oos_rets: list[float] = []
    all_base_oos_rets: list[float] = []
    all_bayes_trades: list[float] = []
    all_base_trades: list[float] = []
    all_bayes_is_sh: list[float] = []
    all_base_is_sh: list[float] = []
    all_bayes_oos_sh: list[float] = []
    all_base_oos_sh: list[float] = []

    for i, sym in enumerate(valid_symbols, 1):
        prices = sym_prices[sym]
        print(f"  [{i:2d}/{len(valid_symbols)}] {sym:6s}  ({len(prices)} bars)", end="\r")
        sys.stdout.flush()

        r = run_wf_symbol(prices, N_FOLDS, TRAIN_WINDOW, OOS_WINDOW, COMMISSION)

        all_bayes_oos_rets.extend(r.bayes_oos_rets)
        all_base_oos_rets.extend(r.base_oos_rets)
        all_bayes_trades.extend(r.bayes_trade_rets)
        all_base_trades.extend(r.base_trade_rets)
        all_bayes_is_sh.extend(r.bayes_is_sharpes)
        all_base_is_sh.extend(r.base_is_sharpes)
        all_bayes_oos_sh.extend(r.bayes_oos_sharpes)
        all_base_oos_sh.extend(r.base_oos_sharpes)

    print()  # clear progress line

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    bayes_sharpe = sharpe_ratio(all_bayes_oos_rets)
    base_sharpe = sharpe_ratio(all_base_oos_rets)
    bayes_pf = profit_factor(all_bayes_trades)
    base_pf = profit_factor(all_base_trades)
    bayes_maxdd = max_drawdown(all_bayes_oos_rets)
    base_maxdd = max_drawdown(all_base_oos_rets)
    bayes_wfe = mean_wfe(all_bayes_is_sh, all_bayes_oos_sh)
    base_wfe = mean_wfe(all_base_is_sh, all_base_oos_sh)

    def fmt_pf(pf: float) -> str:
        return "∞" if math.isinf(pf) else f"{pf:.3f}"

    # ── Results table ─────────────────────────────────────────────────────────
    print("┌──────────────────────────┬──────────────┬──────────────────────┐")
    print("│  Metric                  │   Baseline   │   Bayesian (Ph. 1)   │")
    print("├──────────────────────────┼──────────────┼──────────────────────┤")
    print(f"│  OOS Sharpe (ann.)       │  {base_sharpe:>9.3f}   │  {bayes_sharpe:>9.3f}             │")
    print(f"│  Profit Factor           │  {fmt_pf(base_pf):>9}   │  {fmt_pf(bayes_pf):>9}             │")
    print(f"│  Max Drawdown            │  {base_maxdd * 100:>8.2f} %  │  {bayes_maxdd * 100:>8.2f} %            │")
    print(f"│  WFE                     │  {base_wfe:>9.3f}   │  {bayes_wfe:>9.3f}             │")
    print(f"│  Total OOS Trades        │  {len(all_base_trades):>9}   │  {len(all_bayes_trades):>9}             │")
    print("├──────────────────────────┼──────────────┼──────────────────────┤")
    print("│  Delta (Bayesian − Base) │              │                      │")
    print(f"│    Sharpe  Δ             │              │  {bayes_sharpe - base_sharpe:>+9.3f}             │")
    print(f"│    MaxDD  Δ              │              │  {(bayes_maxdd - base_maxdd) * 100:>+8.2f} %            │")
    print(f"│    WFE  Δ                │              │  {bayes_wfe - base_wfe:>+9.3f}             │")
    print("└──────────────────────────┴──────────────┴──────────────────────┘")

    # ── QUA-58 baseline comparison (runE baseline reference) ─────────────────
    print()
    print("── QUA-58 Baseline Reference ────────────────────────────────────")
    print("  QUA-58 runE_baseline_90_30: Sharpe=0.776, PF=1.23, WFE=0.284")
    print(f"  QUA-69 Baseline:            Sharpe={base_sharpe:.3f}, PF={fmt_pf(base_pf)}, WFE={base_wfe:.3f}")
    print(f"  QUA-69 Bayesian:            Sharpe={bayes_sharpe:.3f}, PF={fmt_pf(bayes_pf)}, WFE={bayes_wfe:.3f}")

    # ── CRO Gate 2 assessment ─────────────────────────────────────────────────
    print()
    print("── CRO Gate 2 Assessment ────────────────────────────────────────")
    bayes_pf_v = 999.0 if math.isinf(bayes_pf) else bayes_pf
    base_pf_v = 999.0 if math.isinf(base_pf) else base_pf
    print(f"  PF target: >= 1.25  |  Baseline PF: {fmt_pf(base_pf)}  |  Bayesian PF: {fmt_pf(bayes_pf)}")
    if bayes_pf_v >= 1.25:
        gate_result = "PASS"
        print(f"  PASS  Bayesian PF >= 1.25 — Phase 1 clears CRO Gate 2 threshold.")
        print("  → Update QUA-55, schedule CRO Gate 2 review.")
    else:
        gate_result = "FAIL"
        gap = 1.25 - bayes_pf_v
        print(f"  FAIL  Bayesian PF {bayes_pf_v:.3f} < 1.25 (gap: {gap:.3f})")
        print(f"  → Phase 1 PF delta vs baseline: {bayes_pf_v - base_pf_v:+.3f}")
        print("  → Recommend Phase 2: hierarchical MCMC (cross-signal covariance).")

    # ── Save results JSON ─────────────────────────────────────────────────────
    results = {
        "run_id": f"qua69-real-{run_ts}",
        "timestamp": run_ts,
        "config": {
            "n_symbols": len(valid_symbols),
            "symbols": valid_symbols,
            "n_folds": N_FOLDS,
            "train_window": TRAIN_WINDOW,
            "oos_window": OOS_WINDOW,
            "commission": COMMISSION,
            "wf_type": "expanding",
            "db": str(DB_PATH),
        },
        "baseline": {
            "sharpe": round(base_sharpe, 4),
            "pf": round(base_pf_v, 4),
            "max_dd": round(base_maxdd, 4),
            "wfe": round(base_wfe, 4),
            "n_trades": len(all_base_trades),
        },
        "bayesian": {
            "sharpe": round(bayes_sharpe, 4),
            "pf": round(bayes_pf_v, 4),
            "max_dd": round(bayes_maxdd, 4),
            "wfe": round(bayes_wfe, 4),
            "n_trades": len(all_bayes_trades),
        },
        "delta": {
            "sharpe": round(bayes_sharpe - base_sharpe, 4),
            "pf": round(bayes_pf_v - base_pf_v, 4),
            "max_dd": round((bayes_maxdd - base_maxdd) * 100, 4),
            "wfe": round(bayes_wfe - base_wfe, 4),
        },
        "gate2": {
            "pf_target": 1.25,
            "result": gate_result,
            "bayesian_pf": round(bayes_pf_v, 4),
        },
        "qua58_baseline_reference": {
            "run": "runE_baseline_90_30",
            "sharpe": 0.776,
            "pf": 1.23,
            "wfe": 0.284,
        },
    }

    out_path = RESULTS_DIR / f"results_qua69_{run_ts}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print()
    print(f"Results saved: {out_path}")


if __name__ == "__main__":
    main()
