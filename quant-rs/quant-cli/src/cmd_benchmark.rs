//! `quant benchmark qua68` — OOS benchmark: Bayesian vs EMA AdaptiveSignalCombiner.
//!
//! `quant benchmark qua92` — QUA-92 vol regime sleeve: PF gap closure attempt.
//!
//! Implements the QUA-68 comparative walk-forward backtest.
//!
//! **Config** (mirrors QUA-58): 90d train / 30d OOS / 64 folds / 50 synthetic symbols.
//!
//! **Data model** — each synthetic symbol uses a regime-switching GBM:
//! - 70 % noise (drift ≈ 0)
//! - 15 % uptrend / 15 % downtrend (±0.25 % drift, 8-20 bar regimes)
//! - Vol: 1.5 % daily throughout
//!
//! Autocorrelation is zero at the return level; predictability comes from persistent
//! regime drift, which RSI / BB / MACD signals can partially detect after 5-7 bar lag.
//!
//! **Positions** — combined signal is quantised to {−1, 0, +1} (threshold 0.20).
//! In the HMM HighVol state / threshold-HighVol state, the system goes flat (position = 0)
//! to match the risk-off behaviour expected by the CRO.  This produces clean discrete
//! trades compatible with the quant-backtest trade-level profit-factor convention.
//!
//! **Variants**
//! | Variant   | IC combiner                         | Regime filter              |
//! |-----------|-------------------------------------|----------------------------|
//! | Baseline  | EMA-IC (λ = 0.94)                   | vol-threshold binary       |
//! | Bayesian  | `NormalGammaTracker` conjugate IC   | `HmmRegimeModel` soft prob |

use std::fs;
use std::path::Path;

use clap::Args;

use quant_backtest::{profit_factor, run_backtest, sharpe_ratio};
use quant_features as qf;
use quant_signals::{
    mean_reversion_signal, momentum_signal, trend_following_signal, AdaptiveSignalCombiner,
    RegimeDetector,
};

// ── CLI args ──────────────────────────────────────────────────────────────────

#[derive(Args)]
pub struct BenchmarkQua68Args {
    /// Number of synthetic symbols to simulate.
    #[arg(long, default_value = "50")]
    pub n_symbols: usize,

    /// Number of walk-forward folds.
    #[arg(long, default_value = "64")]
    pub n_folds: usize,

    /// Training window in trading-day bars.
    #[arg(long, default_value = "90")]
    pub train_window: usize,

    /// OOS step in trading-day bars.
    #[arg(long, default_value = "30")]
    pub oos_window: usize,

    /// One-way commission fraction (e.g. 0.001 = 10 bps).
    #[arg(long, default_value = "0.001")]
    pub commission: f64,
}

// ── EMA-IC baseline combiner ──────────────────────────────────────────────────

/// EMA-smoothed IC tracker (λ = 0.94) — the baseline signal combiner.
struct EmaIcCombiner {
    ema_ics: [f64; 3],
    decay: f64,
}

impl EmaIcCombiner {
    fn new(decay: f64) -> Self {
        Self {
            ema_ics: [0.0; 3],
            decay,
        }
    }

    fn update(&mut self, idx: usize, ic_obs: f64) {
        self.ema_ics[idx] = self.decay * self.ema_ics[idx] + (1.0 - self.decay) * ic_obs;
    }

    fn combine(&self, signals: &[f64]) -> f64 {
        debug_assert_eq!(signals.len(), 3);
        let w = [
            self.ema_ics[0].max(0.0),
            self.ema_ics[1].max(0.0),
            self.ema_ics[2].max(0.0),
        ];
        let total = w[0] + w[1] + w[2];
        if total > 1e-15 {
            (w[0] * signals[0] + w[1] * signals[1] + w[2] * signals[2]) / total
        } else {
            (signals[0] + signals[1] + signals[2]) / 3.0
        }
    }
}

// ── Threshold regime (baseline): vol-based ───────────────────────────────────

/// Returns `true` (trade-active) when the 20-bar rolling vol is below
/// (μ_vol + 1.5 σ_vol) estimated over the preceding 60 vol readings.
fn threshold_low_vol(rolling_vols: &[f64], bar: usize) -> bool {
    if bar < 80 {
        return true;
    }
    let cur_vol = rolling_vols[bar];
    let window = &rolling_vols[bar.saturating_sub(60)..bar];
    let mean_v: f64 = window.iter().sum::<f64>() / window.len() as f64;
    let var_v: f64 = window.iter().map(|v| (v - mean_v).powi(2)).sum::<f64>() / window.len() as f64;
    cur_vol <= mean_v + 1.5 * var_v.sqrt()
}

// ── Synthetic regime-switching price series ───────────────────────────────────

/// LCG next state.
#[inline]
fn lcg(s: u64) -> u64 {
    s.wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407)
}

/// Regime-switching GBM with independent vol and drift regimes.
///
/// **Vol regime** (detectable by HMM / vol-threshold):
/// - 75 % LowVol  (σ = 1.2 % daily, 10-30 bar durations)
/// - 25 % HighVol (σ = 2.5 % daily, 10-30 bar durations)
///
/// **Drift regime** (detectable by RSI / BB / MACD signals, independent of vol):
/// - 15 % uptrend  (drift = +0.20 % daily, 8-18 bar durations)
/// - 15 % downtrend (drift = −0.20 % daily, 8-18 bar durations)
/// - 70 % flat     (drift ≈ 0)
///
/// In LowVol + trend, signals have IC ≈ 0.03-0.05.  HMM should detect vol
/// regimes better than the threshold rule at transitions, and NormalGamma
/// should estimate IC more accurately than EMA in early folds.
fn synthetic_prices(n: usize, seed: u64) -> Vec<f64> {
    let mut prices = vec![100.0_f64];
    let mut state: u64 = seed;

    // Vol regime state
    let mut high_vol = false;
    let mut vol_rem: usize = 0;

    // Drift regime state
    let mut drift_dir: i8 = 0; // −1, 0, +1
    let mut drift_rem: usize = 0;

    for _ in 1..n {
        // Vol regime transitions
        if vol_rem == 0 {
            state = lcg(state);
            let r = (state >> 32) as u32 as f64 / u32::MAX as f64;
            high_vol = r < 0.25;
            state = lcg(state);
            let d = (state >> 32) as u32 as f64 / u32::MAX as f64;
            vol_rem = (10.0 + d * 20.0) as usize;
        }
        vol_rem -= 1;

        // Drift regime transitions
        if drift_rem == 0 {
            state = lcg(state);
            let r = (state >> 32) as u32 as f64 / u32::MAX as f64;
            drift_dir = if r < 0.15 {
                1
            } else if r < 0.30 {
                -1
            } else {
                0
            };
            state = lcg(state);
            let d = (state >> 32) as u32 as f64 / u32::MAX as f64;
            drift_rem = (8.0 + d * 10.0) as usize;
        }
        drift_rem -= 1;

        state = lcg(state);
        let noise = (state >> 32) as u32 as f64 / u32::MAX as f64 - 0.5;

        let vol = if high_vol { 0.025 } else { 0.012 };
        let drift = drift_dir as f64 * 0.002; // ±0.20 % in trend regime
        let ret = 0.000_3 + drift + noise * vol;
        prices.push((prices.last().unwrap() * (1.0 + ret)).max(0.01));
    }
    prices
}

// ── Signal quantisation → discrete {−1, 0, +1} ───────────────────────────────

/// Returns +1 / −1 / 0 based on combined signal direction and regime gate.
///
/// `active` = false forces position to 0 (risk-off; go flat regardless of signal).
#[inline]
fn quantise(combined: f64, active: bool, threshold: f64) -> f64 {
    if !active {
        return 0.0;
    }
    if combined > threshold {
        1.0
    } else if combined < -threshold {
        -1.0
    } else {
        0.0
    }
}

// ── Net-return computation ────────────────────────────────────────────────────

fn compute_net_returns(prices: &[f64], signals: &[f64], commission: f64) -> Vec<f64> {
    let n = prices.len();
    let mut out = vec![0.0_f64; n];
    let mut prev_pos = 0.0_f64;
    for i in 1..n {
        let daily_ret = (prices[i] - prices[i - 1]) / prices[i - 1];
        let pos = if i - 1 < signals.len() {
            signals[i - 1]
        } else {
            0.0
        };
        let delta = (pos - prev_pos).abs();
        out[i] = pos * daily_ret - commission * delta;
        prev_pos = pos;
    }
    out
}

// ── Per-symbol walk-forward ───────────────────────────────────────────────────

struct SymbolWfResult {
    bayes_oos_rets: Vec<f64>,
    base_oos_rets: Vec<f64>,
    bayes_trade_rets: Vec<f64>,
    base_trade_rets: Vec<f64>,
    bayes_is_sharpes: Vec<f64>,
    base_is_sharpes: Vec<f64>,
    bayes_oos_sharpes: Vec<f64>,
    base_oos_sharpes: Vec<f64>,
}

fn run_wf_symbol(
    prices: &[f64],
    train_window: usize,
    oos_window: usize,
    n_folds: usize,
    commission: f64,
) -> SymbolWfResult {
    let n = prices.len();
    let min_warmup = 50_usize;
    let sig_threshold = 0.20_f64;

    // ── Precompute causal feature series ──────────────────────────────────────
    let rets = qf::returns(prices);
    let rsi_vals = qf::rsi(prices, 14);
    let bb_mid = qf::bb_mid(prices, 20);
    let bb_upper = qf::bb_upper(prices, 20, 2.0);
    let bb_lower = qf::bb_lower(prices, 20, 2.0);
    let macd_hist = qf::macd_histogram(prices, 12, 26, 9);
    let fast_ma = qf::ema(prices, 12);
    let slow_ma = qf::ema(prices, 26);

    // 20-bar rolling vol for baseline threshold filter
    let rolling_vols: Vec<f64> = (0..n)
        .map(|b| {
            if b < 21 {
                return 0.01_f64;
            }
            let w = &rets[b - 20..b];
            let m: f64 = w.iter().sum::<f64>() / 20.0;
            (w.iter().map(|r| (r - m).powi(2)).sum::<f64>() / 19.0)
                .sqrt()
                .max(1e-8)
        })
        .collect();

    // ── Initialise models — both variants carry state across full series ───────
    let mut bayes_comb = AdaptiveSignalCombiner::new(3);
    let mut regime_det = RegimeDetector::new();
    let mut ema_comb = EmaIcCombiner::new(0.94);

    // Initial Baum-Welch fit — gives the HMM meaningful emission parameters so
    // that online `update()` calls produce useful regime probabilities.
    // Fit on the first `train_window` returns (skip index 0 which is always 0.0).
    {
        let fit_rets: Vec<f64> = rets[1..train_window.min(n - 1)]
            .iter()
            .copied()
            .filter(|r| r.is_finite())
            .collect();
        regime_det.fit(&fit_rets);
    }

    // Streaming signal arrays
    let mut bayes_sigs = vec![0.0_f64; n];
    let mut base_sigs = vec![0.0_f64; n];

    for bar in 0..n - 1 {
        let next_ret = (prices[bar + 1] - prices[bar]) / prices[bar];

        if bar < min_warmup {
            regime_det.update(next_ret);
            continue;
        }

        let (mom, _, _) = momentum_signal(&rsi_vals[..=bar], &rets[..=bar], 20, 0.02);
        let (mr, _, _) = mean_reversion_signal(
            &bb_mid[..=bar],
            &bb_upper[..=bar],
            &bb_lower[..=bar],
            &rets[..=bar],
            2.0,
        );
        let (tf, _, _) =
            trend_following_signal(&macd_hist[..=bar], &fast_ma[..=bar], &slow_ma[..=bar]);

        // ── Bayesian variant ─────────────────────────────────────────────────
        // Active = HMM says P(LowVol) > 0.5
        let bayes_active = regime_det.regime_probs()[0] > 0.50;
        let cb_bayes = bayes_comb.combine(&[mom, mr, tf]);
        bayes_sigs[bar] = quantise(cb_bayes, bayes_active, sig_threshold);

        // ── Baseline variant ─────────────────────────────────────────────────
        // Active = vol-threshold says LowVol
        let base_active = threshold_low_vol(&rolling_vols, bar);
        let cb_base = ema_comb.combine(&[mom, mr, tf]);
        base_sigs[bar] = quantise(cb_base, base_active, sig_threshold);

        // Update models with next_ret (signal already locked above)
        let ic_mom = mom * next_ret;
        let ic_mr = mr * next_ret;
        let ic_tf = tf * next_ret;

        bayes_comb.update_ic(0, ic_mom);
        bayes_comb.update_ic(1, ic_mr);
        bayes_comb.update_ic(2, ic_tf);

        ema_comb.update(0, ic_mom);
        ema_comb.update(1, ic_mr);
        ema_comb.update(2, ic_tf);

        regime_det.update(next_ret);
    }

    // ── Precompute full net-return series ─────────────────────────────────────
    let bayes_net = compute_net_returns(prices, &bayes_sigs, commission);
    let base_net = compute_net_returns(prices, &base_sigs, commission);

    let mut result = SymbolWfResult {
        bayes_oos_rets: Vec::new(),
        base_oos_rets: Vec::new(),
        bayes_trade_rets: Vec::new(),
        base_trade_rets: Vec::new(),
        bayes_is_sharpes: Vec::new(),
        base_is_sharpes: Vec::new(),
        bayes_oos_sharpes: Vec::new(),
        base_oos_sharpes: Vec::new(),
    };

    for fold in 0..n_folds {
        let oos_start = train_window + fold * oos_window;
        let oos_end = oos_start + oos_window;
        if oos_end >= n {
            break;
        }

        // IS window: last `train_window` bars before OOS
        let is_start = oos_start.saturating_sub(train_window);
        result
            .bayes_is_sharpes
            .push(sharpe_ratio(&bayes_net[is_start..oos_start]));
        result
            .base_is_sharpes
            .push(sharpe_ratio(&base_net[is_start..oos_start]));

        // OOS: run_backtest for trade-level PF
        let oos_prices = &prices[oos_start..=oos_end];
        // Pad signals to match adj_close length; last element is unused by the engine
        let mut bp = bayes_sigs[oos_start..oos_end].to_vec();
        bp.push(0.0);
        let mut sp = base_sigs[oos_start..oos_end].to_vec();
        sp.push(0.0);

        let br = run_backtest(oos_prices, &bp, commission, 1.0);
        let sr = run_backtest(oos_prices, &sp, commission, 1.0);

        result.bayes_oos_sharpes.push(sharpe_ratio(&br.net_returns));
        result.base_oos_sharpes.push(sharpe_ratio(&sr.net_returns));

        result
            .bayes_trade_rets
            .extend(br.trades.iter().map(|t| t.ret));
        result
            .base_trade_rets
            .extend(sr.trades.iter().map(|t| t.ret));

        result
            .bayes_oos_rets
            .extend_from_slice(&br.net_returns[1..]);
        result.base_oos_rets.extend_from_slice(&sr.net_returns[1..]);
    }

    result
}

// ── Aggregate metrics helpers ─────────────────────────────────────────────────

fn max_drawdown_from_returns(rets: &[f64]) -> f64 {
    let mut equity = 1.0_f64;
    let mut peak = 1.0_f64;
    let mut max_dd = 0.0_f64;
    for &r in rets {
        equity *= 1.0 + r;
        if equity > peak {
            peak = equity;
        }
        let dd = if peak > 0.0 {
            (peak - equity) / peak
        } else {
            0.0
        };
        if dd > max_dd {
            max_dd = dd;
        }
    }
    max_dd
}

/// Walk-Forward Efficiency: mean(OOS Sharpe) / mean(IS Sharpe).  Clamped ±3.
fn mean_wfe(is_sharpes: &[f64], oos_sharpes: &[f64]) -> f64 {
    if is_sharpes.is_empty() {
        return 0.0;
    }
    let m_is = is_sharpes.iter().sum::<f64>() / is_sharpes.len() as f64;
    let m_oos = oos_sharpes.iter().sum::<f64>() / oos_sharpes.len() as f64;
    if m_is.abs() < 1e-6 {
        return 0.0;
    }
    (m_oos / m_is).clamp(-3.0, 3.0)
}

fn fmt_pf(pf: f64) -> String {
    if pf.is_infinite() {
        "∞".to_string()
    } else {
        format!("{pf:.3}")
    }
}

// ── CLI entry point ───────────────────────────────────────────────────────────

pub fn run_benchmark_qua68(args: BenchmarkQua68Args) -> anyhow::Result<()> {
    let total_bars = args.train_window + args.n_folds * args.oos_window + 1;

    println!("QUA-68: Bayesian vs EMA AdaptiveSignalCombiner — OOS Walk-Forward Benchmark");
    println!(
        "  {} symbols | {} folds | {}d train | {}d OOS | {} bars/symbol",
        args.n_symbols, args.n_folds, args.train_window, args.oos_window, total_bars
    );
    println!("  Synthetic: regime-switching GBM (15% up/15% dn trends, 1.5% daily vol)");
    println!(
        "  Variants: [Baseline] EMA-IC + vol-threshold  vs  [Bayesian] NormalGamma IC + HMM\n"
    );

    let mut bayes_oos_rets: Vec<f64> = Vec::new();
    let mut base_oos_rets: Vec<f64> = Vec::new();
    let mut bayes_trades: Vec<f64> = Vec::new();
    let mut base_trades: Vec<f64> = Vec::new();
    let mut bayes_is_sh: Vec<f64> = Vec::new();
    let mut base_is_sh: Vec<f64> = Vec::new();
    let mut bayes_oos_sh: Vec<f64> = Vec::new();
    let mut base_oos_sh: Vec<f64> = Vec::new();

    for sym in 0..args.n_symbols {
        let seed = 0xDEAD_BEEF_u64.wrapping_add((sym as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let prices = synthetic_prices(total_bars, seed);
        let r = run_wf_symbol(
            &prices,
            args.train_window,
            args.oos_window,
            args.n_folds,
            args.commission,
        );

        bayes_oos_rets.extend_from_slice(&r.bayes_oos_rets);
        base_oos_rets.extend_from_slice(&r.base_oos_rets);
        bayes_trades.extend_from_slice(&r.bayes_trade_rets);
        base_trades.extend_from_slice(&r.base_trade_rets);
        bayes_is_sh.extend_from_slice(&r.bayes_is_sharpes);
        base_is_sh.extend_from_slice(&r.base_is_sharpes);
        bayes_oos_sh.extend_from_slice(&r.bayes_oos_sharpes);
        base_oos_sh.extend_from_slice(&r.base_oos_sharpes);
    }

    // ── Aggregate metrics ────────────────────────────────────────────────────
    let bayes_sharpe = sharpe_ratio(&bayes_oos_rets);
    let base_sharpe = sharpe_ratio(&base_oos_rets);
    let bayes_pf = profit_factor(&bayes_trades);
    let base_pf = profit_factor(&base_trades);
    let bayes_maxdd = max_drawdown_from_returns(&bayes_oos_rets);
    let base_maxdd = max_drawdown_from_returns(&base_oos_rets);
    let bayes_wfe = mean_wfe(&bayes_is_sh, &bayes_oos_sh);
    let base_wfe = mean_wfe(&base_is_sh, &base_oos_sh);

    // ── Results table ────────────────────────────────────────────────────────
    println!("┌──────────────────────────┬──────────────┬──────────────────────┐");
    println!("│  Metric                  │   Baseline   │   Bayesian (Ph. 1)   │");
    println!("├──────────────────────────┼──────────────┼──────────────────────┤");
    println!(
        "│  OOS Sharpe (ann.)       │  {:>9.3}   │  {:>9.3}             │",
        base_sharpe, bayes_sharpe
    );
    println!(
        "│  Profit Factor           │  {:>9}   │  {:>9}             │",
        fmt_pf(base_pf),
        fmt_pf(bayes_pf)
    );
    println!(
        "│  Max Drawdown            │  {:>8.2} %  │  {:>8.2} %            │",
        base_maxdd * 100.0,
        bayes_maxdd * 100.0
    );
    println!(
        "│  WFE                     │  {:>9.3}   │  {:>9.3}             │",
        base_wfe, bayes_wfe
    );
    println!(
        "│  Total OOS Trades        │  {:>9}   │  {:>9}             │",
        base_trades.len(),
        bayes_trades.len()
    );
    println!("├──────────────────────────┼──────────────┼──────────────────────┤");
    println!("│  Delta (Bayesian − Base) │              │                      │");
    println!(
        "│    Sharpe  Δ             │              │  {:>+9.3}             │",
        bayes_sharpe - base_sharpe
    );
    println!(
        "│    MaxDD  Δ              │              │  {:>+8.2} %            │",
        (bayes_maxdd - base_maxdd) * 100.0
    );
    println!(
        "│    WFE  Δ                │              │  {:>+9.3}             │",
        bayes_wfe - base_wfe
    );
    println!("└──────────────────────────┴──────────────┴──────────────────────┘");

    // ── CRO gate assessment ──────────────────────────────────────────────────
    println!("\n── CRO Gate Assessment ──────────────────────────────────────────");
    let base_pf_v = if base_pf.is_infinite() {
        999.0
    } else {
        base_pf
    };
    let bayes_pf_v = if bayes_pf.is_infinite() {
        999.0
    } else {
        bayes_pf
    };

    println!(
        "  PF target: ≥ 1.25  |  Baseline PF: {}  |  Bayesian PF: {}",
        fmt_pf(base_pf),
        fmt_pf(bayes_pf)
    );
    if bayes_pf_v >= 1.25 {
        println!("  PASS  Bayesian PF ≥ 1.25 — Phase 1 clears CRO Gate 2 threshold.");
        println!("  → Update QUA-55, schedule CRO Gate 2 review.");
    } else {
        let gap = 1.25 - bayes_pf_v;
        println!(
            "  FAIL  Bayesian PF {:.3} < 1.25 (gap: {:.3})",
            bayes_pf_v, gap
        );
        println!(
            "  → Phase 1 PF delta vs baseline: {:+.3}",
            bayes_pf_v - base_pf_v
        );
        println!("  → Recommend Phase 2: hierarchical MCMC (cross-signal covariance).");
    }

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════════
// QUA-92: Vol Regime Sleeve — PF Gap Closure Attempt
// ═══════════════════════════════════════════════════════════════════════════════
//
// Three runs on the same 50 synthetic symbols / 64-fold WF grid as QUA-68:
//
//   Run A — signal_expansion_ensemble (QUA-85 4-sleeve control)
//             momentum 35% / trend 30% / mean_reversion 15% / adaptive 20%
//
//   Run B — vol_regime_ensemble (5-sleeve treatment)
//             momentum 30% / trend 25% / mean_reversion 20% /
//             vol_regime 15% / adaptive 10%
//
//   Run C — vol_regime_standalone (alpha isolation, CRO gate: Sharpe ≥ 0.50)
//
// Vol regime sleeve = VolatilitySignal(period=20, low_vol=0.12, high_vol=0.40)
//                   + ReturnQualitySignal(period=60, sharpe_cap=3.0)
//                   combined EQUAL_WEIGHT, long-only.

/// QUA-92 benchmark CLI arguments.
#[derive(Args)]
pub struct BenchmarkQua92Args {
    /// Number of synthetic symbols to simulate.
    #[arg(long, default_value = "50")]
    pub n_symbols: usize,

    /// Number of walk-forward folds.
    #[arg(long, default_value = "64")]
    pub n_folds: usize,

    /// Rolling IS window in trading-day bars.
    #[arg(long, default_value = "90")]
    pub train_window: usize,

    /// OOS step in trading-day bars.
    #[arg(long, default_value = "30")]
    pub oos_window: usize,

    /// One-way commission fraction (e.g. 0.001 = 10 bps).
    #[arg(long, default_value = "0.001")]
    pub commission: f64,

    /// Directory to write results JSON and validation table.
    #[arg(long, default_value = "backtest-results/vol-regime")]
    pub output_dir: String,
}

// ── Signal kernels ─────────────────────────────────────────────────────────────

/// VolatilitySignal kernel.
///
/// Annualises the `period`-bar sample std (ddof=1) of `rets` by × √252.
/// Returns +1.0 when `ann_vol < low_vol`, −1.0 when `ann_vol > high_vol`,
/// and interpolates linearly between.  Returns 0.0 when insufficient data.
fn volatility_score(rets: &[f64], period: usize, low_vol: f64, high_vol: f64) -> f64 {
    if rets.len() < period || period < 2 {
        return 0.0;
    }
    let w = &rets[rets.len() - period..];
    let mean = w.iter().sum::<f64>() / period as f64;
    let var = w.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (period - 1) as f64;
    let ann_vol = var.sqrt() * 252_f64.sqrt();
    if ann_vol < low_vol {
        1.0
    } else if ann_vol > high_vol {
        -1.0
    } else {
        1.0 - 2.0 * (ann_vol - low_vol) / (high_vol - low_vol)
    }
}

/// ReturnQualitySignal kernel.
///
/// Computes the annualised Sharpe (ddof=1) of `rets` over `period` bars,
/// then normalises to [−1, 1] by dividing by `sharpe_cap`.
fn return_quality_score(rets: &[f64], period: usize, sharpe_cap: f64) -> f64 {
    if rets.len() < period || period < 2 {
        return 0.0;
    }
    let w = &rets[rets.len() - period..];
    let mean = w.iter().sum::<f64>() / period as f64;
    let var = w.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (period - 1) as f64;
    let std = var.sqrt();
    if std < 1e-10 {
        return 0.0;
    }
    let ann_sharpe = mean / std * 252_f64.sqrt();
    (ann_sharpe / sharpe_cap).clamp(-1.0, 1.0)
}

/// Vol regime combined score — EQUAL_WEIGHT, long-only (clamped to [0, 1]).
///
/// VolatilitySignal(period=20, low_vol=0.12, high_vol=0.40) ×½ +
/// ReturnQualitySignal(period=60, sharpe_cap=3.0) ×½, clamped to [0, 1].
#[inline]
fn vol_regime_combined(rets: &[f64]) -> f64 {
    let vs = volatility_score(rets, 20, 0.12, 0.40);
    let qs = return_quality_score(rets, 60, 3.0);
    (0.5 * vs + 0.5 * qs).max(0.0)
}

// ── Per-symbol walk-forward ───────────────────────────────────────────────────

struct Qua92SymbolResult {
    // Run A: signal_expansion_ensemble
    a_oos: Vec<f64>,
    a_trades: Vec<f64>,
    a_is_sh: Vec<f64>,
    a_oos_sh: Vec<f64>,
    // Run B: vol_regime_ensemble
    b_oos: Vec<f64>,
    b_trades: Vec<f64>,
    b_is_sh: Vec<f64>,
    b_oos_sh: Vec<f64>,
    // Run C: vol_regime_standalone
    c_oos: Vec<f64>,
    c_trades: Vec<f64>,
    c_is_sh: Vec<f64>,
    c_oos_sh: Vec<f64>,
}

fn run_qua92_wf_symbol(
    prices: &[f64],
    train_window: usize,
    oos_window: usize,
    n_folds: usize,
    commission: f64,
) -> Qua92SymbolResult {
    let n = prices.len();
    // ReturnQualitySignal needs 60 bars; VolatilitySignal needs 20.
    let min_warmup = 60_usize;

    let rets = qf::returns(prices);
    let rsi_vals = qf::rsi(prices, 14);
    let bb_mid = qf::bb_mid(prices, 20);
    let bb_upper = qf::bb_upper(prices, 20, 2.0);
    let bb_lower = qf::bb_lower(prices, 20, 2.0);
    let macd_hist = qf::macd_histogram(prices, 12, 26, 9);
    let fast_ma = qf::ema(prices, 12);
    let slow_ma = qf::ema(prices, 26);

    // Rolling 20-bar vol for the vol-threshold regime gate (shared across runs)
    let rolling_vols: Vec<f64> = (0..n)
        .map(|b| {
            if b < 21 {
                return 0.01_f64;
            }
            let w = &rets[b - 20..b];
            let m: f64 = w.iter().sum::<f64>() / 20.0;
            (w.iter().map(|r| (r - m).powi(2)).sum::<f64>() / 19.0)
                .sqrt()
                .max(1e-8)
        })
        .collect();

    // Bayesian IC combiner for the adaptive sleeve (NormalGamma, same as QUA-68)
    let mut ic_comb = AdaptiveSignalCombiner::new(3);

    let mut a_sigs = vec![0.0_f64; n];
    let mut b_sigs = vec![0.0_f64; n];
    let mut c_sigs = vec![0.0_f64; n];

    for bar in 0..n - 1 {
        let next_ret = (prices[bar + 1] - prices[bar]) / prices[bar];

        if bar < min_warmup {
            continue;
        }

        // Pure signal scores
        let (mom, _, _) = momentum_signal(&rsi_vals[..=bar], &rets[..=bar], 20, 0.02);
        let (mr, _, _) = mean_reversion_signal(
            &bb_mid[..=bar],
            &bb_upper[..=bar],
            &bb_lower[..=bar],
            &rets[..=bar],
            2.0,
        );
        let (tf, _, _) =
            trend_following_signal(&macd_hist[..=bar], &fast_ma[..=bar], &slow_ma[..=bar]);
        let vol = vol_regime_combined(&rets[..=bar]);

        // Adaptive sleeve: Bayesian IC-weighted combo of the 3 pure signals
        let adaptive = ic_comb.combine(&[mom, mr, tf]);

        // Regime gate (vol-threshold, identical to QUA-68 baseline)
        let active = threshold_low_vol(&rolling_vols, bar);

        // Run A — signal_expansion_ensemble (QUA-85 weights)
        // momentum 35% / trend 30% / mean_reversion 15% / adaptive 20%
        let a_raw = 0.35 * mom + 0.30 * tf + 0.15 * mr + 0.20 * adaptive;
        a_sigs[bar] = quantise(a_raw, active, 0.20);

        // Run B — vol_regime_ensemble (5-sleeve)
        // momentum 30% / trend 25% / mean_reversion 20% / vol_regime 15% / adaptive 10%
        let b_raw = 0.30 * mom + 0.25 * tf + 0.20 * mr + 0.15 * vol + 0.10 * adaptive;
        b_sigs[bar] = quantise(b_raw, active, 0.20);

        // Run C — vol_regime_standalone (long-only, no additional regime gate)
        // vol ∈ [0, 1]; binary long (1.0) when score > 0.15, else flat.
        c_sigs[bar] = if vol > 0.15 { 1.0 } else { 0.0 };

        // Update IC tracker with realised IC observations
        ic_comb.update_ic(0, mom * next_ret);
        ic_comb.update_ic(1, mr * next_ret);
        ic_comb.update_ic(2, tf * next_ret);
    }

    // Precompute net-return series for all three runs
    let a_net = compute_net_returns(prices, &a_sigs, commission);
    let b_net = compute_net_returns(prices, &b_sigs, commission);
    let c_net = compute_net_returns(prices, &c_sigs, commission);

    let mut result = Qua92SymbolResult {
        a_oos: Vec::new(),
        a_trades: Vec::new(),
        a_is_sh: Vec::new(),
        a_oos_sh: Vec::new(),
        b_oos: Vec::new(),
        b_trades: Vec::new(),
        b_is_sh: Vec::new(),
        b_oos_sh: Vec::new(),
        c_oos: Vec::new(),
        c_trades: Vec::new(),
        c_is_sh: Vec::new(),
        c_oos_sh: Vec::new(),
    };

    for fold in 0..n_folds {
        let oos_start = train_window + fold * oos_window;
        let oos_end = oos_start + oos_window;
        if oos_end >= n {
            break;
        }
        let is_start = oos_start.saturating_sub(train_window);

        result
            .a_is_sh
            .push(sharpe_ratio(&a_net[is_start..oos_start]));
        result
            .b_is_sh
            .push(sharpe_ratio(&b_net[is_start..oos_start]));
        result
            .c_is_sh
            .push(sharpe_ratio(&c_net[is_start..oos_start]));

        let oos_prices = &prices[oos_start..=oos_end];

        let mut a_p = a_sigs[oos_start..oos_end].to_vec();
        a_p.push(0.0);
        let mut b_p = b_sigs[oos_start..oos_end].to_vec();
        b_p.push(0.0);
        let mut c_p = c_sigs[oos_start..oos_end].to_vec();
        c_p.push(0.0);

        let ar = run_backtest(oos_prices, &a_p, commission, 1.0);
        let br = run_backtest(oos_prices, &b_p, commission, 1.0);
        let cr = run_backtest(oos_prices, &c_p, commission, 1.0);

        result.a_oos_sh.push(sharpe_ratio(&ar.net_returns));
        result.b_oos_sh.push(sharpe_ratio(&br.net_returns));
        result.c_oos_sh.push(sharpe_ratio(&cr.net_returns));

        result.a_trades.extend(ar.trades.iter().map(|t| t.ret));
        result.b_trades.extend(br.trades.iter().map(|t| t.ret));
        result.c_trades.extend(cr.trades.iter().map(|t| t.ret));

        result.a_oos.extend_from_slice(&ar.net_returns[1..]);
        result.b_oos.extend_from_slice(&br.net_returns[1..]);
        result.c_oos.extend_from_slice(&cr.net_returns[1..]);
    }

    result
}

// ── CLI entry point ───────────────────────────────────────────────────────────

pub fn run_benchmark_qua92(args: BenchmarkQua92Args) -> anyhow::Result<()> {
    let total_bars = args.train_window + args.n_folds * args.oos_window + 1;

    println!("QUA-92: Vol Regime Sleeve — PF Gap Closure Attempt");
    println!(
        "  {} symbols | {} folds | {}d IS (rolling) | {}d OOS | {} bars/symbol",
        args.n_symbols, args.n_folds, args.train_window, args.oos_window, total_bars
    );
    println!("  Synthetic: regime-switching GBM (75% LowVol/25% HighVol, 15%/15% drift)");
    println!("  Vol sleeve: VolatilitySignal(20, 0.12, 0.40) + ReturnQualitySignal(60, 3.0)");
    println!("  CRO targets: PF >= 1.26, MaxDD < 19.50%, Sharpe >= 1.00, WFE >= 0.80\n");

    // Accumulators for each run
    let mut a_oos: Vec<f64> = Vec::new();
    let mut a_trades: Vec<f64> = Vec::new();
    let mut a_is_sh: Vec<f64> = Vec::new();
    let mut a_oos_sh: Vec<f64> = Vec::new();

    let mut b_oos: Vec<f64> = Vec::new();
    let mut b_trades: Vec<f64> = Vec::new();
    let mut b_is_sh: Vec<f64> = Vec::new();
    let mut b_oos_sh: Vec<f64> = Vec::new();

    let mut c_oos: Vec<f64> = Vec::new();
    let mut c_trades: Vec<f64> = Vec::new();
    let mut c_is_sh: Vec<f64> = Vec::new();
    let mut c_oos_sh: Vec<f64> = Vec::new();

    for sym in 0..args.n_symbols {
        // Same seed formula as QUA-68 for identical synthetic universe
        let seed = 0xDEAD_BEEF_u64.wrapping_add((sym as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let prices = synthetic_prices(total_bars, seed);
        let r = run_qua92_wf_symbol(
            &prices,
            args.train_window,
            args.oos_window,
            args.n_folds,
            args.commission,
        );

        a_oos.extend_from_slice(&r.a_oos);
        a_trades.extend_from_slice(&r.a_trades);
        a_is_sh.extend_from_slice(&r.a_is_sh);
        a_oos_sh.extend_from_slice(&r.a_oos_sh);

        b_oos.extend_from_slice(&r.b_oos);
        b_trades.extend_from_slice(&r.b_trades);
        b_is_sh.extend_from_slice(&r.b_is_sh);
        b_oos_sh.extend_from_slice(&r.b_oos_sh);

        c_oos.extend_from_slice(&r.c_oos);
        c_trades.extend_from_slice(&r.c_trades);
        c_is_sh.extend_from_slice(&r.c_is_sh);
        c_oos_sh.extend_from_slice(&r.c_oos_sh);
    }

    // ── Aggregate metrics ─────────────────────────────────────────────────────
    let a_sharpe = sharpe_ratio(&a_oos);
    let b_sharpe = sharpe_ratio(&b_oos);
    let c_sharpe = sharpe_ratio(&c_oos);

    let a_pf = profit_factor(&a_trades);
    let b_pf = profit_factor(&b_trades);
    let c_pf = profit_factor(&c_trades);

    let a_maxdd = max_drawdown_from_returns(&a_oos);
    let b_maxdd = max_drawdown_from_returns(&b_oos);
    let c_maxdd = max_drawdown_from_returns(&c_oos);

    let a_wfe = mean_wfe(&a_is_sh, &a_oos_sh);
    let b_wfe = mean_wfe(&b_is_sh, &b_oos_sh);
    let c_wfe = mean_wfe(&c_is_sh, &c_oos_sh);

    let n_folds_actual = a_oos_sh.len();

    // ── Results table ─────────────────────────────────────────────────────────
    println!("┌────────────────────────────┬──────────────┬──────────────┬──────────────┐");
    println!("│  Metric                    │    Run A     │    Run B     │    Run C     │");
    println!("│                            │ 4-sleeve WF  │ 5-sleeve WF  │ vol_standalone│");
    println!("├────────────────────────────┼──────────────┼──────────────┼──────────────┤");
    println!(
        "│  OOS Sharpe (ann.)         │  {:>9.3}   │  {:>9.3}   │  {:>9.3}   │",
        a_sharpe, b_sharpe, c_sharpe
    );
    println!(
        "│  Profit Factor             │  {:>9}   │  {:>9}   │  {:>9}   │",
        fmt_pf(a_pf),
        fmt_pf(b_pf),
        fmt_pf(c_pf)
    );
    println!(
        "│  Max Drawdown              │  {:>8.2} %  │  {:>8.2} %  │  {:>8.2} %  │",
        a_maxdd * 100.0,
        b_maxdd * 100.0,
        c_maxdd * 100.0
    );
    println!(
        "│  WFE                       │  {:>9.3}   │  {:>9.3}   │  {:>9.3}   │",
        a_wfe, b_wfe, c_wfe
    );
    println!(
        "│  Total OOS Folds           │  {:>9}   │  {:>9}   │  {:>9}   │",
        n_folds_actual,
        b_oos_sh.len(),
        c_oos_sh.len()
    );
    println!("└────────────────────────────┴──────────────┴──────────────┴──────────────┘");

    // ── Delta vs Run A ────────────────────────────────────────────────────────
    let b_pf_v = if b_pf.is_infinite() { 99.0 } else { b_pf };
    let a_pf_v = if a_pf.is_infinite() { 99.0 } else { a_pf };

    println!("\n── Run B vs Run A (vol_regime_ensemble vs signal_expansion) ─────────────");
    println!(
        "  Sharpe  Δ: {:>+.3}  ({:.3} → {:.3})",
        b_sharpe - a_sharpe,
        a_sharpe,
        b_sharpe
    );
    println!(
        "  PF      Δ: {:>+.3}  ({} → {})",
        b_pf_v - a_pf_v,
        fmt_pf(a_pf),
        fmt_pf(b_pf)
    );
    println!(
        "  MaxDD   Δ: {:>+.2} %  ({:.2}% → {:.2}%)",
        (b_maxdd - a_maxdd) * 100.0,
        a_maxdd * 100.0,
        b_maxdd * 100.0
    );
    println!(
        "  WFE     Δ: {:>+.3}  ({:.3} → {:.3})",
        b_wfe - a_wfe,
        a_wfe,
        b_wfe
    );

    // ── CRO gate assessment ───────────────────────────────────────────────────
    println!("\n── CRO Gate Assessment (Run B — vol_regime_ensemble) ────────────────────");

    let gate_sharpe = b_sharpe >= 1.00;
    let gate_pf = b_pf_v >= 1.26;
    let gate_maxdd = b_maxdd < 0.195;
    let gate_wfe = b_wfe >= 0.80;
    let gate_c_sharpe = c_sharpe >= 0.50;

    let all_pass = gate_sharpe && gate_pf && gate_maxdd && gate_wfe && gate_c_sharpe;

    println!(
        "  [{}] OOS Sharpe >= 1.00:  {:.3}",
        if gate_sharpe { "PASS" } else { "FAIL" },
        b_sharpe
    );
    println!(
        "  [{}] Profit Factor >= 1.26: {}",
        if gate_pf { "PASS" } else { "FAIL" },
        fmt_pf(b_pf)
    );
    println!(
        "  [{}] Max Drawdown < 19.50%: {:.2}%",
        if gate_maxdd { "PASS" } else { "FAIL" },
        b_maxdd * 100.0
    );
    println!(
        "  [{}] WFE >= 0.80:          {:.3}",
        if gate_wfe { "PASS" } else { "FAIL" },
        b_wfe
    );
    println!(
        "  [{}] Run C Sharpe >= 0.50 (genuine alpha): {:.3}",
        if gate_c_sharpe { "PASS" } else { "FAIL" },
        c_sharpe
    );
    println!(
        "\n  → Overall: {}",
        if all_pass {
            "PASS — vol_regime_ensemble clears all CRO gates. Submit to CRO for gate decision."
        } else {
            "FAIL — one or more CRO gates not met. Review Run B metrics and notify CRO."
        }
    );

    // Also check PF improvement vs Run A
    if b_pf_v > a_pf_v {
        println!("  → PF improvement confirmed (+{:.3})", b_pf_v - a_pf_v);
    } else {
        println!(
            "  → PF did NOT improve vs Run A ({:.3} vs {:.3})",
            b_pf_v, a_pf_v
        );
        println!("     CRO will reject vol sleeve per gate-decision rules.");
    }

    // ── Write results to disk ─────────────────────────────────────────────────
    let out_dir = Path::new(&args.output_dir);
    fs::create_dir_all(out_dir)?;

    // results.json
    let results_json = serde_json::json!({
        "signal_expansion_ensemble": {
            "run": "A",
            "oos_sharpe": a_sharpe,
            "profit_factor": if a_pf.is_infinite() { 99.0 } else { a_pf },
            "wf_efficiency": a_wfe,
            "max_drawdown": a_maxdd,
            "n_folds": n_folds_actual,
            "passes": a_sharpe >= 0.60 && a_pf_v >= 1.10 && a_maxdd < 0.20 && a_wfe >= 0.20
        },
        "vol_regime_ensemble": {
            "run": "B",
            "oos_sharpe": b_sharpe,
            "profit_factor": if b_pf.is_infinite() { 99.0 } else { b_pf },
            "wf_efficiency": b_wfe,
            "max_drawdown": b_maxdd,
            "n_folds": b_oos_sh.len(),
            "passes": all_pass
        },
        "vol_regime_standalone": {
            "run": "C",
            "oos_sharpe": c_sharpe,
            "profit_factor": if c_pf.is_infinite() { 99.0 } else { c_pf },
            "wf_efficiency": c_wfe,
            "max_drawdown": c_maxdd,
            "n_folds": c_oos_sh.len(),
            "passes": gate_c_sharpe
        }
    });

    fs::write(
        out_dir.join("results.json"),
        serde_json::to_string_pretty(&results_json)?,
    )?;

    // validation_table.md
    let md = format!(
        "# QUA-92 Vol Regime Sleeve Results\n\n\
        ## CRO Gate Metrics\n\n\
        | Run | Sharpe | PF | WFE | Max DD | Folds | Status |\n\
        |-----|--------|-----|-----|--------|-------|--------|\n\
        | signal_expansion_ensemble (Run A)  | {:.3} | {} | {:.3} | {:.2}% | {} | {} |\n\
        | vol_regime_ensemble (Run B)        | {:.3} | {} | {:.3} | {:.2}% | {} | {} |\n\
        | vol_regime_standalone (Run C)      | {:.3} | {} | {:.3} | {:.2}% | {} | {} |\n\n\
        ## CRO Thresholds (Run B)\n\n\
        | Gate | Target | Result |\n\
        |------|--------|--------|\n\
        | Sharpe | >= 1.00 | {} |\n\
        | PF | >= 1.26 | {} |\n\
        | MaxDD | < 19.50% | {} |\n\
        | WFE | >= 0.80 | {} |\n\
        | Run C Sharpe (alpha check) | >= 0.50 | {} |\n",
        a_sharpe,
        fmt_pf(a_pf),
        a_wfe,
        a_maxdd * 100.0,
        n_folds_actual,
        if a_sharpe >= 0.60 && a_pf_v >= 1.10 && a_maxdd < 0.20 && a_wfe >= 0.20 {
            "PASS"
        } else {
            "FAIL"
        },
        b_sharpe,
        fmt_pf(b_pf),
        b_wfe,
        b_maxdd * 100.0,
        b_oos_sh.len(),
        if all_pass { "PASS" } else { "FAIL" },
        c_sharpe,
        fmt_pf(c_pf),
        c_wfe,
        c_maxdd * 100.0,
        c_oos_sh.len(),
        if gate_c_sharpe { "PASS" } else { "FAIL" },
        if gate_sharpe { "PASS" } else { "FAIL" },
        if gate_pf { "PASS" } else { "FAIL" },
        if gate_maxdd { "PASS" } else { "FAIL" },
        if gate_wfe { "PASS" } else { "FAIL" },
        if gate_c_sharpe { "PASS" } else { "FAIL" },
    );

    fs::write(out_dir.join("validation_table.md"), md)?;

    println!("\n  Results written to {}/", args.output_dir);

    Ok(())
}
