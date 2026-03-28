//! `quant benchmark qua68` — OOS benchmark: Bayesian vs EMA AdaptiveSignalCombiner.
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
