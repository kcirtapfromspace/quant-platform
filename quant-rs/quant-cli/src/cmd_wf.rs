//! `quant wf` — real-data walk-forward backtest for CRO gate validation.
//!
//! Generates signals using the same pipeline as `quant run once`
//! (RSI/momentum + Bollinger Band/mean-reversion + MACD/trend-following),
//! then runs an expanding walk-forward on real OHLCV data from DuckDB.
//!
//! # CRO gate spec (CEO-approved)
//! Default config: IS=90d, OOS=30d, step=30d, expanding=true, 64 folds.
//! Gate thresholds: Sharpe ≥ 0.60, PF ≥ 1.10, MaxDD < 20.00%, WFE ≥ 0.20.
//!
//! # No-lookahead guarantee
//! Features (RSI, Bollinger Bands, MACD) are computed for the **full** price
//! series in a single causal pass — `feature[t]` depends only on
//! `adj_close[0..=t]`.  The signal matrix is therefore safe to slice at any
//! fold boundary without re-running the signal pipeline.

use std::collections::{HashMap, HashSet};

use chrono::NaiveDate;
use clap::Args;
use tracing::info;

use quant_backtest::{run_walk_forward, PortfolioBacktestConfig, WalkForwardConfig};
use quant_data::MarketDataStore;
use quant_features as qf;
use quant_signals::{mean_reversion_signal, momentum_signal, trend_following_signal};

// ── CRO gate thresholds (CEO-approved recalibrated values) ───────────────────
// Source: CEO gate decision referenced in QUA-49, QUA-58, QUA-85 gate reviews.
// IMPORTANT: these are the *minimum* pass thresholds. Strategy-specific targets
// (e.g. QUA-92 aspirational PF ≥ 1.26) are higher — a PASS here is necessary
// but not sufficient for CRO sign-off on a specific strategy upgrade.

const GATE_SHARPE: f64 = 0.60;
const GATE_PF: f64 = 1.10;
const GATE_MAXDD: f64 = 0.2000; // 20.00 %
const GATE_WFE: f64 = 0.20;

// ── CLI args ──────────────────────────────────────────────────────────────────

/// Arguments for `quant wf`.
#[derive(Args)]
pub struct WfArgs {
    /// Path to the DuckDB market data file.
    #[arg(long)]
    pub db: String,

    /// Comma-separated symbols to include.  Defaults to the 50-symbol S&P 500 universe.
    #[arg(long)]
    pub symbols: Option<String>,

    /// In-sample window length in trading-day bars (default: 90).
    #[arg(long, default_value = "90")]
    pub is_days: usize,

    /// Out-of-sample window per fold in bars (default: 30).
    #[arg(long, default_value = "30")]
    pub oos_days: usize,

    /// IS step per fold in bars — IS grows by this amount each fold (default: 30).
    #[arg(long, default_value = "30")]
    pub step_days: usize,

    /// Maximum number of folds (default: 64).
    #[arg(long, default_value = "64")]
    pub n_folds: usize,

    /// Start date for data range (YYYY-MM-DD).  Defaults to earliest available.
    #[arg(long)]
    pub start: Option<String>,

    /// End date for data range (YYYY-MM-DD).  Defaults to today.
    #[arg(long)]
    pub end: Option<String>,

    /// One-way commission fraction (default: 0.001 = 10 bps).
    #[arg(long, default_value = "0.001")]
    pub commission: f64,

    /// Starting portfolio value (default: 1 000 000).
    #[arg(long, default_value = "1000000")]
    pub initial_capital: f64,

    /// Rebalance the portfolio every N bars (default: 21 ≈ monthly).
    #[arg(long, default_value = "21")]
    pub rebalance_every: usize,

    /// Write full results to this JSON file.
    #[arg(long)]
    pub output_json: Option<String>,

    /// Comma-separated signals to use: momentum, mean_reversion, trend (default: all three).
    #[arg(long, default_value = "momentum,mean_reversion,trend")]
    pub signals: String,
}

// ── Entry point ───────────────────────────────────────────────────────────────

pub fn run_wf(args: WfArgs) -> anyhow::Result<()> {
    let store = MarketDataStore::open(&args.db)?;
    let universe = resolve_symbols(args.symbols.as_deref());

    let start = args
        .start
        .as_deref()
        .map(|s| {
            NaiveDate::parse_from_str(s, "%Y-%m-%d")
                .map_err(|e| anyhow::anyhow!("invalid --start '{}': {}", s, e))
        })
        .transpose()?
        .unwrap_or_else(|| NaiveDate::from_ymd_opt(1900, 1, 1).unwrap());

    let end = args
        .end
        .as_deref()
        .map(|s| {
            NaiveDate::parse_from_str(s, "%Y-%m-%d")
                .map_err(|e| anyhow::anyhow!("invalid --end '{}': {}", s, e))
        })
        .transpose()?
        .unwrap_or_else(|| chrono::Local::now().date_naive());

    let sig_flags = SignalFlags::from_str(&args.signals);

    info!(
        "Loading data for {} symbols [{} → {}]",
        universe.len(),
        start,
        end
    );

    // ── 1. Load and align price data ──────────────────────────────────────────
    let (symbols, adj_close_matrix, common_dates) =
        load_and_align(&store, &universe, start, end)?;

    let n_assets = symbols.len();
    if n_assets == 0 {
        anyhow::bail!("No symbols with sufficient data found in {}", args.db);
    }
    let n_bars = adj_close_matrix.len() / n_assets;

    info!(
        "Aligned {} symbols × {} bars ({} → {})",
        n_assets,
        n_bars,
        common_dates.first().unwrap_or(&start),
        common_dates.last().unwrap_or(&end),
    );

    let min_required = args.is_days + args.oos_days;
    if n_bars < min_required {
        anyhow::bail!(
            "Insufficient bars: have {n_bars}, need at least {min_required} \
             (is_days={} + oos_days={}). Widen the date range or reduce window sizes.",
            args.is_days,
            args.oos_days,
        );
    }

    // ── 2. Compute returns matrix ─────────────────────────────────────────────
    //
    // returns[t][j] = (adj_close[t][j] - adj_close[t-1][j]) / adj_close[t-1][j]
    // Bar 0 return is undefined — set to 0.  The matrix is then [n_bars × n_assets].
    let returns = build_returns(&adj_close_matrix, n_assets, n_bars);

    // ── 3. Generate signal matrix (no lookahead) ───────────────────────────────
    //
    // For each asset, compute features for the FULL price series (causal) and
    // then emit a signal value per bar using those features.
    let signals = build_signals(&adj_close_matrix, n_assets, n_bars, &sig_flags);

    // ── 4. Run walk-forward ───────────────────────────────────────────────────
    let wf_config = WalkForwardConfig {
        is_days: args.is_days,
        oos_days: args.oos_days,
        step_days: args.step_days,
        expanding: true,
        n_folds: args.n_folds,
    };
    let bt_config = PortfolioBacktestConfig {
        initial_capital: args.initial_capital,
        commission_pct: args.commission,
        rebalance_every: args.rebalance_every,
    };

    info!(
        "Running walk-forward: IS={} OOS={} step={} folds≤{} expanding=true",
        args.is_days, args.oos_days, args.step_days, args.n_folds
    );

    let result = run_walk_forward(&symbols, &returns, &signals, &wf_config, &bt_config);

    // ── 5. Print results ──────────────────────────────────────────────────────
    print_results(&result, &symbols, n_bars, &common_dates);

    // ── 6. Gate check ─────────────────────────────────────────────────────────
    print_gate_check(&result);

    // ── 7. Optionally write JSON ──────────────────────────────────────────────
    if let Some(path) = &args.output_json {
        write_json(path, &result, &symbols, &args, n_bars, &common_dates)?;
        println!("\nResults written to {path}");
    }

    Ok(())
}

// ── Signal generation ──────────────────────────────────────────────────────────

struct SignalFlags {
    momentum: bool,
    mean_reversion: bool,
    trend_following: bool,
}

impl SignalFlags {
    fn from_str(s: &str) -> Self {
        let lower = s.to_lowercase();
        let parts: Vec<&str> = lower.split(',').map(str::trim).collect();
        Self {
            momentum: parts.contains(&"momentum"),
            mean_reversion: parts.contains(&"mean_reversion"),
            trend_following: parts.iter().any(|&x| x == "trend" || x == "trend_following"),
        }
    }
}

/// Compute signal values for all assets and all bars.
///
/// Signal at bar `t` for asset `j` uses features that are causal at bar `t`.
/// Returns a flat `[n_bars × n_assets]` row-major matrix.
fn build_signals(
    adj_close: &[f64],
    n_assets: usize,
    n_bars: usize,
    flags: &SignalFlags,
) -> Vec<f64> {
    let mut signals = vec![0.0_f64; n_bars * n_assets];

    for j in 0..n_assets {
        // Extract price series for asset j.
        let prices: Vec<f64> = (0..n_bars).map(|t| adj_close[t * n_assets + j]).collect();

        // Compute all features for the full series (causal).
        let rets = qf::returns(&prices);
        let rsi_vals = qf::rsi(&prices, 14);
        let bb_mid = qf::bb_mid(&prices, 20);
        let bb_upper = qf::bb_upper(&prices, 20, 2.0);
        let bb_lower = qf::bb_lower(&prices, 20, 2.0);
        let macd_hist = qf::macd_histogram(&prices, 12, 26, 9);
        let fast_ma = qf::ema(&prices, 12);
        let slow_ma = qf::ema(&prices, 26);

        // Emit one signal per bar.
        for t in 0..n_bars {
            let signal = compute_bar_signal(
                t,
                &rets,
                &rsi_vals,
                &bb_mid,
                &bb_upper,
                &bb_lower,
                &macd_hist,
                &fast_ma,
                &slow_ma,
                flags,
            );
            signals[t * n_assets + j] = signal;
        }
    }

    signals
}

/// Compute the combined alpha signal for a single asset at a single bar.
///
/// Uses the prefix `[0..=t]` of each feature series, so there is no lookahead.
#[allow(clippy::too_many_arguments)]
fn compute_bar_signal(
    t: usize,
    rets: &[f64],
    rsi_vals: &[f64],
    bb_mid: &[f64],
    bb_upper: &[f64],
    bb_lower: &[f64],
    macd_hist: &[f64],
    fast_ma: &[f64],
    slow_ma: &[f64],
    flags: &SignalFlags,
) -> f64 {
    if t < 50 {
        return 0.0; // signal warmup — mirrors compute_symbol_alpha minimum
    }

    let end = t + 1; // prefix up to and including bar t

    let mut n_active = 0usize;
    let mut alpha_sum = 0.0_f64;

    if flags.momentum {
        let (score, conf, _) = momentum_signal(&rsi_vals[..end], &rets[..end.min(rets.len())], 20, 0.02);
        let alpha = (score * conf).clamp(-1.0, 1.0);
        alpha_sum += alpha;
        n_active += 1;
    }
    if flags.mean_reversion {
        let (score, conf, _) = mean_reversion_signal(
            &bb_mid[..end],
            &bb_upper[..end],
            &bb_lower[..end],
            &rets[..end.min(rets.len())],
            2.0,
        );
        let alpha = (score * conf).clamp(-1.0, 1.0);
        alpha_sum += alpha;
        n_active += 1;
    }
    if flags.trend_following {
        let (score, conf, _) =
            trend_following_signal(&macd_hist[..end], &fast_ma[..end], &slow_ma[..end]);
        let alpha = (score * conf).clamp(-1.0, 1.0);
        alpha_sum += alpha;
        n_active += 1;
    }

    if n_active == 0 {
        0.0
    } else {
        alpha_sum / n_active as f64
    }
}

// ── Data loading ───────────────────────────────────────────────────────────────

/// Load and align adj-close prices for all symbols onto a shared date grid.
///
/// Returns `(symbols_ordered, flat_adj_close [n_bars × n_assets], common_dates)`.
/// Symbols with fewer than 50 valid bars are dropped.
fn load_and_align(
    store: &MarketDataStore,
    universe: &[String],
    start: NaiveDate,
    end: NaiveDate,
) -> anyhow::Result<(Vec<String>, Vec<f64>, Vec<NaiveDate>)> {
    let mut per_symbol: HashMap<String, Vec<(NaiveDate, f64)>> = HashMap::new();

    for sym in universe {
        let records = store.query(sym, start, end)?;
        if records.len() < 50 {
            info!("{sym}: only {} bars — skipping", records.len());
            continue;
        }
        let series: Vec<(NaiveDate, f64)> = records.iter().map(|r| (r.date, r.adj_close)).collect();
        per_symbol.insert(sym.clone(), series);
    }

    if per_symbol.is_empty() {
        return Ok((vec![], vec![], vec![]));
    }

    // Find dates present in every symbol.
    let mut common: HashSet<NaiveDate> = per_symbol
        .values()
        .next()
        .unwrap()
        .iter()
        .map(|(d, _)| *d)
        .collect();
    for series in per_symbol.values() {
        let dates: HashSet<NaiveDate> = series.iter().map(|(d, _)| *d).collect();
        common = common.intersection(&dates).copied().collect();
    }

    let mut common_dates: Vec<NaiveDate> = common.into_iter().collect();
    common_dates.sort();

    if common_dates.is_empty() {
        return Ok((vec![], vec![], vec![]));
    }

    let date_idx: HashMap<NaiveDate, usize> = common_dates
        .iter()
        .enumerate()
        .map(|(i, d)| (*d, i))
        .collect();

    let n_bars = common_dates.len();
    let mut symbols_ordered: Vec<String> = per_symbol.keys().cloned().collect();
    symbols_ordered.sort();
    let n_assets = symbols_ordered.len();

    let mut matrix = vec![0.0_f64; n_bars * n_assets];
    for (j, sym) in symbols_ordered.iter().enumerate() {
        if let Some(series) = per_symbol.get(sym) {
            for (date, price) in series {
                if let Some(&t) = date_idx.get(date) {
                    matrix[t * n_assets + j] = *price;
                }
            }
        }
    }

    Ok((symbols_ordered, matrix, common_dates))
}

/// Compute returns from a flat adj-close matrix.
///
/// Returns a flat `[n_bars × n_assets]` matrix where bar 0 = 0.0.
fn build_returns(adj_close: &[f64], n_assets: usize, n_bars: usize) -> Vec<f64> {
    let mut rets = vec![0.0_f64; n_bars * n_assets];
    for t in 1..n_bars {
        for j in 0..n_assets {
            let prev = adj_close[(t - 1) * n_assets + j];
            let cur = adj_close[t * n_assets + j];
            rets[t * n_assets + j] = if prev > 0.0 {
                (cur - prev) / prev
            } else {
                0.0
            };
        }
    }
    rets
}

fn resolve_symbols(arg: Option<&str>) -> Vec<String> {
    if let Some(s) = arg {
        return s.split(',').map(|x| x.trim().to_uppercase()).collect();
    }
    // Default 50-symbol S&P 500 universe (mirrors QUA-85 gate).
    vec![
        "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK.B", "UNH", "LLY",
        "JPM", "V", "AVGO", "XOM", "PG", "MA", "HD", "COST", "JNJ", "MRK",
        "ABBV", "CVX", "CRM", "BAC", "NFLX", "PEP", "ADBE", "WMT", "TMO", "AMD",
        "ACN", "CSCO", "ABT", "DIS", "MCD", "WFC", "CAT", "DHR", "INTC", "VZ",
        "INTU", "CMCSA", "NKE", "AMGN", "IBM", "TXN", "GE", "HON", "RTX", "AXP",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

// ── Output helpers ────────────────────────────────────────────────────────────

fn print_results(
    result: &quant_backtest::WalkForwardResult,
    symbols: &[String],
    n_bars: usize,
    dates: &[NaiveDate],
) {
    println!("\n═══════════════════════════════════════════════════════");
    println!("  Walk-Forward Results — Real Data");
    println!("═══════════════════════════════════════════════════════");
    println!(
        "  Universe:   {} symbols × {} bars ({} → {})",
        symbols.len(),
        n_bars,
        dates.first().map(|d| d.to_string()).unwrap_or_default(),
        dates.last().map(|d| d.to_string()).unwrap_or_default(),
    );
    println!("  Folds completed: {}", result.n_folds_completed);
    println!();
    println!(
        "  OOS Sharpe:         {:.3}  (gate ≥ {GATE_SHARPE:.2})",
        result.oos_sharpe
    );
    println!(
        "  OOS Profit Factor:  {:.3}  (gate ≥ {GATE_PF:.2})",
        result.oos_profit_factor
    );
    println!(
        "  OOS MaxDD:         {:.2}%  (gate < {:.2}%)",
        result.oos_max_drawdown * 100.0,
        GATE_MAXDD * 100.0,
    );
    println!(
        "  WFE:                {:.3}  (gate ≥ {GATE_WFE:.2})",
        result.wfe
    );
    println!();

    if result.folds.len() <= 20 {
        println!("  Per-fold detail:");
        println!(
            "  {:>4}  {:>8}  {:>9}  {:>9}  {:>8}  {:>7}",
            "Fold", "IS Sharpe", "OOS Sharpe", "OOS MaxDD", "OOS PF", "WFE"
        );
        for f in &result.folds {
            println!(
                "  {:>4}  {:>8.3}  {:>9.3}  {:>8.2}%  {:>7.3}  {:>7.3}",
                f.fold_idx,
                f.is_sharpe,
                f.oos_sharpe,
                f.oos_max_drawdown * 100.0,
                f.oos_profit_factor.min(5.0),
                if f.wfe_ratio.is_finite() {
                    format!("{:.3}", f.wfe_ratio)
                } else {
                    "  N/A".to_string()
                },
            );
        }
        println!();
    }
}

fn print_gate_check(result: &quant_backtest::WalkForwardResult) {
    let sharpe_ok = result.oos_sharpe >= GATE_SHARPE;
    let pf_ok = result.oos_profit_factor >= GATE_PF;
    let maxdd_ok = result.oos_max_drawdown < GATE_MAXDD;
    let wfe_ok = result.wfe >= GATE_WFE;
    let all_pass = sharpe_ok && pf_ok && maxdd_ok && wfe_ok;

    println!("  CRO Gate Check (QUA-92 thresholds):");
    println!(
        "    Sharpe ≥ {GATE_SHARPE:.2}:     {}  ({:.3})",
        pass_fail(sharpe_ok),
        result.oos_sharpe
    );
    println!(
        "    PF ≥ {GATE_PF:.2}:        {}  ({:.3})",
        pass_fail(pf_ok),
        result.oos_profit_factor
    );
    println!(
        "    MaxDD < {:.2}%:    {}  ({:.2}%)",
        GATE_MAXDD * 100.0,
        pass_fail(maxdd_ok),
        result.oos_max_drawdown * 100.0,
    );
    println!(
        "    WFE ≥ {GATE_WFE:.2}:        {}  ({:.3})",
        pass_fail(wfe_ok),
        result.wfe
    );
    println!();
    println!(
        "  OVERALL: {}",
        if all_pass { "PASS ✓" } else { "FAIL ✗" }
    );
    println!("═══════════════════════════════════════════════════════");
}

fn pass_fail(ok: bool) -> &'static str {
    if ok { "PASS" } else { "FAIL" }
}

fn write_json(
    path: &str,
    result: &quant_backtest::WalkForwardResult,
    symbols: &[String],
    args: &WfArgs,
    n_bars: usize,
    dates: &[NaiveDate],
) -> anyhow::Result<()> {
    use std::io::Write as _;

    let folds_json: Vec<serde_json::Value> = result
        .folds
        .iter()
        .map(|f| {
            serde_json::json!({
                "fold": f.fold_idx,
                "is_sharpe": f.is_sharpe,
                "oos_sharpe": f.oos_sharpe,
                "oos_profit_factor": if f.oos_profit_factor.is_infinite() { 5.0 } else { f.oos_profit_factor },
                "oos_max_drawdown": f.oos_max_drawdown,
                "wfe_ratio": if f.wfe_ratio.is_finite() { f.wfe_ratio } else { 0.0 },
            })
        })
        .collect();

    let doc = serde_json::json!({
        "strategy": "signal_expansion_ensemble",
        "config": {
            "is_days": args.is_days,
            "oos_days": args.oos_days,
            "step_days": args.step_days,
            "expanding": true,
            "n_folds": args.n_folds,
            "commission": args.commission,
            "signals": args.signals,
        },
        "data": {
            "n_symbols": symbols.len(),
            "n_bars": n_bars,
            "start": dates.first().map(|d| d.to_string()),
            "end": dates.last().map(|d| d.to_string()),
        },
        "aggregate": {
            "n_folds_completed": result.n_folds_completed,
            "oos_sharpe": result.oos_sharpe,
            "oos_profit_factor": result.oos_profit_factor,
            "oos_max_drawdown": result.oos_max_drawdown,
            "wfe": result.wfe,
        },
        "gate": {
            "sharpe_pass": result.oos_sharpe >= GATE_SHARPE,
            "pf_pass": result.oos_profit_factor >= GATE_PF,
            "maxdd_pass": result.oos_max_drawdown < GATE_MAXDD,
            "wfe_pass": result.wfe >= GATE_WFE,
            "overall_pass": result.oos_sharpe >= GATE_SHARPE
                && result.oos_profit_factor >= GATE_PF
                && result.oos_max_drawdown < GATE_MAXDD
                && result.wfe >= GATE_WFE,
        },
        "folds": folds_json,
    });

    let mut f = std::fs::File::create(path)?;
    write!(f, "{}", serde_json::to_string_pretty(&doc)?)?;
    Ok(())
}
