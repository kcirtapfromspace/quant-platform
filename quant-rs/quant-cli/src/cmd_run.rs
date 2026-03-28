//! `quant run once` — execute a single portfolio rebalance cycle.
//!
//! Pulls historical returns from the DuckDB store, computes technical features,
//! generates signals (momentum, mean-reversion, trend-following), combines them
//! into alpha scores, optimises the portfolio, and submits orders to a paper OMS.

use std::collections::HashMap;

use chrono::NaiveDate;
use clap::Args;
use tracing::{info, warn};

use quant_data::MarketDataStore;
use quant_features as qf;
use quant_oms::{Order, OrderManagementSystem, OrderSide, OrderType};
use quant_portfolio::alpha::{AlphaCombiner, CombinationMethod, SignalInput};
use quant_portfolio::optimizer::{optimize, PortfolioConstraints};
use quant_portfolio::{estimate_covariance, OptimizationMethod, Rebalancer};
use quant_risk::ExposureLimits;
use quant_signals::{mean_reversion_signal, momentum_signal, trend_following_signal};

#[derive(Args)]
pub struct RunOnceArgs {
    /// Path to the DuckDB market data file.
    #[arg(long)]
    pub db: String,

    /// Initial paper-trading cash balance.
    #[arg(long, default_value = "1000000")]
    pub cash: f64,

    /// Portfolio optimisation method.
    #[arg(long, default_value = "risk_parity",
          value_parser = parse_optimizer)]
    pub optimizer: OptimizationMethod,

    /// Comma-separated symbols (overrides built-in universe).
    #[arg(long)]
    pub symbols: Option<String>,

    /// Number of trading-day bars used for covariance estimation.
    #[arg(long, default_value = "252")]
    pub lookback_days: usize,

    /// Minimum dollar amount per order (skip smaller trades).
    #[arg(long, default_value = "100")]
    pub min_order_value: f64,

    /// Path to write Prometheus textfile metrics after each cycle.
    /// The file is consumed by `quant serve --metrics-file`.
    /// Defaults to /tmp/quant_paper_metrics.prom when not set.
    #[arg(long, default_value = "/tmp/quant_paper_metrics.prom")]
    pub metrics_file: String,
}

pub fn run_once(args: RunOnceArgs) -> anyhow::Result<()> {
    let universe = resolve_symbols(args.symbols.as_deref());

    // ── 1. Load return history from DuckDB ────────────────────────────────
    let store = MarketDataStore::open(&args.db)?;
    let (symbols_ordered, returns_matrix) = load_returns(&store, &universe, args.lookback_days)?;

    let n = symbols_ordered.len();
    if n == 0 {
        anyhow::bail!("No symbols with data found in the database.");
    }

    let bars = returns_matrix.len() / n;
    info!("{} symbols loaded with {} bars", n, bars);

    // ── 2. Signal-driven alpha scores ─────────────────────────────────────
    let alpha_scores = generate_alpha_scores(&store, &symbols_ordered, args.lookback_days)?;

    // ── 3. Covariance matrix ──────────────────────────────────────────────
    // Pass None for shrinkage to use Ledoit-Wolf automatic estimation.
    let cov = estimate_covariance(&returns_matrix, n, None)
        .map_err(|e| anyhow::anyhow!("covariance estimation failed: {}", e))?;

    // ── 4. Portfolio optimisation ─────────────────────────────────────────
    let constraints = PortfolioConstraints {
        long_only: true,
        max_weight: 0.25,
        min_weight: 0.0,
        max_sector_weight: 1.0,
    };
    let opt_result = optimize(
        args.optimizer,
        &symbols_ordered,
        &alpha_scores,
        &cov,
        &constraints,
    )
    .map_err(|e| anyhow::anyhow!("optimiser failed: {}", e))?;

    info!(
        "Optimised: vol={:.2}%  weights={:?}",
        opt_result.risk * 100.0,
        opt_result
            .weights
            .iter()
            .zip(&symbols_ordered)
            .map(|(w, s)| format!("{}={:.1}%", s, w * 100.0))
            .collect::<Vec<_>>()
    );

    // ── 5. Paper OMS + rebalance ──────────────────────────────────────────
    let mut oms = OrderManagementSystem::new_in_memory()?;
    oms.set_cash(args.cash);

    let portfolio_value = args.cash; // fresh start — no existing positions
    let current_weights: HashMap<String, f64> = HashMap::new();

    let target_weights: HashMap<String, f64> = symbols_ordered
        .iter()
        .zip(&opt_result.weights)
        .map(|(s, &w)| (s.clone(), w))
        .collect();

    let rebalancer = Rebalancer::default();
    let rebalance = rebalancer.rebalance(&target_weights, &current_weights, portfolio_value);

    if !rebalance.rebalance_triggered {
        println!("No rebalance needed (portfolio already on target).");
        return Ok(());
    }

    let limits = ExposureLimits::default();
    let mut submitted = 0usize;
    let mut rejected = 0usize;

    for trade in &rebalance.trades {
        if trade.dollar_amount < args.min_order_value {
            warn!(
                "Skip {} {} — ${:.0} below minimum ${:.0}",
                trade.side, trade.symbol, trade.dollar_amount, args.min_order_value
            );
            rejected += 1;
            continue;
        }

        let fraction = trade.dollar_amount / portfolio_value;
        if fraction > limits.max_position_fraction {
            warn!(
                "Risk rejected {} {} — fraction {:.1}% > max {:.1}%",
                trade.side,
                trade.symbol,
                fraction * 100.0,
                limits.max_position_fraction * 100.0
            );
            rejected += 1;
            continue;
        }

        // Use dollar_amount as quantity with a notional price of 1.0.
        let order = Order::new(
            trade.symbol.clone(),
            if trade.side == "BUY" {
                OrderSide::Buy
            } else {
                OrderSide::Sell
            },
            trade.dollar_amount,
            OrderType::Market,
        );

        match oms.submit_order(order) {
            Ok(_) => submitted += 1,
            Err(e) => {
                warn!("OMS rejected {} {}: {}", trade.side, trade.symbol, e);
                rejected += 1;
            }
        }
    }

    // ── 6. Signal-driven output summary ──────────────────────────────────
    println!("Signal alpha scores:");
    for (sym, &alpha) in symbols_ordered.iter().zip(&alpha_scores) {
        println!("  {}: {:.3}", sym, alpha);
    }

    println!(
        "Done: portfolio=${:.0}  submitted={}  rejected={}  vol={:.2}%  turnover={:.2}%",
        portfolio_value,
        submitted,
        rejected,
        opt_result.risk * 100.0,
        rebalance.turnover * 100.0,
    );

    // ── 7. Emit Prometheus metrics ────────────────────────────────────────
    let pnl_cumulative = oms.portfolio_value() - args.cash;
    let daily_pnl_pct = pnl_cumulative / args.cash;
    if let Err(e) = write_paper_metrics(&args.metrics_file, pnl_cumulative, daily_pnl_pct) {
        warn!(
            "failed to write paper metrics to {}: {e}",
            args.metrics_file
        );
    }

    Ok(())
}

/// Write `quant_paper_pnl_cumulative` and `quant_paper_daily_pnl_pct` to a
/// Prometheus text-format file so that `quant serve` can expose them.
fn write_paper_metrics(path: &str, pnl_cumulative: f64, daily_pnl_pct: f64) -> anyhow::Result<()> {
    use std::io::Write as _;
    let mut f = std::fs::File::create(path)?;
    write!(
        f,
        "# HELP quant_paper_pnl_cumulative Running cumulative P&L in USD since strategy inception\n\
         # TYPE quant_paper_pnl_cumulative gauge\n\
         quant_paper_pnl_cumulative {pnl_cumulative}\n\
         # HELP quant_paper_daily_pnl_pct Daily P&L as a fraction of starting notional\n\
         # TYPE quant_paper_daily_pnl_pct gauge\n\
         quant_paper_daily_pnl_pct {daily_pnl_pct}\n"
    )?;
    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Load a flat returns matrix (row-major: bar × symbol) from DuckDB.
/// Returns (symbol_order, flat_matrix).
fn load_returns(
    store: &MarketDataStore,
    universe: &[String],
    lookback_days: usize,
) -> anyhow::Result<(Vec<String>, Vec<f64>)> {
    let today = chrono::Local::now().date_naive();
    let start = today - chrono::Duration::days((lookback_days as i64) * 2);

    let mut per_symbol: HashMap<String, Vec<(NaiveDate, f64)>> = HashMap::new();

    for sym in universe {
        let records = store.query(sym, start, today)?;
        if records.len() < 2 {
            continue;
        }
        let returns: Vec<(NaiveDate, f64)> = records
            .windows(2)
            .map(|w| {
                let ret = if w[0].adj_close > 0.0 {
                    (w[1].adj_close - w[0].adj_close) / w[0].adj_close
                } else {
                    0.0
                };
                (w[1].date, ret)
            })
            .collect();
        per_symbol.insert(sym.clone(), returns);
    }

    // Align on shared dates present in all symbols.
    let common_dates: Vec<NaiveDate> = {
        let mut date_sets: Vec<std::collections::HashSet<NaiveDate>> = per_symbol
            .values()
            .map(|v| v.iter().map(|(d, _)| *d).collect())
            .collect();
        if date_sets.is_empty() {
            return Ok((vec![], vec![]));
        }
        let mut common = date_sets.remove(0);
        for s in date_sets {
            common = common.intersection(&s).copied().collect();
        }
        let mut dates: Vec<NaiveDate> = common.into_iter().collect();
        dates.sort();
        if dates.len() > lookback_days {
            dates = dates[dates.len() - lookback_days..].to_vec();
        }
        dates
    };

    let date_idx: HashMap<NaiveDate, usize> = common_dates
        .iter()
        .enumerate()
        .map(|(i, d)| (*d, i))
        .collect();

    let bars = common_dates.len();
    let symbols_ordered: Vec<String> = {
        let mut s: Vec<String> = per_symbol.keys().cloned().collect();
        s.sort();
        s
    };
    let n = symbols_ordered.len();

    if bars == 0 || n == 0 {
        return Ok((vec![], vec![]));
    }

    // Flat matrix: row = bar, col = symbol → index = bar * n + sym_idx
    let mut matrix = vec![0.0_f64; bars * n];
    for (sym_idx, sym) in symbols_ordered.iter().enumerate() {
        if let Some(returns) = per_symbol.get(sym) {
            for (date, ret) in returns {
                if let Some(&bar_idx) = date_idx.get(date) {
                    matrix[bar_idx * n + sym_idx] = *ret;
                }
            }
        }
    }

    Ok((symbols_ordered, matrix))
}

// ── Signal generation ─────────────────────────────────────────────────────────

/// Per-symbol signal decomposition returned by the signal pipeline.
#[derive(Debug, Clone)]
struct SignalDecomposition {
    pub momentum: (f64, f64),
    pub mean_reversion: (f64, f64),
    pub trend_following: (f64, f64),
    pub combined_alpha: f64,
}

/// Compute the combined alpha score for a single symbol from its price series.
///
/// Runs features → signals → conviction-weighted combination.
/// Returns `None` if the price series is too short (< 50 bars).
fn compute_symbol_alpha(adj_close: &[f64]) -> Option<SignalDecomposition> {
    if adj_close.len() < 50 {
        return None;
    }

    let rets = qf::returns(adj_close);
    let rsi_values = qf::rsi(adj_close, 14);
    let bb_mid = qf::bb_mid(adj_close, 20);
    let bb_upper = qf::bb_upper(adj_close, 20, 2.0);
    let bb_lower = qf::bb_lower(adj_close, 20, 2.0);
    let macd_hist = qf::macd_histogram(adj_close, 12, 26, 9);
    let fast_ma = qf::ema(adj_close, 12);
    let slow_ma = qf::ema(adj_close, 26);

    let (mom_score, mom_conf, _) = momentum_signal(&rsi_values, &rets, 20, 0.02);
    let (mr_score, mr_conf, _) = mean_reversion_signal(&bb_mid, &bb_upper, &bb_lower, &rets, 2.0);
    let (tf_score, tf_conf, _) = trend_following_signal(&macd_hist, &fast_ma, &slow_ma);

    let signal_inputs = vec![
        SignalInput {
            signal_name: "momentum".into(),
            score: mom_score,
            confidence: mom_conf,
            target_position: (mom_score * mom_conf).clamp(-1.0, 1.0),
        },
        SignalInput {
            signal_name: "mean_reversion".into(),
            score: mr_score,
            confidence: mr_conf,
            target_position: (mr_score * mr_conf).clamp(-1.0, 1.0),
        },
        SignalInput {
            signal_name: "trend_following".into(),
            score: tf_score,
            confidence: tf_conf,
            target_position: (tf_score * tf_conf).clamp(-1.0, 1.0),
        },
    ];

    let combiner = AlphaCombiner::new(CombinationMethod::ConvictionWeighted, None);
    let alpha = combiner.combine("_", &signal_inputs);

    Some(SignalDecomposition {
        momentum: (mom_score, mom_conf),
        mean_reversion: (mr_score, mr_conf),
        trend_following: (tf_score, tf_conf),
        combined_alpha: alpha.target_position,
    })
}

/// Compute alpha scores for each symbol by running the full signal pipeline.
fn generate_alpha_scores(
    store: &MarketDataStore,
    symbols: &[String],
    lookback_days: usize,
) -> anyhow::Result<Vec<f64>> {
    let today = chrono::Local::now().date_naive();
    let start = today - chrono::Duration::days((lookback_days as i64) * 2);

    let mut alpha_scores = Vec::with_capacity(symbols.len());

    for sym in symbols {
        let records = store.query(sym, start, today)?;
        let adj_close: Vec<f64> = records.iter().map(|r| r.adj_close).collect();

        match compute_symbol_alpha(&adj_close) {
            Some(decomp) => {
                info!(
                    "{}: mom={:.2}/{:.2}  mr={:.2}/{:.2}  tf={:.2}/{:.2}  → alpha={:.3}",
                    sym,
                    decomp.momentum.0,
                    decomp.momentum.1,
                    decomp.mean_reversion.0,
                    decomp.mean_reversion.1,
                    decomp.trend_following.0,
                    decomp.trend_following.1,
                    decomp.combined_alpha,
                );
                alpha_scores.push(decomp.combined_alpha);
            }
            None => {
                info!(
                    "{}: insufficient history ({} bars), using neutral alpha",
                    sym,
                    adj_close.len()
                );
                alpha_scores.push(0.0);
            }
        }
    }

    Ok(alpha_scores)
}

fn resolve_symbols(arg: Option<&str>) -> Vec<String> {
    if let Some(s) = arg {
        return s.split(',').map(|x| x.trim().to_uppercase()).collect();
    }
    vec!["AAPL", "GOOG", "MSFT", "AMZN", "META", "NVDA", "JPM", "XOM"]
        .into_iter()
        .map(String::from)
        .collect()
}

fn parse_optimizer(s: &str) -> Result<OptimizationMethod, String> {
    match s {
        "equal_weight" => Ok(OptimizationMethod::EqualWeight),
        "risk_parity" => Ok(OptimizationMethod::RiskParity),
        "min_variance" | "minimum_variance" => Ok(OptimizationMethod::MinVariance),
        "mean_variance" => Ok(OptimizationMethod::MeanVariance),
        other => Err(format!(
            "unknown optimizer '{}' (expected: equal_weight, risk_parity, min_variance, mean_variance)",
            other
        )),
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic price series with deterministic drift + noise.
    fn synthetic_prices(n: usize, drift: f64) -> Vec<f64> {
        let mut prices = vec![100.0_f64];
        let mut state: u64 = 42;
        for _ in 1..n {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let noise = ((state >> 33) as f64) / (u32::MAX as f64) - 0.5;
            let ret = drift + noise * 0.02;
            prices.push((prices.last().unwrap() * (1.0 + ret)).max(1.0));
        }
        prices
    }

    #[test]
    fn test_compute_symbol_alpha_returns_none_for_short_series() {
        let prices = vec![100.0; 30];
        assert!(compute_symbol_alpha(&prices).is_none());
    }

    #[test]
    fn test_compute_symbol_alpha_returns_some_for_sufficient_data() {
        let prices = synthetic_prices(252, 0.001);
        let result = compute_symbol_alpha(&prices);
        assert!(result.is_some());
    }

    #[test]
    fn test_alpha_output_in_valid_range() {
        let prices = synthetic_prices(252, 0.001);
        let decomp = compute_symbol_alpha(&prices).unwrap();
        assert!((-1.0..=1.0).contains(&decomp.combined_alpha));
        assert!((-1.0..=1.0).contains(&decomp.momentum.0));
        assert!((0.0..=1.0).contains(&decomp.momentum.1));
        assert!((-1.0..=1.0).contains(&decomp.mean_reversion.0));
        assert!((0.0..=1.0).contains(&decomp.mean_reversion.1));
        assert!((-1.0..=1.0).contains(&decomp.trend_following.0));
        assert!((0.0..=1.0).contains(&decomp.trend_following.1));
    }

    #[test]
    fn test_strong_uptrend_produces_positive_alpha() {
        // Strong consistent uptrend — momentum and trend should be bullish.
        let prices: Vec<f64> = (0..252).map(|i| 100.0 * 1.003_f64.powi(i)).collect();
        let decomp = compute_symbol_alpha(&prices).unwrap();
        // Momentum (RSI-based) should be positive in strong uptrend
        assert!(
            decomp.momentum.0 > 0.0,
            "momentum score={}",
            decomp.momentum.0
        );
    }

    #[test]
    fn test_strong_downtrend_produces_negative_momentum() {
        let prices: Vec<f64> = (0..252).map(|i| 200.0 * 0.997_f64.powi(i)).collect();
        let decomp = compute_symbol_alpha(&prices).unwrap();
        assert!(
            decomp.momentum.0 < 0.0,
            "momentum score={}",
            decomp.momentum.0
        );
    }

    #[test]
    fn test_flat_prices_produce_near_zero_alpha() {
        let prices = vec![100.0_f64; 252];
        let decomp = compute_symbol_alpha(&prices).unwrap();
        // Flat prices → all signals near zero
        assert!(
            decomp.combined_alpha.abs() < 0.3,
            "alpha={}",
            decomp.combined_alpha
        );
    }

    #[test]
    fn test_resolve_symbols_default() {
        let syms = resolve_symbols(None);
        assert_eq!(syms.len(), 8);
        assert!(syms.contains(&"AAPL".to_string()));
    }

    #[test]
    fn test_resolve_symbols_custom() {
        let syms = resolve_symbols(Some("tsla, goog"));
        assert_eq!(syms, vec!["TSLA", "GOOG"]);
    }

    #[test]
    fn test_parse_optimizer_variants() {
        assert_eq!(
            parse_optimizer("equal_weight").unwrap(),
            OptimizationMethod::EqualWeight
        );
        assert_eq!(
            parse_optimizer("risk_parity").unwrap(),
            OptimizationMethod::RiskParity
        );
        assert_eq!(
            parse_optimizer("min_variance").unwrap(),
            OptimizationMethod::MinVariance
        );
        assert!(parse_optimizer("unknown").is_err());
    }
}
