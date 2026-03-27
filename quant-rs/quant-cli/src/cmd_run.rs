//! `quant run once` — execute a single portfolio rebalance cycle.
//!
//! Pulls historical returns from the DuckDB store, runs the Rust portfolio
//! engine, validates proposed trades through the risk engine, and submits
//! approved orders to a paper OMS.

use std::collections::HashMap;

use chrono::NaiveDate;
use clap::Args;
use tracing::{info, warn};

use quant_data::MarketDataStore;
use quant_oms::{Order, OrderManagementSystem, OrderSide, OrderType};
use quant_portfolio::optimizer::{optimize, PortfolioConstraints};
use quant_portfolio::{estimate_covariance, OptimizationMethod, Rebalancer};
use quant_risk::ExposureLimits;

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
}

pub fn run_once(args: RunOnceArgs) -> anyhow::Result<()> {
    let universe = resolve_symbols(args.symbols.as_deref());

    // ── 1. Load return history from DuckDB ────────────────────────────────
    let store = MarketDataStore::open(&args.db)?;
    let (symbols_ordered, returns_matrix) =
        load_returns(&store, &universe, args.lookback_days)?;

    let n = symbols_ordered.len();
    if n == 0 {
        anyhow::bail!("No symbols with data found in the database.");
    }

    let bars = returns_matrix.len() / n;
    info!("{} symbols loaded with {} bars", n, bars);

    // ── 2. Alpha scores (equal-weight flat signal — no live signals wired) ─
    let alpha_scores: Vec<f64> = vec![1.0_f64; n];

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
    let opt_result =
        optimize(args.optimizer, &symbols_ordered, &alpha_scores, &cov, &constraints)
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
            if trade.side == "BUY" { OrderSide::Buy } else { OrderSide::Sell },
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

    println!(
        "Done: portfolio=${:.0}  submitted={}  rejected={}  vol={:.2}%  turnover={:.2}%",
        portfolio_value,
        submitted,
        rejected,
        opt_result.risk * 100.0,
        rebalance.turnover * 100.0,
    );

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

fn resolve_symbols(arg: Option<&str>) -> Vec<String> {
    if let Some(s) = arg {
        return s.split(',').map(|x| x.trim().to_uppercase()).collect();
    }
    vec![
        "AAPL", "GOOG", "MSFT", "AMZN", "META", "NVDA", "JPM", "XOM",
    ]
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
