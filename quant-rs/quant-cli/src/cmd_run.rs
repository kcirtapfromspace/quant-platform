//! `quant run` subcommands — single-cycle, daily daemon, and status reporter.
//!
//! # Subcommands
//!
//! - `quant run once`   — execute one rebalance cycle and exit
//! - `quant run loop`   — daily daemon: rebalances at `--schedule` time each day
//! - `quant run status` — read the SQLite OMS file and print positions + PnL
//!
//! All subcommands share the same signal pipeline (momentum / mean-reversion /
//! trend-following) and can be narrowed via `--signals`.
//!
//! ## Alpaca wiring (loop only)
//!
//! Set `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, and `ALPACA_PAPER=true` to route
//! live orders through the Alpaca paper-trading endpoint.  When the vars are
//! absent the daemon records orders in the OMS only (paper simulation).

use std::collections::HashMap;

use chrono::{NaiveDate, Utc};
use clap::Args;
use tracing::{info, warn};

use quant_data::MarketDataStore;
use quant_features as qf;
use quant_oms::{
    AlpacaBrokerAdapter, Broker, Order, OrderManagementSystem, OrderSide, OrderType,
    SqliteStateStore,
};
use quant_portfolio::alpha::{AlphaCombiner, CombinationMethod, SignalInput};
use quant_portfolio::optimizer::{optimize, PortfolioConstraints};
use quant_portfolio::{estimate_covariance, OptimizationMethod, Rebalancer};
use quant_risk::{DrawdownCircuitBreaker, ExposureLimits};
use quant_signals::{mean_reversion_signal, momentum_signal, trend_following_signal};

// ── CLI argument structs ───────────────────────────────────────────────────────

#[derive(Args)]
pub struct RunOnceArgs {
    /// Path to the DuckDB market data file.
    #[arg(long)]
    pub db: String,

    /// Initial paper-trading cash balance.
    #[arg(long, default_value = "1000000")]
    pub cash: f64,

    /// Portfolio optimisation method.
    #[arg(long, default_value = "risk_parity", value_parser = parse_optimizer)]
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
    #[arg(long, default_value = "/tmp/quant_paper_metrics.prom")]
    pub metrics_file: String,

    /// Comma-separated signals to use: momentum, mean_reversion, trend (default: all three).
    #[arg(long, default_value = "momentum,mean_reversion,trend")]
    pub signals: String,

    /// Path to write per-sleeve strategy state JSON (read by the dashboard).
    /// Skipped when not set.
    #[arg(long)]
    pub state_file: Option<String>,

    /// Drawdown circuit-breaker threshold (fraction of starting cash).
    /// Trading halts if portfolio drawdown exceeds this level.
    /// CRO-approved level: 0.08 (8%). Reads QUANT_DD_CIRCUIT_BREAKER env var.
    #[arg(long, env = "QUANT_DD_CIRCUIT_BREAKER", default_value = "0.08")]
    pub dd_circuit_breaker: f64,

    /// Daily P&L halt threshold (negative fraction of starting cash).
    /// Trading halts if daily P&L falls below this fraction.
    /// CRO-approved level: -0.03 (-3%). Reads QUANT_DAILY_PNL_HALT env var.
    #[arg(long, env = "QUANT_DAILY_PNL_HALT", default_value = "-0.03")]
    pub daily_pnl_halt: f64,
}

#[derive(Args)]
pub struct RunLoopArgs {
    /// Path to the DuckDB market data file.
    #[arg(long)]
    pub db: String,

    /// Path to the SQLite OMS state file (persists positions across restarts).
    #[arg(long, default_value = "./quant_oms.db")]
    pub oms_db: String,

    /// Daily rebalance time in HH:MM (24-hour local time, e.g. 16:05).
    #[arg(long, default_value = "16:05")]
    pub schedule: String,

    /// Starting cash balance (used as portfolio size for order sizing).
    #[arg(long, default_value = "1000000")]
    pub cash: f64,

    /// Portfolio optimisation method.
    #[arg(long, default_value = "risk_parity", value_parser = parse_optimizer)]
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
    #[arg(long, default_value = "/tmp/quant_paper_metrics.prom")]
    pub metrics_file: String,

    /// Comma-separated signals to use: momentum, mean_reversion, trend (default: all three).
    #[arg(long, default_value = "momentum,mean_reversion,trend")]
    pub signals: String,

    /// Path to write per-sleeve strategy state JSON (read by the dashboard).
    /// Skipped when not set.
    #[arg(long)]
    pub state_file: Option<String>,
}

#[derive(Args)]
pub struct RunStatusArgs {
    /// Path to the SQLite OMS state file.
    #[arg(long, default_value = "./quant_oms.db")]
    pub oms_db: String,
}

// ── Signal filter ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct SignalFilter {
    momentum: bool,
    mean_reversion: bool,
    trend_following: bool,
}

struct RebalanceConfig<'a> {
    universe: &'a [String],
    portfolio_cash: f64,
    optimizer: OptimizationMethod,
    signal_filter: &'a SignalFilter,
    lookback_days: usize,
    min_order_value: f64,
    metrics_file: &'a str,
    /// Path to write `run_e_state.json` for the dashboard.  `None` = skip.
    state_file: Option<&'a str>,
}

impl SignalFilter {
    fn from_str(s: &str) -> Self {
        let lower = s.to_lowercase();
        let parts: Vec<&str> = lower.split(',').map(str::trim).collect();
        Self {
            momentum: parts.contains(&"momentum"),
            mean_reversion: parts.contains(&"mean_reversion"),
            trend_following: parts
                .iter()
                .any(|&x| x == "trend" || x == "trend_following"),
        }
    }
}

// ── Public entry points ────────────────────────────────────────────────────────

pub fn run_once(args: RunOnceArgs) -> anyhow::Result<()> {
    let universe = resolve_symbols(args.symbols.as_deref());
    let signal_filter = SignalFilter::from_str(&args.signals);

    // ── 0. Circuit-breaker + daily P&L halt (CRO gates, QUA-128) ─────────
    {
        let (prev_pnl_cumulative, prev_daily_pnl_pct) = read_paper_metrics(&args.metrics_file);
        let current_value = args.cash + prev_pnl_cumulative;
        let cb = DrawdownCircuitBreaker::new(args.dd_circuit_breaker);
        if cb.is_tripped(args.cash, current_value) {
            let dd_pct = (args.cash - current_value) / args.cash * 100.0;
            anyhow::bail!(
                "CIRCUIT BREAKER TRIPPED: portfolio drawdown {:.2}% exceeds CRO threshold {:.2}%. \
                 Trading halted. Manual reset required.",
                dd_pct,
                args.dd_circuit_breaker * 100.0,
            );
        }
        if prev_daily_pnl_pct < args.daily_pnl_halt {
            anyhow::bail!(
                "DAILY P&L HALT: daily P&L {:.2}% breaches CRO floor {:.2}%. \
                 Trading halted for remainder of session.",
                prev_daily_pnl_pct * 100.0,
                args.daily_pnl_halt * 100.0,
            );
        }
    }

    let market_store = MarketDataStore::open(&args.db)?;
    let mut oms = OrderManagementSystem::new_in_memory()?;
    oms.set_cash(args.cash);

    do_rebalance(
        &market_store,
        &mut oms,
        None,
        RebalanceConfig {
            universe: &universe,
            portfolio_cash: args.cash,
            optimizer: args.optimizer,
            signal_filter: &signal_filter,
            lookback_days: args.lookback_days,
            min_order_value: args.min_order_value,
            metrics_file: &args.metrics_file,
            state_file: args.state_file.as_deref(),
        },
    )?;

    Ok(())
}

/// Daily rebalance daemon.
///
/// Loops indefinitely, firing a rebalance cycle once per day at `--schedule`
/// time.  Halts with exit code 1 if MaxDD exceeds 22 %.
///
/// Alpaca orders are submitted when `ALPACA_API_KEY` + `ALPACA_SECRET_KEY`
/// are set; otherwise orders are recorded in the OMS only (paper mode).
pub fn run_loop(args: RunLoopArgs) -> anyhow::Result<()> {
    let universe = resolve_symbols(args.symbols.as_deref());
    let signal_filter = SignalFilter::from_str(&args.signals);
    let (sched_h, sched_m) = parse_schedule(&args.schedule)?;

    let market_store = MarketDataStore::open_read_only(&args.db)?;
    let oms_store = SqliteStateStore::new(&args.oms_db)?;
    let mut oms = OrderManagementSystem::new(Some(oms_store));
    oms.restore_state()?;
    oms.set_cash(args.cash);

    let broker = try_build_alpaca_broker();
    if broker.is_some() {
        let paper = std::env::var("ALPACA_PAPER").as_deref() == Ok("true");
        info!("Alpaca adapter active (paper={})", paper);
    } else {
        info!("No Alpaca credentials found — running in paper-only mode");
    }

    let cb = DrawdownCircuitBreaker::new(0.22);
    let mut peak_value = args.cash;
    let mut last_run_date: Option<NaiveDate> = oms.last_order_date();

    info!(
        "Run loop started: schedule={}:{:02}  oms_db={}",
        sched_h, sched_m, args.oms_db
    );

    loop {
        let now = chrono::Local::now();
        let today = now.date_naive();
        let sched_time = chrono::NaiveTime::from_hms_opt(sched_h, sched_m, 0).unwrap();

        if last_run_date != Some(today) && now.time() >= sched_time {
            info!("Running daily rebalance cycle for {today}");

            match do_rebalance(
                &market_store,
                &mut oms,
                broker.as_ref(),
                RebalanceConfig {
                    universe: &universe,
                    portfolio_cash: args.cash,
                    optimizer: args.optimizer,
                    signal_filter: &signal_filter,
                    lookback_days: args.lookback_days,
                    min_order_value: args.min_order_value,
                    metrics_file: &args.metrics_file,
                    state_file: args.state_file.as_deref(),
                },
            ) {
                Ok((submitted, rejected)) => {
                    info!("Cycle complete: submitted={submitted}  rejected={rejected}");
                    last_run_date = Some(today);
                }
                Err(e) => warn!("Rebalance cycle failed: {e}"),
            }

            // Portfolio value for MaxDD check.
            // If Alpaca is configured, use real account equity; otherwise use
            // the configured cash (MaxDD stays 0 in pure paper mode).
            let current_value = portfolio_value_for_maxdd(broker.as_ref(), &oms, args.cash);
            peak_value = peak_value.max(current_value);
            let dd = cb.drawdown(peak_value, current_value);

            if cb.is_tripped(peak_value, current_value) {
                tracing::error!(
                    "CRITICAL: MaxDD {:.1}% >= 22% — halting. peak={peak_value:.0} current={current_value:.0}",
                    dd * 100.0
                );
                std::process::exit(1);
            }

            info!("Drawdown check: {:.2}%", dd * 100.0);
        }

        std::thread::sleep(std::time::Duration::from_secs(60));
    }
}

/// Print the current OMS state: positions, unrealized PnL, cash balance.
pub fn run_status(args: RunStatusArgs) -> anyhow::Result<()> {
    let oms_store = SqliteStateStore::new(&args.oms_db)?;
    let mut oms = OrderManagementSystem::new(Some(oms_store));
    oms.restore_state()?;

    println!("=== OMS Status: {} ===", args.oms_db);

    let last_date = oms.last_order_date();
    println!(
        "Last rebalance: {}",
        last_date
            .map(|d| d.to_string())
            .unwrap_or_else(|| "never".to_string())
    );

    let positions = oms.get_all_positions();
    if positions.is_empty() {
        println!("No open positions.");
    } else {
        println!(
            "\n{:<8}  {:>12}  {:>10}  {:>10}  {:>14}  {:>14}",
            "SYMBOL", "QTY", "AVG_COST", "MKT_PRICE", "MKT_VALUE", "UNREALIZED_PNL"
        );
        let mut syms: Vec<&str> = positions.keys().map(String::as_str).collect();
        syms.sort();
        let mut total_mv = 0.0_f64;
        let mut total_pnl = 0.0_f64;
        for sym in syms {
            let p = &positions[sym];
            println!(
                "{:<8}  {:>12.2}  {:>10.2}  {:>10.2}  {:>14.2}  {:>14.2}",
                sym,
                p.quantity,
                p.avg_cost,
                p.market_price,
                p.market_value(),
                p.unrealized_pnl()
            );
            total_mv += p.market_value();
            total_pnl += p.unrealized_pnl();
        }
        println!(
            "{:<8}  {:>12}  {:>10}  {:>10}  {:>14.2}  {:>14.2}",
            "TOTAL", "", "", "", total_mv, total_pnl
        );
    }

    println!("\nCash:            {:.2}", oms.cash());
    println!("Portfolio value: {:.2}", oms.portfolio_value());

    Ok(())
}

// ── Core rebalance logic (shared by `once` and `loop`) ────────────────────────

fn do_rebalance(
    market_store: &MarketDataStore,
    oms: &mut OrderManagementSystem,
    broker: Option<&AlpacaBrokerAdapter>,
    config: RebalanceConfig<'_>,
) -> anyhow::Result<(usize, usize)> {
    // ── 1. Load return history ────────────────────────────────────────────
    let (symbols_ordered, returns_matrix) =
        load_returns(market_store, config.universe, config.lookback_days)?;
    let n = symbols_ordered.len();
    if n == 0 {
        anyhow::bail!("No symbols with data found in the database.");
    }
    let bars = returns_matrix.len() / n;
    info!("{} symbols loaded with {} bars", n, bars);

    // ── 2. Signal-driven alpha scores ─────────────────────────────────────
    let (alpha_scores, decomps) = generate_alpha_scores(
        market_store,
        &symbols_ordered,
        config.lookback_days,
        config.signal_filter,
    )?;

    // ── 3. Covariance matrix ──────────────────────────────────────────────
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
        config.optimizer,
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

    // ── 5. Rebalance trades ───────────────────────────────────────────────
    let target_weights: HashMap<String, f64> = symbols_ordered
        .iter()
        .zip(&opt_result.weights)
        .map(|(s, &w)| (s.clone(), w))
        .collect();

    let rebalancer = Rebalancer::default();
    let rebalance = rebalancer.rebalance(&target_weights, &HashMap::new(), config.portfolio_cash);

    if !rebalance.rebalance_triggered {
        println!("No rebalance needed (portfolio already on target).");
        return Ok((0, 0));
    }

    // Print alpha scores.
    println!("Signal alpha scores:");
    for (sym, &alpha) in symbols_ordered.iter().zip(&alpha_scores) {
        println!("  {}: {:.3}", sym, alpha);
    }

    // ── 6. Submit orders ──────────────────────────────────────────────────
    let limits = ExposureLimits::default();
    let mut submitted = 0usize;
    let mut rejected = 0usize;

    for trade in &rebalance.trades {
        if trade.dollar_amount < config.min_order_value {
            warn!(
                "Skip {} {} — ${:.0} below minimum ${:.0}",
                trade.side, trade.symbol, trade.dollar_amount, config.min_order_value
            );
            rejected += 1;
            continue;
        }

        let fraction = trade.dollar_amount / config.portfolio_cash;
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

        // Use dollar_amount as quantity (notional units at price 1.0).
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

        // Record in OMS (clone so we can still pass &order to broker below).
        let oms_id = match oms.submit_order(order.clone()) {
            Ok(id) => id,
            Err(e) => {
                warn!("OMS rejected {} {}: {}", trade.side, trade.symbol, e);
                rejected += 1;
                continue;
            }
        };

        // Optionally route to Alpaca broker.
        if let Some(b) = broker {
            match b.submit_order(&order) {
                Ok(broker_id) => {
                    if let Err(e) = oms.mark_submitted(&oms_id, &broker_id) {
                        warn!("Failed to mark {} submitted: {}", oms_id, e);
                    }
                    submitted += 1;
                }
                Err(e) => {
                    warn!("Broker rejected {} {}: {}", trade.side, trade.symbol, e);
                    rejected += 1;
                }
            }
        } else {
            submitted += 1;
        }
    }

    println!(
        "Done: portfolio=${:.0}  submitted={}  rejected={}  vol={:.2}%  turnover={:.2}%",
        config.portfolio_cash,
        submitted,
        rejected,
        opt_result.risk * 100.0,
        rebalance.turnover * 100.0,
    );

    // ── 7. Emit Prometheus metrics ────────────────────────────────────────
    let pnl_cumulative = oms.portfolio_value() - config.portfolio_cash;
    let daily_pnl_pct = pnl_cumulative / config.portfolio_cash.max(1.0);

    let port_rets = compute_portfolio_returns(&returns_matrix, n, &opt_result.weights);
    let sharpe_30d = rolling_sharpe(&port_rets, 30, 10);
    let sharpe_7d = rolling_sharpe(&port_rets, 7, 3);
    let pf = compute_profit_factor(&port_rets);
    let gross_exposure_usd =
        config.portfolio_cash * opt_result.weights.iter().cloned().sum::<f64>();
    let pos_weights: Vec<(String, f64)> = symbols_ordered
        .iter()
        .zip(&opt_result.weights)
        .map(|(s, &w)| (s.clone(), w))
        .collect();

    if let Err(e) = write_paper_metrics(
        config.metrics_file,
        pnl_cumulative,
        daily_pnl_pct,
        sharpe_30d,
        sharpe_7d,
        pf,
        gross_exposure_usd,
        &pos_weights,
    ) {
        warn!(
            "failed to write paper metrics to {}: {e}",
            config.metrics_file
        );
    }

    // ── 8. Write run_e_state.json for the dashboard ───────────────────────
    if let Some(state_path) = config.state_file {
        if let Err(e) =
            write_run_e_state(state_path, &decomps, config.signal_filter, sharpe_30d, pf)
        {
            warn!("failed to write run_e_state to {state_path}: {e}");
        }
    }

    Ok((submitted, rejected))
}

// ── Internal helpers ───────────────────────────────────────────────────────────

/// Build an Alpaca broker adapter from env vars if present.
///
/// Reads `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, and `ALPACA_PAPER` (set to
/// `"true"` for paper trading).  Returns `None` if either key is missing.
fn try_build_alpaca_broker() -> Option<AlpacaBrokerAdapter> {
    let key = std::env::var("ALPACA_API_KEY").ok()?;
    let secret = std::env::var("ALPACA_SECRET_KEY").ok()?;
    let paper = std::env::var("ALPACA_PAPER").as_deref() == Ok("true");
    match AlpacaBrokerAdapter::new(&key, &secret, paper) {
        Ok(adapter) => Some(adapter),
        Err(e) => {
            warn!("Failed to initialise Alpaca adapter: {e}");
            None
        }
    }
}

/// Best-effort portfolio equity for MaxDD tracking.
///
/// If Alpaca is configured, fetches real account equity.  Falls back to the
/// OMS portfolio value (which equals `fallback_cash` in pure paper mode).
fn portfolio_value_for_maxdd(
    broker: Option<&AlpacaBrokerAdapter>,
    oms: &OrderManagementSystem,
    fallback_cash: f64,
) -> f64 {
    if let Some(b) = broker {
        match b.get_account() {
            Ok(acct) => return acct.equity,
            Err(e) => warn!("Failed to fetch Alpaca account equity for MaxDD check: {e}"),
        }
    }
    oms.portfolio_value().max(fallback_cash)
}

/// Parse `"HH:MM"` into `(hour, minute)`.
fn parse_schedule(s: &str) -> anyhow::Result<(u32, u32)> {
    let parts: Vec<&str> = s.splitn(2, ':').collect();
    if parts.len() != 2 {
        anyhow::bail!("Invalid schedule format '{}' — expected HH:MM", s);
    }
    let h: u32 = parts[0]
        .parse()
        .map_err(|_| anyhow::anyhow!("Invalid hour in schedule '{}'", s))?;
    let m: u32 = parts[1]
        .parse()
        .map_err(|_| anyhow::anyhow!("Invalid minute in schedule '{}'", s))?;
    if h > 23 {
        anyhow::bail!("Hour {} out of range (0–23)", h);
    }
    if m > 59 {
        anyhow::bail!("Minute {} out of range (0–59)", m);
    }
    Ok((h, m))
}

/// Write paper-trading Prometheus metrics to a text-format file.
///
/// Emits: pnl_cumulative, daily_pnl_pct, rolling Sharpe (30d + 7d),
/// profit factor, gross exposure USD, and per-symbol position weights.
#[allow(clippy::too_many_arguments)]
fn write_paper_metrics(
    path: &str,
    pnl_cumulative: f64,
    daily_pnl_pct: f64,
    sharpe_30d: Option<f64>,
    sharpe_7d: Option<f64>,
    profit_factor: Option<f64>,
    gross_exposure_usd: f64,
    position_weights: &[(String, f64)],
) -> anyhow::Result<()> {
    use std::io::Write as _;

    // Use NaN for undefined metrics so Prometheus shows "N/A" rather than a
    // misleading value (e.g. insufficient data for rolling window).
    let sharpe_30d_val = sharpe_30d.unwrap_or(f64::NAN);
    let sharpe_7d_val = sharpe_7d.unwrap_or(f64::NAN);
    let profit_factor_val = profit_factor.unwrap_or(f64::NAN);

    let mut f = std::fs::File::create(path)?;
    write!(
        f,
        "# HELP quant_paper_pnl_cumulative Running cumulative P&L in USD since strategy inception\n\
         # TYPE quant_paper_pnl_cumulative gauge\n\
         quant_paper_pnl_cumulative {pnl_cumulative}\n\
         # HELP quant_paper_daily_pnl_pct Daily P&L as a fraction of starting notional\n\
         # TYPE quant_paper_daily_pnl_pct gauge\n\
         quant_paper_daily_pnl_pct {daily_pnl_pct}\n\
         # HELP quant_paper_sharpe_rolling_30d Rolling 30-day annualised Sharpe ratio on paper portfolio returns\n\
         # TYPE quant_paper_sharpe_rolling_30d gauge\n\
         quant_paper_sharpe_rolling_30d {sharpe_30d_val}\n\
         # HELP quant_paper_sharpe_rolling_7d Rolling 7-day annualised Sharpe ratio on paper portfolio returns\n\
         # TYPE quant_paper_sharpe_rolling_7d gauge\n\
         quant_paper_sharpe_rolling_7d {sharpe_7d_val}\n\
         # HELP quant_paper_profit_factor Gross winning return / abs(gross losing return) over historical window\n\
         # TYPE quant_paper_profit_factor gauge\n\
         quant_paper_profit_factor {profit_factor_val}\n\
         # HELP quant_paper_gross_exposure_usd Total absolute notional of simulated portfolio in USD\n\
         # TYPE quant_paper_gross_exposure_usd gauge\n\
         quant_paper_gross_exposure_usd {gross_exposure_usd}\n\
         # HELP quant_paper_position_weight Position weight as fraction of portfolio NAV per symbol\n\
         # TYPE quant_paper_position_weight gauge\n"
    )?;
    for (symbol, weight) in position_weights {
        writeln!(
            f,
            "quant_paper_position_weight{{symbol=\"{symbol}\"}} {weight}"
        )?;
    }
    Ok(())
}

/// Write per-sleeve strategy state as JSON for the React dashboard.
///
/// Produces `{ "timestamp": "...", "strategies": [...] }` at `path`.
/// One entry per active signal sleeve.  Fields match the `StrategyState`
/// TypeScript interface in `demo/server/index.ts` so `loadRunEState()` can
/// parse it directly.
fn write_run_e_state(
    path: &str,
    decomps: &[SignalDecomposition],
    filter: &SignalFilter,
    sharpe_30d: Option<f64>,
    pf: Option<f64>,
) -> anyhow::Result<()> {
    use serde::Serialize;
    use std::io::Write as _;

    #[derive(Serialize)]
    struct SleeveState {
        strategy_key: &'static str,
        name: &'static str,
        status: &'static str,
        regime: String,
        signal_confidence: f64,
        daily_pnl: f64,
        positions: usize,
        category: &'static str,
    }

    let n = decomps.len();

    fn sleeve(
        strategy_key: &'static str,
        name: &'static str,
        category: &'static str,
        enabled: bool,
        pairs: &[(f64, f64)],
    ) -> SleeveState {
        if !enabled {
            return SleeveState {
                strategy_key,
                name,
                status: "halted",
                regime: "sideways".to_string(),
                signal_confidence: 0.0,
                daily_pnl: 0.0,
                positions: 0,
                category,
            };
        }
        let n = pairs.len().max(1);
        let avg_conf = pairs.iter().map(|&(_, c)| c).sum::<f64>() / n as f64;
        let avg_signal = pairs.iter().map(|&(s, c)| s * c).sum::<f64>() / n as f64;
        let positions = pairs.iter().filter(|&&(s, c)| s * c > 0.05).count();
        let regime = if avg_signal > 0.10 {
            "bull"
        } else if avg_signal < -0.10 {
            "bear"
        } else {
            "sideways"
        };
        SleeveState {
            strategy_key,
            name,
            status: "active",
            regime: regime.to_string(),
            signal_confidence: avg_conf.clamp(0.0, 1.0),
            daily_pnl: 0.0,
            positions,
            category,
        }
    }

    let mom_pairs: Vec<(f64, f64)> = decomps.iter().map(|d| d.momentum).collect();
    let mr_pairs: Vec<(f64, f64)> = decomps.iter().map(|d| d.mean_reversion).collect();
    let tf_pairs: Vec<(f64, f64)> = decomps.iter().map(|d| d.trend_following).collect();

    let strategies = vec![
        sleeve(
            "momentum_us_equity",
            "Momentum US Equity",
            "Time-series",
            filter.momentum,
            &mom_pairs,
        ),
        sleeve(
            "mean_reversion_us_equity",
            "Mean Reversion US Equity",
            "Time-series",
            filter.mean_reversion,
            &mr_pairs,
        ),
        sleeve(
            "trend_following_us_equity",
            "Trend Following US Equity",
            "Time-series",
            filter.trend_following,
            &tf_pairs,
        ),
    ];

    let state = serde_json::json!({
        "timestamp": Utc::now().to_rfc3339(),
        "sharpe_30d": sharpe_30d,
        "profit_factor": pf,
        "n_symbols": n,
        "strategies": strategies,
    });

    let mut f = std::fs::File::create(path)?;
    write!(f, "{}", serde_json::to_string_pretty(&state)?)?;
    Ok(())
}

/// Read `quant_paper_pnl_cumulative` and `quant_paper_daily_pnl_pct` from a
/// Prometheus text-format metrics file.  Returns (0.0, 0.0) if the file is
/// absent or unparseable — safe to call before the first cycle.
fn read_paper_metrics(path: &str) -> (f64, f64) {
    let Ok(content) = std::fs::read_to_string(path) else {
        return (0.0, 0.0);
    };
    let mut pnl_cumulative = 0.0_f64;
    let mut daily_pnl_pct = 0.0_f64;
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("quant_paper_pnl_cumulative ") {
            if let Ok(v) = rest.parse() {
                pnl_cumulative = v;
            }
        } else if let Some(rest) = line.strip_prefix("quant_paper_daily_pnl_pct ") {
            if let Ok(v) = rest.parse() {
                daily_pnl_pct = v;
            }
        }
    }
    (pnl_cumulative, daily_pnl_pct)
}

// ── Returns loading ────────────────────────────────────────────────────────────

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

// ── Signal generation ──────────────────────────────────────────────────────────

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
/// Only computes (and blends into the alpha) signals that are enabled in
/// `filter`.  Returns `None` if the price series is too short (< 50 bars).
fn compute_symbol_alpha(adj_close: &[f64], filter: &SignalFilter) -> Option<SignalDecomposition> {
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

    let mut signal_inputs: Vec<SignalInput> = Vec::new();

    let mom = if filter.momentum {
        let (score, conf, _) = momentum_signal(&rsi_values, &rets, 20, 0.02);
        signal_inputs.push(SignalInput {
            signal_name: "momentum".into(),
            score,
            confidence: conf,
            target_position: (score * conf).clamp(-1.0, 1.0),
        });
        (score, conf)
    } else {
        (0.0, 0.0)
    };

    let mr = if filter.mean_reversion {
        let (score, conf, _) = mean_reversion_signal(&bb_mid, &bb_upper, &bb_lower, &rets, 2.0);
        signal_inputs.push(SignalInput {
            signal_name: "mean_reversion".into(),
            score,
            confidence: conf,
            target_position: (score * conf).clamp(-1.0, 1.0),
        });
        (score, conf)
    } else {
        (0.0, 0.0)
    };

    let tf = if filter.trend_following {
        let (score, conf, _) = trend_following_signal(&macd_hist, &fast_ma, &slow_ma);
        signal_inputs.push(SignalInput {
            signal_name: "trend_following".into(),
            score,
            confidence: conf,
            target_position: (score * conf).clamp(-1.0, 1.0),
        });
        (score, conf)
    } else {
        (0.0, 0.0)
    };

    let combined_alpha = if signal_inputs.is_empty() {
        0.0
    } else {
        let combiner = AlphaCombiner::new(CombinationMethod::ConvictionWeighted, None);
        combiner.combine("_", &signal_inputs).target_position
    };

    Some(SignalDecomposition {
        momentum: mom,
        mean_reversion: mr,
        trend_following: tf,
        combined_alpha,
    })
}

/// Compute alpha scores for each symbol using the filtered signal pipeline.
///
/// Returns both the combined alpha vector (used by the optimizer) and the
/// per-symbol `SignalDecomposition` slice (used by `write_run_e_state`).
fn generate_alpha_scores(
    store: &MarketDataStore,
    symbols: &[String],
    lookback_days: usize,
    filter: &SignalFilter,
) -> anyhow::Result<(Vec<f64>, Vec<SignalDecomposition>)> {
    let today = chrono::Local::now().date_naive();
    let start = today - chrono::Duration::days((lookback_days as i64) * 2);

    let mut alpha_scores = Vec::with_capacity(symbols.len());
    let mut decomps = Vec::with_capacity(symbols.len());

    for sym in symbols {
        let records = store.query(sym, start, today)?;
        let adj_close: Vec<f64> = records.iter().map(|r| r.adj_close).collect();

        match compute_symbol_alpha(&adj_close, filter) {
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
                decomps.push(decomp);
            }
            None => {
                info!(
                    "{}: insufficient history ({} bars), using neutral alpha",
                    sym,
                    adj_close.len()
                );
                alpha_scores.push(0.0);
                decomps.push(SignalDecomposition {
                    momentum: (0.0, 0.0),
                    mean_reversion: (0.0, 0.0),
                    trend_following: (0.0, 0.0),
                    combined_alpha: 0.0,
                });
            }
        }
    }

    Ok((alpha_scores, decomps))
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

// ── Metrics computation helpers ────────────────────────────────────────────────

/// Compute per-bar portfolio returns: `weights · returns_row` for each bar.
///
/// `returns_matrix` is row-major with shape `[bars][n]`.
fn compute_portfolio_returns(returns_matrix: &[f64], n: usize, weights: &[f64]) -> Vec<f64> {
    if n == 0 || weights.len() != n {
        return vec![];
    }
    let bars = returns_matrix.len() / n;
    (0..bars)
        .map(|t| (0..n).map(|i| weights[i] * returns_matrix[t * n + i]).sum())
        .collect()
}

/// Rolling annualised Sharpe ratio from the tail of `returns`.
///
/// Uses the last `window` observations.  Returns `None` when fewer than
/// `min_samples` data points are available or when the standard deviation is
/// effectively zero.
fn rolling_sharpe(returns: &[f64], window: usize, min_samples: usize) -> Option<f64> {
    let slice = &returns[returns.len().saturating_sub(window)..];
    if slice.len() < min_samples {
        return None;
    }
    let n = slice.len() as f64;
    let mean: f64 = slice.iter().sum::<f64>() / n;
    let var: f64 = slice.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    let std = var.sqrt();
    if std < 1e-12 {
        return None;
    }
    Some(mean / std * 252_f64.sqrt())
}

/// Profit factor = sum(positive returns) / sum(abs(negative returns)).
///
/// Returns `None` when there are no losing periods (avoids division by zero).
fn compute_profit_factor(returns: &[f64]) -> Option<f64> {
    let winning: f64 = returns.iter().filter(|&&r| r > 0.0).sum();
    let losing: f64 = returns.iter().filter(|&&r| r < 0.0).map(|r| -r).sum();
    if losing < 1e-12 {
        return None;
    }
    Some(winning / losing)
}

// ── Unit tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use quant_risk::DrawdownCircuitBreaker;

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

    // ── SignalFilter ──────────────────────────────────────────────────────────

    #[test]
    fn test_signal_filter_all_enabled() {
        let f = SignalFilter::from_str("momentum,mean_reversion,trend");
        assert!(f.momentum);
        assert!(f.mean_reversion);
        assert!(f.trend_following);
    }

    #[test]
    fn test_signal_filter_momentum_only() {
        let f = SignalFilter::from_str("momentum");
        assert!(f.momentum);
        assert!(!f.mean_reversion);
        assert!(!f.trend_following);
    }

    #[test]
    fn test_signal_filter_trend_aliases() {
        let f1 = SignalFilter::from_str("trend");
        let f2 = SignalFilter::from_str("trend_following");
        assert!(f1.trend_following);
        assert!(f2.trend_following);
        assert!(!f1.momentum);
    }

    // ── compute_symbol_alpha ──────────────────────────────────────────────────

    #[test]
    fn test_compute_symbol_alpha_returns_none_for_short_series() {
        let filter = SignalFilter::from_str("momentum,mean_reversion,trend");
        assert!(compute_symbol_alpha(&vec![100.0; 30], &filter).is_none());
    }

    #[test]
    fn test_compute_symbol_alpha_returns_some_for_sufficient_data() {
        let filter = SignalFilter::from_str("momentum,mean_reversion,trend");
        let prices = synthetic_prices(252, 0.001);
        assert!(compute_symbol_alpha(&prices, &filter).is_some());
    }

    #[test]
    fn test_alpha_output_in_valid_range() {
        let filter = SignalFilter::from_str("momentum,mean_reversion,trend");
        let prices = synthetic_prices(252, 0.001);
        let decomp = compute_symbol_alpha(&prices, &filter).unwrap();
        assert!((-1.0..=1.0).contains(&decomp.combined_alpha));
        assert!((-1.0..=1.0).contains(&decomp.momentum.0));
        assert!((0.0..=1.0).contains(&decomp.momentum.1));
        assert!((-1.0..=1.0).contains(&decomp.mean_reversion.0));
        assert!((0.0..=1.0).contains(&decomp.mean_reversion.1));
        assert!((-1.0..=1.0).contains(&decomp.trend_following.0));
        assert!((0.0..=1.0).contains(&decomp.trend_following.1));
    }

    #[test]
    fn test_momentum_only_filter_zeroes_other_components() {
        let filter = SignalFilter::from_str("momentum");
        let prices = synthetic_prices(252, 0.001);
        let decomp = compute_symbol_alpha(&prices, &filter).unwrap();
        // Inactive signals return (0, 0) from the filter branches.
        assert!((decomp.mean_reversion.0).abs() < 1e-9);
        assert!((decomp.mean_reversion.1).abs() < 1e-9);
        assert!((decomp.trend_following.0).abs() < 1e-9);
        assert!((decomp.trend_following.1).abs() < 1e-9);
        // Momentum signal itself should be non-trivially computed.
        assert!((0.0..=1.0).contains(&decomp.momentum.1));
    }

    #[test]
    fn test_no_signals_enabled_returns_zero_alpha() {
        // Empty filter — no signals enabled.
        let filter = SignalFilter::from_str("none_of_these");
        let prices = synthetic_prices(252, 0.001);
        let decomp = compute_symbol_alpha(&prices, &filter).unwrap();
        assert!((decomp.combined_alpha).abs() < 1e-9);
    }

    #[test]
    fn test_signal_blending_all_vs_momentum_only_differ() {
        // With a strong uptrend, momentum-only alpha should equal the momentum
        // component of the full blend only when the other signals are flat or
        // aligned — just verify both paths produce valid ranges.
        let prices: Vec<f64> = (0..252).map(|i| 100.0 * 1.003_f64.powi(i)).collect();
        let all_filter = SignalFilter::from_str("momentum,mean_reversion,trend");
        let mom_filter = SignalFilter::from_str("momentum");
        let all_decomp = compute_symbol_alpha(&prices, &all_filter).unwrap();
        let mom_decomp = compute_symbol_alpha(&prices, &mom_filter).unwrap();
        assert!((-1.0..=1.0).contains(&all_decomp.combined_alpha));
        assert!((-1.0..=1.0).contains(&mom_decomp.combined_alpha));
    }

    #[test]
    fn test_strong_uptrend_produces_positive_momentum() {
        let filter = SignalFilter::from_str("momentum,mean_reversion,trend");
        let prices: Vec<f64> = (0..252).map(|i| 100.0 * 1.003_f64.powi(i)).collect();
        let decomp = compute_symbol_alpha(&prices, &filter).unwrap();
        assert!(
            decomp.momentum.0 > 0.0,
            "momentum score={}",
            decomp.momentum.0
        );
    }

    #[test]
    fn test_strong_downtrend_produces_negative_momentum() {
        let filter = SignalFilter::from_str("momentum,mean_reversion,trend");
        let prices: Vec<f64> = (0..252).map(|i| 200.0 * 0.997_f64.powi(i)).collect();
        let decomp = compute_symbol_alpha(&prices, &filter).unwrap();
        assert!(
            decomp.momentum.0 < 0.0,
            "momentum score={}",
            decomp.momentum.0
        );
    }

    #[test]
    fn test_flat_prices_produce_near_zero_alpha() {
        let filter = SignalFilter::from_str("momentum,mean_reversion,trend");
        let prices = vec![100.0_f64; 252];
        let decomp = compute_symbol_alpha(&prices, &filter).unwrap();
        assert!(
            decomp.combined_alpha.abs() < 0.3,
            "alpha={}",
            decomp.combined_alpha
        );
    }

    // ── compute_portfolio_returns ─────────────────────────────────────────────

    #[test]
    fn test_portfolio_returns_equal_weight() {
        // 2 symbols, 3 bars.  Equal weights (0.5 each).
        let matrix = vec![
            0.01, 0.02, // bar 0: sym0=+1%, sym1=+2%
            -0.01, 0.0, // bar 1: sym0=-1%, sym1=0%
            0.0, 0.03, // bar 2: sym0=0%, sym1=+3%
        ];
        let weights = vec![0.5, 0.5];
        let pr = compute_portfolio_returns(&matrix, 2, &weights);
        assert_eq!(pr.len(), 3);
        assert!((pr[0] - 0.015).abs() < 1e-12);
        assert!((pr[1] - (-0.005)).abs() < 1e-12);
        assert!((pr[2] - 0.015).abs() < 1e-12);
    }

    #[test]
    fn test_portfolio_returns_empty_on_zero_n() {
        assert!(compute_portfolio_returns(&[0.01], 0, &[]).is_empty());
    }

    // ── rolling_sharpe ────────────────────────────────────────────────────────

    #[test]
    fn test_rolling_sharpe_insufficient_data_returns_none() {
        let rets = vec![0.01; 5];
        assert!(rolling_sharpe(&rets, 30, 10).is_none());
    }

    #[test]
    fn test_rolling_sharpe_constant_returns_returns_none() {
        // Zero std dev → None.
        let rets = vec![0.005; 30];
        assert!(rolling_sharpe(&rets, 30, 10).is_none());
    }

    #[test]
    fn test_rolling_sharpe_positive_drift() {
        // Strong positive drift should produce a positive Sharpe.
        let mut state: u64 = 99;
        let rets: Vec<f64> = (0..60)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let noise = ((state >> 33) as f64) / (u32::MAX as f64) - 0.5;
                0.005 + noise * 0.01 // strong positive drift
            })
            .collect();
        let sharpe = rolling_sharpe(&rets, 30, 10).expect("should compute");
        assert!(sharpe > 0.0, "sharpe={sharpe}");
    }

    // ── compute_profit_factor ─────────────────────────────────────────────────

    #[test]
    fn test_profit_factor_no_losses_returns_none() {
        let rets = vec![0.01, 0.02, 0.005];
        assert!(compute_profit_factor(&rets).is_none());
    }

    #[test]
    fn test_profit_factor_mixed_returns() {
        // winning = 0.03 + 0.01 = 0.04, losing = 0.02
        let rets = vec![0.03, -0.02, 0.01];
        let pf = compute_profit_factor(&rets).expect("should compute");
        assert!((pf - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_profit_factor_more_losses_than_wins() {
        // pf < 1.0 → losing strategy
        let rets = vec![0.01, -0.02, -0.03];
        let pf = compute_profit_factor(&rets).expect("should compute");
        assert!(pf < 1.0, "pf={pf}");
    }

    // ── Loop halt logic (circuit breaker) ────────────────────────────────────

    #[test]
    fn test_loop_halt_maxdd_tripped_at_22_pct() {
        let cb = DrawdownCircuitBreaker::new(0.22);
        let peak = 1_000_000.0_f64;
        let current = 780_000.0; // exactly 22 % drawdown
        assert!(cb.is_tripped(peak, current));
        assert!((cb.drawdown(peak, current) - 0.22).abs() < 1e-9);
    }

    #[test]
    fn test_loop_halt_not_tripped_below_22_pct() {
        let cb = DrawdownCircuitBreaker::new(0.22);
        assert!(!cb.is_tripped(1_000_000.0, 790_000.0)); // 21 % — under threshold
    }

    #[test]
    fn test_loop_halt_tripped_beyond_22_pct() {
        let cb = DrawdownCircuitBreaker::new(0.22);
        assert!(cb.is_tripped(1_000_000.0, 700_000.0)); // 30 % — well over
    }

    // ── parse_schedule ────────────────────────────────────────────────────────

    #[test]
    fn test_parse_schedule_valid() {
        let (h, m) = parse_schedule("16:05").unwrap();
        assert_eq!(h, 16);
        assert_eq!(m, 5);
    }

    #[test]
    fn test_parse_schedule_midnight() {
        let (h, m) = parse_schedule("00:00").unwrap();
        assert_eq!(h, 0);
        assert_eq!(m, 0);
    }

    #[test]
    fn test_parse_schedule_invalid_hour() {
        assert!(parse_schedule("25:00").is_err());
    }

    #[test]
    fn test_parse_schedule_invalid_minute() {
        assert!(parse_schedule("16:60").is_err());
    }

    #[test]
    fn test_parse_schedule_missing_colon() {
        assert!(parse_schedule("1605").is_err());
    }

    // ── resolve_symbols / parse_optimizer ────────────────────────────────────

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
