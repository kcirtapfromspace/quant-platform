//! `quant backtest` — single-asset bar-replay backtest from DuckDB data.

use chrono::NaiveDate;
use clap::Args;

use quant_backtest::run_backtest;
use quant_data::MarketDataStore;

#[derive(Args)]
pub struct BacktestArgs {
    /// Path to the DuckDB market data file.
    #[arg(long)]
    pub db: String,

    /// Symbol to backtest.
    #[arg(long)]
    pub symbol: String,

    /// Start date (YYYY-MM-DD). Defaults to earliest available.
    #[arg(long)]
    pub start: Option<String>,

    /// End date (YYYY-MM-DD). Defaults to latest available.
    #[arg(long)]
    pub end: Option<String>,

    /// Signal: 1 = always long, -1 = always short, 0 = flat.
    #[arg(long, default_value = "1")]
    pub signal: f64,

    /// One-way commission fraction (e.g. 0.001 = 10 bps).
    #[arg(long, default_value = "0.001")]
    pub commission: f64,

    /// Starting portfolio value.
    #[arg(long, default_value = "1000000")]
    pub initial_capital: f64,
}

pub fn run_backtest_cmd(args: BacktestArgs) -> anyhow::Result<()> {
    let store = MarketDataStore::open(&args.db)?;

    let start = match &args.start {
        Some(s) => NaiveDate::parse_from_str(s, "%Y-%m-%d")
            .map_err(|e| anyhow::anyhow!("invalid --start '{}': {}", s, e))?,
        None => NaiveDate::from_ymd_opt(1900, 1, 1).unwrap(),
    };
    let end = match &args.end {
        Some(s) => NaiveDate::parse_from_str(s, "%Y-%m-%d")
            .map_err(|e| anyhow::anyhow!("invalid --end '{}': {}", s, e))?,
        None => chrono::Local::now().date_naive(),
    };

    let records = store.query(&args.symbol.to_uppercase(), start, end)?;
    if records.len() < 2 {
        anyhow::bail!(
            "Insufficient data for {} ({} bars). Run `quant ingest run` first.",
            args.symbol,
            records.len()
        );
    }

    let adj_close: Vec<f64> = records.iter().map(|r| r.adj_close).collect();
    let signals: Vec<f64> = vec![args.signal; adj_close.len()];

    let result = run_backtest(&adj_close, &signals, args.commission, args.initial_capital);

    println!(
        "Backtest: {} ({} bars)",
        args.symbol.to_uppercase(),
        records.len()
    );
    println!("  Total return:  {:.2}%", result.total_return * 100.0);
    println!("  CAGR:          {:.2}%", result.cagr * 100.0);
    println!("  Sharpe ratio:  {:.3}", result.sharpe_ratio);
    println!("  Max drawdown:  {:.2}%", result.max_drawdown * 100.0);
    println!("  Win rate:      {:.1}%", result.win_rate * 100.0);
    println!(
        "  Profit factor: {}",
        if result.profit_factor.is_infinite() {
            "∞".to_string()
        } else {
            format!("{:.2}", result.profit_factor)
        }
    );
    println!("  Trades:        {}", result.trades.len());

    Ok(())
}
