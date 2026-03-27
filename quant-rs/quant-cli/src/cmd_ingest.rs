use chrono::NaiveDate;
use clap::Args;
use quant_data::{IngestMode, IngestionPipeline, MarketDataStore};
use quant_data::ingest::yahoo::YahooFinanceSource;
use tracing::info;

// ── ingest run ────────────────────────────────────────────────────────────────

#[derive(Args)]
pub struct IngestRunArgs {
    /// Path to the DuckDB market data file.
    #[arg(long)]
    pub db: String,

    /// Ingestion mode.
    #[arg(long, default_value = "incremental", value_parser = parse_mode)]
    pub mode: IngestMode,

    /// Start date for full-mode fetch (YYYY-MM-DD).
    #[arg(long)]
    pub start: Option<String>,

    /// End date (YYYY-MM-DD, default: today).
    #[arg(long)]
    pub end: Option<String>,

    /// Comma-separated symbols (overrides built-in universe).
    #[arg(long)]
    pub symbols: Option<String>,
}

pub fn run_ingest(args: IngestRunArgs) -> anyhow::Result<()> {
    let symbols = resolve_symbols(args.symbols.as_deref());
    let start = args.start.as_deref().map(parse_date).transpose()?;
    let end = args.end.as_deref().map(parse_date).transpose()?;

    let store = MarketDataStore::open(&args.db)?;
    let source = YahooFinanceSource::new();
    let pipeline = IngestionPipeline::new(store, source);

    info!(
        "Ingesting {} symbols (mode={:?})",
        symbols.len(),
        args.mode
    );

    let result = pipeline.run(&symbols, args.mode, start, end)?;
    println!("{}", result.summary());

    if !result.gaps_detected.is_empty() {
        println!(
            "Gaps detected in {} symbol(s):",
            result.gaps_detected.len()
        );
        for (sym, gaps) in &result.gaps_detected {
            let preview: Vec<String> =
                gaps.iter().take(5).map(|d| d.to_string()).collect();
            let suffix = if gaps.len() > 5 { "..." } else { "" };
            println!("  {}: {}{}", sym, preview.join(", "), suffix);
        }
        anyhow::bail!("gaps detected — check source data");
    }

    Ok(())
}

// ── ingest status ─────────────────────────────────────────────────────────────

#[derive(Args)]
pub struct IngestStatusArgs {
    /// Path to the DuckDB market data file.
    #[arg(long)]
    pub db: String,
}

pub fn run_status(args: IngestStatusArgs) -> anyhow::Result<()> {
    let store = MarketDataStore::open(&args.db)?;
    let symbols = store.symbols()?;

    if symbols.is_empty() {
        println!("No data in database.");
        return Ok(());
    }

    println!("Symbols: {}", symbols.len());
    for sym in &symbols {
        let count = store.count(Some(sym))?;
        let latest = store.latest_date(sym)?;
        println!(
            "  {:<8} rows={:<6} latest={}",
            sym,
            count,
            latest.map(|d| d.to_string()).unwrap_or_else(|| "—".into())
        );
    }

    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn resolve_symbols(arg: Option<&str>) -> Vec<String> {
    if let Some(s) = arg {
        return s.split(',').map(|x| x.trim().to_uppercase()).collect();
    }
    // Default universe — can be overridden via --symbols
    vec![
        "AAPL", "GOOG", "MSFT", "AMZN", "META", "NVDA", "JPM", "XOM",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

fn parse_date(s: &str) -> anyhow::Result<NaiveDate> {
    NaiveDate::parse_from_str(s, "%Y-%m-%d")
        .map_err(|e| anyhow::anyhow!("invalid date '{}': {}", s, e))
}

fn parse_mode(s: &str) -> Result<IngestMode, String> {
    match s {
        "incremental" => Ok(IngestMode::Incremental),
        "full" => Ok(IngestMode::Full),
        other => Err(format!("unknown mode '{}' (expected: incremental, full)", other)),
    }
}
