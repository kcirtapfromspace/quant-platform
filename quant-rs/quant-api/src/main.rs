//! `quant-api` binary — start the REST + WebSocket API server.
//!
//! ```
//! quant-api --db market.ddb --port 8080
//! quant-api --db market.ddb --oms-db quant_oms.db --port 8080
//! ```

use clap::Parser;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "quant-api", version, about = "Quant REST + WebSocket API server")]
struct Cli {
    /// Path to the DuckDB market data file.
    #[arg(long)]
    db: String,

    /// Path to the SQLite OMS state file (enables order/position endpoints).
    #[arg(long)]
    oms_db: Option<String>,

    /// Port to listen on.
    #[arg(long, default_value = "8080")]
    port: u16,

    /// Prometheus textfile written by `quant run once`.
    #[arg(long, default_value = "/tmp/quant_paper_metrics.prom")]
    metrics_file: String,

    /// Directory containing backtest result sub-folders.
    #[arg(long, default_value = "./backtest-results")]
    backtest_results_dir: String,

    /// Enable verbose (DEBUG) logging.
    #[arg(long)]
    verbose: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let level = if cli.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(level)),
        )
        .with_target(false)
        .init();

    // CRO Finding 1: fail fast if no API key is configured.
    // The auth middleware is fail-closed, but we also assert here so the
    // operator gets an explicit startup error rather than a silent 401 on
    // every request.
    let api_key = std::env::var("QUANT_API_KEY").unwrap_or_else(|_| {
        eprintln!(
            "ERROR: QUANT_API_KEY environment variable is not set.\n\
             quant-api refuses to start without authentication configured.\n\
             Set QUANT_API_KEY to a strong random secret before deploying."
        );
        std::process::exit(1);
    });

    let state = quant_api::AppState::new_with_key(
        cli.db,
        cli.oms_db,
        cli.metrics_file,
        cli.backtest_results_dir,
        Some(api_key),
    );

    quant_api::serve(state, cli.port).await
}
