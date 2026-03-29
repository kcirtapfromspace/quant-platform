//! `quant serve` — Axum REST + WebSocket API server.
//!
//! Exposes the quant engine over HTTP for the frontend dashboard.
//! Delegates all routing and serving to the `quant-api` library crate.

use clap::Args;

#[derive(Args)]
pub struct ServeArgs {
    /// Path to the DuckDB market data file.
    #[arg(long)]
    pub db: String,

    /// Path to the SQLite OMS state file (enables order/position endpoints).
    #[arg(long, default_value = "./quant_oms.db")]
    pub oms_db: String,

    /// Port to listen on.
    #[arg(long, default_value = "8080")]
    pub port: u16,

    /// Prometheus textfile written by `quant run once`.
    #[arg(long, default_value = "/tmp/quant_paper_metrics.prom")]
    pub metrics_file: String,

    /// Directory containing per-run backtest result sub-folders.
    #[arg(long, default_value = "./backtest-results")]
    pub backtest_results_dir: String,
}

pub fn run_serve(args: ServeArgs) -> anyhow::Result<()> {
    // CRO Finding 1: refuse to start without an API key.
    let api_key = std::env::var("QUANT_API_KEY").unwrap_or_else(|_| {
        eprintln!(
            "ERROR: QUANT_API_KEY environment variable is not set.\n\
             quant serve refuses to start without authentication configured."
        );
        std::process::exit(1);
    });

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(async move {
            let oms_db = if std::path::Path::new(&args.oms_db).exists() {
                Some(args.oms_db)
            } else {
                None
            };

            let state = quant_api::AppState::new_with_key(
                args.db,
                oms_db,
                args.metrics_file,
                args.backtest_results_dir,
                Some(api_key),
            );

            quant_api::serve(state, args.port).await
        })
}
