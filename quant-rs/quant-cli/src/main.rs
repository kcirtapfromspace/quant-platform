//! Quant Infrastructure CLI
//!
//! Replaces the Python `quant-ingest` and `quant-run` entry points.
//!
//! # Subcommands
//!
//! ```
//! quant ingest run    --db <path> [--mode full] [--start YYYY-MM-DD] [--symbols A,B]
//! quant ingest status --db <path>
//! quant run once      --db <path> [--cash 1000000] [--optimizer risk_parity] [--signals momentum,mean_reversion,trend]
//! quant run loop      --db <path> --schedule 16:05 [--cash 1000000] [--optimizer risk_parity] [--oms-db ./quant_oms.db]
//! quant run status    [--oms-db ./quant_oms.db]
//! quant backtest      --db <path> --symbol AAPL [--start YYYY-MM-DD] [--end YYYY-MM-DD]
//! ```

mod cmd_backtest;
mod cmd_benchmark;
mod cmd_ingest;
mod cmd_run;
mod cmd_serve;

use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "quant", version, about = "Quant Infrastructure CLI")]
struct Cli {
    /// Enable verbose (DEBUG) logging.
    #[arg(long, global = true)]
    verbose: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Market data ingestion commands.
    Ingest {
        #[command(subcommand)]
        action: IngestAction,
    },
    /// Strategy runner commands.
    Run {
        #[command(subcommand)]
        action: RunAction,
    },
    /// Backtesting commands.
    Backtest(cmd_backtest::BacktestArgs),
    /// OOS benchmark suite.
    Benchmark {
        #[command(subcommand)]
        suite: BenchmarkSuite,
    },
    /// HTTP status server for k8s deployments.
    Serve(cmd_serve::ServeArgs),
}

#[derive(Subcommand)]
pub enum BenchmarkSuite {
    /// QUA-68: Bayesian vs EMA AdaptiveSignalCombiner walk-forward benchmark.
    Qua68(cmd_benchmark::BenchmarkQua68Args),
}

#[derive(Subcommand)]
pub enum IngestAction {
    /// Fetch and store OHLCV data.
    Run(cmd_ingest::IngestRunArgs),
    /// Show database coverage summary.
    Status(cmd_ingest::IngestStatusArgs),
}

#[derive(Subcommand)]
pub enum RunAction {
    /// Execute a single strategy cycle and exit.
    Once(cmd_run::RunOnceArgs),
    /// Run the strategy as a daily rebalance daemon (loops forever).
    Loop(cmd_run::RunLoopArgs),
    /// Print the current OMS positions and unrealized PnL.
    Status(cmd_run::RunStatusArgs),
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let level = if cli.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(level)),
        )
        .with_target(false)
        .init();

    match cli.command {
        Commands::Ingest { action } => match action {
            IngestAction::Run(args) => cmd_ingest::run_ingest(args),
            IngestAction::Status(args) => cmd_ingest::run_status(args),
        },
        Commands::Run { action } => match action {
            RunAction::Once(args) => cmd_run::run_once(args),
            RunAction::Loop(args) => cmd_run::run_loop(args),
            RunAction::Status(args) => cmd_run::run_status(args),
        },
        Commands::Backtest(args) => cmd_backtest::run_backtest_cmd(args),
        Commands::Benchmark { suite } => match suite {
            BenchmarkSuite::Qua68(args) => cmd_benchmark::run_benchmark_qua68(args),
        },
        Commands::Serve(args) => cmd_serve::run_serve(args),
    }
}
