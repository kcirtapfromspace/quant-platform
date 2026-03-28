//! `quant serve` — lightweight HTTP status server for k8s deployments.
//!
//! Serves health and status endpoints so the deployment can be probed
//! by Kubernetes liveness checks and accessed via Tailscale Ingress.

use std::io::{Read, Write};
use std::net::TcpListener;

use clap::Args;
use tracing::info;

use quant_data::MarketDataStore;

#[derive(Args)]
pub struct ServeArgs {
    /// Port to listen on.
    #[arg(long, default_value = "8000")]
    pub port: u16,

    /// Path to the DuckDB market data file (optional).
    #[arg(long)]
    pub db: Option<String>,

    /// Path to the Prometheus textfile written by `quant run once`.
    /// Contents are appended to the /metrics response on each scrape.
    #[arg(long, default_value = "/tmp/quant_paper_metrics.prom")]
    pub metrics_file: String,
}

pub fn run_serve(args: ServeArgs) -> anyhow::Result<()> {
    let addr = format!("0.0.0.0:{}", args.port);
    let listener = TcpListener::bind(&addr)?;
    info!("quant serve listening on {addr}");

    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("accept error: {e}");
                continue;
            }
        };

        let mut buf = [0u8; 2048];
        let _ = stream.read(&mut buf);
        let request = String::from_utf8_lossy(&buf);

        let path = request
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .unwrap_or("/");

        let (status, content_type, body) = match path {
            "/health" | "/healthz" => (
                "200 OK",
                "application/json",
                r#"{"status":"ok"}"#.to_string(),
            ),
            "/metrics" => (
                "200 OK",
                "text/plain; version=0.0.4; charset=utf-8",
                build_metrics(&args.metrics_file),
            ),
            _ => ("200 OK", "application/json", build_status(&args.db)),
        };

        let response = format!(
            "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        let _ = stream.write_all(response.as_bytes());
    }
    Ok(())
}

fn build_status(db_path: &Option<String>) -> String {
    let db_status = match db_path {
        Some(path) => match MarketDataStore::open(path) {
            Ok(_) => serde_json::json!("connected"),
            Err(e) => serde_json::json!(format!("error: {e}")),
        },
        None => serde_json::json!("no database configured"),
    };

    serde_json::json!({
        "service": "quant-rs",
        "version": env!("CARGO_PKG_VERSION"),
        "status": "running",
        "database": db_status,
    })
    .to_string()
}

fn build_metrics(metrics_file: &str) -> String {
    let mut out = String::from(
        "# HELP quant_up Whether the quant service is up\n\
         # TYPE quant_up gauge\n\
         quant_up 1\n",
    );
    if let Ok(extra) = std::fs::read_to_string(metrics_file) {
        out.push_str(&extra);
    }
    out
}
