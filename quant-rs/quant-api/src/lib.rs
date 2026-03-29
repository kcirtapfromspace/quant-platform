//! `quant-api` — Axum REST + WebSocket API gateway for the quant engine.
//!
//! # Architecture
//!
//! - [`AppState`] is shared across all handlers via `Arc`.
//! - REST routes are under `/api/v1/` and protected by optional `X-API-Key` auth.
//! - The WebSocket endpoint at `/ws` pushes live price ticks and position updates.
//! - A background Tokio task drives the WebSocket broadcast channel.

pub mod auth;
pub mod error;
pub mod routes;
pub mod ws;

use std::sync::Arc;

use axum::{
    http::{header, HeaderValue, Method},
    middleware,
    routing::get,
    Router,
};
use tokio::sync::broadcast;
use tower_http::{compression::CompressionLayer, cors::CorsLayer};
use tracing::info;

use ws::WsEvent;

/// Shared application state threaded through all handlers.
pub struct AppState {
    /// Path to the DuckDB market data file (`:memory:` for testing).
    pub db_path: String,
    /// Path to the SQLite OMS state file; `None` → no order/position data.
    pub oms_db_path: Option<String>,
    /// Optional API key loaded from `QUANT_API_KEY` env var.
    pub api_key: Option<String>,
    /// Prometheus textfile written by `quant run once` (for risk metrics).
    pub metrics_file: String,
    /// Directory that contains per-run backtest results sub-folders.
    pub backtest_results_dir: String,
    /// Broadcast channel sender for WebSocket push events.
    pub broadcast_tx: broadcast::Sender<WsEvent>,
}

impl AppState {
    /// Create state reading `QUANT_API_KEY` from the environment (production use).
    pub fn new(
        db_path: impl Into<String>,
        oms_db_path: Option<String>,
        metrics_file: impl Into<String>,
        backtest_results_dir: impl Into<String>,
    ) -> Arc<Self> {
        let api_key = std::env::var("QUANT_API_KEY").ok();
        Self::new_with_key(
            db_path,
            oms_db_path,
            metrics_file,
            backtest_results_dir,
            api_key,
        )
    }

    /// Create state with an explicit API key (used in tests to avoid env var pollution).
    pub fn new_with_key(
        db_path: impl Into<String>,
        oms_db_path: Option<String>,
        metrics_file: impl Into<String>,
        backtest_results_dir: impl Into<String>,
        api_key: Option<String>,
    ) -> Arc<Self> {
        let (broadcast_tx, _) = ws::new_broadcast();
        Arc::new(AppState {
            db_path: db_path.into(),
            oms_db_path,
            api_key,
            metrics_file: metrics_file.into(),
            backtest_results_dir: backtest_results_dir.into(),
            broadcast_tx,
        })
    }
}

/// Build the Axum router.  Call [`serve`] to bind and run it.
///
/// Security properties:
/// - `/health` is public (k8s liveness probe).
/// - All `/api/v1/*` routes require `X-API-Key` header (fail-closed when no
///   key is configured).
/// - `/ws` requires `X-API-Key` header **or** `?api_key=` query param
///   (browsers cannot set headers on WebSocket upgrades).
/// - CORS is restricted to `QUANT_API_CORS_ORIGIN` env var (default:
///   `https://dashboard.tail16ecc2.ts.net`).  Wildcard origin is rejected.
pub fn build_router(state: Arc<AppState>) -> Router {
    let auth_layer = middleware::from_fn_with_state(state.clone(), auth::require_api_key);

    let api_routes = Router::new()
        .route("/portfolio", get(routes::portfolio::get_portfolio))
        .route(
            "/portfolio/history",
            get(routes::portfolio::get_portfolio_history),
        )
        .route("/orders", get(routes::orders::get_orders))
        .route("/risk", get(routes::risk::get_risk))
        .route("/risk/snapshot", get(routes::risk::get_risk_snapshot))
        .route("/signals", get(routes::signals::get_signals))
        .route("/strategies", get(routes::strategies::get_strategies))
        .route("/history/{symbol}", get(routes::market::get_history))
        .route(
            "/backtest/latest",
            get(routes::backtest::get_backtest_latest),
        )
        .route("/market/quotes", get(routes::market::get_quotes))
        .layer(auth_layer.clone());

    // CORS: read allowed origin from env; no wildcard fallback.
    let cors_origin = std::env::var("QUANT_API_CORS_ORIGIN")
        .unwrap_or_else(|_| "https://dashboard.tail16ecc2.ts.net".to_string());
    let allowed_origin: HeaderValue = cors_origin
        .parse()
        .expect("QUANT_API_CORS_ORIGIN is not a valid header value");

    Router::new()
        .route("/health", get(routes::health::health))
        .nest("/api/v1", api_routes)
        // WebSocket: protected by the same auth middleware.
        // Clients must supply ?api_key=<key> in the upgrade URL.
        .route("/ws", get(ws::ws_handler).layer(auth_layer))
        .layer(
            CorsLayer::new()
                .allow_origin(allowed_origin)
                .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
                .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION])
                .allow_credentials(false),
        )
        .layer(CompressionLayer::new())
        .with_state(state)
}

/// Bind, spawn the WebSocket background task, and serve until shutdown.
pub async fn serve(state: Arc<AppState>, port: u16) -> anyhow::Result<()> {
    let addr = format!("0.0.0.0:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    info!("quant-api listening on {addr}");

    // Spawn the broadcast task that pushes price + position updates over WS
    tokio::spawn(ws::broadcast_task(state.clone()));

    let app = build_router(state);
    axum::serve(listener, app).await?;
    Ok(())
}
