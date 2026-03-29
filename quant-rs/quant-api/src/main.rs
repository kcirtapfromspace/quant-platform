use std::{
    collections::HashMap,
    path::PathBuf,
    sync::Arc,
    time::Duration,
};

use anyhow::Result;
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Path, Query, State,
    },
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Json, Response},
    routing::{delete, get},
    Router,
};
use chrono::Utc;
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::time::interval;
use tower_http::cors::{Any, CorsLayer};
use tracing::{info, warn};
use uuid::Uuid;

// ── Config ────────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct Config {
    api_key: String,
    oms_db: PathBuf,
    quant_db: PathBuf,
    backtest_dir: PathBuf,
    metrics_path: PathBuf,
    port: u16,
}

impl Config {
    fn from_env() -> Self {
        Config {
            api_key: std::env::var("API_KEY").unwrap_or_else(|_| "changeme".into()),
            oms_db: std::env::var("OMS_DB_PATH")
                .unwrap_or_else(|_| "/data/oms.db".into())
                .into(),
            quant_db: std::env::var("QUANT_DB_PATH")
                .unwrap_or_else(|_| "/data/quant.duckdb".into())
                .into(),
            backtest_dir: std::env::var("BACKTEST_RESULTS_DIR")
                .unwrap_or_else(|_| "/data/backtest-results".into())
                .into(),
            metrics_path: std::env::var("METRICS_PATH")
                .unwrap_or_else(|_| "/data/metrics.prom".into())
                .into(),
            port: std::env::var("PORT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(8080),
        }
    }
}

// ── Response types ────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Clone)]
struct Order {
    id: String,
    symbol: String,
    side: String,
    #[serde(rename = "type")]
    order_type: String,
    quantity: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    limit_price: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fill_price: Option<f64>,
    status: String,
    created_at: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    filled_at: Option<i64>,
}

#[derive(Serialize)]
struct Position {
    symbol: String,
    quantity: f64,
    avg_cost: f64,
    current_price: f64,
    market_value: f64,
    unrealized_pnl: f64,
    unrealized_pnl_percent: f64,
    weight: f64,
}

#[derive(Serialize)]
struct Portfolio {
    cash: f64,
    equity: f64,
    total_value: f64,
    daily_pnl: f64,
    daily_pnl_percent: f64,
    positions: Vec<Position>,
}

#[derive(Serialize, Clone)]
struct OhlcvBar {
    time: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
}

#[derive(Serialize)]
struct QuoteRow {
    symbol: String,
    close: f64,
    date: String,
}

#[derive(Serialize)]
struct RiskMetrics {
    gross_exposure: f64,
    net_exposure: f64,
    unrealized_pnl: f64,
    max_drawdown: f64,
    vol_30d: f64,
    sharpe_30d: f64,
}

#[derive(Serialize)]
struct SignalRow {
    symbol: String,
    momentum: f64,
    trend: f64,
    mean_reversion: f64,
    score: f64,
}

#[derive(Deserialize)]
struct PlaceOrderBody {
    symbol: String,
    side: String,
    #[serde(rename = "type")]
    order_type: String,
    quantity: f64,
    #[serde(rename = "limitPrice")]
    limit_price: Option<f64>,
}

#[derive(Deserialize)]
struct HistoryQuery {
    range: Option<String>,
}

#[derive(Deserialize)]
struct OhlcvQuery {
    symbol: Option<String>,
    interval: Option<String>,
}

// ── Auth middleware ───────────────────────────────────────────────────────────

async fn auth_middleware(
    State(cfg): State<Arc<Config>>,
    headers: HeaderMap,
    request: axum::extract::Request,
    next: Next,
) -> Response {
    let path = request.uri().path().to_owned();
    if path == "/health" || path == "/metrics" {
        return next.run(request).await;
    }
    let key = headers
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if key != cfg.api_key {
        return (StatusCode::UNAUTHORIZED, Json(serde_json::json!({"error":"unauthorized"}))).into_response();
    }
    next.run(request).await
}

// ── Handlers ──────────────────────────────────────────────────────────────────

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "ok", "version": "0.1.0"}))
}

async fn get_orders(State(cfg): State<Arc<Config>>) -> impl IntoResponse {
    let path = cfg.oms_db.clone();
    let orders = tokio::task::spawn_blocking(move || -> Result<Vec<Order>> {
        if !path.exists() {
            return Ok(vec![]);
        }
        let conn = rusqlite::Connection::open_with_flags(
            &path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
        )?;
        let mut stmt = conn.prepare(
            "SELECT id, symbol, side, order_type, quantity, limit_price, avg_fill_price,
                    status, created_at, updated_at
             FROM orders
             ORDER BY created_at DESC
             LIMIT 100",
        )?;
        let rows = stmt.query_map([], |row| {
            let created_str: String = row.get(8)?;
            let created_ts = chrono::DateTime::parse_from_rfc3339(&created_str)
                .map(|dt| dt.timestamp())
                .unwrap_or(0);
            Ok(Order {
                id: row.get(0)?,
                symbol: row.get(1)?,
                side: row.get(2)?,
                order_type: row.get(3)?,
                quantity: row.get(4)?,
                limit_price: row.get(5)?,
                fill_price: {
                    let p: f64 = row.get(6)?;
                    if p == 0.0 { None } else { Some(p) }
                },
                status: row.get(7)?,
                created_at: created_ts,
                filled_at: None,
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>().map_err(Into::into)
    })
    .await
    .unwrap_or_else(|_| Ok(vec![]));

    match orders {
        Ok(v) => Json(v).into_response(),
        Err(e) => {
            warn!("orders query failed: {e}");
            Json(Vec::<Order>::new()).into_response()
        }
    }
}

async fn place_order(
    State(cfg): State<Arc<Config>>,
    Json(body): Json<PlaceOrderBody>,
) -> impl IntoResponse {
    let path = cfg.oms_db.clone();
    let id = Uuid::new_v4().to_string();
    let now = Utc::now();
    let id2 = id.clone();
    let sym = body.symbol.clone();
    let side = body.side.clone();
    let otype = body.order_type.clone();
    let result = tokio::task::spawn_blocking(move || -> Result<()> {
        let conn = rusqlite::Connection::open(&path)?;
        conn.execute(
            "INSERT INTO orders (id, symbol, side, quantity, order_type, limit_price,
              stop_price, time_in_force, status, filled_quantity, avg_fill_price,
              created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, NULL, 'day', 'pending', 0.0, 0.0, ?7, ?7)",
            rusqlite::params![
                id2,
                sym,
                side,
                body.quantity,
                otype,
                body.limit_price,
                now.to_rfc3339(),
            ],
        )?;
        Ok(())
    })
    .await;

    match result {
        Ok(Ok(())) => Json(Order {
            id,
            symbol: body.symbol,
            side: body.side,
            order_type: body.order_type,
            quantity: body.quantity,
            limit_price: body.limit_price,
            fill_price: None,
            status: "pending".into(),
            created_at: now.timestamp(),
            filled_at: None,
        })
        .into_response(),
        _ => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": "failed to place order"})),
        )
            .into_response(),
    }
}

async fn cancel_order(
    State(cfg): State<Arc<Config>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let path = cfg.oms_db.clone();
    let result = tokio::task::spawn_blocking(move || -> Result<()> {
        if !path.exists() {
            return Ok(());
        }
        let conn = rusqlite::Connection::open(&path)?;
        conn.execute(
            "UPDATE orders SET status='cancelled', updated_at=?1 WHERE id=?2 AND status='pending'",
            rusqlite::params![Utc::now().to_rfc3339(), id],
        )?;
        Ok(())
    })
    .await;

    match result {
        Ok(Ok(())) => StatusCode::NO_CONTENT.into_response(),
        _ => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}

async fn get_portfolio(State(cfg): State<Arc<Config>>) -> impl IntoResponse {
    let path = cfg.oms_db.clone();
    let portfolio = tokio::task::spawn_blocking(move || -> Result<Portfolio> {
        if !path.exists() {
            return Ok(empty_portfolio());
        }
        let conn = rusqlite::Connection::open_with_flags(
            &path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
        )?;
        let mut stmt = conn.prepare(
            "SELECT symbol, quantity, avg_cost, market_price FROM positions WHERE quantity != 0",
        )?;
        let positions: Vec<Position> = stmt
            .query_map([], |row| {
                let symbol: String = row.get(0)?;
                let quantity: f64 = row.get(1)?;
                let avg_cost: f64 = row.get(2)?;
                let market_price: f64 = row.get(3)?;
                let market_value = quantity * market_price;
                let cost_basis = quantity * avg_cost;
                let unrealized_pnl = market_value - cost_basis;
                let unrealized_pnl_percent = if cost_basis != 0.0 {
                    unrealized_pnl / cost_basis.abs() * 100.0
                } else {
                    0.0
                };
                Ok(Position {
                    symbol,
                    quantity,
                    avg_cost,
                    current_price: market_price,
                    market_value,
                    unrealized_pnl,
                    unrealized_pnl_percent,
                    weight: 0.0, // computed below
                })
            })?
            .filter_map(|r| r.ok())
            .collect();

        let equity: f64 = positions.iter().map(|p| p.market_value).sum();
        let cash = 1_000_000.0_f64; // TODO: track cash in OMS
        let total_value = equity + cash;
        let positions = positions
            .into_iter()
            .map(|mut p| {
                p.weight = if total_value > 0.0 { p.market_value / total_value * 100.0 } else { 0.0 };
                p
            })
            .collect();

        Ok(Portfolio {
            cash,
            equity,
            total_value,
            daily_pnl: 0.0,
            daily_pnl_percent: 0.0,
            positions,
        })
    })
    .await
    .unwrap_or_else(|_| Ok(empty_portfolio()));

    Json(portfolio.unwrap_or_else(|_| empty_portfolio()))
}

fn empty_portfolio() -> Portfolio {
    Portfolio {
        cash: 1_000_000.0,
        equity: 0.0,
        total_value: 1_000_000.0,
        daily_pnl: 0.0,
        daily_pnl_percent: 0.0,
        positions: vec![],
    }
}

async fn get_quotes(State(cfg): State<Arc<Config>>) -> impl IntoResponse {
    let path = cfg.quant_db.clone();
    let quotes = tokio::task::spawn_blocking(move || -> Result<Vec<QuoteRow>> {
        if !path.exists() {
            return Ok(vec![]);
        }
        let conn = duckdb::Connection::open_with_flags(&path, duckdb::Config::default())?;
        let mut stmt = conn.prepare(
            "SELECT symbol, close, date FROM ohlcv
             WHERE (symbol, date) IN (
               SELECT symbol, MAX(date) FROM ohlcv GROUP BY symbol
             )
             ORDER BY symbol
             LIMIT 100",
        )?;
        let rows: Vec<QuoteRow> = stmt
            .query_map([], |row| {
                Ok(QuoteRow {
                    symbol: row.get(0)?,
                    close: row.get(1)?,
                    date: row.get(2)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    })
    .await
    .unwrap_or_else(|_| Ok(vec![]));

    Json(quotes.unwrap_or_default())
}

async fn get_history(
    State(cfg): State<Arc<Config>>,
    Path(symbol): Path<String>,
    Query(params): Query<HistoryQuery>,
) -> impl IntoResponse {
    let days: i64 = match params.range.as_deref().unwrap_or("6mo") {
        "1mo" => 30,
        "3mo" => 90,
        "1y" => 365,
        "2y" => 730,
        _ => 180, // 6mo
    };
    let path = cfg.quant_db.clone();
    let bars = tokio::task::spawn_blocking(move || -> Result<Vec<OhlcvBar>> {
        if !path.exists() {
            return Ok(vec![]);
        }
        let conn = duckdb::Connection::open_with_flags(&path, duckdb::Config::default())?;
        let cutoff = (Utc::now() - chrono::Duration::days(days))
            .format("%Y-%m-%d")
            .to_string();
        let mut stmt = conn.prepare(
            "SELECT date, open, high, low, close, volume FROM ohlcv
             WHERE symbol = ? AND date >= ?
             ORDER BY date ASC",
        )?;
        let bars: Vec<OhlcvBar> = stmt
            .query_map([symbol.as_str(), cutoff.as_str()], |row| {
                let date_str: String = row.get(0)?;
                let ts = chrono::NaiveDate::parse_from_str(&date_str, "%Y-%m-%d")
                    .map(|d| d.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp())
                    .unwrap_or(0);
                Ok(OhlcvBar {
                    time: ts,
                    open: row.get(1)?,
                    high: row.get(2)?,
                    low: row.get(3)?,
                    close: row.get(4)?,
                    volume: row.get(5)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();
        Ok(bars)
    })
    .await
    .unwrap_or_else(|_| Ok(vec![]));

    Json(bars.unwrap_or_default())
}

async fn get_ohlcv(
    State(cfg): State<Arc<Config>>,
    Query(params): Query<OhlcvQuery>,
) -> impl IntoResponse {
    let symbol = params.symbol.unwrap_or_else(|| "AAPL".into());
    let _interval = params.interval.unwrap_or_else(|| "1d".into());
    // DuckDB store is daily; return last 60 bars
    let path = cfg.quant_db.clone();
    let bars = tokio::task::spawn_blocking(move || -> Result<Vec<OhlcvBar>> {
        if !path.exists() {
            return Ok(vec![]);
        }
        let conn = duckdb::Connection::open_with_flags(&path, duckdb::Config::default())?;
        let mut stmt = conn.prepare(
            "SELECT date, open, high, low, close, volume FROM ohlcv
             WHERE symbol = ?
             ORDER BY date DESC
             LIMIT 60",
        )?;
        let mut bars: Vec<OhlcvBar> = stmt
            .query_map([symbol.as_str()], |row| {
                let date_str: String = row.get(0)?;
                let ts = chrono::NaiveDate::parse_from_str(&date_str, "%Y-%m-%d")
                    .map(|d| d.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp())
                    .unwrap_or(0);
                Ok(OhlcvBar {
                    time: ts,
                    open: row.get(1)?,
                    high: row.get(2)?,
                    low: row.get(3)?,
                    close: row.get(4)?,
                    volume: row.get(5)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();
        bars.sort_by_key(|b| b.time);
        Ok(bars)
    })
    .await
    .unwrap_or_else(|_| Ok(vec![]));

    Json(bars.unwrap_or_default())
}

async fn get_risk(State(cfg): State<Arc<Config>>) -> impl IntoResponse {
    let path = cfg.metrics_path.clone();
    let metrics = tokio::task::spawn_blocking(move || -> RiskMetrics {
        if !path.exists() {
            return RiskMetrics::default();
        }
        let content = std::fs::read_to_string(&path).unwrap_or_default();
        parse_prom_metrics(&content)
    })
    .await
    .unwrap_or_default();
    Json(metrics)
}

fn parse_prom_metrics(content: &str) -> RiskMetrics {
    let mut m: HashMap<&str, f64> = HashMap::new();
    for line in content.lines() {
        if line.starts_with('#') {
            continue;
        }
        if let Some((key, val)) = line.split_once(' ') {
            if let Ok(v) = val.trim().parse::<f64>() {
                m.insert(key, v);
            }
        }
    }
    RiskMetrics {
        gross_exposure: m.get("gross_exposure").copied().unwrap_or(0.0),
        net_exposure: m.get("net_exposure").copied().unwrap_or(0.0),
        unrealized_pnl: m.get("unrealized_pnl").copied().unwrap_or(0.0),
        max_drawdown: m.get("max_drawdown").copied().unwrap_or(0.0),
        vol_30d: m.get("vol_30d").copied().unwrap_or(0.0),
        sharpe_30d: m.get("sharpe_30d").copied().unwrap_or(0.0),
    }
}

impl Default for RiskMetrics {
    fn default() -> Self {
        RiskMetrics {
            gross_exposure: 0.0,
            net_exposure: 0.0,
            unrealized_pnl: 0.0,
            max_drawdown: 0.0,
            vol_30d: 0.0,
            sharpe_30d: 0.0,
        }
    }
}

async fn get_signals(State(cfg): State<Arc<Config>>) -> impl IntoResponse {
    let path = cfg.quant_db.clone();
    let signals = tokio::task::spawn_blocking(move || -> Result<Vec<SignalRow>> {
        if !path.exists() {
            return Ok(vec![]);
        }
        let conn = duckdb::Connection::open_with_flags(&path, duckdb::Config::default())?;
        // Simple momentum: close vs 20-day moving average
        let mut stmt = conn.prepare(
            "WITH latest AS (
               SELECT symbol, close, date,
                      AVG(close) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS ma20,
                      AVG(close) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) AS ma60
               FROM ohlcv
             ),
             ranked AS (
               SELECT symbol, close, ma20, ma60,
                      ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date DESC) AS rn
               FROM latest
             )
             SELECT symbol,
                    (close - ma20) / NULLIF(ma20, 0) AS momentum,
                    (ma20 - ma60) / NULLIF(ma60, 0) AS trend,
                    (ma20 - close) / NULLIF(close, 0) AS mean_rev
             FROM ranked
             WHERE rn = 1
             ORDER BY momentum DESC
             LIMIT 30",
        )?;
        let rows: Vec<SignalRow> = stmt
            .query_map([], |row| {
                let momentum: f64 = row.get::<_, Option<f64>>(1)?.unwrap_or(0.0);
                let trend: f64 = row.get::<_, Option<f64>>(2)?.unwrap_or(0.0);
                let mean_rev: f64 = row.get::<_, Option<f64>>(3)?.unwrap_or(0.0);
                Ok(SignalRow {
                    symbol: row.get(0)?,
                    momentum,
                    trend,
                    mean_reversion: mean_rev,
                    score: momentum * 0.4 + trend * 0.4 + mean_rev * 0.2,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    })
    .await
    .unwrap_or_else(|_| Ok(vec![]));

    Json(signals.unwrap_or_default())
}

async fn get_backtest_latest(State(cfg): State<Arc<Config>>) -> impl IntoResponse {
    let dir = cfg.backtest_dir.clone();
    let result = tokio::task::spawn_blocking(move || -> Option<serde_json::Value> {
        let entries = std::fs::read_dir(&dir).ok()?;
        let latest = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".json"))
            .max_by_key(|e| e.metadata().and_then(|m| m.modified()).ok());
        let path = latest?.path();
        let content = std::fs::read_to_string(path).ok()?;
        serde_json::from_str(&content).ok()
    })
    .await
    .unwrap_or(None);

    match result {
        Some(v) => Json(v).into_response(),
        None => Json(serde_json::json!({"error": "no backtest results found"})).into_response(),
    }
}

// ── WebSocket ─────────────────────────────────────────────────────────────────

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(cfg): State<Arc<Config>>,
    headers: HeaderMap,
) -> Response {
    // Allow WS auth via query param or header (for browser WebSocket which can't set headers)
    let authorized = headers
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .map(|k| k == cfg.api_key)
        .unwrap_or(false);

    if !authorized && cfg.api_key != "changeme" {
        return (StatusCode::UNAUTHORIZED, "unauthorized").into_response();
    }

    ws.on_upgrade(move |socket| handle_ws(socket, cfg))
}

async fn handle_ws(socket: WebSocket, cfg: Arc<Config>) {
    let (mut sender, mut receiver) = socket.split();

    let mut quote_tick = interval(Duration::from_secs(2));
    let mut portfolio_tick = interval(Duration::from_secs(10));
    let mut heartbeat_tick = interval(Duration::from_secs(30));
    // skip the immediate first tick
    quote_tick.tick().await;
    portfolio_tick.tick().await;
    heartbeat_tick.tick().await;

    loop {
        tokio::select! {
            msg = receiver.next() => {
                match msg {
                    None | Some(Err(_)) => break,
                    Some(Ok(Message::Close(_))) => break,
                    _ => {}
                }
            }
            _ = quote_tick.tick() => {
                if let Some(msg) = build_quote_message(&cfg).await {
                    if sender.send(Message::Text(msg.into())).await.is_err() {
                        break;
                    }
                }
            }
            _ = portfolio_tick.tick() => {
                if let Some(msg) = build_portfolio_message(&cfg).await {
                    if sender.send(Message::Text(msg.into())).await.is_err() {
                        break;
                    }
                }
            }
            _ = heartbeat_tick.tick() => {
                let hb = serde_json::json!({"type":"heartbeat","ts": Utc::now().to_rfc3339()}).to_string();
                if sender.send(Message::Text(hb.into())).await.is_err() {
                    break;
                }
            }
        }
    }
}

async fn build_quote_message(cfg: &Config) -> Option<String> {
    let path = cfg.quant_db.clone();
    let quotes = tokio::task::spawn_blocking(move || -> Option<Vec<QuoteRow>> {
        if !path.exists() {
            return None;
        }
        let conn = duckdb::Connection::open_with_flags(&path, duckdb::Config::default()).ok()?;
        let mut stmt = conn.prepare(
            "SELECT symbol, close, date FROM ohlcv
             WHERE (symbol, date) IN (
               SELECT symbol, MAX(date) FROM ohlcv GROUP BY symbol
             )
             LIMIT 20",
        ).ok()?;
        let rows: Vec<QuoteRow> = stmt
            .query_map([], |row| Ok(QuoteRow {
                symbol: row.get(0)?,
                close: row.get(1)?,
                date: row.get(2)?,
            }))
            .ok()?
            .filter_map(|r| r.ok())
            .collect();
        Some(rows)
    })
    .await
    .ok()
    .flatten()?;

    // Emit one quote message per symbol
    let messages: Vec<serde_json::Value> = quotes
        .into_iter()
        .map(|q| serde_json::json!({
            "type": "quote",
            "data": {
                "symbol": q.symbol,
                "price": q.close,
                "change": 0.0,
                "changePercent": 0.0,
                "high": q.close,
                "low": q.close,
                "open": q.close,
                "previousClose": q.close,
                "volume": 0,
                "timestamp": Utc::now().timestamp_millis()
            }
        }))
        .collect();

    Some(serde_json::to_string(&messages).unwrap_or_default())
}

async fn build_portfolio_message(cfg: &Config) -> Option<String> {
    let path = cfg.oms_db.clone();
    let portfolio = tokio::task::spawn_blocking(move || -> Option<Portfolio> {
        if !path.exists() {
            return None;
        }
        let conn = rusqlite::Connection::open_with_flags(
            &path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
        ).ok()?;
        let mut stmt = conn.prepare(
            "SELECT symbol, quantity, avg_cost, market_price FROM positions WHERE quantity != 0",
        ).ok()?;
        let positions: Vec<Position> = stmt
            .query_map([], |row| {
                let qty: f64 = row.get(1)?;
                let avg: f64 = row.get(2)?;
                let price: f64 = row.get(3)?;
                let mv = qty * price;
                let cost = qty * avg;
                let pnl = mv - cost;
                let pnl_pct = if cost != 0.0 { pnl / cost.abs() * 100.0 } else { 0.0 };
                Ok(Position {
                    symbol: row.get(0)?,
                    quantity: qty,
                    avg_cost: avg,
                    current_price: price,
                    market_value: mv,
                    unrealized_pnl: pnl,
                    unrealized_pnl_percent: pnl_pct,
                    weight: 0.0,
                })
            })
            .ok()?
            .filter_map(|r| r.ok())
            .collect();

        let equity: f64 = positions.iter().map(|p| p.market_value).sum();
        let total = equity + 1_000_000.0;
        let positions = positions
            .into_iter()
            .map(|mut p| { p.weight = if total > 0.0 { p.market_value / total * 100.0 } else { 0.0 }; p })
            .collect();

        Some(Portfolio {
            cash: 1_000_000.0,
            equity,
            total_value: total,
            daily_pnl: 0.0,
            daily_pnl_percent: 0.0,
            positions,
        })
    })
    .await
    .ok()
    .flatten()?;

    Some(serde_json::json!({
        "type": "portfolio",
        "data": portfolio
    }).to_string())
}

// ── Main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("quant_api=info".parse().unwrap()),
        )
        .init();

    let cfg = Arc::new(Config::from_env());
    let port = cfg.port;

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/health", get(health))
        .route("/ws", get(ws_handler))
        .route("/api/v1/orders", get(get_orders).post(place_order))
        .route("/api/v1/orders/:id", delete(cancel_order))
        .route("/api/v1/portfolio", get(get_portfolio))
        .route("/api/v1/market/quotes", get(get_quotes))
        .route("/api/v1/market/history/:symbol", get(get_history))
        .route("/api/v1/market/ohlcv", get(get_ohlcv))
        .route("/api/v1/risk", get(get_risk))
        .route("/api/v1/signals", get(get_signals))
        .route("/api/v1/backtest/latest", get(get_backtest_latest))
        .layer(middleware::from_fn_with_state(cfg.clone(), auth_middleware))
        .layer(cors)
        .with_state(cfg);

    let addr = format!("0.0.0.0:{port}");
    info!("quant-api listening on {addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
