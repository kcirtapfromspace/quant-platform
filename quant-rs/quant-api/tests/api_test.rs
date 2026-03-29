use std::sync::Arc;

use axum::{
    body::Body,
    http::{Request, StatusCode},
    response::Response,
};
use chrono::{Duration, TimeZone, Utc};
use http_body_util::BodyExt;
use quant_data::{MarketDataStore, OhlcvRecord};
use quant_oms::{
    Order, OrderSide, OrderStatus, OrderType, Position, SqliteStateStore, TimeInForce,
};
use serde_json::{json, Value};
use tempfile::TempDir;
use tower::ServiceExt;

const API_KEY: &str = "test-api-key";

struct TestFixture {
    _tmp: TempDir,
    state: Arc<quant_api::AppState>,
}

impl TestFixture {
    fn seeded() -> Self {
        let tmp = TempDir::new().unwrap();
        let market_db = tmp.path().join("market.duckdb");
        let oms_db = tmp.path().join("oms.sqlite");
        let metrics_file = tmp.path().join("metrics.prom");
        let results_dir = tmp.path().join("backtests");
        let run_dir = results_dir.join("run-001");

        seed_market_data(&market_db);
        seed_oms_data(&oms_db);
        std::fs::write(
            &metrics_file,
            "quant_max_drawdown 0.1234\nquant_rolling_vol_30d 0.1875\n",
        )
        .unwrap();
        std::fs::create_dir_all(&run_dir).unwrap();
        std::fs::write(
            run_dir.join("results.json"),
            serde_json::to_vec(&json!({
                "symbol": "AAPL",
                "sharpe_ratio": 1.42,
                "max_drawdown": 0.1234,
                "total_return": 0.311,
            }))
            .unwrap(),
        )
        .unwrap();

        let state = quant_api::AppState::new_with_key(
            market_db.to_string_lossy().into_owned(),
            Some(oms_db.to_string_lossy().into_owned()),
            metrics_file.to_string_lossy().into_owned(),
            results_dir.to_string_lossy().into_owned(),
            Some(API_KEY.to_string()),
        );

        Self { _tmp: tmp, state }
    }

    fn auth_only() -> Self {
        let tmp = TempDir::new().unwrap();
        let metrics_file = tmp.path().join("metrics.prom");
        let results_dir = tmp.path().join("backtests");
        std::fs::create_dir_all(&results_dir).unwrap();
        std::fs::write(&metrics_file, "").unwrap();

        let state = quant_api::AppState::new_with_key(
            ":memory:",
            None,
            metrics_file.to_string_lossy().into_owned(),
            results_dir.to_string_lossy().into_owned(),
            Some(API_KEY.to_string()),
        );

        Self { _tmp: tmp, state }
    }
}

fn seed_market_data(path: &std::path::Path) {
    let store = MarketDataStore::open(path).unwrap();
    let end = chrono::Local::now().date_naive();
    let start = end - Duration::days(279);

    let aapl: Vec<OhlcvRecord> = (0..280)
        .map(|idx| {
            let date = start + Duration::days(idx as i64);
            let base = 150.0 + idx as f64 * 0.35;
            OhlcvRecord {
                symbol: "AAPL".to_string(),
                date,
                open: base,
                high: base + 1.5,
                low: base - 1.0,
                close: base + 0.8,
                volume: 1_000_000.0 + idx as f64 * 2500.0,
                adj_close: base + 0.8,
            }
        })
        .collect();

    let msft = OhlcvRecord {
        symbol: "MSFT".to_string(),
        date: end,
        open: 410.0,
        high: 414.0,
        low: 408.5,
        close: 412.25,
        volume: 875_000.0,
        adj_close: 412.25,
    };

    let mut all = aapl;
    all.push(msft);
    store.upsert(&all).unwrap();
}

fn seed_oms_data(path: &std::path::Path) {
    let store = SqliteStateStore::new(path).unwrap();

    let mut order = Order::new("AAPL", OrderSide::Buy, 25.0, OrderType::Limit);
    order.id = "ord-aapl-001".to_string();
    order.limit_price = Some(182.5);
    order.avg_fill_price = 183.1;
    order.filled_quantity = 25.0;
    order.status = OrderStatus::Filled;
    order.time_in_force = TimeInForce::Day;
    order.created_at = Utc
        .with_ymd_and_hms(2026, 3, 27, 14, 0, 0)
        .single()
        .unwrap();
    order.updated_at = Utc
        .with_ymd_and_hms(2026, 3, 27, 14, 1, 0)
        .single()
        .unwrap();
    store.save_order(&order).unwrap();

    let mut position = Position::new("AAPL");
    position.quantity = 25.0;
    position.avg_cost = 180.0;
    position.market_price = 183.1;
    store.save_position(&position).unwrap();
}

async fn json_response(
    state: Arc<quant_api::AppState>,
    uri: &str,
    api_key: Option<&str>,
) -> (StatusCode, Value) {
    let response = send_request(state, uri, api_key).await;
    let status = response.status();
    let json = body_json(response.into_body()).await;
    (status, json)
}

async fn send_request(
    state: Arc<quant_api::AppState>,
    uri: &str,
    api_key: Option<&str>,
) -> Response {
    let mut builder = Request::builder().uri(uri);
    if let Some(api_key) = api_key {
        builder = builder.header("X-API-Key", api_key);
    }

    quant_api::build_router(state)
        .oneshot(builder.body(Body::empty()).unwrap())
        .await
        .unwrap()
}

async fn body_json(body: Body) -> Value {
    let bytes = body.collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap()
}

fn assert_order_shape(order: &Value) {
    assert_eq!(order["id"], "ord-aapl-001");
    assert_eq!(order["symbol"], "AAPL");
    assert_eq!(order["side"], "buy");
    assert_eq!(order["order_type"], "limit");
    assert_eq!(order["status"], "filled");
    assert_eq!(order["filled_quantity"], 25.0);
    assert_eq!(order["avg_fill_price"], 183.1);
}

#[tokio::test]
async fn health_returns_public_status_payload() {
    let fixture = TestFixture::auth_only();
    let (status, json) = json_response(fixture.state, "/health", None).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["status"], "ok");
    assert_eq!(json["service"], "quant-api");
    assert!(json["version"].as_str().is_some());
}

#[tokio::test]
async fn protected_routes_require_a_valid_api_key() {
    let fixture = TestFixture::auth_only();

    for uri in [
        "/api/v1/portfolio",
        "/api/v1/orders",
        "/api/v1/risk",
        "/api/v1/signals",
        "/api/v1/market/quotes",
        "/api/v1/backtest/latest",
    ] {
        let (missing_status, missing_json) = json_response(fixture.state.clone(), uri, None).await;
        assert_eq!(missing_status, StatusCode::UNAUTHORIZED, "{uri}");
        assert_eq!(missing_json["error"], "invalid or missing X-API-Key");

        let (wrong_status, wrong_json) =
            json_response(fixture.state.clone(), uri, Some("wrong-key")).await;
        assert_eq!(wrong_status, StatusCode::UNAUTHORIZED, "{uri}");
        assert_eq!(wrong_json["error"], "invalid or missing X-API-Key");
    }
}

#[tokio::test]
async fn portfolio_returns_positions_and_weights() {
    let fixture = TestFixture::seeded();
    let (status, json) = json_response(fixture.state, "/api/v1/portfolio", Some(API_KEY)).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["n_positions"], 1);
    assert_eq!(json["cash"], 0.0);
    assert!(json["portfolio_value"].as_f64().unwrap() > 0.0);

    let positions = json["positions"].as_array().unwrap();
    assert_eq!(positions.len(), 1);
    assert_eq!(positions[0]["symbol"], "AAPL");
    assert_eq!(positions[0]["quantity"], 25.0);
    assert_eq!(positions[0]["avg_cost"], 180.0);
    assert!(positions[0]["weight"].as_f64().unwrap() > 0.99);
}

#[tokio::test]
async fn orders_returns_recent_oms_rows() {
    let fixture = TestFixture::seeded();
    let (status, json) = json_response(fixture.state, "/api/v1/orders", Some(API_KEY)).await;

    assert_eq!(status, StatusCode::OK);
    let orders = json.as_array().unwrap();
    assert_eq!(orders.len(), 1);
    assert_order_shape(&orders[0]);
}

#[tokio::test]
async fn risk_returns_metrics_and_prometheus_gauges() {
    let fixture = TestFixture::seeded();
    let (status, json) = json_response(fixture.state, "/api/v1/risk", Some(API_KEY)).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["n_long"], 1);
    assert_eq!(json["n_short"], 0);
    assert_eq!(json["max_drawdown"], 0.1234);
    assert_eq!(json["rolling_vol_30d"], 0.1875);
    assert!(json["gross_exposure"].as_f64().unwrap() > 0.0);
    assert!(json["portfolio_value"].as_f64().unwrap() > 0.0);
}

#[tokio::test]
async fn signals_returns_ranked_signal_views() {
    let fixture = TestFixture::seeded();
    let (status, json) = json_response(fixture.state, "/api/v1/signals", Some(API_KEY)).await;

    assert_eq!(status, StatusCode::OK);
    let signals = json.as_array().unwrap();
    assert!(!signals.is_empty());
    assert_eq!(signals[0]["symbol"], "AAPL");
    assert!(signals[0]["combined_target"].as_f64().is_some());
    assert!(signals[0]["rsi"].as_f64().is_some());
    assert!(signals[0]["momentum_confidence"].as_f64().is_some());
}

#[tokio::test]
async fn market_quotes_returns_latest_bar_per_symbol() {
    let fixture = TestFixture::seeded();
    let (status, json) = json_response(fixture.state, "/api/v1/market/quotes", Some(API_KEY)).await;

    assert_eq!(status, StatusCode::OK);
    let quotes = json.as_array().unwrap();
    assert_eq!(quotes.len(), 2);
    assert_eq!(quotes[0]["symbol"], "AAPL");
    assert_eq!(quotes[1]["symbol"], "MSFT");
    assert!(quotes.iter().all(|quote| quote["date"].as_str().is_some()));
}

#[tokio::test]
async fn backtest_latest_returns_latest_results_fixture() {
    let fixture = TestFixture::seeded();
    let (status, json) =
        json_response(fixture.state, "/api/v1/backtest/latest", Some(API_KEY)).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["symbol"], "AAPL");
    assert_eq!(json["sharpe_ratio"], 1.42);
    assert_eq!(json["max_drawdown"], 0.1234);
}

#[tokio::test]
async fn backtest_latest_returns_not_found_when_results_are_missing() {
    let fixture = TestFixture::auth_only();
    let (status, json) =
        json_response(fixture.state, "/api/v1/backtest/latest", Some(API_KEY)).await;

    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(json["error"], "no backtest results found");
}
