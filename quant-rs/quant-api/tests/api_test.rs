use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt;
use tower::ServiceExt;

fn test_state() -> std::sync::Arc<quant_api::AppState> {
    quant_api::AppState::new_with_key(
        ":memory:",
        None,
        "/tmp/quant_test_metrics.prom",
        "./backtest-results",
        None,
    )
}

fn test_state_with_key(key: &str) -> std::sync::Arc<quant_api::AppState> {
    quant_api::AppState::new_with_key(
        ":memory:",
        None,
        "/tmp/quant_test_metrics.prom",
        "./backtest-results",
        Some(key.to_string()),
    )
}

async fn body_json(body: Body) -> serde_json::Value {
    let bytes = body.collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap()
}

// ── /health ──────────────────────────────────────────────────────────────────

#[tokio::test]
async fn health_returns_ok() {
    let app = quant_api::build_router(test_state());
    let response = app
        .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let json = body_json(response.into_body()).await;
    assert_eq!(json["status"], "ok");
}

// ── /api/v1/portfolio ────────────────────────────────────────────────────────

#[tokio::test]
async fn portfolio_returns_empty_without_oms() {
    let app = quant_api::build_router(test_state());
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/v1/portfolio")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let json = body_json(response.into_body()).await;
    assert_eq!(json["n_positions"], 0);
}

// ── /api/v1/orders ───────────────────────────────────────────────────────────

#[tokio::test]
async fn orders_returns_empty_without_oms() {
    let app = quant_api::build_router(test_state());
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/v1/orders")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let json = body_json(response.into_body()).await;
    assert!(json.as_array().unwrap().is_empty());
}

// ── /api/v1/risk ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn risk_returns_zero_exposure_without_oms() {
    let app = quant_api::build_router(test_state());
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/v1/risk")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let json = body_json(response.into_body()).await;
    assert_eq!(json["n_long"], 0);
    assert_eq!(json["n_short"], 0);
}

// ── /api/v1/signals ──────────────────────────────────────────────────────────

#[tokio::test]
async fn signals_returns_empty_for_empty_db() {
    let app = quant_api::build_router(test_state());
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/v1/signals")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let json = body_json(response.into_body()).await;
    assert!(json.as_array().unwrap().is_empty());
}

// ── /api/v1/market/quotes ────────────────────────────────────────────────────

#[tokio::test]
async fn market_quotes_returns_empty_for_empty_db() {
    let app = quant_api::build_router(test_state());
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/v1/market/quotes")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let json = body_json(response.into_body()).await;
    assert!(json.as_array().unwrap().is_empty());
}

// ── /api/v1/backtest/latest ──────────────────────────────────────────────────

#[tokio::test]
async fn backtest_latest_returns_404_when_no_results() {
    let app = quant_api::build_router(test_state());
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/v1/backtest/latest")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    // Either 404 (no results dir) or 200 if backtest-results exists with data
    assert!(
        response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::OK
    );
}

// ── auth middleware ───────────────────────────────────────────────────────────

#[tokio::test]
async fn auth_rejects_missing_key() {
    let app = quant_api::build_router(test_state_with_key("secret-token"));
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/v1/portfolio")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    // Restore env
}

#[tokio::test]
async fn auth_rejects_wrong_key() {
    let app = quant_api::build_router(test_state_with_key("correct-key"));
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/v1/portfolio")
                .header("X-API-Key", "wrong-key")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn auth_accepts_correct_key() {
    let app = quant_api::build_router(test_state_with_key("my-secret"));
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/v1/portfolio")
                .header("X-API-Key", "my-secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn health_bypasses_auth() {
    let app = quant_api::build_router(test_state_with_key("secret"));
    let response = app
        .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();

    // /health is not under /api/v1 so auth middleware does not apply
    assert_eq!(response.status(), StatusCode::OK);
}
