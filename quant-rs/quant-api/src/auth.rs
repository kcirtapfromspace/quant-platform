use axum::{
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use std::sync::Arc;

use crate::AppState;

/// API key middleware.
///
/// Checks `X-API-Key` request header first, then falls back to the `api_key`
/// query parameter (needed for WebSocket upgrades where browsers cannot set
/// custom headers).
///
/// Fail-closed: if no API key is configured in [`AppState`] the request is
/// rejected with 401 regardless of what the client sends.
pub async fn require_api_key(
    State(state): State<Arc<AppState>>,
    req: Request,
    next: Next,
) -> Response {
    let Some(ref expected) = state.api_key else {
        // No key configured — fail closed. The server should have been started
        // with QUANT_API_KEY set; if not, deny everything rather than exposing
        // the API unprotected.
        return (
            StatusCode::UNAUTHORIZED,
            Json(json!({ "error": "server is not configured with an API key" })),
        )
            .into_response();
    };

    // Accept key from header (REST) or query param (WebSocket upgrade).
    let provided = req
        .headers()
        .get("X-API-Key")
        .and_then(|v| v.to_str().ok())
        .or_else(|| {
            req.uri()
                .query()
                .and_then(|q| {
                    q.split('&').find_map(|part| {
                        let (k, v) = part.split_once('=')?;
                        (k == "api_key").then_some(v)
                    })
                })
        })
        .unwrap_or("");

    if provided != expected {
        return (
            StatusCode::UNAUTHORIZED,
            Json(json!({ "error": "invalid or missing X-API-Key" })),
        )
            .into_response();
    }

    next.run(req).await
}
