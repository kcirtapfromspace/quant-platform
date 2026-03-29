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

pub async fn require_api_key(
    State(state): State<Arc<AppState>>,
    req: Request,
    next: Next,
) -> Response {
    if let Some(ref expected) = state.api_key {
        let provided = req
            .headers()
            .get("X-API-Key")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        if provided != expected {
            return (
                StatusCode::UNAUTHORIZED,
                Json(json!({ "error": "invalid or missing X-API-Key" })),
            )
                .into_response();
        }
    }
    next.run(req).await
}
