use axum::{extract::State, Json};
use serde_json::Value;
use std::sync::Arc;

use crate::{
    error::{ApiError, ApiResult},
    AppState,
};

pub async fn get_backtest_latest(
    State(state): State<Arc<AppState>>,
) -> ApiResult<Json<Value>> {
    let results_dir = state.backtest_results_dir.clone();

    let result = tokio::task::spawn_blocking(move || {
        find_latest_results(&results_dir)
    })
    .await?;

    match result {
        Some(v) => Ok(Json(v)),
        None => Err(ApiError::NotFound(
            "no backtest results found".to_string(),
        )),
    }
}

/// Walk the backtest-results directory for the most recently modified results.json.
fn find_latest_results(dir: &str) -> Option<Value> {
    let mut best: Option<(std::time::SystemTime, std::path::PathBuf)> = None;

    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path().join("results.json");
        if path.exists() {
            if let Ok(meta) = std::fs::metadata(&path) {
                if let Ok(modified) = meta.modified() {
                    let is_better = best.as_ref().is_none_or(|(t, _)| modified > *t);
                    if is_better {
                        best = Some((modified, path));
                    }
                }
            }
        }
    }

    let (_, path) = best?;
    let text = std::fs::read_to_string(&path).ok()?;
    serde_json::from_str(&text).ok()
}
