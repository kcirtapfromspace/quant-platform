use axum::{extract::State, Json};
use quant_oms::Order;
use std::sync::Arc;

use crate::{error::ApiResult, AppState};

pub async fn get_orders(State(state): State<Arc<AppState>>) -> ApiResult<Json<Vec<Order>>> {
    let oms_path = match &state.oms_db_path {
        Some(p) => p.clone(),
        None => return Ok(Json(vec![])),
    };

    let mut orders = tokio::task::spawn_blocking(move || {
        let store = quant_oms::SqliteStateStore::new(&oms_path)?;
        store.load_orders().map_err(anyhow::Error::from)
    })
    .await??;

    // Most recent first, cap at 100
    orders.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    orders.truncate(100);

    Ok(Json(orders))
}
