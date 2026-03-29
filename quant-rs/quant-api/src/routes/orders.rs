use axum::{extract::State, Json};
use serde::Serialize;
use std::sync::Arc;

use crate::{error::ApiResult, AppState};

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OrderView {
    pub id: String,
    pub symbol: String,
    pub side: String,
    pub r#type: String,
    pub quantity: f64,
    pub limit_price: Option<f64>,
    pub fill_price: Option<f64>,
    pub status: String,
    pub created_at: i64,
    pub filled_at: Option<i64>,
}

pub async fn get_orders(State(state): State<Arc<AppState>>) -> ApiResult<Json<Vec<OrderView>>> {
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

    let views = orders
        .into_iter()
        .map(|order| {
            let filled = order.filled_quantity > 0.0;
            OrderView {
                id: order.id,
                symbol: order.symbol,
                side: order.side.as_str().to_string(),
                r#type: order.order_type.as_str().to_string(),
                quantity: order.quantity,
                limit_price: order.limit_price,
                fill_price: filled.then_some(order.avg_fill_price),
                status: order.status.as_str().to_string(),
                created_at: order.created_at.timestamp_millis(),
                filled_at: filled.then_some(order.updated_at.timestamp_millis()),
            }
        })
        .collect();

    Ok(Json(views))
}
