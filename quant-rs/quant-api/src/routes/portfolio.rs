use axum::{extract::State, Json};
use serde::Serialize;
use std::sync::Arc;

use crate::{error::ApiResult, AppState};

#[derive(Serialize)]
pub struct PositionView {
    pub symbol: String,
    pub quantity: f64,
    pub avg_cost: f64,
    pub market_price: f64,
    pub market_value: f64,
    pub unrealized_pnl: f64,
    pub weight: f64,
}

#[derive(Serialize)]
pub struct PortfolioResponse {
    pub positions: Vec<PositionView>,
    pub portfolio_value: f64,
    pub cash: f64,
    pub n_positions: usize,
}

pub async fn get_portfolio(
    State(state): State<Arc<AppState>>,
) -> ApiResult<Json<PortfolioResponse>> {
    let oms_path = match &state.oms_db_path {
        Some(p) => p.clone(),
        None => {
            return Ok(Json(PortfolioResponse {
                positions: vec![],
                portfolio_value: 0.0,
                cash: 0.0,
                n_positions: 0,
            }))
        }
    };

    let result = tokio::task::spawn_blocking(move || {
        let store = quant_oms::SqliteStateStore::new(&oms_path)?;
        let positions = store.load_positions()?;
        Ok::<_, anyhow::Error>(positions)
    })
    .await??;

    let total_market_value: f64 = result.values().map(|p| p.market_value()).sum();
    let portfolio_value = total_market_value; // Cash not directly accessible from store; show equity only

    let mut views: Vec<PositionView> = result
        .into_values()
        .map(|p| {
            let mv = p.market_value();
            let weight = if portfolio_value > 0.0 {
                mv / portfolio_value
            } else {
                0.0
            };
            PositionView {
                symbol: p.symbol.clone(),
                quantity: p.quantity,
                avg_cost: p.avg_cost,
                market_price: p.market_price,
                market_value: mv,
                unrealized_pnl: p.unrealized_pnl(),
                weight,
            }
        })
        .collect();

    views.sort_by(|a, b| {
        b.market_value
            .abs()
            .partial_cmp(&a.market_value.abs())
            .unwrap()
    });

    let n_positions = views.len();
    Ok(Json(PortfolioResponse {
        positions: views,
        portfolio_value,
        cash: 0.0,
        n_positions,
    }))
}
