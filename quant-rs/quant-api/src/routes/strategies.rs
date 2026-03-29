use axum::Json;
use serde::Serialize;

use crate::error::ApiResult;

#[derive(Serialize)]
pub struct StrategyStateView {
    pub strategy_key: String,
    pub name: String,
    pub status: String,
    pub regime: String,
    pub signal_confidence: f64,
    pub daily_pnl: f64,
    pub positions: u32,
    pub category: String,
}

pub async fn get_strategies() -> ApiResult<Json<Vec<StrategyStateView>>> {
    let strategies = vec![
        StrategyStateView {
            strategy_key: "momentum".to_string(),
            name: "Momentum Signal".to_string(),
            status: "paper".to_string(),
            regime: "bull".to_string(),
            signal_confidence: 0.65,
            daily_pnl: 0.0,
            positions: 0,
            category: "Time-series".to_string(),
        },
        StrategyStateView {
            strategy_key: "mean_reversion".to_string(),
            name: "Mean Reversion Signal".to_string(),
            status: "paper".to_string(),
            regime: "sideways".to_string(),
            signal_confidence: 0.55,
            daily_pnl: 0.0,
            positions: 0,
            category: "Time-series".to_string(),
        },
        StrategyStateView {
            strategy_key: "trend_following".to_string(),
            name: "Trend Following Signal".to_string(),
            status: "paper".to_string(),
            regime: "bull".to_string(),
            signal_confidence: 0.60,
            daily_pnl: 0.0,
            positions: 0,
            category: "Time-series".to_string(),
        },
    ];
    Ok(Json(strategies))
}
