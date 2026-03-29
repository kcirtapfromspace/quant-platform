use axum::{
    extract::{Path, Query, State},
    Json,
};
use chrono::Local;
use quant_data::OhlcvRecord;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::{error::ApiResult, AppState};

#[derive(Deserialize)]
pub struct HistoryParams {
    range: Option<String>,
}

/// OHLCV bar shaped for the frontend chart (Unix timestamps, adj_close as close).
#[derive(Serialize)]
pub struct OhlcvBarView {
    pub time: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

pub async fn get_history(
    State(state): State<Arc<AppState>>,
    Path(symbol): Path<String>,
    Query(params): Query<HistoryParams>,
) -> ApiResult<Json<Vec<OhlcvBarView>>> {
    let db_path = state.db_path.clone();
    let range = params.range.unwrap_or_else(|| "6mo".to_string());

    let bars = tokio::task::spawn_blocking(move || {
        let store = quant_data::MarketDataStore::open_read_only(&db_path)?;
        let end = Local::now().date_naive();
        let days: i64 = match range.as_str() {
            "1mo" => 30,
            "3mo" => 90,
            "6mo" => 180,
            "1y" => 365,
            "2y" => 730,
            "5y" => 1825,
            _ => 180,
        };
        let start = end - chrono::Duration::days(days);
        let records = store.query(&symbol.to_uppercase(), start, end)?;
        let views: Vec<OhlcvBarView> = records
            .into_iter()
            .map(|r| OhlcvBarView {
                time: r.date.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp(),
                open: r.open,
                high: r.high,
                low: r.low,
                close: r.adj_close,
                volume: r.volume,
            })
            .collect();
        Ok::<_, anyhow::Error>(views)
    })
    .await??;

    Ok(Json(bars))
}

pub async fn get_quotes(State(state): State<Arc<AppState>>) -> ApiResult<Json<Vec<OhlcvRecord>>> {
    let db_path = state.db_path.clone();

    let quotes = tokio::task::spawn_blocking(move || {
        let store = quant_data::MarketDataStore::open_read_only(&db_path)?;
        let symbols = store.symbols()?;

        let mut result = Vec::new();
        for sym in &symbols {
            if let Ok(Some(date)) = store.latest_date(sym) {
                if let Ok(mut bars) = store.query(sym, date, date) {
                    result.append(&mut bars);
                }
            }
        }
        result.sort_by(|a, b| a.symbol.cmp(&b.symbol));
        Ok::<_, anyhow::Error>(result)
    })
    .await??;

    Ok(Json(quotes))
}
