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

#[derive(Deserialize)]
pub struct OhlcvParams {
    symbol: Option<String>,
    interval: Option<String>,
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

fn range_days(range: &str) -> i64 {
    match range {
        "1mo" => 30,
        "3mo" => 90,
        "6mo" => 180,
        "1y" => 365,
        "2y" => 730,
        "5y" => 1825,
        _ => 180,
    }
}

fn interval_days(interval: &str) -> i64 {
    match interval {
        "1m" => 5,
        "5m" => 14,
        "15m" => 30,
        "1h" => 90,
        "1d" => 365,
        _ => 180,
    }
}

fn ohlcv_view_from_records(records: Vec<OhlcvRecord>) -> Vec<OhlcvBarView> {
    records
        .into_iter()
        .map(|r| OhlcvBarView {
            time: r.date.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp(),
            open: r.open,
            high: r.high,
            low: r.low,
            close: r.adj_close,
            volume: r.volume,
        })
        .collect()
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
        let days = range_days(&range);
        let start = end - chrono::Duration::days(days);
        let records = store.query(&symbol.to_uppercase(), start, end)?;
        Ok::<_, anyhow::Error>(ohlcv_view_from_records(records))
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

pub async fn get_ohlcv(
    State(state): State<Arc<AppState>>,
    Query(params): Query<OhlcvParams>,
) -> ApiResult<Json<Vec<OhlcvBarView>>> {
    let db_path = state.db_path.clone();
    let symbol = params.symbol.unwrap_or_else(|| "AAPL".to_string());
    let interval = params.interval.unwrap_or_else(|| "5m".to_string());

    let bars = tokio::task::spawn_blocking(move || {
        let store = quant_data::MarketDataStore::open_read_only(&db_path)?;
        let end = Local::now().date_naive();
        let days = interval_days(&interval);
        let start = end - chrono::Duration::days(days);
        let records = store.query(&symbol.to_uppercase(), start, end)?;
        Ok::<_, anyhow::Error>(ohlcv_view_from_records(records))
    })
    .await??;

    Ok(Json(bars))
}

pub async fn get_watchlist(State(state): State<Arc<AppState>>) -> ApiResult<Json<Vec<String>>> {
    let db_path = state.db_path.clone();

    let symbols = tokio::task::spawn_blocking(move || {
        let store = quant_data::MarketDataStore::open_read_only(&db_path)?;
        let mut symbols = store.symbols()?;
        symbols.sort();
        symbols.dedup();
        Ok::<_, anyhow::Error>(symbols)
    })
    .await??;

    Ok(Json(symbols))
}
