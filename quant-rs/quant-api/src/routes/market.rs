use axum::{extract::State, Json};
use quant_data::OhlcvRecord;
use std::sync::Arc;

use crate::{error::ApiResult, AppState};

pub async fn get_quotes(State(state): State<Arc<AppState>>) -> ApiResult<Json<Vec<OhlcvRecord>>> {
    let db_path = state.db_path.clone();

    let quotes = tokio::task::spawn_blocking(move || {
        let store = quant_data::MarketDataStore::open(&db_path)?;
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
