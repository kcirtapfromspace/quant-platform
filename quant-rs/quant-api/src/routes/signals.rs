use axum::{extract::State, Json};
use serde::Serialize;
use std::sync::Arc;

use crate::{error::ApiResult, AppState};

const LOOKBACK: usize = 252;
const RSI_PERIOD: usize = 14;
const MACD_FAST: usize = 12;
const MACD_SLOW: usize = 26;
const MACD_SIGNAL_PERIOD: usize = 9;
const BB_PERIOD: usize = 20;
const BB_STD: f64 = 2.0;

#[derive(Serialize)]
pub struct SignalView {
    pub symbol: String,
    pub momentum_score: f64,
    pub momentum_confidence: f64,
    pub mean_reversion_score: f64,
    pub mean_reversion_confidence: f64,
    pub trend_score: f64,
    pub trend_confidence: f64,
    pub combined_target: f64,
    pub rsi: f64,
}

pub async fn get_signals(State(state): State<Arc<AppState>>) -> ApiResult<Json<Vec<SignalView>>> {
    let db_path = state.db_path.clone();

    let signals = tokio::task::spawn_blocking(move || {
        let store = quant_data::MarketDataStore::open(&db_path)?;
        let symbols = store.symbols()?;

        let end = chrono::Local::now().date_naive();
        let start = end - chrono::Duration::days((LOOKBACK + 60) as i64);

        let mut views = Vec::new();
        for sym in &symbols {
            if let Ok(bars) = store.query(sym, start, end) {
                if bars.len() < LOOKBACK {
                    continue;
                }
                let closes: Vec<f64> = bars.iter().map(|b| b.adj_close).collect();
                let view = compute_signals_for(sym, &closes);
                views.push(view);
            }
        }
        views.sort_by(|a, b| {
            b.combined_target
                .abs()
                .partial_cmp(&a.combined_target.abs())
                .unwrap()
        });
        Ok::<_, anyhow::Error>(views)
    })
    .await??;

    Ok(Json(signals))
}

fn compute_signals_for(symbol: &str, closes: &[f64]) -> SignalView {
    use quant_features as qf;
    use quant_signals::{mean_reversion_signal, momentum_signal, trend_following_signal};

    let rsi_vals = qf::rsi(closes, RSI_PERIOD);
    let returns = qf::returns(closes);
    let macd_hist = qf::macd_histogram(closes, MACD_FAST, MACD_SLOW, MACD_SIGNAL_PERIOD);
    let fast_ma = qf::ema(closes, MACD_FAST);
    let slow_ma = qf::ema(closes, MACD_SLOW);
    let bb_mid = qf::bb_mid(closes, BB_PERIOD);
    let bb_upper = qf::bb_upper(closes, BB_PERIOD, BB_STD);
    let bb_lower = qf::bb_lower(closes, BB_PERIOD, BB_STD);

    let (mom_score, mom_conf, _) = momentum_signal(&rsi_vals, &returns, 5, 0.05);
    let (mr_score, mr_conf, _) =
        mean_reversion_signal(&bb_mid, &bb_upper, &bb_lower, &returns, BB_STD);
    let (trend_score, trend_conf, _) = trend_following_signal(&macd_hist, &fast_ma, &slow_ma);

    let combined_target = (mom_score * mom_conf + mr_score * mr_conf + trend_score * trend_conf)
        / (mom_conf + mr_conf + trend_conf).max(1e-9);

    let last_rsi = rsi_vals
        .iter()
        .rev()
        .find(|v| v.is_finite())
        .copied()
        .unwrap_or(50.0);

    SignalView {
        symbol: symbol.to_string(),
        momentum_score: mom_score,
        momentum_confidence: mom_conf,
        mean_reversion_score: mr_score,
        mean_reversion_confidence: mr_conf,
        trend_score,
        trend_confidence: trend_conf,
        combined_target,
        rsi: last_rsi,
    }
}
