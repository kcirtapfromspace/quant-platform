use axum::{extract::State, Json};
use serde::Serialize;
use std::sync::Arc;

use crate::{error::ApiResult, AppState};

#[derive(Serialize)]
pub struct RiskMetrics {
    pub gross_exposure: f64,
    pub net_exposure: f64,
    pub gross_exposure_pct: f64,
    pub net_exposure_pct: f64,
    pub unrealized_pnl: f64,
    pub n_long: usize,
    pub n_short: usize,
    pub portfolio_value: f64,
    /// 30-day rolling portfolio vol (annualised). None if insufficient history.
    pub rolling_vol_30d: Option<f64>,
    /// Max drawdown from Prometheus textfile if available.
    pub max_drawdown: Option<f64>,
}

pub async fn get_risk(State(state): State<Arc<AppState>>) -> ApiResult<Json<RiskMetrics>> {
    let oms_path = state.oms_db_path.clone();
    let metrics_file = state.metrics_file.clone();

    let (positions, prom_metrics) = tokio::task::spawn_blocking(move || {
        let positions = oms_path
            .as_deref()
            .and_then(|p| quant_oms::SqliteStateStore::new(p).ok())
            .and_then(|s| s.load_positions().ok())
            .unwrap_or_default();

        let prom = std::fs::read_to_string(&metrics_file).unwrap_or_default();
        (positions, prom)
    })
    .await?;

    let gross_exposure: f64 = positions.values().map(|p| p.market_value().abs()).sum();
    let net_exposure: f64 = positions.values().map(|p| p.market_value()).sum();
    let unrealized_pnl: f64 = positions.values().map(|p| p.unrealized_pnl()).sum();
    let n_long = positions.values().filter(|p| p.quantity > 0.0).count();
    let n_short = positions.values().filter(|p| p.quantity < 0.0).count();
    let portfolio_value = gross_exposure;

    let (gross_pct, net_pct) = if portfolio_value > 0.0 {
        (gross_exposure / portfolio_value, net_exposure / portfolio_value)
    } else {
        (0.0, 0.0)
    };

    // Parse max_drawdown and vol from Prometheus textfile if present
    let max_drawdown = parse_prom_gauge(&prom_metrics, "quant_max_drawdown");
    let rolling_vol_30d = parse_prom_gauge(&prom_metrics, "quant_rolling_vol_30d");

    Ok(Json(RiskMetrics {
        gross_exposure,
        net_exposure,
        gross_exposure_pct: gross_pct,
        net_exposure_pct: net_pct,
        unrealized_pnl,
        n_long,
        n_short,
        portfolio_value,
        rolling_vol_30d,
        max_drawdown,
    }))
}

/// Extract the scalar value of a gauge metric from a Prometheus textfile.
fn parse_prom_gauge(text: &str, name: &str) -> Option<f64> {
    text.lines()
        .find(|l| l.starts_with(name) && !l.starts_with('#'))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse::<f64>().ok())
}
