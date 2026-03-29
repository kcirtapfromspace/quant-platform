use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::Response,
};
use futures_util::{SinkExt, StreamExt};
use serde::Serialize;
use std::sync::Arc;
use tokio::sync::broadcast;

use crate::AppState;

/// Events pushed to WebSocket clients.
#[derive(Clone, Serialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum WsEvent {
    Quote(QuotePayload),
    Ohlcv(OhlcvPayload),
    Heartbeat,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct QuotePayload {
    pub symbol: String,
    pub price: f64,
    pub change: f64,
    pub change_percent: f64,
    pub high: f64,
    pub low: f64,
    pub open: f64,
    pub previous_close: f64,
    pub volume: f64,
    pub timestamp: i64,
}

#[derive(Clone, Serialize)]
pub struct OhlcvPayload {
    pub time: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

pub fn new_broadcast() -> (broadcast::Sender<WsEvent>, broadcast::Receiver<WsEvent>) {
    broadcast::channel(256)
}

pub async fn ws_handler(ws: WebSocketUpgrade, State(state): State<Arc<AppState>>) -> Response {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, state: Arc<AppState>) {
    let mut rx = state.broadcast_tx.subscribe();
    let (mut sender, mut receiver) = socket.split();

    // Receive task — drain client pings/close frames
    let recv_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = receiver.next().await {
            if let Message::Close(_) = msg {
                break;
            }
        }
    });

    // Send task — forward broadcast events to this client
    let send_task = tokio::spawn(async move {
        loop {
            match rx.recv().await {
                Ok(event) => {
                    let text = match serde_json::to_string(&event) {
                        Ok(s) => s,
                        Err(_) => continue,
                    };
                    if sender.send(Message::Text(text.into())).await.is_err() {
                        break;
                    }
                }
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
                Err(broadcast::error::RecvError::Closed) => break,
            }
        }
    });

    tokio::select! {
        _ = recv_task => {},
        _ = send_task => {},
    }
}

/// Background task: polls DuckDB for latest quotes and OMS for positions,
/// broadcasts updates every second.
pub async fn broadcast_task(state: Arc<AppState>) {
    let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(1));
    loop {
        interval.tick().await;

        // Broadcast heartbeat so clients know the connection is alive
        let _ = state.broadcast_tx.send(WsEvent::Heartbeat);

        // Fetch latest OHLCV quotes from DuckDB
        let db_path = state.db_path.clone();
        let quotes = tokio::task::spawn_blocking(move || {
            let store = quant_data::MarketDataStore::open_read_only(&db_path).ok()?;
            let symbols = store.symbols().ok()?;
            let mut result = Vec::new();
            for sym in &symbols {
                if let Ok(Some(date)) = store.latest_date(sym) {
                    if let Ok(bars) = store.query(sym, date, date) {
                        if let Some(bar) = bars.into_iter().next() {
                            result.push(bar);
                        }
                    }
                }
            }
            Some(result)
        })
        .await
        .ok()
        .flatten()
        .unwrap_or_default();

        for bar in quotes {
            let previous_close = bar.open;
            let price = bar.adj_close;
            let change = price - previous_close;
            let change_percent = if previous_close.abs() > f64::EPSILON {
                (change / previous_close) * 100.0
            } else {
                0.0
            };
            let timestamp = bar.date.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp();

            let _ = state.broadcast_tx.send(WsEvent::Quote(QuotePayload {
                symbol: bar.symbol.clone(),
                price,
                change,
                change_percent,
                high: bar.high,
                low: bar.low,
                open: bar.open,
                previous_close,
                volume: bar.volume,
                timestamp,
            }));

            let _ = state.broadcast_tx.send(WsEvent::Ohlcv(OhlcvPayload {
                time: timestamp,
                open: bar.open,
                high: bar.high,
                low: bar.low,
                close: price,
                volume: bar.volume,
            }));
        }

        // TODO: add explicit portfolio/orderbook strategy_state websocket payloads if/when
        // frontend consumers require live updates beyond quote/ohlcv streaming.
    }
}
