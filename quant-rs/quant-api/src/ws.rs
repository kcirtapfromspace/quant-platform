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
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WsEvent {
    PriceTick {
        symbol: String,
        close: f64,
        date: String,
    },
    PositionUpdate {
        symbol: String,
        quantity: f64,
        market_value: f64,
        unrealized_pnl: f64,
    },
    Heartbeat,
}

pub fn new_broadcast() -> (broadcast::Sender<WsEvent>, broadcast::Receiver<WsEvent>) {
    broadcast::channel(256)
}

pub async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> Response {
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
            let store = quant_data::MarketDataStore::open(&db_path).ok()?;
            let symbols = store.symbols().ok()?;
            let mut result = Vec::new();
            for sym in &symbols {
                if let Ok(Some(date)) = store.latest_date(sym) {
                    if let Ok(bars) = store.query(sym, date, date) {
                        if let Some(bar) = bars.into_iter().next() {
                            result.push((sym.clone(), bar.close, date.to_string()));
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

        for (symbol, close, date) in quotes {
            let _ = state
                .broadcast_tx
                .send(WsEvent::PriceTick { symbol, close, date });
        }

        // Fetch current positions from OMS SQLite
        if let Some(ref oms_path) = state.oms_db_path {
            let path = oms_path.clone();
            let positions = tokio::task::spawn_blocking(move || {
                let store = quant_oms::SqliteStateStore::new(&path).ok()?;
                store.load_positions().ok()
            })
            .await
            .ok()
            .flatten()
            .unwrap_or_default();

            for (symbol, pos) in positions {
                let _ = state.broadcast_tx.send(WsEvent::PositionUpdate {
                    symbol,
                    quantity: pos.quantity,
                    market_value: pos.market_value(),
                    unrealized_pnl: pos.unrealized_pnl(),
                });
            }
        }
    }
}
