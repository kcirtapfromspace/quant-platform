//! Alpaca Markets REST broker adapter.
//!
//! Implements the [`Broker`] trait against the Alpaca v2 REST API.
//!
//! # Authentication
//! Pass your key and secret directly to [`AlpacaBrokerAdapter::new`], or let
//! [`AlpacaBrokerAdapter::from_env`] read `APCA_API_KEY_ID` and
//! `APCA_API_SECRET_KEY` from the environment.
//!
//! # Paper trading
//! Pass `paper: true` to route requests to `https://paper-api.alpaca.markets`
//! instead of the live endpoint.

use reqwest::blocking::Client;
use reqwest::header::{HeaderMap, HeaderValue, InvalidHeaderValue};
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::broker::{AccountInfo, Broker, BrokerError, BrokerPosition};
use crate::models::Order;

// ── Constants ─────────────────────────────────────────────────────────────────

const LIVE_BASE_URL: &str = "https://api.alpaca.markets";
const PAPER_BASE_URL: &str = "https://paper-api.alpaca.markets";

// ── Adapter ───────────────────────────────────────────────────────────────────

pub struct AlpacaBrokerAdapter {
    client: Client,
    base_url: String,
}

impl AlpacaBrokerAdapter {
    /// Create a new adapter with explicit credentials.
    ///
    /// Set `paper = true` to use Alpaca's paper-trading endpoint.
    pub fn new(key_id: &str, secret_key: &str, paper: bool) -> Result<Self, BrokerError> {
        let base_url = if paper { PAPER_BASE_URL } else { LIVE_BASE_URL };
        Self::with_base_url(key_id, secret_key, base_url.to_string())
    }

    /// Create a new adapter reading credentials from environment variables
    /// (`APCA_API_KEY_ID` and `APCA_API_SECRET_KEY`).
    pub fn from_env(paper: bool) -> Result<Self, BrokerError> {
        let key_id = std::env::var("APCA_API_KEY_ID")
            .map_err(|_| BrokerError::Config("APCA_API_KEY_ID not set".into()))?;
        let secret_key = std::env::var("APCA_API_SECRET_KEY")
            .map_err(|_| BrokerError::Config("APCA_API_SECRET_KEY not set".into()))?;
        Self::new(&key_id, &secret_key, paper)
    }

    fn with_base_url(
        key_id: &str,
        secret_key: &str,
        base_url: String,
    ) -> Result<Self, BrokerError> {
        let mut headers = HeaderMap::new();
        headers.insert(
            "APCA-API-KEY-ID",
            HeaderValue::from_str(key_id).map_err(header_err)?,
        );
        headers.insert(
            "APCA-API-SECRET-KEY",
            HeaderValue::from_str(secret_key).map_err(header_err)?,
        );

        let client = Client::builder()
            .default_headers(headers)
            .build()
            .map_err(BrokerError::Http)?;

        Ok(Self { client, base_url })
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }
}

fn header_err(e: InvalidHeaderValue) -> BrokerError {
    BrokerError::Config(format!("invalid header value: {e}"))
}

// ── Alpaca API request/response shapes ────────────────────────────────────────

#[derive(Serialize)]
struct AlpacaOrderRequest<'a> {
    symbol: &'a str,
    /// Fractional quantities are supported; Alpaca accepts decimal strings.
    qty: String,
    side: &'a str,
    #[serde(rename = "type")]
    order_type: &'a str,
    time_in_force: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    limit_price: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_price: Option<String>,
}

#[derive(Deserialize)]
struct AlpacaOrderResponse {
    id: String,
}

#[derive(Deserialize)]
struct AlpacaAccountResponse {
    id: String,
    equity: String,
    cash: String,
    buying_power: String,
    portfolio_value: String,
    currency: String,
    status: String,
}

#[derive(Deserialize)]
struct AlpacaPositionResponse {
    symbol: String,
    qty: String,
    avg_entry_price: String,
    market_value: String,
    unrealized_pl: String,
}

// ── Broker implementation ─────────────────────────────────────────────────────

impl Broker for AlpacaBrokerAdapter {
    fn submit_order(&self, order: &Order) -> Result<String, BrokerError> {
        let qty = format_qty(order.quantity);
        let req = AlpacaOrderRequest {
            symbol: &order.symbol,
            qty,
            side: order.side.as_str(),
            order_type: order.order_type.as_str(),
            time_in_force: order.time_in_force.as_str(),
            limit_price: order.limit_price.map(|p| format!("{p:.2}")),
            stop_price: order.stop_price.map(|p| format!("{p:.2}")),
        };

        debug!(symbol = %order.symbol, qty = %req.qty, side = order.side.as_str(), "submitting order to Alpaca");

        let resp = self.client.post(self.url("/v2/orders")).json(&req).send()?;

        check_status(&resp)?;
        let body: AlpacaOrderResponse = resp.json()?;
        Ok(body.id)
    }

    fn get_account(&self) -> Result<AccountInfo, BrokerError> {
        let resp = self.client.get(self.url("/v2/account")).send()?;
        check_status(&resp)?;

        let body: AlpacaAccountResponse = resp.json()?;
        Ok(AccountInfo {
            id: body.id,
            equity: parse_f64(&body.equity),
            cash: parse_f64(&body.cash),
            buying_power: parse_f64(&body.buying_power),
            portfolio_value: parse_f64(&body.portfolio_value),
            currency: body.currency,
            status: body.status,
        })
    }

    fn get_positions(&self) -> Result<Vec<BrokerPosition>, BrokerError> {
        let resp = self.client.get(self.url("/v2/positions")).send()?;
        check_status(&resp)?;

        let body: Vec<AlpacaPositionResponse> = resp.json()?;
        Ok(body
            .into_iter()
            .map(|p| BrokerPosition {
                symbol: p.symbol,
                qty: parse_f64(&p.qty),
                avg_entry_price: parse_f64(&p.avg_entry_price),
                market_value: parse_f64(&p.market_value),
                unrealized_pl: parse_f64(&p.unrealized_pl),
            })
            .collect())
    }

    fn cancel_order(&self, broker_order_id: &str) -> Result<(), BrokerError> {
        debug!(broker_order_id, "cancelling Alpaca order");
        let resp = self
            .client
            .delete(self.url(&format!("/v2/orders/{broker_order_id}")))
            .send()?;
        check_status(&resp)?;
        Ok(())
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Format a quantity for the Alpaca API.  Whole numbers are sent as integers;
/// fractional quantities keep up to 9 decimal places.
fn format_qty(qty: f64) -> String {
    if qty.fract() == 0.0 {
        format!("{}", qty as u64)
    } else {
        format!("{qty:.9}")
    }
}

/// Parse a decimal string returned by Alpaca into f64, defaulting to 0.0 on
/// failure (Alpaca occasionally returns `""` for unavailable fields).
fn parse_f64(s: &str) -> f64 {
    s.parse().unwrap_or(0.0)
}

/// Consume the response status, returning `BrokerError::Api` on non-2xx.
fn check_status(resp: &reqwest::blocking::Response) -> Result<(), BrokerError> {
    if resp.status().is_success() {
        return Ok(());
    }
    Err(BrokerError::Api {
        status: resp.status().as_u16(),
        message: resp
            .status()
            .canonical_reason()
            .unwrap_or("unknown")
            .to_string(),
    })
}

// ── Unit tests (mock HTTP) ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Order, OrderSide, OrderType};
    use mockito::Server;

    fn make_adapter(server: &mockito::ServerGuard) -> AlpacaBrokerAdapter {
        AlpacaBrokerAdapter {
            client: Client::new(),
            base_url: server.url(),
        }
    }

    // ── submit_order ─────────────────────────────────────────────────────────

    #[test]
    fn test_submit_order_returns_broker_id() {
        let mut server = Server::new();
        let _m = server
            .mock("POST", "/v2/orders")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"id":"alpaca-order-abc123","status":"pending_new"}"#)
            .create();

        let adapter = make_adapter(&server);
        let order = Order::new("AAPL", OrderSide::Buy, 10.0, OrderType::Market);
        let broker_id = adapter.submit_order(&order).unwrap();
        assert_eq!(broker_id, "alpaca-order-abc123");
    }

    #[test]
    fn test_submit_order_api_error_propagates() {
        let mut server = Server::new();
        let _m = server
            .mock("POST", "/v2/orders")
            .with_status(422)
            .with_header("content-type", "application/json")
            .with_body(r#"{"code":40010001,"message":"qty is required"}"#)
            .create();

        let adapter = make_adapter(&server);
        let order = Order::new("AAPL", OrderSide::Buy, 10.0, OrderType::Market);
        let err = adapter.submit_order(&order).unwrap_err();
        assert!(matches!(err, BrokerError::Api { status: 422, .. }));
    }

    // ── get_account ──────────────────────────────────────────────────────────

    #[test]
    fn test_get_account_parses_fields() {
        let mut server = Server::new();
        let _m = server
            .mock("GET", "/v2/account")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                    "id": "acct-001",
                    "equity": "125000.50",
                    "cash": "50000.00",
                    "buying_power": "200000.00",
                    "portfolio_value": "125000.50",
                    "currency": "USD",
                    "status": "ACTIVE"
                }"#,
            )
            .create();

        let adapter = make_adapter(&server);
        let info = adapter.get_account().unwrap();
        assert_eq!(info.id, "acct-001");
        assert!((info.equity - 125_000.50).abs() < 1e-6);
        assert!((info.cash - 50_000.0).abs() < 1e-6);
        assert_eq!(info.currency, "USD");
        assert_eq!(info.status, "ACTIVE");
    }

    // ── get_positions ────────────────────────────────────────────────────────

    #[test]
    fn test_get_positions_returns_all_positions() {
        let mut server = Server::new();
        let _m = server
            .mock("GET", "/v2/positions")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"[
                    {
                        "symbol": "AAPL",
                        "qty": "10",
                        "avg_entry_price": "150.00",
                        "market_value": "1600.00",
                        "unrealized_pl": "100.00"
                    },
                    {
                        "symbol": "TSLA",
                        "qty": "5",
                        "avg_entry_price": "700.00",
                        "market_value": "3600.00",
                        "unrealized_pl": "100.00"
                    }
                ]"#,
            )
            .create();

        let adapter = make_adapter(&server);
        let positions = adapter.get_positions().unwrap();
        assert_eq!(positions.len(), 2);
        assert_eq!(positions[0].symbol, "AAPL");
        assert!((positions[0].qty - 10.0).abs() < 1e-9);
        assert!((positions[0].avg_entry_price - 150.0).abs() < 1e-6);
        assert_eq!(positions[1].symbol, "TSLA");
    }

    // ── cancel_order ─────────────────────────────────────────────────────────

    #[test]
    fn test_cancel_order_success() {
        let mut server = Server::new();
        let _m = server
            .mock("DELETE", "/v2/orders/broker-order-xyz")
            .with_status(204)
            .create();

        let adapter = make_adapter(&server);
        adapter.cancel_order("broker-order-xyz").unwrap();
    }

    // ── format_qty helper ────────────────────────────────────────────────────

    #[test]
    fn test_format_qty_whole_number() {
        assert_eq!(format_qty(10.0), "10");
        assert_eq!(format_qty(1.0), "1");
    }

    #[test]
    fn test_format_qty_fractional() {
        assert_eq!(format_qty(1.5), "1.500000000");
    }
}
