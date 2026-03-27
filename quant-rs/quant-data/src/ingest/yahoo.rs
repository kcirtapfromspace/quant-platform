/// Yahoo Finance OHLCV data source via the `v8/finance/chart` HTTP API.
///
/// Downloads bars for each symbol individually using the blocking `reqwest`
/// client. Batch downloads are not available in the public JSON API, so we
/// fetch one ticker per request with a small inter-request delay to stay
/// within Yahoo's informal rate limits.
use std::collections::HashMap;
use std::thread;
use std::time::Duration;

use chrono::{NaiveDate, TimeZone, Utc};
use reqwest::blocking::Client;
use serde::Deserialize;
use tracing::{debug, warn};

use crate::error::DataError;
use crate::ingest::DataSource;
use crate::models::OhlcvRecord;

const DELAY_BETWEEN_SYMBOLS: Duration = Duration::from_millis(300);

// Yahoo Finance chart API response shapes ─────────────────────────────────

#[derive(Deserialize)]
struct YfResponse {
    chart: YfChart,
}

#[derive(Deserialize)]
struct YfChart {
    result: Option<Vec<YfResult>>,
    error: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct YfResult {
    timestamp: Vec<i64>,
    indicators: YfIndicators,
}

#[derive(Deserialize)]
struct YfIndicators {
    quote: Vec<YfQuote>,
    adjclose: Option<Vec<YfAdjClose>>,
}

#[derive(Deserialize)]
struct YfQuote {
    open: Vec<Option<f64>>,
    high: Vec<Option<f64>>,
    low: Vec<Option<f64>>,
    close: Vec<Option<f64>>,
    volume: Vec<Option<f64>>,
}

#[derive(Deserialize)]
struct YfAdjClose {
    adjclose: Vec<Option<f64>>,
}

// ─────────────────────────────────────────────────────────────────────────

/// Synchronous Yahoo Finance data source.
pub struct YahooFinanceSource {
    client: Client,
}

impl YahooFinanceSource {
    pub fn new() -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("Mozilla/5.0 (compatible; quant-data/0.1)")
            .build()
            .expect("failed to build HTTP client");
        Self { client }
    }

    fn fetch_one(
        &self,
        symbol: &str,
        start: NaiveDate,
        end: NaiveDate,
    ) -> Result<Vec<OhlcvRecord>, DataError> {
        // Yahoo end timestamp is exclusive; add one day.
        let start_ts = Utc
            .from_utc_datetime(&start.and_hms_opt(0, 0, 0).unwrap())
            .timestamp();
        let end_ts = Utc
            .from_utc_datetime(
                &end.succ_opt()
                    .unwrap_or(end)
                    .and_hms_opt(0, 0, 0)
                    .unwrap(),
            )
            .timestamp();

        let url = format!(
            "https://query1.finance.yahoo.com/v8/finance/chart/{}?interval=1d&period1={}&period2={}&includeAdjustedClose=true",
            symbol, start_ts, end_ts
        );

        debug!("Fetching {} from Yahoo Finance", symbol);

        let resp: YfResponse = self
            .client
            .get(&url)
            .send()
            .map_err(DataError::Http)?
            .error_for_status()
            .map_err(DataError::Http)?
            .json()
            .map_err(DataError::Http)?;

        if let Some(err) = resp.chart.error {
            return Err(DataError::Parse(format!(
                "Yahoo API error for {}: {}",
                symbol, err
            )));
        }

        let results = match resp.chart.result {
            Some(r) if !r.is_empty() => r,
            _ => return Ok(vec![]),
        };

        let result = &results[0];
        let quote = match result.indicators.quote.first() {
            Some(q) => q,
            None => return Ok(vec![]),
        };
        let adj_closes: Vec<Option<f64>> = result
            .indicators
            .adjclose
            .as_ref()
            .and_then(|a| a.first())
            .map(|a| a.adjclose.clone())
            .unwrap_or_default();

        let mut records = Vec::new();
        for (i, &ts) in result.timestamp.iter().enumerate() {
            let open = match quote.open.get(i).copied().flatten() {
                Some(v) => v,
                None => continue,
            };
            let high = match quote.high.get(i).copied().flatten() {
                Some(v) => v,
                None => continue,
            };
            let low = match quote.low.get(i).copied().flatten() {
                Some(v) => v,
                None => continue,
            };
            let close = match quote.close.get(i).copied().flatten() {
                Some(v) => v,
                None => continue,
            };
            let volume = quote
                .volume
                .get(i)
                .copied()
                .flatten()
                .unwrap_or(0.0);
            let adj_close = adj_closes
                .get(i)
                .copied()
                .flatten()
                .unwrap_or(close);

            let date = chrono::DateTime::from_timestamp(ts, 0)
                .map(|dt| dt.naive_utc().date())
                .unwrap_or_else(|| {
                    warn!("Could not parse timestamp {} for {}", ts, symbol);
                    start
                });

            records.push(OhlcvRecord {
                symbol: symbol.to_uppercase(),
                date,
                open,
                high,
                low,
                close,
                volume,
                adj_close,
            });
        }

        records.sort_by_key(|r| r.date);
        Ok(records)
    }
}

impl Default for YahooFinanceSource {
    fn default() -> Self {
        Self::new()
    }
}

impl DataSource for YahooFinanceSource {
    fn name(&self) -> &str {
        "yahoo_finance"
    }

    fn fetch(
        &self,
        symbols: &[String],
        start: NaiveDate,
        end: NaiveDate,
    ) -> Result<Vec<OhlcvRecord>, DataError> {
        let mut all: Vec<OhlcvRecord> = Vec::new();
        let mut errors: HashMap<String, String> = HashMap::new();

        for (i, symbol) in symbols.iter().enumerate() {
            match self.fetch_one(symbol, start, end) {
                Ok(records) => {
                    debug!("{}: {} bars", symbol, records.len());
                    all.extend(records);
                }
                Err(e) => {
                    warn!("Failed to fetch {}: {}", symbol, e);
                    errors.insert(symbol.clone(), e.to_string());
                }
            }

            if i + 1 < symbols.len() {
                thread::sleep(DELAY_BETWEEN_SYMBOLS);
            }
        }

        if all.is_empty() && !symbols.is_empty() {
            return Err(DataError::NoData(symbols.to_vec()));
        }

        all.sort_by(|a, b| a.symbol.cmp(&b.symbol).then(a.date.cmp(&b.date)));
        Ok(all)
    }
}
