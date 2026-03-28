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
            .from_utc_datetime(&end.succ_opt().unwrap_or(end).and_hms_opt(0, 0, 0).unwrap())
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

        parse_yahoo_records(symbol, start, resp)
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

fn parse_yahoo_records(
    symbol: &str,
    start: NaiveDate,
    resp: YfResponse,
) -> Result<Vec<OhlcvRecord>, DataError> {
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
        let volume = quote.volume.get(i).copied().flatten().unwrap_or(0.0);
        let adj_close = adj_closes.get(i).copied().flatten().unwrap_or(close);

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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn nd(year: i32, month: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(year, month, day).unwrap()
    }

    fn sample_response(
        timestamps: Vec<i64>,
        open: Vec<Option<f64>>,
        high: Vec<Option<f64>>,
        low: Vec<Option<f64>>,
        close: Vec<Option<f64>>,
        volume: Vec<Option<f64>>,
        adjclose: Option<Vec<Option<f64>>>,
    ) -> YfResponse {
        YfResponse {
            chart: YfChart {
                result: Some(vec![YfResult {
                    timestamp: timestamps,
                    indicators: YfIndicators {
                        quote: vec![YfQuote {
                            open,
                            high,
                            low,
                            close,
                            volume,
                        }],
                        adjclose: adjclose.map(|adjclose| vec![YfAdjClose { adjclose }]),
                    },
                }]),
                error: None,
            },
        }
    }

    #[test]
    fn parse_yahoo_records_returns_parse_error_when_api_reports_one() {
        let resp = YfResponse {
            chart: YfChart {
                result: None,
                error: Some(json!({"code": "Not Found", "description": "symbol missing"})),
            },
        };

        let err = parse_yahoo_records("msft", nd(2024, 1, 2), resp).unwrap_err();

        match err {
            DataError::Parse(message) => {
                assert!(message.contains("Yahoo API error for msft"));
                assert!(message.contains("Not Found"));
            }
            other => panic!("expected parse error, got {other:?}"),
        }
    }

    #[test]
    fn parse_yahoo_records_skips_incomplete_rows_and_sorts_by_date() {
        let jan_3 = nd(2024, 1, 3)
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp();
        let jan_2 = nd(2024, 1, 2)
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp();
        let resp = sample_response(
            vec![jan_3, jan_2],
            vec![Some(11.0), None],
            vec![Some(12.0), Some(11.5)],
            vec![Some(10.5), Some(10.0)],
            vec![Some(11.5), Some(11.0)],
            vec![Some(2000.0), Some(1500.0)],
            Some(vec![Some(11.4), Some(10.9)]),
        );

        let records = parse_yahoo_records("aapl", nd(2024, 1, 1), resp).unwrap();

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].symbol, "AAPL");
        assert_eq!(records[0].date, nd(2024, 1, 3));
        assert_eq!(records[0].adj_close, 11.4);
    }

    #[test]
    fn parse_yahoo_records_falls_back_to_close_and_start_date() {
        let resp = sample_response(
            vec![i64::MAX],
            vec![Some(101.0)],
            vec![Some(102.0)],
            vec![Some(99.0)],
            vec![Some(100.0)],
            vec![None],
            None,
        );

        let records = parse_yahoo_records("spy", nd(2024, 2, 5), resp).unwrap();

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].date, nd(2024, 2, 5));
        assert_eq!(records[0].volume, 0.0);
        assert_eq!(records[0].adj_close, 100.0);
    }
}
