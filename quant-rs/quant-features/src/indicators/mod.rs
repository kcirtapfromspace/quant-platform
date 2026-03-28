//! Technical indicator kernels.
//!
//! All functions follow the same contract:
//! - Input: price/volume slices in ascending chronological order.
//! - Output: `Vec<f64>` of the same length; NaN during warm-up.
//! - No panics on valid input (period >= minimum requirement).

// ─── Return-based ────────────────────────────────────────────────────────────

/// Simple period returns: `(p[t] - p[t-1]) / p[t-1]`. NaN at index 0.
pub fn returns(prices: &[f64]) -> Vec<f64> {
    let n = prices.len();
    let mut out = vec![f64::NAN; n];
    for i in 1..n {
        let prev = prices[i - 1];
        if prev != 0.0 {
            out[i] = (prices[i] - prev) / prev;
        }
    }
    out
}

/// Log returns: `ln(p[t] / p[t-1])`. NaN at index 0 or when p ≤ 0.
pub fn log_returns(prices: &[f64]) -> Vec<f64> {
    let n = prices.len();
    let mut out = vec![f64::NAN; n];
    for i in 1..n {
        let prev = prices[i - 1];
        let curr = prices[i];
        if prev > 0.0 && curr > 0.0 {
            out[i] = (curr / prev).ln();
        }
    }
    out
}

// ─── Rolling statistics ───────────────────────────────────────────────────────

/// Rolling mean over `period` bars. NaN for the first `period-1` values.
///
/// # Panics
/// Panics if `period < 1`.
pub fn rolling_mean(prices: &[f64], period: usize) -> Vec<f64> {
    assert!(period >= 1, "period must be >= 1");
    let n = prices.len();
    let mut out = vec![f64::NAN; n];
    if period > n {
        return out;
    }
    let mut sum: f64 = prices[..period].iter().sum();
    out[period - 1] = sum / period as f64;
    for i in period..n {
        sum += prices[i] - prices[i - period];
        out[i] = sum / period as f64;
    }
    out
}

/// Rolling sample std (ddof=1) over `period` bars. NaN for the first `period-1` values.
///
/// # Panics
/// Panics if `period < 2`.
pub fn rolling_std(prices: &[f64], period: usize) -> Vec<f64> {
    assert!(period >= 2, "period must be >= 2");
    let n = prices.len();
    let mut out = vec![f64::NAN; n];
    if period > n {
        return out;
    }
    for i in (period - 1)..n {
        let window = &prices[(i + 1 - period)..=i];
        let mean = window.iter().sum::<f64>() / period as f64;
        let var = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (period - 1) as f64;
        out[i] = var.sqrt();
    }
    out
}

// ─── EMA / MACD ───────────────────────────────────────────────────────────────

/// Exponential Moving Average with `span`. `alpha = 2 / (span + 1)`, `adjust=False`.
///
/// # Panics
/// Panics if `span < 1`.
pub fn ema(prices: &[f64], span: usize) -> Vec<f64> {
    assert!(span >= 1, "span must be >= 1");
    if prices.is_empty() {
        return vec![];
    }
    let alpha = 2.0 / (span as f64 + 1.0);
    let mut out = vec![0.0f64; prices.len()];
    out[0] = prices[0];
    for i in 1..prices.len() {
        out[i] = alpha * prices[i] + (1.0 - alpha) * out[i - 1];
    }
    out
}

/// MACD line: `EMA(fast) - EMA(slow)`.
///
/// # Panics
/// Panics if `fast >= slow` or either period < 1.
pub fn macd(prices: &[f64], fast: usize, slow: usize) -> Vec<f64> {
    assert!(fast < slow, "fast must be < slow");
    let ef = ema(prices, fast);
    let es = ema(prices, slow);
    ef.iter().zip(es.iter()).map(|(f, s)| f - s).collect()
}

/// MACD signal line: `EMA(signal)` of the MACD line.
pub fn macd_signal(prices: &[f64], fast: usize, slow: usize, signal: usize) -> Vec<f64> {
    let macd_line = macd(prices, fast, slow);
    ema(&macd_line, signal)
}

/// MACD histogram: MACD line − signal line.
pub fn macd_histogram(prices: &[f64], fast: usize, slow: usize, signal: usize) -> Vec<f64> {
    let macd_line = macd(prices, fast, slow);
    let sig = ema(&macd_line, signal);
    macd_line
        .iter()
        .zip(sig.iter())
        .map(|(m, s)| m - s)
        .collect()
}

// ─── RSI ─────────────────────────────────────────────────────────────────────

/// RSI with EWM smoothing matching `pandas.ewm(alpha=1/period, adjust=False)`.
///
/// Initialises the EWM from the first diff value (not a simple-average seed).
/// Output is NaN until `period` non-NaN differences have been accumulated,
/// matching pandas `min_periods=period` behaviour.
///
/// # Panics
/// Panics if `period < 2`.
pub fn rsi(prices: &[f64], period: usize) -> Vec<f64> {
    assert!(period >= 2, "period must be >= 2");
    let n = prices.len();
    let mut out = vec![f64::NAN; n];
    if n <= period {
        return out;
    }

    let alpha = 1.0 / period as f64;
    let mut avg_gain = 0.0f64;
    let mut avg_loss = 0.0f64;
    let mut count = 0usize;

    for i in 1..n {
        let d = prices[i] - prices[i - 1];
        let gain = if d > 0.0 { d } else { 0.0 };
        let loss = if d < 0.0 { -d } else { 0.0 };

        count += 1;
        if count == 1 {
            // Seed EWM with the first observation (matches pandas adjust=False)
            avg_gain = gain;
            avg_loss = loss;
        } else {
            avg_gain = alpha * gain + (1.0 - alpha) * avg_gain;
            avg_loss = alpha * loss + (1.0 - alpha) * avg_loss;
        }

        if count >= period {
            out[i] = rsi_from_avgs(avg_gain, avg_loss);
        }
    }
    out
}

#[inline]
fn rsi_from_avgs(avg_gain: f64, avg_loss: f64) -> f64 {
    if avg_gain == 0.0 && avg_loss == 0.0 {
        50.0
    } else if avg_loss == 0.0 {
        100.0
    } else if avg_gain == 0.0 {
        0.0
    } else {
        100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    }
}

// ─── Bollinger Bands ─────────────────────────────────────────────────────────

/// Bollinger mid band (SMA).
pub fn bb_mid(prices: &[f64], period: usize) -> Vec<f64> {
    rolling_mean(prices, period)
}

/// Bollinger upper band: `SMA + num_std * std`.
pub fn bb_upper(prices: &[f64], period: usize, num_std: f64) -> Vec<f64> {
    let mid = rolling_mean(prices, period);
    let std = rolling_std(prices, period);
    mid.iter()
        .zip(std.iter())
        .map(|(m, s)| m + num_std * s)
        .collect()
}

/// Bollinger lower band: `SMA - num_std * std`.
pub fn bb_lower(prices: &[f64], period: usize, num_std: f64) -> Vec<f64> {
    let mid = rolling_mean(prices, period);
    let std = rolling_std(prices, period);
    mid.iter()
        .zip(std.iter())
        .map(|(m, s)| m - num_std * s)
        .collect()
}

/// Bollinger bandwidth: `(upper - lower) / mid = 2 * num_std * std / mid`.
pub fn bb_bandwidth(prices: &[f64], period: usize, num_std: f64) -> Vec<f64> {
    let mid = rolling_mean(prices, period);
    let std = rolling_std(prices, period);
    mid.iter()
        .zip(std.iter())
        .map(|(m, s)| {
            if m.is_nan() || s.is_nan() || *m == 0.0 {
                f64::NAN
            } else {
                2.0 * num_std * s / m
            }
        })
        .collect()
}

// ─── Volume ───────────────────────────────────────────────────────────────────

/// Volume simple moving average.
pub fn volume_sma(volume: &[f64], period: usize) -> Vec<f64> {
    rolling_mean(volume, period)
}

/// Volume ratio: `volume / rolling_mean(volume, period)`. NaN during warm-up.
pub fn volume_ratio(volume: &[f64], period: usize) -> Vec<f64> {
    let sma = rolling_mean(volume, period);
    volume
        .iter()
        .zip(sma.iter())
        .map(|(v, s)| {
            if s.is_nan() || *s == 0.0 {
                f64::NAN
            } else {
                v / s
            }
        })
        .collect()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        if a.is_nan() && b.is_nan() {
            return true;
        }
        (a - b).abs() < 1e-9
    }

    #[test]
    fn test_returns_basic() {
        let p = vec![100.0, 110.0, 99.0];
        let r = returns(&p);
        assert!(r[0].is_nan());
        assert!(approx(r[1], 0.1));
        assert!(approx(r[2], (99.0 - 110.0) / 110.0));
    }

    #[test]
    fn test_log_returns() {
        let p = vec![100.0, 110.0];
        let r = log_returns(&p);
        assert!(r[0].is_nan());
        assert!(approx(r[1], (110.0_f64 / 100.0).ln()));
    }

    #[test]
    fn test_rolling_mean() {
        let p = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let r = rolling_mean(&p, 3);
        assert!(r[0].is_nan());
        assert!(r[1].is_nan());
        assert!(approx(r[2], 2.0));
        assert!(approx(r[3], 3.0));
        assert!(approx(r[4], 4.0));
    }

    #[test]
    fn test_rolling_mean_period_one() {
        let p = vec![3.0, 5.0, 7.0];
        let r = rolling_mean(&p, 1);
        assert!(approx(r[0], 3.0));
        assert!(approx(r[1], 5.0));
        assert!(approx(r[2], 7.0));
    }

    #[test]
    fn test_rolling_std() {
        let p = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let r = rolling_std(&p, 4);
        // First valid value at index 3
        assert!(r[0].is_nan());
        assert!(r[2].is_nan());
        assert!(!r[3].is_nan());
    }

    #[test]
    fn test_ema_span1_equals_input() {
        let p = vec![1.0, 2.0, 3.0, 4.0];
        let e = ema(&p, 1);
        for (a, b) in p.iter().zip(e.iter()) {
            assert!(approx(*a, *b));
        }
    }

    #[test]
    fn test_rsi_all_gains_is_100() {
        let prices: Vec<f64> = (1..=30).map(|i| i as f64).collect();
        let r = rsi(&prices, 14);
        // After warm-up, all gains → RSI = 100
        assert!(r[..14].iter().all(|x| x.is_nan()));
        assert!(approx(r[14], 100.0));
    }

    #[test]
    fn test_rsi_all_losses_is_0() {
        let prices: Vec<f64> = (1..=30).map(|i| (30 - i + 1) as f64).collect();
        let r = rsi(&prices, 14);
        assert!(approx(r[14], 0.0));
    }

    #[test]
    fn test_macd_output_length() {
        let p: Vec<f64> = (1..=50).map(|i| i as f64).collect();
        let m = macd(&p, 12, 26);
        assert_eq!(m.len(), p.len());
    }

    #[test]
    fn test_bb_mid_equals_rolling_mean() {
        let p: Vec<f64> = (1..=30).map(|i| i as f64).collect();
        let mid = bb_mid(&p, 20);
        let rm = rolling_mean(&p, 20);
        for (a, b) in mid.iter().zip(rm.iter()) {
            if a.is_nan() {
                assert!(b.is_nan());
            } else {
                assert!(approx(*a, *b));
            }
        }
    }

    #[test]
    fn test_volume_ratio_warm_up() {
        let v = vec![100.0, 200.0, 150.0, 100.0, 50.0];
        let r = volume_ratio(&v, 3);
        assert!(r[0].is_nan());
        assert!(r[1].is_nan());
        assert!(!r[2].is_nan());
    }

    #[test]
    fn test_output_lengths_match_input() {
        let p: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        assert_eq!(returns(&p).len(), p.len());
        assert_eq!(log_returns(&p).len(), p.len());
        assert_eq!(rolling_mean(&p, 20).len(), p.len());
        assert_eq!(rolling_std(&p, 20).len(), p.len());
        assert_eq!(ema(&p, 12).len(), p.len());
        assert_eq!(rsi(&p, 14).len(), p.len());
        assert_eq!(macd(&p, 12, 26).len(), p.len());
        assert_eq!(bb_upper(&p, 20, 2.0).len(), p.len());
        assert_eq!(bb_lower(&p, 20, 2.0).len(), p.len());
        assert_eq!(bb_bandwidth(&p, 20, 2.0).len(), p.len());
    }

    #[test]
    fn test_empty_input() {
        let p: Vec<f64> = vec![];
        assert!(returns(&p).is_empty());
        assert!(log_returns(&p).is_empty());
        assert!(ema(&p, 5).is_empty());
    }
}
