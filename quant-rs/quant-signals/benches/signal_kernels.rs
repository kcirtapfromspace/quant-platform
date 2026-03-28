/// Criterion benchmarks for `quant-signals` Phase-4 kernels.
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use quant_signals::{mean_reversion_signal, momentum_signal, trend_following_signal};

// ── Synthetic data helpers ────────────────────────────────────────────────

fn synthetic_prices(n: usize) -> Vec<f64> {
    let mut prices = vec![150.0_f64];
    // Deterministic LCG so benchmarks are reproducible without a dep.
    let mut state: u64 = 42;
    for _ in 1..n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let norm = ((state >> 33) as f64) / (u32::MAX as f64) - 0.5; // ~[-0.5, 0.5]
        prices.push((prices.last().unwrap() * (1.0 + norm * 0.02)).max(1.0));
    }
    prices
}

fn make_returns(prices: &[f64]) -> Vec<f64> {
    let mut out = vec![f64::NAN];
    for w in prices.windows(2) {
        out.push((w[1] - w[0]) / w[0]);
    }
    out
}

fn make_rsi(prices: &[f64], period: usize) -> Vec<f64> {
    let n = prices.len();
    let mut out = vec![f64::NAN; n];
    let alpha = 1.0 / period as f64;
    let mut avg_gain = 0.0_f64;
    let mut avg_loss = 0.0_f64;
    for i in 1..n {
        let d = prices[i] - prices[i - 1];
        let gain = if d > 0.0 { d } else { 0.0 };
        let loss = if d < 0.0 { -d } else { 0.0 };
        if i == 1 {
            avg_gain = gain;
            avg_loss = loss;
        } else {
            avg_gain = alpha * gain + (1.0 - alpha) * avg_gain;
            avg_loss = alpha * loss + (1.0 - alpha) * avg_loss;
        }
        if i >= period {
            out[i] = if avg_gain == 0.0 && avg_loss == 0.0 {
                50.0
            } else if avg_loss == 0.0 {
                100.0
            } else if avg_gain == 0.0 {
                0.0
            } else {
                100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
            };
        }
    }
    out
}

fn make_rolling_mean(prices: &[f64], period: usize) -> Vec<f64> {
    let n = prices.len();
    let mut out = vec![f64::NAN; n];
    for i in (period - 1)..n {
        out[i] = prices[i - period + 1..=i].iter().sum::<f64>() / period as f64;
    }
    out
}

fn make_macd_histogram(prices: &[f64], fast: usize, slow: usize, signal: usize) -> Vec<f64> {
    let alpha_fast = 2.0 / (fast + 1) as f64;
    let alpha_slow = 2.0 / (slow + 1) as f64;
    let alpha_sig = 2.0 / (signal + 1) as f64;

    let mut ema_f = prices[0];
    let mut ema_s = prices[0];
    let mut macd_line: Vec<f64> = Vec::with_capacity(prices.len());
    for &p in prices {
        ema_f = alpha_fast * p + (1.0 - alpha_fast) * ema_f;
        ema_s = alpha_slow * p + (1.0 - alpha_slow) * ema_s;
        macd_line.push(ema_f - ema_s);
    }
    let mut sig_line = macd_line[0];
    let mut hist = Vec::with_capacity(prices.len());
    for &m in &macd_line {
        sig_line = alpha_sig * m + (1.0 - alpha_sig) * sig_line;
        hist.push(m - sig_line);
    }
    hist
}

fn make_bb(prices: &[f64], period: usize, num_std: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = prices.len();
    let mut mid = vec![f64::NAN; n];
    let mut upper = vec![f64::NAN; n];
    let mut lower = vec![f64::NAN; n];
    for i in (period - 1)..n {
        let w = &prices[i - period + 1..=i];
        let m = w.iter().sum::<f64>() / period as f64;
        let var = w.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (period - 1) as f64;
        let s = var.sqrt();
        mid[i] = m;
        upper[i] = m + num_std * s;
        lower[i] = m - num_std * s;
    }
    (mid, upper, lower)
}

// ── Benchmarks ────────────────────────────────────────────────────────────

fn bench_momentum_signal(c: &mut Criterion) {
    let prices = synthetic_prices(500);
    let rsi = make_rsi(&prices, 14);
    let rets = make_returns(&prices);
    c.bench_function("momentum_signal", |b| {
        b.iter(|| momentum_signal(black_box(&rsi), black_box(&rets), 5, 0.05))
    });
}

fn bench_mean_reversion_signal(c: &mut Criterion) {
    let prices = synthetic_prices(500);
    let (bb_mid, bb_upper, bb_lower) = make_bb(&prices, 20, 2.0);
    let rets = make_returns(&prices);
    c.bench_function("mean_reversion_signal", |b| {
        b.iter(|| {
            mean_reversion_signal(
                black_box(&bb_mid),
                black_box(&bb_upper),
                black_box(&bb_lower),
                black_box(&rets),
                2.0,
            )
        })
    });
}

fn bench_trend_following_signal(c: &mut Criterion) {
    let prices = synthetic_prices(500);
    let hist = make_macd_histogram(&prices, 12, 26, 9);
    let fast_ma = make_rolling_mean(&prices, 20);
    let slow_ma = make_rolling_mean(&prices, 50);
    c.bench_function("trend_following_signal", |b| {
        b.iter(|| {
            trend_following_signal(black_box(&hist), black_box(&fast_ma), black_box(&slow_ma))
        })
    });
}

criterion_group!(
    benches,
    bench_momentum_signal,
    bench_mean_reversion_signal,
    bench_trend_following_signal,
);
criterion_main!(benches);
