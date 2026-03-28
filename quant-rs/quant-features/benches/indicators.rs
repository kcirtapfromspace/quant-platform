//! Micro-benchmarks for `quant-features` indicator kernels.
//!
//! Run with: `cargo bench -p quant-features`
//!
//! Results are written to `target/criterion/` as HTML reports.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use quant_features as qf;

fn make_prices(n: usize) -> Vec<f64> {
    // Simple deterministic random-walk for reproducible benchmarks.
    let mut prices = Vec::with_capacity(n);
    prices.push(100.0f64);
    for i in 1..n {
        // |sin| keeps prices bounded and positive
        let r = (i as f64 * 0.1).sin() * 0.005;
        prices.push(prices[i - 1] * (1.0 + r));
    }
    prices
}

fn make_volumes(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 1_000_000.0 + (i as f64 * 7.0).sin() * 200_000.0)
        .collect()
}

const N: usize = 10_000;

fn bench_returns(c: &mut Criterion) {
    let p = make_prices(N);
    c.bench_function("returns_10k", |b| b.iter(|| qf::returns(black_box(&p))));
}

fn bench_log_returns(c: &mut Criterion) {
    let p = make_prices(N);
    c.bench_function("log_returns_10k", |b| {
        b.iter(|| qf::log_returns(black_box(&p)))
    });
}

fn bench_rolling_mean(c: &mut Criterion) {
    let p = make_prices(N);
    c.bench_function("rolling_mean_20_10k", |b| {
        b.iter(|| qf::rolling_mean(black_box(&p), 20))
    });
}

fn bench_rolling_std(c: &mut Criterion) {
    let p = make_prices(N);
    c.bench_function("rolling_std_20_10k", |b| {
        b.iter(|| qf::rolling_std(black_box(&p), 20))
    });
}

fn bench_ema(c: &mut Criterion) {
    let p = make_prices(N);
    c.bench_function("ema_12_10k", |b| b.iter(|| qf::ema(black_box(&p), 12)));
}

fn bench_rsi(c: &mut Criterion) {
    let p = make_prices(N);
    c.bench_function("rsi_14_10k", |b| b.iter(|| qf::rsi(black_box(&p), 14)));
}

fn bench_macd(c: &mut Criterion) {
    let p = make_prices(N);
    c.bench_function("macd_12_26_10k", |b| {
        b.iter(|| qf::macd(black_box(&p), 12, 26))
    });
}

fn bench_macd_signal(c: &mut Criterion) {
    let p = make_prices(N);
    c.bench_function("macd_signal_12_26_9_10k", |b| {
        b.iter(|| qf::macd_signal(black_box(&p), 12, 26, 9))
    });
}

fn bench_macd_histogram(c: &mut Criterion) {
    let p = make_prices(N);
    c.bench_function("macd_hist_12_26_9_10k", |b| {
        b.iter(|| qf::macd_histogram(black_box(&p), 12, 26, 9))
    });
}

fn bench_bollinger(c: &mut Criterion) {
    let p = make_prices(N);
    let mut group = c.benchmark_group("bollinger_20_10k");
    group.bench_function("bb_mid", |b| b.iter(|| qf::bb_mid(black_box(&p), 20)));
    group.bench_function("bb_upper", |b| {
        b.iter(|| qf::bb_upper(black_box(&p), 20, 2.0))
    });
    group.bench_function("bb_lower", |b| {
        b.iter(|| qf::bb_lower(black_box(&p), 20, 2.0))
    });
    group.bench_function("bb_bandwidth", |b| {
        b.iter(|| qf::bb_bandwidth(black_box(&p), 20, 2.0))
    });
    group.finish();
}

fn bench_volume(c: &mut Criterion) {
    let v = make_volumes(N);
    let mut group = c.benchmark_group("volume_20_10k");
    group.bench_function("volume_sma", |b| {
        b.iter(|| qf::volume_sma(black_box(&v), 20))
    });
    group.bench_function("volume_ratio", |b| {
        b.iter(|| qf::volume_ratio(black_box(&v), 20))
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_returns,
    bench_log_returns,
    bench_rolling_mean,
    bench_rolling_std,
    bench_ema,
    bench_rsi,
    bench_macd,
    bench_macd_signal,
    bench_macd_histogram,
    bench_bollinger,
    bench_volume,
);
criterion_main!(benches);
