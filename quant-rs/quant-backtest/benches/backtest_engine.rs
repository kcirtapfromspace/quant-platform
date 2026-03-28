//! Criterion benchmarks for the `quant-backtest` Phase-5 engine.
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use quant_backtest::run_backtest;

// ── Synthetic data helpers ────────────────────────────────────────────────────

fn synthetic_prices(n: usize) -> Vec<f64> {
    let mut prices = vec![150.0_f64];
    // Deterministic LCG for reproducible benchmarks without extra deps.
    let mut state: u64 = 42;
    for _ in 1..n {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let norm = ((state >> 33) as f64) / (u32::MAX as f64) - 0.5; // ~[-0.5, 0.5]
        prices.push((prices.last().unwrap() * (1.0 + norm * 0.015)).max(1.0));
    }
    prices
}

/// Toy momentum signal: +1 if last price > rolling-20 mean, else -1.
fn momentum_signals(prices: &[f64]) -> Vec<f64> {
    let n = prices.len();
    let period = 20;
    let mut out = vec![0.0_f64; n];
    for i in period..n {
        let mean: f64 = prices[i - period..i].iter().sum::<f64>() / period as f64;
        out[i] = if prices[i] > mean { 1.0 } else { -1.0 };
    }
    out
}

// ── Benchmarks ────────────────────────────────────────────────────────────────

fn bench_backtest_252(c: &mut Criterion) {
    let prices = synthetic_prices(252);
    let signals = momentum_signals(&prices);
    c.bench_function("run_backtest_252bars", |b| {
        b.iter(|| run_backtest(black_box(&prices), black_box(&signals), 0.001, 1_000_000.0))
    });
}

fn bench_backtest_2520(c: &mut Criterion) {
    let prices = synthetic_prices(2520);
    let signals = momentum_signals(&prices);
    c.bench_function("run_backtest_2520bars", |b| {
        b.iter(|| run_backtest(black_box(&prices), black_box(&signals), 0.001, 1_000_000.0))
    });
}

fn bench_backtest_10000(c: &mut Criterion) {
    let prices = synthetic_prices(10_000);
    let signals = momentum_signals(&prices);
    c.bench_function("run_backtest_10000bars", |b| {
        b.iter(|| run_backtest(black_box(&prices), black_box(&signals), 0.001, 1_000_000.0))
    });
}

criterion_group!(
    benches,
    bench_backtest_252,
    bench_backtest_2520,
    bench_backtest_10000,
);
criterion_main!(benches);
