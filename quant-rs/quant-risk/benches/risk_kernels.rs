//! Micro-benchmarks for `quant-risk` kernels.
//!
//! Run with: `cargo bench -p quant-risk`
//!
//! Results are written to `target/criterion/` as HTML reports.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use quant_risk::{kelly_fraction, position_size_fixed_fraction, position_size_vol_target};
use quant_risk::{DrawdownCircuitBreaker, ExposureLimits};

const CAPITAL: f64 = 1_000_000.0;
const PRICE: f64 = 150.0;

fn bench_fixed_fraction(c: &mut Criterion) {
    c.bench_function("position_size_fixed_fraction", |b| {
        b.iter(|| {
            position_size_fixed_fraction(black_box(CAPITAL), black_box(PRICE), black_box(0.02))
        })
    });
}

fn bench_kelly(c: &mut Criterion) {
    c.bench_function("kelly_fraction", |b| {
        b.iter(|| kelly_fraction(black_box(0.6), black_box(2.0)))
    });
}

fn bench_vol_target(c: &mut Criterion) {
    c.bench_function("position_size_vol_target", |b| {
        b.iter(|| {
            position_size_vol_target(
                black_box(CAPITAL),
                black_box(PRICE),
                black_box(0.20),
                black_box(0.10),
            )
        })
    });
}

fn bench_exposure_check(c: &mut Criterion) {
    let limits = ExposureLimits::default();
    c.bench_function("exposure_check_approved", |b| {
        b.iter(|| {
            limits.check(
                black_box(CAPITAL),
                black_box(500_000.0),
                black_box(100_000.0),
                black_box(50_000.0),
            )
        })
    });
}

fn bench_circuit_breaker(c: &mut Criterion) {
    let cb = DrawdownCircuitBreaker::new(0.10);
    c.bench_function("circuit_breaker_is_tripped", |b| {
        b.iter(|| cb.is_tripped(black_box(CAPITAL), black_box(950_000.0)))
    });
}

fn bench_drawdown(c: &mut Criterion) {
    let cb = DrawdownCircuitBreaker::new(0.20);
    c.bench_function("drawdown_calculation", |b| {
        b.iter(|| cb.drawdown(black_box(CAPITAL), black_box(900_000.0)))
    });
}

criterion_group!(
    benches,
    bench_fixed_fraction,
    bench_kelly,
    bench_vol_target,
    bench_exposure_check,
    bench_circuit_breaker,
    bench_drawdown,
);
criterion_main!(benches);
