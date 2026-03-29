# syntax=docker/dockerfile:1
# Multi-stage ARM64 build using cargo-chef for dependency caching.
# Build context must be the quant-rs/ workspace root.
#
# Cold first build: ~5-6h (DuckDB C++ under QEMU + cargo-chef cook).
# Warm subsequent builds: ~10-15m (cook layer cached in registry; only Rust source recompiled).

# ── Stage 0: cargo-chef base ──────────────────────────────────────────────────
FROM --platform=linux/arm64 rust:1.93.1-slim-bookworm AS chef

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    g++ \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN cargo install cargo-chef --locked

WORKDIR /build

# ── Stage 1: Dependency recipe planner ───────────────────────────────────────
FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

# ── Stage 2: Dependency cook (cached layer) ───────────────────────────────────
# Invalidated only when Cargo.lock changes.
# Compiles all workspace deps including bundled DuckDB C++ source (~5-6h cold under QEMU,
# cached on subsequent builds via type=registry cache in CI).
FROM chef AS builder
COPY --from=planner /build/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json

# Copy real workspace source and build quant-api binary.
COPY . .
RUN cargo build --release --bin quant-api

# ── Stage 3: Minimal runtime image ───────────────────────────────────────────
FROM --platform=linux/arm64 debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/quant-api /usr/local/bin/quant-api

# Environment defaults — override at runtime via k8s env / docker -e.
ENV QUANT_API_KEY=changeme \
    QUANT_DB_PATH=/data/quant.duckdb \
    OMS_DB_PATH=/data/oms.db \
    METRICS_PATH=/data/metrics.prom \
    BACKTEST_RESULTS_DIR=/data/backtest-results \
    PORT=8080

EXPOSE 8080

CMD ["/usr/local/bin/quant-api"]
