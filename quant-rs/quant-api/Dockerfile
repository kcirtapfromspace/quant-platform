# syntax=docker/dockerfile:1
# Multi-stage ARM64 build for quant-api
# Context: quant-rs/ workspace root (set in CI: context: ./quant-rs)
# Stage 1: Build the Rust binary
FROM rust:1.82-slim AS builder
WORKDIR /build

RUN apt-get update && apt-get install -y pkg-config libssl-dev && apt-get clean

# Copy workspace manifests first for dependency caching
COPY Cargo.toml Cargo.lock ./
COPY quant-features/Cargo.toml quant-features/
COPY quant-risk/Cargo.toml quant-risk/
COPY quant-signals/Cargo.toml quant-signals/
COPY quant-backtest/Cargo.toml quant-backtest/
COPY quant-oms/Cargo.toml quant-oms/
COPY quant-portfolio/Cargo.toml quant-portfolio/
COPY quant-data/Cargo.toml quant-data/
COPY quant-cli/Cargo.toml quant-cli/
COPY quant-bayes/Cargo.toml quant-bayes/
COPY quant-api/Cargo.toml quant-api/

# Fake lib/main stubs for all crates so cargo can resolve deps
RUN for d in quant-features quant-risk quant-signals quant-backtest quant-oms quant-portfolio quant-data quant-bayes; do \
      mkdir -p $d/src && echo "pub fn stub(){}" > $d/src/lib.rs; \
    done && \
    mkdir -p quant-cli/src && echo "fn main(){}" > quant-cli/src/main.rs && \
    mkdir -p quant-api/src && echo "fn main(){}" > quant-api/src/main.rs

# Pre-build deps only (layer cache)
RUN cargo build --release --bin quant-api 2>/dev/null || true

# Copy real sources and build
COPY . .
RUN cargo build --release --bin quant-api

# Stage 2: Minimal runtime image
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && apt-get clean

COPY --from=builder /build/target/release/quant-api /usr/local/bin/quant-api

ENV API_KEY=changeme
ENV OMS_DB_PATH=/data/oms.db
ENV QUANT_DB_PATH=/data/quant.duckdb
ENV METRICS_PATH=/data/metrics.prom
ENV BACKTEST_RESULTS_DIR=/data/backtest-results
ENV PORT=8080
ENV RUST_LOG=quant_api=info

EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/quant-api"]
