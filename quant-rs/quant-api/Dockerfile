# syntax=docker/dockerfile:1
# Multi-stage ARM64 build. Build context must be the quant-rs/ workspace root.

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM rust:1.82-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    g++ \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY . .

RUN cargo build --release --bin quant-api

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM debian:bookworm-slim

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
