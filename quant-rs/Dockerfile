# syntax=docker/dockerfile:1
#
# Multi-stage ARM64 Dockerfile for quant-rs
# Target: aarch64-unknown-linux-gnu (Turing Pi cluster, Talos Linux)
#
# Build:
#   docker buildx build --platform linux/arm64 -t quant-rs:latest .
#
# Push to local registry:
#   docker buildx build --platform linux/arm64 \
#     -t mirror.registry:5000/quant-rs:latest --push .

# ── Stage 0: cargo-chef for dependency caching ────────────────────────────────
# cargo-chef pre-compiles all dependencies (including bundled DuckDB C++ source)
# so source-only changes don't recompile the heavy deps layer.
FROM --platform=linux/arm64 rust:1.93.1-slim-bookworm AS chef

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    g++ \
    pkg-config \
    libssl-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN cargo install cargo-chef --locked

WORKDIR /build

# ── Stage 1: Dependency recipe planner ───────────────────────────────────────
FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

# ── Stage 2: Dependency cook (cached layer) ───────────────────────────────────
# This layer is invalidated only when Cargo.lock changes.
# Compiles all deps including bundled DuckDB (~10-20 min first run, cached after).
FROM chef AS builder

COPY --from=planner /build/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json

# Copy real workspace source and build the quant binary
COPY . .
RUN cargo build --release -p quant-cli

# ── Stage 3: Minimal runtime image ───────────────────────────────────────────
FROM --platform=linux/arm64 debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for least-privilege operation
RUN groupadd -r quant && useradd -r -g quant quant

COPY --from=builder /build/target/release/quant /usr/local/bin/quant

USER quant

ENTRYPOINT ["/usr/local/bin/quant"]
CMD ["--help"]
