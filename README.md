# Quant Infrastructure

Quantitative trading platform — data ingestion, signal processing, risk engine, OMS, and backtesting.

## Architecture

```
quant/
├── config.py          # env-var config + universe loader
├── data/              # market data ingestion (yfinance → DuckDB / TimescaleDB)
├── features/          # feature engine with registry and Redis cache
├── signals/           # signal registry and strategy framework
├── risk/              # position sizing, exposure limits, circuit breaker
├── oms/               # order management system + broker adapters
├── backtest/          # vectorised backtesting engine (Rust core)
├── monitoring/        # Prometheus metrics instrumentation
└── db/                # SQLAlchemy models + session factory (PostgreSQL)

quant-rs/              # Rust compute kernels (required)
├── quant-features/    # Technical indicators: RSI, MACD, Bollinger Bands, etc.
├── quant-risk/        # Position sizing, exposure limits, circuit breaker
├── quant-signals/     # Momentum, mean-reversion, trend-following kernels
├── quant-backtest/    # Vectorised backtest loop + performance metrics
└── src/               # PyO3 extension module (quant_rs Python package)
```

Infrastructure:

- **TimescaleDB** — time-series OHLCV storage
- **Redis** — feature cache, signal pub/sub
- **MinIO** — parquet data lake, backtest results, model artifacts
- **DuckDB** — local OLAP for backtesting
- **Prometheus + Grafana** — observability stack (`monitoring/`)

---

## Prerequisites

| Tool | Minimum version | Install |
|------|----------------|---------|
| Python | 3.10 | [python.org](https://python.org) |
| Rust | 1.78 | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` |
| maturin | latest | `pip install maturin` |
| uv | latest | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Docker + Compose | Docker 24 | [docker.com](https://docker.com) |
| pre-commit | 3.7 | installed via `uv` below |

> **Rust is required.** All hot-path compute (feature indicators, risk kernels,
> signal scoring, backtest loop) runs in the `quant_rs` extension built from
> `quant-rs/` via PyO3 + maturin.

---

## Quick Start

### 1. Clone and set up environment

```bash
git clone <repo-url>
cd quant-infrastructure

# Copy env file and fill in any custom values
cp .env.example .env
```

### 2. Install Python dependencies and build the Rust extension

```bash
# With uv (recommended)
uv sync --extra dev

# Or with pip
pip install -e ".[dev]"
```

Build and install the `quant_rs` Rust extension (required):

```bash
cd quant-rs
maturin develop --release
cd ..
```

### 3. Start the local dev stack

```bash
docker compose up -d
```

Services started:

| Service | URL |
|---------|-----|
| TimescaleDB | `localhost:5432` |
| Redis | `localhost:6379` |
| MinIO API | `http://localhost:9000` |
| MinIO Console | `http://localhost:9001` (admin/minioadmin) |

Wait for services to be healthy:

```bash
docker compose ps
```

### 4. Run database migrations

```bash
uv run alembic upgrade head
```

### 5. Install pre-commit hooks

```bash
uv run pre-commit install
```

### 6. Run tests

```bash
uv run pytest
```

---

## Development Workflow

### Running linters manually

```bash
# Lint
uv run ruff check quant/

# Format
uv run ruff format quant/

# Type check
uv run mypy quant/
```

### Creating a new Alembic migration

```bash
# Auto-generate from model changes
uv run alembic revision --autogenerate -m "add my_table"

# Edit the generated file in alembic/versions/
# Then apply
uv run alembic upgrade head
```

### Rolling back a migration

```bash
uv run alembic downgrade -1
```

---

## Monitoring Stack

The Prometheus + Grafana observability stack lives in `monitoring/`.

```bash
cd monitoring
docker compose up -d
```

| Service | URL |
|---------|-----|
| Prometheus | `http://localhost:9090` |
| Grafana | `http://localhost:3000` (admin/changeme) |
| Alertmanager | `http://localhost:9093` |
| Pushgateway | `http://localhost:9091` |

---

## CI/CD

GitHub Actions runs on every push and PR to `main` / `develop`:

1. **Rust** — `cargo fmt --check` + `cargo clippy` + `cargo test` for all Rust crates
2. **Lint** — `ruff check` + `ruff format --check`
3. **Type Check** — `mypy`
4. **Tests** — `maturin develop --release` then `pytest` with TimescaleDB + Redis service containers
5. **Docker Build** — validates the image builds cleanly (depends on all prior jobs)

See `.github/workflows/ci.yml` for full configuration.

---

## Environment Variables

All variables are documented in `.env.example`. Key ones:

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL DSN for SQLAlchemy / Alembic |
| `REDIS_URL` | Redis connection URL |
| `S3_ENDPOINT_URL` | MinIO / S3 endpoint |
| `S3_ACCESS_KEY_ID` | MinIO access key |
| `S3_SECRET_ACCESS_KEY` | MinIO secret key |
| `QUANT_DB_PATH` | Local DuckDB file path |
| `QUANT_ENV` | `development` / `test` / `production` |
