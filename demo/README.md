# QUA Trading Demo

Real-time paper trading dashboard backed by live market data.

## Features

- **Live Market Data** — Real-time quotes from Yahoo Finance for major US equities
- **Paper Trading** — Place market and limit orders with $1M virtual cash
- **Position Tracking** — Live P&L, market value, and cost basis per position
- **Portfolio Dashboard** — Total value, daily returns, cash and equity breakdown
- **Candlestick Charts** — 6-month historical charts powered by TradingView lightweight-charts
- **WebSocket Streaming** — Sub-second price updates pushed to the browser

## Quick Start

```bash
cd demo
npm install
npm run dev
```

Open http://localhost:5173

## Architecture

```
Browser (React + Vite)
  ├─ WebSocket ← real-time quotes + portfolio updates
  └─ REST API  → place orders, fetch history

Express Server (localhost:3001)
  ├─ Yahoo Finance proxy (quotes + OHLCV)
  ├─ Paper trading engine (OMS + positions)
  └─ WebSocket broadcaster (5s poll interval)
```

## Tech Stack

| Layer    | Technology                                    |
|----------|-----------------------------------------------|
| Frontend | React 19, Vite 6, TailwindCSS, lightweight-charts |
| Backend  | Express 5, WebSocket (ws), TypeScript         |
| Data     | Yahoo Finance public API (no key required)    |
| Runtime  | Node.js 25                                    |

## Default Watchlist

AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, JPM, V, SPY
