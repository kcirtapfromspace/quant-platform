import express from 'express';
import cors from 'cors';
import { createServer } from 'http';
import { WebSocketServer, WebSocket } from 'ws';
import { fetchQuote, fetchHistory, fetchBatchQuotes, QuoteData, OhlcvBar } from './marketData.js';
import { PaperTradingEngine } from './paperTrading.js';

// ── Backtest engine (mirrors quant-backtest Rust logic) ───────────────────────

const TRADING_DAYS_PER_YEAR = 252;

interface BacktestResult {
  symbol: string;
  range: string;
  bars: number;
  equityCurve: Array<{ time: number; value: number }>;
  drawdownCurve: Array<{ time: number; value: number }>;
  sharpeRatio: number;
  maxDrawdown: number;
  cagr: number;
  winRate: number;
  profitFactor: number;
  totalReturn: number;
  tradeCount: number;
}

function computeBacktest(
  bars: OhlcvBar[],
  signal: number,
  commissionPct: number,
  initialCapital: number,
  range: string,
  symbol: string,
): BacktestResult {
  const n = bars.length;
  const closes = bars.map((b) => b.close);

  // Daily returns (pct_change; first bar = 0)
  const dailyReturns = closes.map((c, i) =>
    i === 0 ? 0 : (c - closes[i - 1]) / closes[i - 1],
  );

  // Positions = signal shifted 1 bar forward (no lookahead)
  const positions = closes.map((_, i) => (i === 0 ? 0 : signal));

  // Net returns = gross - commission on position change
  const netReturns: number[] = [];
  for (let i = 0; i < n; i++) {
    const gross = positions[i] * dailyReturns[i];
    const delta = i === 0 ? 0 : Math.abs(positions[i] - positions[i - 1]);
    netReturns.push(gross - delta * commissionPct);
  }

  // Equity curve
  let equity = initialCapital;
  const equityCurve = bars.map((b, i) => {
    equity *= 1 + netReturns[i];
    return { time: b.time, value: equity };
  });

  // Drawdown curve (non-positive fraction of running peak)
  let peak = initialCapital;
  const drawdownCurve = equityCurve.map((pt) => {
    if (pt.value > peak) peak = pt.value;
    const dd = peak > 0 ? (pt.value - peak) / peak : 0;
    return { time: pt.time, value: dd };
  });

  const maxDrawdown = Math.max(...drawdownCurve.map((d) => Math.abs(d.value)));
  const totalReturn = (equityCurve[n - 1].value - initialCapital) / initialCapital;
  const years = n / TRADING_DAYS_PER_YEAR;
  const cagr = years > 0 ? Math.pow(1 + totalReturn, 1 / years) - 1 : 0;

  // Sharpe (annualised, rf=0, ddof=1)
  const mean = netReturns.reduce((s, r) => s + r, 0) / n;
  const variance =
    netReturns.reduce((s, r) => s + (r - mean) ** 2, 0) / Math.max(n - 1, 1);
  const sharpeRatio =
    variance > 0 ? (mean / Math.sqrt(variance)) * Math.sqrt(TRADING_DAYS_PER_YEAR) : 0;

  // Round-trip trades: contiguous non-zero position blocks
  let inTrade = false;
  let tradeEntry = 0;
  const trades: Array<{ ret: number }> = [];
  for (let i = 0; i < n; i++) {
    const pos = positions[i];
    if (!inTrade && pos !== 0) {
      inTrade = true;
      tradeEntry = i;
    } else if (inTrade && pos === 0) {
      const tradeReturns = netReturns.slice(tradeEntry, i);
      const tradeRet = tradeReturns.reduce((acc, r) => acc * (1 + r), 1) - 1;
      trades.push({ ret: tradeRet });
      inTrade = false;
    }
  }
  if (inTrade) {
    const tradeReturns = netReturns.slice(tradeEntry);
    trades.push({ ret: tradeReturns.reduce((acc, r) => acc * (1 + r), 1) - 1 });
  }

  const wins = trades.filter((t) => t.ret > 0);
  const losses = trades.filter((t) => t.ret <= 0);
  const grossProfit = wins.reduce((s, t) => s + t.ret, 0);
  const grossLoss = Math.abs(losses.reduce((s, t) => s + t.ret, 0));
  const winRate = trades.length > 0 ? wins.length / trades.length : 0;
  const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : Infinity;

  return {
    symbol,
    range,
    bars: n,
    equityCurve,
    drawdownCurve,
    sharpeRatio,
    maxDrawdown,
    cagr,
    winRate,
    profitFactor: isFinite(profitFactor) ? profitFactor : 999,
    totalReturn,
    tradeCount: trades.length,
  };
}

const app = express();
const PORT = 3001;

app.use(cors());
app.use(express.json());

const WATCHLIST = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'V', 'SPY'];

const latestQuotes = new Map<string, QuoteData>();
const clients = new Set<WebSocket>();

const engine = new PaperTradingEngine({
  initialCash: 1_000_000,
  onFill: (order) => {
    broadcast({ type: 'fill', data: order });
    broadcastPortfolio();
  },
});

function broadcast(msg: object) {
  const payload = JSON.stringify(msg);
  for (const ws of clients) {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(payload);
    }
  }
}

function broadcastPortfolio() {
  const portfolio = engine.getPortfolio(latestQuotes);
  broadcast({ type: 'portfolio', data: portfolio });
}

async function pollQuotes() {
  const quotes = await fetchBatchQuotes(WATCHLIST);
  for (const [sym, q] of quotes) {
    latestQuotes.set(sym, q);
    broadcast({ type: 'quote', data: q });
  }
  engine.processQuotes(latestQuotes);
  broadcastPortfolio();
}

// --- REST API ---

app.get('/api/watchlist', (_req, res) => {
  res.json(WATCHLIST);
});

app.get('/api/quotes', (_req, res) => {
  res.json(Object.fromEntries(latestQuotes));
});

app.get('/api/quote/:symbol', async (req, res) => {
  const quote = latestQuotes.get(req.params.symbol.toUpperCase()) ?? (await fetchQuote(req.params.symbol));
  if (!quote) return res.status(404).json({ error: 'Symbol not found' });
  res.json(quote);
});

app.get('/api/history/:symbol', async (req, res) => {
  const range = (req.query.range as string) || '6mo';
  const interval = (req.query.interval as string) || '1d';
  const bars = await fetchHistory(req.params.symbol, range, interval);
  res.json(bars);
});

app.get('/api/portfolio', (_req, res) => {
  res.json(engine.getPortfolio(latestQuotes));
});

app.get('/api/orders', (_req, res) => {
  res.json(engine.getOrders());
});

app.post('/api/orders', (req, res) => {
  const { symbol, side, type, quantity, limitPrice } = req.body;
  if (!symbol || !side || !type || !quantity) {
    return res.status(400).json({ error: 'Missing required fields: symbol, side, type, quantity' });
  }
  const order = engine.placeOrder(symbol, side, type, quantity, limitPrice);

  // Immediately try to fill market orders
  if (type === 'market' && latestQuotes.size > 0) {
    engine.processQuotes(latestQuotes);
  }

  res.status(201).json(order);
});

app.delete('/api/orders/:id', (req, res) => {
  const ok = engine.cancelOrder(req.params.id);
  if (!ok) return res.status(404).json({ error: 'Order not found or not cancellable' });
  res.json({ cancelled: true });
});

app.get('/api/backtest/:symbol', async (req, res) => {
  const symbol = req.params.symbol.toUpperCase();
  const range = (req.query.range as string) || '5y';
  const signal = parseFloat((req.query.signal as string) || '1');
  const commission = parseFloat((req.query.commission as string) || '0.001');
  const initialCapital = parseFloat((req.query.initialCapital as string) || '1000000');

  const bars = await fetchHistory(symbol, range, '1d');
  if (bars.length < 2) {
    return res.status(422).json({ error: `Insufficient data for ${symbol} (${bars.length} bars)` });
  }

  const result = computeBacktest(bars, signal, commission, initialCapital, range, symbol);
  res.json(result);
});

// ── Analytics endpoints ────────────────────────────────────────────────────────

app.get('/api/analytics/performance', async (req, res) => {
  const symbol = ((req.query.symbol as string) || 'AAPL').toUpperCase();
  const range = (req.query.range as string) || '1y';
  const benchmarkParam = ((req.query.benchmark as string) || 'SPY').toUpperCase();
  const benchmarkSymbol = benchmarkParam === 'BTC' ? 'BTC-USD' : benchmarkParam;

  const [stratBars, benchBars] = await Promise.all([
    fetchHistory(symbol, range, '1d'),
    fetchHistory(benchmarkSymbol, range, '1d'),
  ]);

  if (stratBars.length < 30) {
    return res.status(422).json({ error: `Insufficient data for ${symbol}` });
  }

  // Align by time — use intersection
  const benchByTime = new Map(benchBars.map((b) => [b.time, b.close]));
  const alignedTimes = stratBars.map((b) => b.time).filter((t) => benchByTime.has(t));
  const stratByTime = new Map(stratBars.map((b) => [b.time, b.close]));

  const stratCloses = alignedTimes.map((t) => stratByTime.get(t)!);
  const benchCloses = alignedTimes.map((t) => benchByTime.get(t)!);
  const n = alignedTimes.length;

  const stratReturns = stratCloses.map((c, i) =>
    i === 0 ? 0 : (c - stratCloses[i - 1]) / stratCloses[i - 1],
  );
  const benchReturns = benchCloses.map((c, i) =>
    i === 0 ? 0 : (c - benchCloses[i - 1]) / benchCloses[i - 1],
  );

  // Cumulative returns (percentage from start)
  let sc = 1;
  let bc = 1;
  const cumulativeReturns: Array<{ time: number; value: number }> = [];
  const benchmarkReturns: Array<{ time: number; value: number }> = [];
  for (let i = 0; i < n; i++) {
    sc *= 1 + stratReturns[i];
    bc *= 1 + benchReturns[i];
    cumulativeReturns.push({ time: alignedTimes[i], value: (sc - 1) * 100 });
    benchmarkReturns.push({ time: alignedTimes[i], value: (bc - 1) * 100 });
  }

  // Relative performance (strategy − benchmark, cumulative)
  const relativePerf = alignedTimes.map((t, i) => ({
    time: t,
    value: cumulativeReturns[i].value - benchmarkReturns[i].value,
  }));

  // Rolling 30-day metrics (window = 30 bars)
  const WINDOW = 30;
  const rolling30dReturns: Array<{ time: number; value: number }> = [];
  const rolling30dSharpe: Array<{ time: number; value: number }> = [];

  for (let i = WINDOW - 1; i < n; i++) {
    const window = stratReturns.slice(i - WINDOW + 1, i + 1);
    const compound = window.reduce((acc, r) => acc * (1 + r), 1) - 1;
    rolling30dReturns.push({ time: alignedTimes[i], value: compound * 100 });

    const wMean = window.reduce((a, b) => a + b, 0) / WINDOW;
    const wVar = window.reduce((a, r) => a + (r - wMean) ** 2, 0) / Math.max(WINDOW - 1, 1);
    const sharpe = wVar > 0 ? (wMean / Math.sqrt(wVar)) * Math.sqrt(252) : 0;
    rolling30dSharpe.push({ time: alignedTimes[i], value: sharpe });
  }

  // Distribution stats (skip first return = 0)
  const returns = stratReturns.slice(1);
  const rn = returns.length;
  const mean = returns.reduce((a, b) => a + b, 0) / rn;
  const variance = returns.reduce((a, r) => a + (r - mean) ** 2, 0) / Math.max(rn - 1, 1);
  const stddev = Math.sqrt(variance);
  const skewness =
    stddev > 0 ? returns.reduce((a, r) => a + ((r - mean) / stddev) ** 3, 0) / rn : 0;
  const kurtosis =
    stddev > 0 ? returns.reduce((a, r) => a + ((r - mean) / stddev) ** 4, 0) / rn - 3 : 0;

  res.json({
    cumulativeReturns,
    benchmarkReturns,
    relativePerf,
    rolling30dReturns,
    rolling30dSharpe,
    dailyReturns: returns,
    stats: { mean, stddev, skewness, kurtosis },
  });
});

app.get('/api/analytics/attribution', (_req, res) => {
  const rows = [
    { strategy: 'Momentum — AAPL/NVDA', return: 0.142, weight: 0.30 },
    { strategy: 'Mean Reversion — SPY', return: 0.067, weight: 0.25 },
    { strategy: 'Pairs — MSFT/GOOGL', return: 0.089, weight: 0.20 },
    { strategy: 'Trend — NVDA/TSLA', return: 0.231, weight: 0.15 },
    { strategy: 'Stat Arb — JPM/V', return: 0.034, weight: 0.10 },
  ].map((r) => ({ ...r, contribution: r.return * r.weight }));
  res.json(rows);
});

// --- Server + WebSocket ---

const server = createServer(app);
const wss = new WebSocketServer({ server, path: '/ws' });

wss.on('connection', (ws) => {
  clients.add(ws);

  // Send current state on connect
  for (const q of latestQuotes.values()) {
    ws.send(JSON.stringify({ type: 'quote', data: q }));
  }
  ws.send(JSON.stringify({ type: 'portfolio', data: engine.getPortfolio(latestQuotes) }));

  ws.on('close', () => clients.delete(ws));
});

server.listen(PORT, () => {
  console.log(`[QUA Demo] API server running on http://localhost:${PORT}`);
  console.log(`[QUA Demo] WebSocket on ws://localhost:${PORT}/ws`);
  console.log(`[QUA Demo] Watchlist: ${WATCHLIST.join(', ')}`);

  // Initial fetch, then poll every 5 seconds
  pollQuotes();
  setInterval(pollQuotes, 5000);
});
