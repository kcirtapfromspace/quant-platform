import express from 'express';
import cors from 'cors';
import { createServer } from 'http';
import { WebSocketServer, WebSocket } from 'ws';
import { fetchQuote, fetchHistory, fetchBatchQuotes, QuoteData } from './marketData.js';
import { PaperTradingEngine } from './paperTrading.js';

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
