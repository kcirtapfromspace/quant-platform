import type { Order, OhlcvBar } from '../types';

const BASE = '/api';

export async function placeOrder(params: {
  symbol: string;
  side: 'buy' | 'sell';
  type: 'market' | 'limit';
  quantity: number;
  limitPrice?: number;
}): Promise<Order> {
  const res = await fetch(`${BASE}/orders`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function cancelOrder(id: string): Promise<void> {
  const res = await fetch(`${BASE}/orders/${id}`, { method: 'DELETE' });
  if (!res.ok) throw new Error(await res.text());
}

export async function fetchOrders(): Promise<Order[]> {
  const res = await fetch(`${BASE}/orders`);
  return res.json();
}

export async function fetchHistory(symbol: string, range = '6mo'): Promise<OhlcvBar[]> {
  const res = await fetch(`${BASE}/history/${symbol}?range=${range}`);
  return res.json();
}

export async function fetchWatchlist(): Promise<string[]> {
  const res = await fetch(`${BASE}/watchlist`);
  return res.json();
}

export async function fetchOhlcv(symbol: string, interval = '5m'): Promise<OhlcvBar[]> {
  const res = await fetch(`${BASE}/market/ohlcv?symbol=${encodeURIComponent(symbol)}&interval=${interval}`);
  if (!res.ok) return [];
  return res.json();
}
