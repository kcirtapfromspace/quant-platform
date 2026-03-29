import type { Order, OhlcvBar } from '../types';

const BASE = import.meta.env.VITE_API_URL ?? '';
const API_KEY = import.meta.env.VITE_API_KEY ?? '';

function apiHeaders(extra?: Record<string, string>): Record<string, string> {
  return { 'X-API-Key': API_KEY, ...extra };
}

export async function placeOrder(params: {
  symbol: string;
  side: 'buy' | 'sell';
  type: 'market' | 'limit';
  quantity: number;
  limitPrice?: number;
}): Promise<Order> {
  const res = await fetch(`${BASE}/api/v1/orders`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...apiHeaders() },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function cancelOrder(id: string): Promise<void> {
  const res = await fetch(`${BASE}/api/v1/orders/${id}`, {
    method: 'DELETE',
    headers: apiHeaders(),
  });
  if (!res.ok) throw new Error(await res.text());
}

export async function fetchOrders(): Promise<Order[]> {
  const res = await fetch(`${BASE}/api/v1/orders`, { headers: apiHeaders() });
  return res.json();
}

export async function fetchHistory(symbol: string, range = '6mo'): Promise<OhlcvBar[]> {
  const res = await fetch(
    `${BASE}/api/v1/market/history/${symbol}?range=${range}`,
    { headers: apiHeaders() },
  );
  return res.json();
}

export async function fetchWatchlist(): Promise<string[]> {
  const res = await fetch(`${BASE}/api/v1/market/quotes`, { headers: apiHeaders() });
  if (!res.ok) return [];
  const data: Array<{ symbol: string }> = await res.json();
  return data.map((q) => q.symbol);
}

export async function fetchOhlcv(symbol: string, interval = '5m'): Promise<OhlcvBar[]> {
  const res = await fetch(
    `${BASE}/api/v1/market/ohlcv?symbol=${encodeURIComponent(symbol)}&interval=${interval}`,
    { headers: apiHeaders() },
  );
  if (!res.ok) return [];
  return res.json();
}
