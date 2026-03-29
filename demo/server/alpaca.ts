/**
 * Alpaca paper trading REST client.
 *
 * Credentials are read from env:
 *   ALPACA_API_KEY     — Alpaca API key ID
 *   ALPACA_SECRET_KEY  — Alpaca API secret key
 *
 * All calls go to the paper-trading base URL.  If credentials are absent
 * (dev mode) the exported helpers return null / empty so callers can fall
 * back to the in-memory engine.
 */

const BASE_URL = 'https://paper-api.alpaca.markets/v2';

function alpacaHeaders(): Record<string, string> {
  return {
    'APCA-API-KEY-ID': process.env.ALPACA_API_KEY ?? '',
    'APCA-API-SECRET-KEY': process.env.ALPACA_SECRET_KEY ?? '',
    Accept: 'application/json',
  };
}

export function hasCredentials(): boolean {
  return !!(process.env.ALPACA_API_KEY && process.env.ALPACA_SECRET_KEY);
}

// ── Alpaca response shapes ─────────────────────────────────────────────────

export interface AlpacaAccount {
  /** Total portfolio value (cash + long positions). */
  portfolio_value: string;
  /** Cash balance. */
  cash: string;
  /** Total equity (same as portfolio_value for cash accounts). */
  equity: string;
  /** Previous trading day equity — used to compute daily P&L. */
  last_equity: string;
}

export interface AlpacaPosition {
  symbol: string;
  /** Number of shares (string in Alpaca API). */
  qty: string;
  avg_entry_price: string;
  current_price: string;
  market_value: string;
  unrealized_pl: string;
  /** Unrealized P&L as a fraction (0.10 = 10%). */
  unrealized_plpc: string;
}

// ── API calls ──────────────────────────────────────────────────────────────

export async function getAccount(): Promise<AlpacaAccount | null> {
  try {
    const res = await fetch(`${BASE_URL}/account`, { headers: alpacaHeaders() });
    if (!res.ok) {
      console.error(`[alpaca] getAccount failed: ${res.status} ${res.statusText}`);
      return null;
    }
    return (await res.json()) as AlpacaAccount;
  } catch (err) {
    console.error('[alpaca] getAccount error:', err);
    return null;
  }
}

export async function getPositions(): Promise<AlpacaPosition[]> {
  try {
    const res = await fetch(`${BASE_URL}/positions`, { headers: alpacaHeaders() });
    if (!res.ok) {
      console.error(`[alpaca] getPositions failed: ${res.status} ${res.statusText}`);
      return [];
    }
    return (await res.json()) as AlpacaPosition[];
  } catch (err) {
    console.error('[alpaca] getPositions error:', err);
    return [];
  }
}
