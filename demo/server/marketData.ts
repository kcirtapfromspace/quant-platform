export interface QuoteData {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  high: number;
  low: number;
  open: number;
  previousClose: number;
  volume: number;
  timestamp: number;
}

export interface OhlcvBar {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

const YAHOO_QUOTE_URL = 'https://query1.finance.yahoo.com/v8/finance/chart/';

export async function fetchQuote(symbol: string): Promise<QuoteData | null> {
  try {
    const url = `${YAHOO_QUOTE_URL}${encodeURIComponent(symbol)}?interval=1m&range=1d`;
    const res = await fetch(url, {
      headers: { 'User-Agent': 'Mozilla/5.0' },
    });
    if (!res.ok) return null;

    const json = await res.json() as any;
    const result = json.chart?.result?.[0];
    if (!result) return null;

    const meta = result.meta;
    const price = meta.regularMarketPrice ?? 0;
    const previousClose = meta.chartPreviousClose ?? meta.previousClose ?? price;
    const change = price - previousClose;
    const changePercent = previousClose !== 0 ? (change / previousClose) * 100 : 0;

    return {
      symbol: meta.symbol ?? symbol.toUpperCase(),
      price,
      change,
      changePercent,
      high: meta.regularMarketDayHigh ?? price,
      low: meta.regularMarketDayLow ?? price,
      open: meta.regularMarketOpen ?? price,
      previousClose,
      volume: meta.regularMarketVolume ?? 0,
      timestamp: Date.now(),
    };
  } catch {
    return null;
  }
}

export async function fetchHistory(symbol: string, range = '6mo', interval = '1d'): Promise<OhlcvBar[]> {
  try {
    const url = `${YAHOO_QUOTE_URL}${encodeURIComponent(symbol)}?interval=${interval}&range=${range}`;
    const res = await fetch(url, {
      headers: { 'User-Agent': 'Mozilla/5.0' },
    });
    if (!res.ok) return [];

    const json = await res.json() as any;
    const result = json.chart?.result?.[0];
    if (!result) return [];

    const timestamps = result.timestamp ?? [];
    const ohlcv = result.indicators?.quote?.[0];
    if (!ohlcv) return [];

    const bars: OhlcvBar[] = [];
    for (let i = 0; i < timestamps.length; i++) {
      if (ohlcv.open[i] == null) continue;
      bars.push({
        time: timestamps[i],
        open: ohlcv.open[i],
        high: ohlcv.high[i],
        low: ohlcv.low[i],
        close: ohlcv.close[i],
        volume: ohlcv.volume[i] ?? 0,
      });
    }
    return bars;
  } catch {
    return [];
  }
}

export async function fetchBatchQuotes(symbols: string[]): Promise<Map<string, QuoteData>> {
  const results = new Map<string, QuoteData>();
  const fetches = symbols.map(async (sym) => {
    const q = await fetchQuote(sym);
    if (q) results.set(sym.toUpperCase(), q);
  });
  await Promise.allSettled(fetches);
  return results;
}
