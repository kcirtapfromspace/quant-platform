import { useEffect, useRef, useState, useCallback } from 'react';
import {
  createChart,
  IChartApi,
  ISeriesApi,
  CandlestickData,
  HistogramData,
  Time,
} from 'lightweight-charts';
import { fetchOhlcv, fetchWatchlist } from '../hooks/useApi';
import type { OhlcvBar, OrderBook, Quote } from '../types';

// ---------------------------------------------------------------------------
// Mock helpers (used when backend endpoints aren't available yet)
// ---------------------------------------------------------------------------

function generateMockBars(count = 200): OhlcvBar[] {
  const bars: OhlcvBar[] = [];
  let close = 150 + Math.random() * 50;
  const now = Math.floor(Date.now() / 1000);
  const step = 300; // 5-minute bars
  for (let i = count; i >= 0; i--) {
    const open = close;
    const change = (Math.random() - 0.49) * 3;
    close = Math.max(1, open + change);
    const high = Math.max(open, close) + Math.random() * 1.5;
    const low = Math.min(open, close) - Math.random() * 1.5;
    const volume = Math.floor(50_000 + Math.random() * 200_000);
    bars.push({ time: now - i * step, open, high, low, close, volume });
  }
  return bars;
}

function generateMockOrderBook(symbol: string, midPrice: number): OrderBook {
  const bids = Array.from({ length: 10 }, (_, i) => ({
    price: +(midPrice - 0.01 * (i + 1)).toFixed(2),
    size: Math.floor(100 + Math.random() * 2000),
  }));
  const asks = Array.from({ length: 10 }, (_, i) => ({
    price: +(midPrice + 0.01 * (i + 1)).toFixed(2),
    size: Math.floor(100 + Math.random() * 2000),
  }));
  return { symbol, bids, asks };
}

// ---------------------------------------------------------------------------
// Timeframe selector
// ---------------------------------------------------------------------------

type Interval = '1m' | '5m' | '15m' | '1h' | '1d';

const INTERVALS: { label: string; value: Interval }[] = [
  { label: '1m', value: '1m' },
  { label: '5m', value: '5m' },
  { label: '15m', value: '15m' },
  { label: '1h', value: '1h' },
  { label: '1D', value: '1d' },
];

// ---------------------------------------------------------------------------
// CandlestickChart
// ---------------------------------------------------------------------------

interface CandlestickChartProps {
  symbol: string;
  interval: Interval;
  liveBar: OhlcvBar | null;
  quote?: Quote;
}

function CandlestickChart({ symbol, interval, liveBar, quote }: CandlestickChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const volumeContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const volumeChartRef = useRef<IChartApi | null>(null);
  const candleRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const volumeRef = useRef<ISeriesApi<'Histogram'> | null>(null);
  const [loading, setLoading] = useState(true);

  // Create charts once
  useEffect(() => {
    if (!containerRef.current || !volumeContainerRef.current) return;

    const sharedOptions = {
      layout: {
        background: { color: '#0f172a' },
        textColor: '#94a3b8',
        fontSize: 11,
        fontFamily: 'JetBrains Mono, monospace',
      },
      grid: {
        vertLines: { color: '#1e293b' },
        horzLines: { color: '#1e293b' },
      },
      crosshair: {
        mode: 0,
        vertLine: { color: '#475569', labelBackgroundColor: '#334155' },
        horzLine: { color: '#475569', labelBackgroundColor: '#334155' },
      },
      rightPriceScale: { borderColor: '#1e293b' },
      timeScale: { borderColor: '#1e293b', timeVisible: true },
    };

    const priceChart = createChart(containerRef.current, {
      ...sharedOptions,
      height: containerRef.current.clientHeight || 320,
    });

    const volChart = createChart(volumeContainerRef.current, {
      ...sharedOptions,
      height: volumeContainerRef.current.clientHeight || 80,
      timeScale: { ...sharedOptions.timeScale, visible: false },
      rightPriceScale: { ...sharedOptions.rightPriceScale, scaleMargins: { top: 0.1, bottom: 0 } },
      crosshair: { ...sharedOptions.crosshair, horzLine: { visible: false, labelVisible: false } },
    });

    const candle = priceChart.addCandlestickSeries({
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderDownColor: '#ef4444',
      borderUpColor: '#22c55e',
      wickDownColor: '#ef4444',
      wickUpColor: '#22c55e',
    });

    const volume = volChart.addHistogramSeries({
      color: '#334155',
      priceFormat: { type: 'volume' },
    });

    chartRef.current = priceChart;
    volumeChartRef.current = volChart;
    candleRef.current = candle;
    volumeRef.current = volume;

    const handleResize = () => {
      if (containerRef.current) {
        priceChart.applyOptions({ width: containerRef.current.clientWidth });
      }
      if (volumeContainerRef.current) {
        volChart.applyOptions({ width: volumeContainerRef.current.clientWidth });
      }
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      priceChart.remove();
      volChart.remove();
      chartRef.current = null;
      volumeChartRef.current = null;
      candleRef.current = null;
      volumeRef.current = null;
    };
  }, []);

  // Load historical data whenever symbol or interval changes
  useEffect(() => {
    if (!candleRef.current || !volumeRef.current) return;
    setLoading(true);

    fetchOhlcv(symbol, interval)
      .then((bars) => {
        if (!candleRef.current || !volumeRef.current) return;
        if (bars.length === 0) bars = generateMockBars();

        const candleData: CandlestickData<Time>[] = bars.map((b) => ({
          time: b.time as Time,
          open: b.open,
          high: b.high,
          low: b.low,
          close: b.close,
        }));

        const volData: HistogramData<Time>[] = bars.map((b) => ({
          time: b.time as Time,
          value: b.volume,
          color: b.close >= b.open ? '#166534' : '#7f1d1d',
        }));

        candleRef.current.setData(candleData);
        volumeRef.current.setData(volData);
        chartRef.current?.timeScale().fitContent();
      })
      .finally(() => setLoading(false));
  }, [symbol, interval]);

  // Stream live bar updates
  useEffect(() => {
    if (!liveBar || !candleRef.current || !volumeRef.current) return;
    candleRef.current.update({
      time: liveBar.time as Time,
      open: liveBar.open,
      high: liveBar.high,
      low: liveBar.low,
      close: liveBar.close,
    });
    volumeRef.current.update({
      time: liveBar.time as Time,
      value: liveBar.volume,
      color: liveBar.close >= liveBar.open ? '#166534' : '#7f1d1d',
    });
  }, [liveBar]);

  return (
    <div className="flex flex-col flex-1 min-w-0 overflow-hidden">
      {/* Price chart */}
      <div className="relative flex-1 min-h-0">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-slate-900/80 z-10">
            <span className="text-slate-400 text-sm">Loading chart…</span>
          </div>
        )}
        <div ref={containerRef} className="w-full h-full" style={{ minHeight: 280 }} />
      </div>
      {/* Volume histogram */}
      <div ref={volumeContainerRef} className="w-full" style={{ height: 80 }} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// OrderBookPanel
// ---------------------------------------------------------------------------

interface OrderBookPanelProps {
  book: OrderBook | null;
}

function OrderBookPanel({ book }: OrderBookPanelProps) {
  if (!book) {
    return (
      <div className="flex items-center justify-center h-full text-slate-500 text-sm">
        No order book data
      </div>
    );
  }

  const maxSize = Math.max(
    ...book.bids.map((b) => b.size),
    ...book.asks.map((a) => a.size),
    1
  );

  const spread =
    book.asks.length && book.bids.length
      ? book.asks[0].price - book.bids[0].price
      : null;

  const spreadPct =
    spread !== null && book.bids.length
      ? ((spread / book.bids[0].price) * 100).toFixed(3)
      : null;

  return (
    <div className="flex flex-col h-full text-xs font-mono">
      {/* Header */}
      <div className="grid grid-cols-2 text-slate-500 pb-1 border-b border-slate-700/50 mb-1 px-1">
        <span>Size</span>
        <span className="text-right">Price</span>
      </div>

      {/* Asks (reversed so best ask is closest to spread) */}
      <div className="flex-1 overflow-y-auto flex flex-col-reverse gap-px">
        {[...book.asks].reverse().map((entry, i) => (
          <div key={i} className="relative flex items-center h-5">
            <div
              className="absolute right-0 top-0 bottom-0 bg-red-900/30"
              style={{ width: `${(entry.size / maxSize) * 100}%` }}
            />
            <span className="relative z-10 flex-1 text-slate-400 pl-1">
              {entry.size.toLocaleString()}
            </span>
            <span className="relative z-10 text-red-400 pr-1">
              {entry.price.toFixed(2)}
            </span>
          </div>
        ))}
      </div>

      {/* Spread */}
      <div className="py-1 text-center text-slate-500 border-y border-slate-700/50 my-1">
        {spread !== null
          ? `Spread ${spread.toFixed(2)} (${spreadPct}%)`
          : '—'}
      </div>

      {/* Bids */}
      <div className="flex-1 overflow-y-auto flex flex-col gap-px">
        {book.bids.map((entry, i) => (
          <div key={i} className="relative flex items-center h-5">
            <div
              className="absolute right-0 top-0 bottom-0 bg-green-900/30"
              style={{ width: `${(entry.size / maxSize) * 100}%` }}
            />
            <span className="relative z-10 flex-1 text-slate-400 pl-1">
              {entry.size.toLocaleString()}
            </span>
            <span className="relative z-10 text-green-400 pr-1">
              {entry.price.toFixed(2)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// MarketDataPage
// ---------------------------------------------------------------------------

interface MarketDataPageProps {
  quotes: Map<string, Quote>;
  orderBooks: Map<string, OrderBook>;
  lastOhlcv: OhlcvBar | null;
}

export function MarketDataPage({ quotes, orderBooks, lastOhlcv }: MarketDataPageProps) {
  const [symbol, setSymbol] = useState('AAPL');
  const [inputValue, setInputValue] = useState('AAPL');
  const [interval, setInterval] = useState<Interval>('5m');
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [allSymbols, setAllSymbols] = useState<string[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const quote = quotes.get(symbol);

  // Compute or mock order book for selected symbol
  const book: OrderBook | null = (() => {
    const ws = orderBooks.get(symbol);
    if (ws) return ws;
    const mid = quote?.price ?? 150;
    return generateMockOrderBook(symbol, mid);
  })();

  // Live bar only applies if it matches current symbol+interval (best-effort)
  const liveBar = lastOhlcv;

  useEffect(() => {
    fetchWatchlist()
      .then((syms) => setAllSymbols(syms))
      .catch(() => setAllSymbols(['AAPL', 'TSLA', 'NVDA', 'MSFT', 'AMZN', 'GOOG', 'META', 'BTC-USD', 'ETH-USD']));
  }, []);

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const val = e.target.value.toUpperCase();
      setInputValue(val);
      if (val.length > 0) {
        setSuggestions(allSymbols.filter((s) => s.startsWith(val)).slice(0, 6));
        setShowSuggestions(true);
      } else {
        setShowSuggestions(false);
      }
    },
    [allSymbols]
  );

  const commitSymbol = useCallback(
    (sym: string) => {
      const upper = sym.toUpperCase().trim();
      if (!upper) return;
      setSymbol(upper);
      setInputValue(upper);
      setShowSuggestions(false);
    },
    []
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'Enter') commitSymbol(inputValue);
      if (e.key === 'Escape') setShowSuggestions(false);
    },
    [inputValue, commitSymbol]
  );

  return (
    <div className="flex flex-col flex-1 overflow-hidden bg-surface-900">
      {/* Top bar */}
      <div className="flex items-center gap-4 px-4 py-2.5 border-b border-slate-700/50 flex-shrink-0">
        {/* Symbol search */}
        <div className="relative">
          <input
            ref={inputRef}
            value={inputValue}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            onBlur={() => setTimeout(() => setShowSuggestions(false), 150)}
            placeholder="Symbol…"
            className="bg-slate-800 border border-slate-600 rounded px-3 py-1.5 text-sm font-mono text-slate-100 w-36 focus:outline-none focus:border-sky-500"
          />
          {showSuggestions && suggestions.length > 0 && (
            <div className="absolute top-full left-0 mt-1 bg-slate-800 border border-slate-600 rounded shadow-lg z-20 w-36">
              {suggestions.map((s) => (
                <button
                  key={s}
                  onMouseDown={() => commitSymbol(s)}
                  className="w-full text-left px-3 py-1.5 text-sm font-mono text-slate-300 hover:bg-slate-700"
                >
                  {s}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Quote summary */}
        {quote && (
          <div className="flex items-baseline gap-3">
            <span className="font-mono text-lg font-semibold">${quote.price.toFixed(2)}</span>
            <span
              className={`font-mono text-sm ${quote.change >= 0 ? 'text-gain' : 'text-loss'}`}
            >
              {quote.change >= 0 ? '+' : ''}
              {quote.change.toFixed(2)} ({quote.change >= 0 ? '+' : ''}
              {quote.changePercent.toFixed(2)}%)
            </span>
            <span className="text-slate-500 text-xs">
              O {quote.open.toFixed(2)} H {quote.high.toFixed(2)} L {quote.low.toFixed(2)}
            </span>
          </div>
        )}

        <div className="flex-1" />

        {/* Timeframe selector */}
        <div className="flex items-center gap-1">
          {INTERVALS.map(({ label, value }) => (
            <button
              key={value}
              onClick={() => setInterval(value)}
              className={`px-2.5 py-1 rounded text-xs font-mono transition-colors ${
                interval === value
                  ? 'bg-sky-600 text-white'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Main content: chart + order book */}
      <div className="flex flex-1 overflow-hidden">
        {/* Chart area */}
        <div className="flex flex-1 flex-col overflow-hidden p-2">
          <div className="bg-surface-800 rounded-lg border border-slate-700/50 flex flex-col flex-1 overflow-hidden p-1">
            <div className="px-3 pt-2 pb-1 text-sm font-semibold text-slate-300 flex-shrink-0">
              {symbol} · {interval}
            </div>
            <CandlestickChart
              symbol={symbol}
              interval={interval}
              liveBar={liveBar}
              quote={quote}
            />
          </div>
        </div>

        {/* Order book */}
        <div className="w-56 flex-shrink-0 p-2 pl-0">
          <div className="bg-surface-800 rounded-lg border border-slate-700/50 flex flex-col h-full overflow-hidden">
            <div className="px-3 py-2 border-b border-slate-700/50 flex-shrink-0">
              <span className="text-sm font-semibold text-slate-300">Order Book</span>
              <span className="ml-2 text-xs text-slate-500">{symbol}</span>
            </div>
            <div className="flex-1 overflow-hidden p-2">
              <OrderBookPanel book={book} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
