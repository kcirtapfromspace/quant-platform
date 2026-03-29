import { useEffect, useRef, useState } from 'react';
import { createChart, IChartApi, ISeriesApi, CandlestickData, Time } from 'lightweight-charts';
import { fetchHistory } from '../hooks/useApi';
import type { Quote } from '../types';

interface PriceChartProps {
  symbol: string;
  currentQuote?: Quote;
}

export function PriceChart({ symbol, currentQuote }: PriceChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      layout: {
        background: { color: '#0f172a' },
        textColor: '#94a3b8',
        fontSize: 12,
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
      timeScale: { borderColor: '#1e293b', timeVisible: false },
      rightPriceScale: { borderColor: '#1e293b' },
    });

    const series = chart.addCandlestickSeries({
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderDownColor: '#ef4444',
      borderUpColor: '#22c55e',
      wickDownColor: '#ef4444',
      wickUpColor: '#22c55e',
    });

    chartRef.current = chart;
    seriesRef.current = series;

    const handleResize = () => {
      if (containerRef.current) chart.applyOptions({ width: containerRef.current.clientWidth });
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!seriesRef.current) return;
    setLoading(true);
    setError(null);

    fetchHistory(symbol, '6mo')
      .then((bars) => {
        if (!seriesRef.current) return;
        const data: CandlestickData<Time>[] = bars.map((b) => ({
          time: b.time as Time,
          open: b.open,
          high: b.high,
          low: b.low,
          close: b.close,
        }));
        seriesRef.current.setData(data);
        chartRef.current?.timeScale().fitContent();
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : 'Failed to load chart'))
      .finally(() => setLoading(false));
  }, [symbol]);

  return (
    <div className="bg-surface-800 rounded-lg border border-slate-700/50 overflow-hidden flex flex-col">
      <div className="px-4 py-2.5 border-b border-slate-700/50 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h2 className="text-base font-semibold">{symbol}</h2>
          {currentQuote && (
            <>
              <span className="font-mono text-lg font-semibold">${currentQuote.price.toFixed(2)}</span>
              <span className={`font-mono text-sm ${currentQuote.change >= 0 ? 'text-gain' : 'text-loss'}`}>
                {currentQuote.change >= 0 ? '+' : ''}{currentQuote.change.toFixed(2)}
                ({currentQuote.change >= 0 ? '+' : ''}{currentQuote.changePercent.toFixed(2)}%)
              </span>
            </>
          )}
        </div>
        <div className="flex items-center gap-2 text-xs text-slate-500">
          {currentQuote && (
            <>
              <span>O {currentQuote.open.toFixed(2)}</span>
              <span>H {currentQuote.high.toFixed(2)}</span>
              <span>L {currentQuote.low.toFixed(2)}</span>
            </>
          )}
        </div>
      </div>
      <div className="relative flex-1 min-h-[350px]">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-surface-800/80 z-10">
            <div className="text-slate-400 text-sm">Loading chart…</div>
          </div>
        )}
        {error && (
          <div className="absolute inset-0 flex items-center justify-center z-10">
            <div className="text-red-400 text-sm">{error}</div>
          </div>
        )}
        <div ref={containerRef} className="w-full h-full" style={{ minHeight: 350 }} />
      </div>
    </div>
  );
}
