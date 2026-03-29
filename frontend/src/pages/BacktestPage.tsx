import { useEffect, useRef, useState, useCallback } from 'react';
import { createChart, IChartApi, ISeriesApi, Time } from 'lightweight-charts';

// ── Types ─────────────────────────────────────────────────────────────────────

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

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmtPct(v: number, decimals = 2) {
  return `${v >= 0 ? '+' : ''}${(v * 100).toFixed(decimals)}%`;
}

function fmtNum(v: number, decimals = 3) {
  return v.toFixed(decimals);
}

// ── Line chart component ──────────────────────────────────────────────────────

interface LineChartProps {
  data: Array<{ time: number; value: number }>;
  color: string;
  label: string;
  formatValue?: (v: number) => string;
  fillArea?: boolean;
}

function LineChart({ data, color, label, formatValue, fillArea = false }: LineChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<'Area' | 'Line'> | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      layout: {
        background: { color: 'transparent' },
        textColor: '#94a3b8',
        fontSize: 11,
        fontFamily: 'JetBrains Mono, ui-monospace, monospace',
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
      timeScale: {
        borderColor: '#1e293b',
        timeVisible: false,
      },
      rightPriceScale: {
        borderColor: '#1e293b',
        scaleMargins: { top: 0.1, bottom: 0.1 },
      },
      handleScroll: false,
      handleScale: false,
    });

    const series = fillArea
      ? chart.addAreaSeries({
          lineColor: color,
          topColor: color + '33',
          bottomColor: color + '00',
          lineWidth: 2,
          priceFormat: formatValue
            ? { type: 'custom', formatter: formatValue, minMove: 0.0001 }
            : { type: 'price' },
        })
      : chart.addLineSeries({
          color,
          lineWidth: 2,
          priceFormat: formatValue
            ? { type: 'custom', formatter: formatValue, minMove: 0.0001 }
            : { type: 'price' },
        });

    chartRef.current = chart;
    seriesRef.current = series as ISeriesApi<'Area' | 'Line'>;

    const handleResize = () => {
      if (containerRef.current) {
        chart.applyOptions({ width: containerRef.current.clientWidth });
      }
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
    };
  }, [color, fillArea]);

  useEffect(() => {
    if (!seriesRef.current || data.length === 0) return;
    const pts = data.map((d) => ({ time: d.time as Time, value: d.value }));
    seriesRef.current.setData(pts);
    chartRef.current?.timeScale().fitContent();
  }, [data]);

  return (
    <div className="bg-slate-800/60 rounded-lg border border-slate-700/50 overflow-hidden">
      <div className="px-4 py-2.5 border-b border-slate-700/50">
        <span className="text-sm font-semibold text-slate-300">{label}</span>
      </div>
      <div ref={containerRef} className="w-full" style={{ height: 220 }} />
    </div>
  );
}

// ── Metrics card ──────────────────────────────────────────────────────────────

interface MetricProps {
  label: string;
  value: string;
  positive?: boolean | null;
}

function Metric({ label, value, positive }: MetricProps) {
  const color =
    positive === null || positive === undefined
      ? 'text-slate-200'
      : positive
        ? 'text-emerald-400'
        : 'text-red-400';
  return (
    <div className="bg-slate-800/60 border border-slate-700/50 rounded-lg px-4 py-3 flex flex-col gap-0.5">
      <span className="text-xs text-slate-500 uppercase tracking-wide">{label}</span>
      <span className={`font-mono text-lg font-semibold ${color}`}>{value}</span>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

const RANGES = ['1y', '2y', '5y', '10y', 'max'] as const;
type Range = (typeof RANGES)[number];

const SIGNALS = [
  { label: 'Long (1)', value: '1' },
  { label: 'Short (-1)', value: '-1' },
  { label: 'Flat (0)', value: '0' },
] as const;

export function BacktestPage() {
  const [symbol, setSymbol] = useState('AAPL');
  const [symbolInput, setSymbolInput] = useState('AAPL');
  const [range, setRange] = useState<Range>('5y');
  const [signal, setSignal] = useState('1');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<BacktestResult | null>(null);

  const runBacktest = useCallback(async () => {
    const sym = symbolInput.trim().toUpperCase();
    if (!sym) return;
    setSymbol(sym);
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(
        `/api/backtest/${encodeURIComponent(sym)}?range=${range}&signal=${signal}`,
      );
      if (!res.ok) {
        const body = await res.json().catch(() => ({ error: res.statusText }));
        throw new Error(body.error ?? res.statusText);
      }
      const data: BacktestResult = await res.json();
      setResult(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [symbolInput, range, signal]);

  // Run on mount with defaults
  useEffect(() => {
    runBacktest();
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    runBacktest();
  };

  return (
    <div className="flex flex-col gap-4 p-4 min-h-0 overflow-y-auto">
      {/* Controls */}
      <div className="bg-slate-800/60 border border-slate-700/50 rounded-lg px-4 py-3">
        <form onSubmit={handleSubmit} className="flex flex-wrap items-end gap-3">
          <div className="flex flex-col gap-1">
            <label className="text-xs text-slate-500 uppercase tracking-wide">Symbol</label>
            <input
              type="text"
              value={symbolInput}
              onChange={(e) => setSymbolInput(e.target.value.toUpperCase())}
              className="bg-slate-900 border border-slate-600 rounded px-3 py-1.5 text-sm font-mono w-28 text-slate-100 focus:outline-none focus:border-sky-500"
              placeholder="AAPL"
              maxLength={10}
            />
          </div>

          <div className="flex flex-col gap-1">
            <label className="text-xs text-slate-500 uppercase tracking-wide">Range</label>
            <div className="flex rounded overflow-hidden border border-slate-600">
              {RANGES.map((r) => (
                <button
                  key={r}
                  type="button"
                  onClick={() => setRange(r)}
                  className={`px-3 py-1.5 text-xs font-mono transition-colors ${
                    range === r
                      ? 'bg-sky-600 text-white'
                      : 'bg-slate-900 text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {r}
                </button>
              ))}
            </div>
          </div>

          <div className="flex flex-col gap-1">
            <label className="text-xs text-slate-500 uppercase tracking-wide">Signal</label>
            <select
              value={signal}
              onChange={(e) => setSignal(e.target.value)}
              className="bg-slate-900 border border-slate-600 rounded px-3 py-1.5 text-sm font-mono text-slate-100 focus:outline-none focus:border-sky-500"
            >
              {SIGNALS.map((s) => (
                <option key={s.value} value={s.value}>
                  {s.label}
                </option>
              ))}
            </select>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="px-5 py-1.5 bg-sky-600 hover:bg-sky-500 disabled:bg-slate-700 disabled:text-slate-500 text-white text-sm font-semibold rounded transition-colors"
          >
            {loading ? 'Running…' : 'Run Backtest'}
          </button>
        </form>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-900/30 border border-red-700/50 rounded-lg px-4 py-3 text-red-400 text-sm">
          {error}
        </div>
      )}

      {/* Results */}
      {result && !loading && (
        <>
          {/* Header */}
          <div className="flex items-baseline gap-3">
            <h2 className="text-lg font-bold text-slate-100 font-mono">{result.symbol}</h2>
            <span className="text-sm text-slate-500">{result.bars} bars · {result.range}</span>
          </div>

          {/* Metrics */}
          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-2">
            <Metric
              label="Total Return"
              value={fmtPct(result.totalReturn)}
              positive={result.totalReturn >= 0}
            />
            <Metric
              label="CAGR"
              value={fmtPct(result.cagr)}
              positive={result.cagr >= 0}
            />
            <Metric
              label="Sharpe"
              value={fmtNum(result.sharpeRatio)}
              positive={result.sharpeRatio >= 1}
            />
            <Metric
              label="Max Drawdown"
              value={fmtPct(-result.maxDrawdown)}
              positive={false}
            />
            <Metric
              label="Win Rate"
              value={fmtPct(result.winRate, 1)}
              positive={result.winRate >= 0.5}
            />
            <Metric
              label="Profit Factor"
              value={result.profitFactor >= 99 ? '∞' : fmtNum(result.profitFactor, 2)}
              positive={result.profitFactor >= 1}
            />
            <Metric
              label="Trades"
              value={String(result.tradeCount)}
              positive={null}
            />
          </div>

          {/* Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
            <LineChart
              data={result.equityCurve}
              color="#38bdf8"
              label="Equity Curve"
              formatValue={(v) => `$${(v / 1000).toFixed(0)}k`}
              fillArea
            />
            <LineChart
              data={result.drawdownCurve}
              color="#f87171"
              label="Drawdown"
              formatValue={(v) => `${(v * 100).toFixed(1)}%`}
            />
          </div>

          {/* Footer */}
          <div className="text-xs text-slate-600 pb-2">
            Buy-and-hold simulation · commission 10 bps · initial capital $1M ·
            data via Yahoo Finance
          </div>
        </>
      )}

      {/* Loading skeleton */}
      {loading && (
        <div className="flex items-center justify-center py-20 text-slate-500 text-sm">
          Running backtest for {symbolInput.trim().toUpperCase() || 'AAPL'}…
        </div>
      )}
    </div>
  );
}
