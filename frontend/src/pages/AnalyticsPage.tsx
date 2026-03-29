import { useEffect, useRef, useState, useCallback } from 'react';
import { createChart, IChartApi, ISeriesApi, Time } from 'lightweight-charts';

// ── Types ─────────────────────────────────────────────────────────────────────

interface PerformanceData {
  cumulativeReturns: Array<{ time: number; value: number }>;
  benchmarkReturns: Array<{ time: number; value: number }>;
  relativePerf: Array<{ time: number; value: number }>;
  rolling30dReturns: Array<{ time: number; value: number }>;
  rolling30dSharpe: Array<{ time: number; value: number }>;
  dailyReturns: number[];
  stats: {
    mean: number;
    stddev: number;
    skewness: number;
    kurtosis: number;
  };
}

interface AttributionRow {
  strategy: string;
  return: number;
  weight: number;
  contribution: number;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmtPct(v: number, decimals = 2) {
  return `${v >= 0 ? '+' : ''}${(v * 100).toFixed(decimals)}%`;
}

function fmtNum(v: number, decimals = 3) {
  return v.toFixed(decimals);
}

// ── Single-series line chart ──────────────────────────────────────────────────

interface LineChartProps {
  data: Array<{ time: number; value: number }>;
  color: string;
  label: string;
  formatValue?: (v: number) => string;
  fillArea?: boolean;
  height?: number;
}

function LineChart({ data, color, label, formatValue, fillArea = false, height = 200 }: LineChartProps) {
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
      timeScale: { borderColor: '#1e293b', timeVisible: false },
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
    seriesRef.current.setData(data.map((d) => ({ time: d.time as Time, value: d.value })));
    chartRef.current?.timeScale().fitContent();
  }, [data]);

  return (
    <div className="bg-slate-800/60 rounded-lg border border-slate-700/50 overflow-hidden">
      <div className="px-4 py-2.5 border-b border-slate-700/50">
        <span className="text-sm font-semibold text-slate-300">{label}</span>
      </div>
      <div ref={containerRef} className="w-full" style={{ height }} />
    </div>
  );
}

// ── Comparison chart (two series) ─────────────────────────────────────────────

interface ComparisonChartProps {
  stratData: Array<{ time: number; value: number }>;
  benchData: Array<{ time: number; value: number }>;
  stratLabel: string;
  benchLabel: string;
  label: string;
  height?: number;
}

function ComparisonChart({
  stratData,
  benchData,
  stratLabel,
  benchLabel,
  label,
  height = 220,
}: ComparisonChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const stratSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const benchSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);

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
      timeScale: { borderColor: '#1e293b', timeVisible: false },
      rightPriceScale: {
        borderColor: '#1e293b',
        scaleMargins: { top: 0.1, bottom: 0.1 },
      },
      handleScroll: false,
      handleScale: false,
    });

    const fmt = (v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`;

    stratSeriesRef.current = chart.addLineSeries({
      color: '#38bdf8',
      lineWidth: 2,
      title: stratLabel,
      priceFormat: { type: 'custom', formatter: fmt, minMove: 0.0001 },
    });

    benchSeriesRef.current = chart.addLineSeries({
      color: '#fb923c',
      lineWidth: 2,
      title: benchLabel,
      priceFormat: { type: 'custom', formatter: fmt, minMove: 0.0001 },
    });

    chartRef.current = chart;

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
      stratSeriesRef.current = null;
      benchSeriesRef.current = null;
    };
  }, [stratLabel, benchLabel]);

  useEffect(() => {
    if (!stratSeriesRef.current || stratData.length === 0) return;
    stratSeriesRef.current.setData(stratData.map((d) => ({ time: d.time as Time, value: d.value })));
    chartRef.current?.timeScale().fitContent();
  }, [stratData]);

  useEffect(() => {
    if (!benchSeriesRef.current || benchData.length === 0) return;
    benchSeriesRef.current.setData(benchData.map((d) => ({ time: d.time as Time, value: d.value })));
    chartRef.current?.timeScale().fitContent();
  }, [benchData]);

  return (
    <div className="bg-slate-800/60 rounded-lg border border-slate-700/50 overflow-hidden">
      <div className="px-4 py-2.5 border-b border-slate-700/50 flex items-center justify-between">
        <span className="text-sm font-semibold text-slate-300">{label}</span>
        <div className="flex items-center gap-4 text-xs">
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-4 h-0.5 bg-sky-400 rounded" />
            <span className="text-slate-400">{stratLabel}</span>
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-4 h-0.5 bg-orange-400 rounded" />
            <span className="text-slate-400">{benchLabel}</span>
          </span>
        </div>
      </div>
      <div ref={containerRef} className="w-full" style={{ height }} />
    </div>
  );
}

// ── Return Distribution (SVG histogram + normal fit) ──────────────────────────

interface ReturnDistributionProps {
  dailyReturns: number[];
  stats: { mean: number; stddev: number; skewness: number; kurtosis: number };
}

function ReturnDistribution({ dailyReturns, stats }: ReturnDistributionProps) {
  if (dailyReturns.length === 0) return null;

  const NUM_BINS = 40;
  const minR = Math.min(...dailyReturns);
  const maxR = Math.max(...dailyReturns);
  const binWidth = (maxR - minR) / NUM_BINS;

  const bins = Array.from({ length: NUM_BINS }, (_, i) => ({
    x: minR + i * binWidth,
    count: 0,
  }));

  for (const r of dailyReturns) {
    const idx = Math.min(Math.floor((r - minR) / binWidth), NUM_BINS - 1);
    if (idx >= 0) bins[idx].count++;
  }

  const maxCount = Math.max(...bins.map((b) => b.count));
  const n = dailyReturns.length;

  // SVG dimensions
  const W = 480;
  const H = 180;
  const PADDING = { top: 12, right: 16, bottom: 28, left: 36 };
  const plotW = W - PADDING.left - PADDING.right;
  const plotH = H - PADDING.top - PADDING.bottom;

  // Normal PDF scaled to match histogram density
  const normalPdf = (x: number) => {
    const z = (x - stats.mean) / stats.stddev;
    return (1 / (stats.stddev * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * z * z);
  };

  // Max PDF value for scaling
  const pdfPeak = normalPdf(stats.mean);
  // Histogram bar height in "density" terms: count / (n * binWidth)
  const maxDensity = maxCount / (n * binWidth);
  const yScale = plotH / Math.max(pdfPeak, maxDensity);

  // Generate normal curve path
  const curvePoints: string[] = [];
  const CURVE_STEPS = 100;
  for (let i = 0; i <= CURVE_STEPS; i++) {
    const x = minR + (i / CURVE_STEPS) * (maxR - minR);
    const density = normalPdf(x);
    const px = PADDING.left + ((x - minR) / (maxR - minR)) * plotW;
    const py = PADDING.top + plotH - density * yScale;
    curvePoints.push(`${i === 0 ? 'M' : 'L'}${px.toFixed(1)},${py.toFixed(1)}`);
  }

  // X-axis tick values
  const xTicks = [-0.04, -0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.04].filter(
    (t) => t >= minR - binWidth && t <= maxR + binWidth,
  );

  return (
    <div className="bg-slate-800/60 rounded-lg border border-slate-700/50 overflow-hidden">
      <div className="px-4 py-2.5 border-b border-slate-700/50">
        <span className="text-sm font-semibold text-slate-300">Return Distribution</span>
      </div>
      <div className="p-4">
        {/* Histogram SVG */}
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ height: H }}>
          {/* Bars */}
          {bins.map((bin, i) => {
            const density = bin.count / (n * binWidth);
            const bH = density * yScale;
            const bX = PADDING.left + (i / NUM_BINS) * plotW;
            const bW = plotW / NUM_BINS - 0.5;
            const isNegative = bin.x + binWidth / 2 < 0;
            return (
              <rect
                key={i}
                x={bX}
                y={PADDING.top + plotH - bH}
                width={bW}
                height={bH}
                fill={isNegative ? '#f8717133' : '#34d39933'}
                stroke={isNegative ? '#f87171' : '#34d399'}
                strokeWidth={0.5}
              />
            );
          })}

          {/* Normal curve */}
          <path d={curvePoints.join(' ')} fill="none" stroke="#fbbf24" strokeWidth={1.5} />

          {/* Zero line */}
          {minR < 0 && maxR > 0 && (
            <line
              x1={PADDING.left + ((-minR) / (maxR - minR)) * plotW}
              y1={PADDING.top}
              x2={PADDING.left + ((-minR) / (maxR - minR)) * plotW}
              y2={PADDING.top + plotH}
              stroke="#475569"
              strokeWidth={1}
              strokeDasharray="3,3"
            />
          )}

          {/* X-axis */}
          <line
            x1={PADDING.left}
            y1={PADDING.top + plotH}
            x2={PADDING.left + plotW}
            y2={PADDING.top + plotH}
            stroke="#334155"
            strokeWidth={1}
          />
          {xTicks.map((t) => {
            const px = PADDING.left + ((t - minR) / (maxR - minR)) * plotW;
            return (
              <g key={t}>
                <line
                  x1={px}
                  y1={PADDING.top + plotH}
                  x2={px}
                  y2={PADDING.top + plotH + 4}
                  stroke="#475569"
                  strokeWidth={1}
                />
                <text
                  x={px}
                  y={PADDING.top + plotH + 14}
                  textAnchor="middle"
                  fontSize={9}
                  fill="#64748b"
                  fontFamily="ui-monospace, monospace"
                >
                  {(t * 100).toFixed(1)}%
                </text>
              </g>
            );
          })}
        </svg>

        {/* Legend + stats */}
        <div className="flex flex-wrap items-center justify-between gap-3 mt-2">
          <div className="flex items-center gap-4 text-xs">
            <span className="flex items-center gap-1.5 text-slate-400">
              <span className="inline-block w-3 h-3 rounded-sm border border-emerald-400 bg-emerald-400/20" />
              Positive
            </span>
            <span className="flex items-center gap-1.5 text-slate-400">
              <span className="inline-block w-3 h-3 rounded-sm border border-red-400 bg-red-400/20" />
              Negative
            </span>
            <span className="flex items-center gap-1.5 text-slate-400">
              <span className="inline-block w-4 h-0.5 bg-amber-400 rounded" />
              Normal fit
            </span>
          </div>
          <div className="flex gap-4 text-xs font-mono">
            <span className="text-slate-500">
              μ <span className="text-slate-300">{fmtPct(stats.mean, 3)}</span>
            </span>
            <span className="text-slate-500">
              σ <span className="text-slate-300">{fmtPct(stats.stddev, 3)}</span>
            </span>
            <span className="text-slate-500">
              skew <span className={stats.skewness < 0 ? 'text-red-400' : 'text-emerald-400'}>{fmtNum(stats.skewness, 2)}</span>
            </span>
            <span className="text-slate-500">
              kurt <span className={stats.kurtosis > 0 ? 'text-amber-400' : 'text-slate-300'}>{fmtNum(stats.kurtosis, 2)}</span>
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Attribution table ─────────────────────────────────────────────────────────

function AttributionTable({ rows }: { rows: AttributionRow[] }) {
  const totalContrib = rows.reduce((s, r) => s + r.contribution, 0);

  return (
    <div className="bg-slate-800/60 rounded-lg border border-slate-700/50 overflow-hidden">
      <div className="px-4 py-2.5 border-b border-slate-700/50 flex items-center gap-2">
        <span className="text-sm font-semibold text-slate-300">Return Attribution</span>
        <span className="text-xs text-slate-600 font-normal">(stub — mock data)</span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs font-mono">
          <thead>
            <tr className="border-b border-slate-700/50 text-slate-500 uppercase tracking-wide">
              <th className="px-4 py-2 text-left">Strategy</th>
              <th className="px-4 py-2 text-right">Return</th>
              <th className="px-4 py-2 text-right">Weight</th>
              <th className="px-4 py-2 text-right">Contribution</th>
              <th className="px-4 py-2 text-right w-24">Bar</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const barWidth = totalContrib > 0 ? (row.contribution / totalContrib) * 100 : 0;
              return (
                <tr key={row.strategy} className="border-b border-slate-700/30 hover:bg-slate-700/20 transition-colors">
                  <td className="px-4 py-2.5 text-slate-300">{row.strategy}</td>
                  <td className={`px-4 py-2.5 text-right ${row.return >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                    {fmtPct(row.return)}
                  </td>
                  <td className="px-4 py-2.5 text-right text-slate-400">{(row.weight * 100).toFixed(0)}%</td>
                  <td className={`px-4 py-2.5 text-right font-semibold ${row.contribution >= 0 ? 'text-emerald-300' : 'text-red-300'}`}>
                    {fmtPct(row.contribution)}
                  </td>
                  <td className="px-4 py-2.5">
                    <div className="h-2 bg-slate-700/50 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-sky-500/70 rounded-full"
                        style={{ width: `${barWidth.toFixed(1)}%` }}
                      />
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
          <tfoot>
            <tr className="text-slate-400 font-semibold">
              <td className="px-4 py-2.5">Total</td>
              <td className="px-4 py-2.5" />
              <td className="px-4 py-2.5 text-right">100%</td>
              <td className={`px-4 py-2.5 text-right ${totalContrib >= 0 ? 'text-emerald-300' : 'text-red-300'}`}>
                {fmtPct(totalContrib)}
              </td>
              <td className="px-4 py-2.5" />
            </tr>
          </tfoot>
        </table>
      </div>
    </div>
  );
}

// ── Controls ──────────────────────────────────────────────────────────────────

const RANGES = ['3mo', '6mo', '1y', '2y', '5y'] as const;
type Range = (typeof RANGES)[number];

const BENCHMARKS = [
  { label: 'SPY', value: 'SPY' },
  { label: 'BTC', value: 'BTC' },
  { label: 'QQQ', value: 'QQQ' },
] as const;

// ── Main page ─────────────────────────────────────────────────────────────────

export function AnalyticsPage() {
  const [symbolInput, setSymbolInput] = useState('AAPL');
  const [symbol, setSymbol] = useState('AAPL');
  const [range, setRange] = useState<Range>('1y');
  const [benchmark, setBenchmark] = useState('SPY');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [perfData, setPerfData] = useState<PerformanceData | null>(null);
  const [attribution, setAttribution] = useState<AttributionRow[]>([]);

  const fetchAnalytics = useCallback(async (sym: string, rng: Range, bench: string) => {
    setLoading(true);
    setError(null);
    try {
      const [perfRes, attrRes] = await Promise.all([
        fetch(`/api/analytics/performance?symbol=${encodeURIComponent(sym)}&range=${rng}&benchmark=${bench}`),
        fetch('/api/analytics/attribution'),
      ]);

      if (!perfRes.ok) {
        const body = await perfRes.json().catch(() => ({ error: perfRes.statusText }));
        throw new Error(body.error ?? perfRes.statusText);
      }

      const [perf, attr] = await Promise.all([perfRes.json(), attrRes.json()]);
      setPerfData(perf);
      setAttribution(attr);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchAnalytics(symbol, range, benchmark);
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const sym = symbolInput.trim().toUpperCase();
    if (!sym) return;
    setSymbol(sym);
    fetchAnalytics(sym, range, benchmark);
  };

  const handleRangeChange = (r: Range) => {
    setRange(r);
    fetchAnalytics(symbol, r, benchmark);
  };

  const handleBenchmarkChange = (b: string) => {
    setBenchmark(b);
    fetchAnalytics(symbol, range, b);
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
                  onClick={() => handleRangeChange(r)}
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
            <label className="text-xs text-slate-500 uppercase tracking-wide">Benchmark</label>
            <div className="flex rounded overflow-hidden border border-slate-600">
              {BENCHMARKS.map((b) => (
                <button
                  key={b.value}
                  type="button"
                  onClick={() => handleBenchmarkChange(b.value)}
                  className={`px-3 py-1.5 text-xs font-mono transition-colors ${
                    benchmark === b.value
                      ? 'bg-orange-600 text-white'
                      : 'bg-slate-900 text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {b.label}
                </button>
              ))}
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="px-5 py-1.5 bg-sky-600 hover:bg-sky-500 disabled:bg-slate-700 disabled:text-slate-500 text-white text-sm font-semibold rounded transition-colors"
          >
            {loading ? 'Loading…' : 'Load'}
          </button>
        </form>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-900/30 border border-red-700/50 rounded-lg px-4 py-3 text-red-400 text-sm">
          {error}
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div className="flex items-center justify-center py-20 text-slate-500 text-sm">
          Loading analytics for {symbolInput.trim().toUpperCase() || 'AAPL'}…
        </div>
      )}

      {/* Content */}
      {perfData && !loading && (
        <>
          {/* Section label */}
          <div className="flex items-baseline gap-3">
            <h2 className="text-lg font-bold text-slate-100 font-mono">{symbol}</h2>
            <span className="text-sm text-slate-500">vs {benchmark} · {range}</span>
          </div>

          {/* Row 1: Cumulative returns comparison */}
          <ComparisonChart
            stratData={perfData.cumulativeReturns}
            benchData={perfData.benchmarkReturns}
            stratLabel={symbol}
            benchLabel={benchmark}
            label="Cumulative Returns"
            height={220}
          />

          {/* Row 2: Relative performance + rolling 30d returns */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
            <LineChart
              data={perfData.relativePerf}
              color="#a78bfa"
              label={`Relative Performance (${symbol} − ${benchmark})`}
              formatValue={(v) => `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`}
              height={200}
            />
            <LineChart
              data={perfData.rolling30dReturns}
              color="#34d399"
              label="Rolling 30d Returns"
              formatValue={(v) => `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`}
              height={200}
            />
          </div>

          {/* Row 3: Rolling Sharpe + Return Distribution */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
            <LineChart
              data={perfData.rolling30dSharpe}
              color="#fbbf24"
              label="Rolling 30d Sharpe Ratio"
              formatValue={(v) => v.toFixed(2)}
              height={200}
            />
            <ReturnDistribution dailyReturns={perfData.dailyReturns} stats={perfData.stats} />
          </div>

          {/* Row 4: Attribution */}
          {attribution.length > 0 && <AttributionTable rows={attribution} />}

          <div className="text-xs text-slate-600 pb-2">
            Prices via Yahoo Finance · returns are price-only (no dividends)
          </div>
        </>
      )}
    </div>
  );
}
