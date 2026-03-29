import { useEffect, useRef } from 'react';
import { createChart, IChartApi, ISeriesApi, Time } from 'lightweight-charts';
import { useRiskSnapshot } from '../hooks/useApi';
import type { RiskSnapshot } from '../hooks/useApi';

// ── Static stress scenarios (displayed while backend wires real data) ──────────

interface StressScenario {
  name: string;
  year: string;
  estimatedPnl: number;
  pnlPct: number;
}

const STRESS_SCENARIOS: StressScenario[] = [
  { name: '2008 Financial Crisis', year: '2008', estimatedPnl: -184_200, pnlPct: -0.1842 },
  { name: 'COVID Crash',           year: '2020', estimatedPnl: -96_400,  pnlPct: -0.0964 },
  { name: '2022 Rate Shock',       year: '2022', estimatedPnl: -142_300, pnlPct: -0.1423 },
  { name: 'Tech Bubble Burst',     year: '2000', estimatedPnl: -221_000, pnlPct: -0.2210 },
  { name: 'Flash Crash',           year: '2010', estimatedPnl: -38_500,  pnlPct: -0.0385 },
];

const MOCK_CORR_SYMBOLS = ['AAPL', 'NVDA', 'MSFT', 'GOOGL', 'TSLA'];
const MOCK_CORR_MATRIX: number[][] = [
  [1.00, 0.71, 0.82, 0.78, 0.52],
  [0.71, 1.00, 0.68, 0.65, 0.61],
  [0.82, 0.68, 1.00, 0.89, 0.47],
  [0.78, 0.65, 0.89, 1.00, 0.44],
  [0.52, 0.61, 0.47, 0.44, 1.00],
];

// ── Circular gauge ────────────────────────────────────────────────────────────

function CircularGauge({ value, max, label, sublabel, dangerThreshold = 0.8 }: {
  value: number; max: number; label: string; sublabel: string; dangerThreshold?: number;
}) {
  const pct = Math.min(value / max, 1);
  const danger = pct >= dangerThreshold;
  const warning = pct >= 0.6 && !danger;
  const RADIUS = 44;
  const arcColor = danger ? '#f87171' : warning ? '#fbbf24' : '#38bdf8';
  const START_ANGLE = 210;
  const SPAN = 300;
  const sweep = pct * SPAN;
  const toRad = (deg: number) => (deg * Math.PI) / 180;
  const cx = 60; const cy = 60;
  const arcPath = (fromDeg: number, toDeg: number, r: number) => {
    const f = toRad(fromDeg - 90); const t = toRad(toDeg - 90);
    const x1 = cx + r * Math.cos(f); const y1 = cy + r * Math.sin(f);
    const x2 = cx + r * Math.cos(t); const y2 = cy + r * Math.sin(t);
    const large = toDeg - fromDeg > 180 ? 1 : 0;
    return `M ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2}`;
  };

  return (
    <div className="bg-slate-800/60 border border-slate-700/50 rounded-lg p-4 flex flex-col items-center gap-1">
      <svg viewBox="0 0 120 120" className="w-28 h-28">
        <path d={arcPath(START_ANGLE, START_ANGLE + SPAN, RADIUS)} fill="none" stroke="#1e293b" strokeWidth={8} strokeLinecap="round" />
        {sweep > 0 && (
          <path d={arcPath(START_ANGLE, START_ANGLE + sweep, RADIUS)} fill="none" stroke={arcColor} strokeWidth={8} strokeLinecap="round" />
        )}
        <text x={cx} y={cy - 4} textAnchor="middle" fontSize={14} fontWeight="600" fill={arcColor} fontFamily="ui-monospace, monospace">
          {(value * 100).toFixed(2)}%
        </text>
        <text x={cx} y={cy + 12} textAnchor="middle" fontSize={9} fill="#64748b" fontFamily="ui-monospace, monospace">
          of {(max * 100).toFixed(0)}% lim
        </text>
      </svg>
      <div className="text-center">
        <div className="text-sm font-semibold text-slate-200">{label}</div>
        <div className="text-xs text-slate-500">{sublabel}</div>
      </div>
    </div>
  );
}

// ── Circuit breaker ───────────────────────────────────────────────────────────

function CircuitBreakerIndicator({ armed }: { armed: boolean }) {
  return (
    <div className="bg-slate-800/60 border border-slate-700/50 rounded-lg p-4 flex flex-col items-center justify-center gap-3">
      <div className={`w-16 h-16 rounded-full flex items-center justify-center border-4 ${armed ? 'border-red-500 bg-red-500/20 animate-pulse' : 'border-emerald-500 bg-emerald-500/20'}`}>
        {armed ? (
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#f87171" strokeWidth="2.5" strokeLinecap="round">
            <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
            <line x1="12" y1="9" x2="12" y2="13" /><line x1="12" y1="17" x2="12.01" y2="17" />
          </svg>
        ) : (
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#34d399" strokeWidth="2.5" strokeLinecap="round">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
            <polyline points="9 12 11 14 15 10" />
          </svg>
        )}
      </div>
      <div className="text-center">
        <div className={`text-sm font-semibold ${armed ? 'text-red-400' : 'text-emerald-400'}`}>
          {armed ? 'CIRCUIT BREAKER ARMED' : 'CIRCUIT SAFE'}
        </div>
        <div className="text-xs text-slate-500 mt-0.5">
          {armed ? 'Trading halted — manual override required' : 'All systems nominal'}
        </div>
      </div>
    </div>
  );
}

// ── Drawdown bar ──────────────────────────────────────────────────────────────

function DrawdownBar({ current, max }: { current: number; max: number }) {
  const CIRCUIT_THRESHOLD = 0.15;
  const currentPct = (current / CIRCUIT_THRESHOLD) * 100;
  const maxPct = (max / CIRCUIT_THRESHOLD) * 100;
  const currentColor = current >= CIRCUIT_THRESHOLD * 0.8 ? 'bg-red-500' : current >= CIRCUIT_THRESHOLD * 0.5 ? 'bg-amber-500' : 'bg-sky-500';

  return (
    <div className="bg-slate-800/60 border border-slate-700/50 rounded-lg p-4">
      <div className="text-xs text-slate-500 uppercase tracking-wide mb-3">Drawdown vs Limit (15%)</div>
      <div className="space-y-2">
        <div className="flex items-center gap-3">
          <span className="text-xs font-mono text-slate-400 w-16">Current</span>
          <div className="flex-1 h-2.5 bg-slate-700/50 rounded-full overflow-hidden">
            <div className={`h-full rounded-full ${currentColor}`} style={{ width: `${Math.min(currentPct, 100)}%` }} />
          </div>
          <span className={`text-xs font-mono w-12 text-right ${current >= CIRCUIT_THRESHOLD * 0.8 ? 'text-red-400' : 'text-slate-300'}`}>
            {(current * 100).toFixed(2)}%
          </span>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs font-mono text-slate-500 w-16">Max</span>
          <div className="flex-1 h-2.5 bg-slate-700/50 rounded-full overflow-hidden">
            <div className="h-full rounded-full bg-slate-500/60" style={{ width: `${Math.min(maxPct, 100)}%` }} />
          </div>
          <span className="text-xs font-mono w-12 text-right text-slate-500">
            {(max * 100).toFixed(2)}%
          </span>
        </div>
        <div className="text-xs text-slate-600 text-right">circuit breaker @ 15%</div>
      </div>
    </div>
  );
}

// ── Drawdown chart ────────────────────────────────────────────────────────────

function DrawdownChart({ drawdown, maxDrawdown, circuitThreshold }: {
  drawdown: number; maxDrawdown: number; circuitThreshold: number;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<'Area'> | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      layout: { background: { color: 'transparent' }, textColor: '#94a3b8', fontSize: 11, fontFamily: 'JetBrains Mono, ui-monospace, monospace' },
      grid: { vertLines: { color: '#1e293b' }, horzLines: { color: '#1e293b' } },
      crosshair: { mode: 0, vertLine: { color: '#475569', labelBackgroundColor: '#334155' }, horzLine: { color: '#475569', labelBackgroundColor: '#334155' } },
      timeScale: { borderColor: '#1e293b', timeVisible: false },
      rightPriceScale: { borderColor: '#1e293b', scaleMargins: { top: 0.05, bottom: 0.05 } },
      handleScroll: false,
      handleScale: false,
    });

    const series = chart.addAreaSeries({
      lineColor: '#f87171',
      topColor: '#f8717133',
      bottomColor: '#f8717100',
      lineWidth: 2,
      priceFormat: { type: 'custom', formatter: (v: number) => `${(v * 100).toFixed(2)}%`, minMove: 0.0001 },
    });

    series.createPriceLine({
      price: circuitThreshold,
      color: '#ef4444',
      lineWidth: 1,
      lineStyle: 2,
      axisLabelVisible: true,
      title: 'Circuit Breaker',
    });

    // Build a simple 2-point series from current/max so the chart isn't empty
    const now = Math.floor(Date.now() / 1000);
    series.setData([
      { time: (now - 86400) as Time, value: maxDrawdown },
      { time: now as Time, value: drawdown },
    ]);
    chart.timeScale().fitContent();

    seriesRef.current = series;
    chartRef.current = chart;

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
  }, [drawdown, maxDrawdown, circuitThreshold]);

  return (
    <div className="bg-slate-800/60 rounded-lg border border-slate-700/50 overflow-hidden">
      <div className="px-4 py-2.5 border-b border-slate-700/50">
        <span className="text-sm font-semibold text-slate-300">Portfolio Drawdown</span>
      </div>
      <div ref={containerRef} className="w-full" style={{ height: 220 }} />
    </div>
  );
}

// ── Position limits ───────────────────────────────────────────────────────────

function PositionLimits({ positions }: { positions: Array<{ symbol: string; utilization: number }> }) {
  return (
    <div className="bg-slate-800/60 rounded-lg border border-slate-700/50 overflow-hidden">
      <div className="px-4 py-2.5 border-b border-slate-700/50">
        <span className="text-sm font-semibold text-slate-300">Position Limit Utilization</span>
      </div>
      <div className="p-4 space-y-2.5">
        {positions.map((p) => {
          const pct = p.utilization * 100;
          const barColor = pct >= 90 ? 'bg-red-500' : pct >= 70 ? 'bg-amber-500' : 'bg-emerald-500';
          const labelColor = pct >= 90 ? 'text-red-400' : pct >= 70 ? 'text-amber-400' : 'text-emerald-400';
          return (
            <div key={p.symbol} className="flex items-center gap-3">
              <span className="text-xs font-mono text-slate-400 w-12">{p.symbol}</span>
              <div className="flex-1 h-2 bg-slate-700/50 rounded-full overflow-hidden">
                <div className={`h-full rounded-full ${barColor}`} style={{ width: `${pct.toFixed(0)}%` }} />
              </div>
              <span className={`text-xs font-mono w-10 text-right ${labelColor}`}>{pct.toFixed(0)}%</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Correlation matrix ────────────────────────────────────────────────────────

function corrColor(v: number): { bg: string; text: string } {
  if (v === 1)   return { bg: 'bg-slate-600/50',     text: 'text-slate-400'   };
  if (v >= 0.8)  return { bg: 'bg-red-500/30',       text: 'text-red-300'     };
  if (v >= 0.6)  return { bg: 'bg-amber-500/20',     text: 'text-amber-300'   };
  if (v >= 0.3)  return { bg: 'bg-sky-500/15',       text: 'text-sky-300'     };
  if (v >= 0)    return { bg: 'bg-emerald-500/15',   text: 'text-emerald-300' };
  return              { bg: 'bg-violet-500/20',    text: 'text-violet-300'  };
}

function CorrelationMatrix() {
  return (
    <div className="bg-slate-800/60 rounded-lg border border-slate-700/50 overflow-hidden">
      <div className="px-4 py-2.5 border-b border-slate-700/50 flex items-center justify-between">
        <span className="text-sm font-semibold text-slate-300">Correlation Matrix</span>
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <span className="inline-block w-3 h-3 rounded-sm bg-red-500/30 border border-red-500/50" /> high
          <span className="inline-block w-3 h-3 rounded-sm bg-amber-500/20 border border-amber-500/50" /> mid
          <span className="inline-block w-3 h-3 rounded-sm bg-emerald-500/15 border border-emerald-500/50" /> low
        </div>
      </div>
      <div className="p-4 overflow-x-auto">
        <table className="text-xs font-mono">
          <thead>
            <tr>
              <th className="w-14 pb-1" />
              {MOCK_CORR_SYMBOLS.map((s) => (
                <th key={s} className="w-14 text-center text-slate-500 pb-1 font-normal">{s}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {MOCK_CORR_MATRIX.map((row, i) => (
              <tr key={i}>
                <td className="text-slate-500 pr-2 font-normal">{MOCK_CORR_SYMBOLS[i]}</td>
                {row.map((v, j) => {
                  const { bg, text } = corrColor(v);
                  return (
                    <td key={j} className={`w-14 h-9 text-center rounded ${bg} ${text} font-semibold`}>
                      {v.toFixed(2)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── Stress scenarios ──────────────────────────────────────────────────────────

function StressTable({ scenarios }: { scenarios: StressScenario[] }) {
  return (
    <div className="bg-slate-800/60 rounded-lg border border-slate-700/50 overflow-hidden">
      <div className="px-4 py-2.5 border-b border-slate-700/50 flex items-center gap-2">
        <span className="text-sm font-semibold text-slate-300">Stress Scenarios</span>
        <span className="text-xs text-slate-600">(estimated P&L impact)</span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs font-mono">
          <thead>
            <tr className="border-b border-slate-700/50 text-slate-500 uppercase tracking-wide">
              <th className="px-4 py-2 text-left">Scenario</th>
              <th className="px-4 py-2 text-right">Year</th>
              <th className="px-4 py-2 text-right">Est. P&L</th>
              <th className="px-4 py-2 text-right">Portfolio %</th>
            </tr>
          </thead>
          <tbody>
            {scenarios.map((s) => (
              <tr key={s.name} className="border-b border-slate-700/30 hover:bg-slate-700/20 transition-colors">
                <td className="px-4 py-2.5 text-slate-300">{s.name}</td>
                <td className="px-4 py-2.5 text-right text-slate-500">{s.year}</td>
                <td className="px-4 py-2.5 text-right text-red-400">-${Math.abs(s.estimatedPnl).toLocaleString()}</td>
                <td className="px-4 py-2.5 text-right text-red-400">{(s.pnlPct * 100).toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── Default snapshot (shown while API loads) ──────────────────────────────────

const DEFAULT_RISK: RiskSnapshot = {
  var95: 0,
  var99: 0,
  drawdown: 0,
  maxDrawdown: 0,
  circuitBreakerArmed: false,
  positionLimitUtilization: [],
};

// ── Main page ─────────────────────────────────────────────────────────────────

export function RiskDashboardPage() {
  const { data: risk, isLoading, isError } = useRiskSnapshot();
  const snapshot = risk ?? DEFAULT_RISK;

  return (
    <div className="flex flex-col gap-4 p-4 min-h-0 overflow-y-auto">
      {isLoading && (
        <div className="text-xs text-slate-500 bg-slate-800/60 border border-slate-700/50 rounded px-3 py-1.5 font-mono self-start">
          loading risk metrics…
        </div>
      )}
      {isError && (
        <div className="text-xs text-red-400 bg-red-900/20 border border-red-700/40 rounded px-3 py-1.5 font-mono self-start">
          risk API unavailable — backend wiring pending
        </div>
      )}

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <CircularGauge value={snapshot.var95} max={0.04} label="VaR 95%" sublabel="1-day, historical" dangerThreshold={0.75} />
        <CircularGauge value={snapshot.var99} max={0.06} label="VaR 99%" sublabel="1-day, historical" dangerThreshold={0.75} />
        <div className="col-span-2">
          <CircuitBreakerIndicator armed={snapshot.circuitBreakerArmed} />
        </div>
      </div>

      <DrawdownBar current={snapshot.drawdown} max={snapshot.maxDrawdown} />
      <DrawdownChart drawdown={snapshot.drawdown} maxDrawdown={snapshot.maxDrawdown} circuitThreshold={0.15} />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        <PositionLimits positions={snapshot.positionLimitUtilization} />
        <CorrelationMatrix />
      </div>

      <StressTable scenarios={STRESS_SCENARIOS} />
    </div>
  );
}
