import { useEffect, useRef, useState } from 'react';
import { createChart, IChartApi, ISeriesApi, Time } from 'lightweight-charts';

// ── Types ─────────────────────────────────────────────────────────────────────

type StrategyStatus = 'active' | 'halted' | 'paper';
type Regime = 'bull' | 'bear' | 'sideways';

interface StrategyState {
  id: string;
  name: string;
  status: StrategyStatus;
  regime: Regime;
  dailyPnl: number;
  signalConfidence: number; // 0–100
}

interface RegimePoint {
  time: number;
  regime: Regime;
  equityValue: number;
}

// ── Mock data ─────────────────────────────────────────────────────────────────

function buildMockStrategies(): StrategyState[] {
  return [
    { id: 's1', name: 'Momentum — AAPL/NVDA', status: 'active', regime: 'bull', dailyPnl: 4820, signalConfidence: 82 },
    { id: 's2', name: 'Mean Reversion — SPY', status: 'active', regime: 'sideways', dailyPnl: -1230, signalConfidence: 61 },
    { id: 's3', name: 'Pairs — MSFT/GOOGL', status: 'paper', regime: 'sideways', dailyPnl: 890, signalConfidence: 74 },
    { id: 's4', name: 'Trend Following — NVDA', status: 'active', regime: 'bull', dailyPnl: 9340, signalConfidence: 91 },
    { id: 's5', name: 'Stat Arb — JPM/V', status: 'halted', regime: 'bear', dailyPnl: -2450, signalConfidence: 38 },
  ];
}

function buildMockRegimeTimeline(): RegimePoint[] {
  // ~12 months of daily data
  const points: RegimePoint[] = [];
  const DAYS = 252;
  const now = Math.floor(Date.now() / 1000);
  const DAY_S = 86400;
  const startTime = now - DAYS * DAY_S;

  const regimeSequence: Array<{ regime: Regime; len: number }> = [
    { regime: 'bull', len: 60 },
    { regime: 'sideways', len: 40 },
    { regime: 'bull', len: 70 },
    { regime: 'bear', len: 30 },
    { regime: 'sideways', len: 20 },
    { regime: 'bull', len: 32 },
  ];

  let equity = 1_000_000;
  let dayIdx = 0;

  for (const seg of regimeSequence) {
    const drift = seg.regime === 'bull' ? 0.0008 : seg.regime === 'bear' ? -0.001 : 0.0001;
    const vol = seg.regime === 'sideways' ? 0.007 : 0.012;
    for (let d = 0; d < seg.len && dayIdx < DAYS; d++, dayIdx++) {
      const ret = drift + (Math.random() * 2 - 1) * vol;
      equity *= 1 + ret;
      points.push({
        time: startTime + dayIdx * DAY_S,
        regime: seg.regime,
        equityValue: equity,
      });
    }
  }

  return points;
}

// ── Status badge ──────────────────────────────────────────────────────────────

function StatusBadge({ status }: { status: StrategyStatus }) {
  const cfg: Record<StrategyStatus, { label: string; className: string }> = {
    active: { label: 'ACTIVE', className: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/40' },
    halted: { label: 'HALTED', className: 'bg-red-500/20 text-red-400 border-red-500/40' },
    paper: { label: 'PAPER', className: 'bg-amber-500/20 text-amber-400 border-amber-500/40' },
  };
  const { label, className } = cfg[status];
  return (
    <span className={`text-xs font-mono font-semibold px-2 py-0.5 rounded border ${className}`}>
      {label}
    </span>
  );
}

// ── Regime badge ──────────────────────────────────────────────────────────────

function RegimeBadge({ regime }: { regime: Regime }) {
  const cfg: Record<Regime, { label: string; className: string }> = {
    bull: { label: '↑ BULL', className: 'bg-sky-500/20 text-sky-300 border-sky-500/40' },
    bear: { label: '↓ BEAR', className: 'bg-red-500/20 text-red-400 border-red-500/40' },
    sideways: { label: '→ SIDE', className: 'bg-slate-500/20 text-slate-400 border-slate-500/40' },
  };
  const { label, className } = cfg[regime];
  return (
    <span className={`text-xs font-mono font-semibold px-2 py-0.5 rounded border ${className}`}>
      {label}
    </span>
  );
}

// ── Signal confidence bar ─────────────────────────────────────────────────────

function ConfidenceBar({ value }: { value: number }) {
  const color = value >= 75 ? 'bg-emerald-500' : value >= 50 ? 'bg-amber-500' : 'bg-red-500';
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 bg-slate-700/70 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${value}%` }} />
      </div>
      <span className="text-xs font-mono text-slate-400 w-8 text-right">{value}%</span>
    </div>
  );
}

// ── Strategy card ─────────────────────────────────────────────────────────────

interface StrategyCardProps {
  strategy: StrategyState;
  onToggle: (id: string) => void;
}

function StrategyCard({ strategy, onToggle }: StrategyCardProps) {
  const pnlPos = strategy.dailyPnl >= 0;
  return (
    <div className="bg-slate-800/60 border border-slate-700/50 rounded-lg p-4 flex flex-col gap-3">
      <div className="flex items-start justify-between gap-2">
        <div>
          <div className="text-sm font-semibold text-slate-200 font-mono leading-tight">
            {strategy.name}
          </div>
          <div className="flex items-center gap-2 mt-1.5">
            <StatusBadge status={strategy.status} />
            <RegimeBadge regime={strategy.regime} />
          </div>
        </div>
        <div className={`text-right font-mono ${pnlPos ? 'text-emerald-400' : 'text-red-400'}`}>
          <div className="text-xs text-slate-500 uppercase tracking-wide">Daily P&L</div>
          <div className="text-base font-semibold">
            {pnlPos ? '+' : ''}
            {strategy.dailyPnl >= 0
              ? `$${strategy.dailyPnl.toLocaleString()}`
              : `-$${Math.abs(strategy.dailyPnl).toLocaleString()}`}
          </div>
        </div>
      </div>

      <div>
        <div className="text-xs text-slate-500 uppercase tracking-wide mb-1">Signal Confidence</div>
        <ConfidenceBar value={strategy.signalConfidence} />
      </div>

      <div className="border-t border-slate-700/50 pt-2">
        <button
          onClick={() => onToggle(strategy.id)}
          disabled={strategy.status === 'halted'}
          className={`text-xs font-mono px-3 py-1 rounded border transition-colors ${
            strategy.status === 'halted'
              ? 'border-slate-700 text-slate-600 cursor-not-allowed'
              : strategy.status === 'active'
                ? 'border-red-500/40 text-red-400 hover:bg-red-500/10'
                : 'border-emerald-500/40 text-emerald-400 hover:bg-emerald-500/10'
          }`}
        >
          {strategy.status === 'active' ? 'Halt Strategy' : strategy.status === 'paper' ? 'Go Live' : 'Halted'}
        </button>
      </div>
    </div>
  );
}

// ── Regime timeline chart ─────────────────────────────────────────────────────

const REGIME_COLORS: Record<Regime, string> = {
  bull: '#38bdf8',
  bear: '#f87171',
  sideways: '#94a3b8',
};

interface RegimeChartProps {
  points: RegimePoint[];
}

function RegimeChart({ points }: RegimeChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const equitySeriesRef = useRef<ISeriesApi<'Line'> | null>(null);

  useEffect(() => {
    if (!containerRef.current || points.length === 0) return;

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

    // Color each segment by regime
    const equitySeries = chart.addLineSeries({
      lineWidth: 2,
      color: '#38bdf8',
      priceFormat: { type: 'custom', formatter: (v: number) => `$${(v / 1000).toFixed(0)}k`, minMove: 1 },
    });

    // Build colored area segments by splitting the data by regime change
    let segStart = 0;
    for (let i = 1; i <= points.length; i++) {
      const regimeChanged = i === points.length || points[i].regime !== points[segStart].regime;
      if (regimeChanged) {
        const segment = points.slice(segStart, i);
        const color = REGIME_COLORS[points[segStart].regime];
        const seg = chart.addAreaSeries({
          lineColor: color,
          topColor: color + '28',
          bottomColor: color + '00',
          lineWidth: 0,
          crosshairMarkerVisible: false,
          priceLineVisible: false,
          lastValueVisible: false,
        });
        seg.setData(segment.map((p) => ({ time: p.time as Time, value: p.equityValue })));
        segStart = i;
      }
    }

    equitySeriesRef.current = equitySeries;
    equitySeries.setData(points.map((p) => ({ time: p.time as Time, value: p.equityValue })));
    chart.timeScale().fitContent();
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
      equitySeriesRef.current = null;
    };
  }, [points]);

  return (
    <div className="bg-slate-800/60 rounded-lg border border-slate-700/50 overflow-hidden">
      <div className="px-4 py-2.5 border-b border-slate-700/50 flex items-center justify-between">
        <span className="text-sm font-semibold text-slate-300">Regime Timeline</span>
        <div className="flex items-center gap-4 text-xs">
          {Object.entries(REGIME_COLORS).map(([regime, color]) => (
            <span key={regime} className="flex items-center gap-1.5 text-slate-400">
              <span className="inline-block w-3 h-3 rounded-sm" style={{ background: color + '55', border: `1px solid ${color}` }} />
              {regime}
            </span>
          ))}
        </div>
      </div>
      <div ref={containerRef} className="w-full" style={{ height: 240 }} />
    </div>
  );
}

// ── Confirmation modal ────────────────────────────────────────────────────────

interface ConfirmModalProps {
  strategyName: string;
  action: 'halt' | 'live';
  onConfirm: () => void;
  onCancel: () => void;
}

function ConfirmModal({ strategyName, action, onConfirm, onCancel }: ConfirmModalProps) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="bg-slate-800 border border-slate-600 rounded-xl p-6 w-80 shadow-2xl">
        <h3 className="text-base font-semibold text-slate-100 mb-2">
          {action === 'halt' ? 'Halt Strategy?' : 'Go Live?'}
        </h3>
        <p className="text-sm text-slate-400 mb-5">
          {action === 'halt'
            ? `This will immediately halt "${strategyName}" and cancel any open orders.`
            : `This will switch "${strategyName}" from paper trading to live execution.`}
        </p>
        <div className="flex gap-3 justify-end">
          <button
            onClick={onCancel}
            className="px-4 py-1.5 text-sm text-slate-400 hover:text-slate-200 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            className={`px-4 py-1.5 text-sm font-semibold rounded transition-colors ${
              action === 'halt'
                ? 'bg-red-600 hover:bg-red-500 text-white'
                : 'bg-emerald-600 hover:bg-emerald-500 text-white'
            }`}
          >
            {action === 'halt' ? 'Halt' : 'Go Live'}
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export function StrategyMonitorPage() {
  const [strategies, setStrategies] = useState<StrategyState[]>(buildMockStrategies);
  const [regimePoints] = useState<RegimePoint[]>(buildMockRegimeTimeline);
  const [pendingToggle, setPendingToggle] = useState<{ id: string; action: 'halt' | 'live' } | null>(null);

  const handleToggle = (id: string) => {
    const strat = strategies.find((s) => s.id === id);
    if (!strat || strat.status === 'halted') return;
    setPendingToggle({ id, action: strat.status === 'active' ? 'halt' : 'live' });
  };

  const handleConfirm = () => {
    if (!pendingToggle) return;
    setStrategies((prev) =>
      prev.map((s) => {
        if (s.id !== pendingToggle.id) return s;
        return { ...s, status: pendingToggle.action === 'halt' ? 'halted' : 'active' };
      }),
    );
    setPendingToggle(null);
  };

  const pendingStrategy = pendingToggle ? strategies.find((s) => s.id === pendingToggle.id) : null;

  const totalDailyPnl = strategies.reduce((s, r) => s + r.dailyPnl, 0);
  const activeCount = strategies.filter((s) => s.status === 'active').length;

  return (
    <div className="flex flex-col gap-4 p-4 min-h-0 overflow-y-auto">
      {/* Header stats */}
      <div className="flex items-center gap-4 flex-wrap">
        <div className="bg-slate-800/60 border border-slate-700/50 rounded-lg px-4 py-2.5 flex flex-col gap-0.5">
          <span className="text-xs text-slate-500 uppercase tracking-wide">Active Strategies</span>
          <span className="font-mono text-lg font-semibold text-slate-100">{activeCount} / {strategies.length}</span>
        </div>
        <div className="bg-slate-800/60 border border-slate-700/50 rounded-lg px-4 py-2.5 flex flex-col gap-0.5">
          <span className="text-xs text-slate-500 uppercase tracking-wide">Total Daily P&L</span>
          <span className={`font-mono text-lg font-semibold ${totalDailyPnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
            {totalDailyPnl >= 0 ? '+' : ''}${totalDailyPnl.toLocaleString()}
          </span>
        </div>
        <div className="ml-auto">
          <span className="text-xs text-slate-600 bg-slate-800/60 border border-slate-700/50 rounded px-2 py-1 font-mono">
            mock data · wiring blocked on QUA-22
          </span>
        </div>
      </div>

      {/* Strategy cards grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-3">
        {strategies.map((s) => (
          <StrategyCard key={s.id} strategy={s} onToggle={handleToggle} />
        ))}
      </div>

      {/* Regime timeline */}
      <RegimeChart points={regimePoints} />

      {/* Confirmation modal */}
      {pendingToggle && pendingStrategy && (
        <ConfirmModal
          strategyName={pendingStrategy.name}
          action={pendingToggle.action}
          onConfirm={handleConfirm}
          onCancel={() => setPendingToggle(null)}
        />
      )}
    </div>
  );
}
