import { useEffect, useState, useMemo } from 'react';
import type { Order } from '../types';

// ── Helpers ──────────────────────────────────────────────────────────────────

function computeSlippage(order: Order): number | null {
  if (order.status !== 'filled' || order.fillPrice == null || order.limitPrice == null) return null;
  return ((order.fillPrice - order.limitPrice) / order.limitPrice) * 10_000;
}

function fmtTime(ts: number): string {
  return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function fmtDate(ts: number): string {
  return new Date(ts).toLocaleDateString([], { month: 'short', day: '2-digit' });
}

// ── Mock data ─────────────────────────────────────────────────────────────────

function generateMockOrders(): Order[] {
  const symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META'];
  const statuses: Order['status'][] = ['filled', 'filled', 'filled', 'filled', 'cancelled', 'pending'];
  const sides: Order['side'][] = ['buy', 'sell'];
  const types: Order['type'][] = ['market', 'limit', 'limit'];
  const now = Date.now();
  const MS = 60_000;

  return Array.from({ length: 60 }, (_, i) => {
    const symbol = symbols[i % symbols.length];
    const side = sides[i % 2];
    const type = types[i % 3];
    const status = statuses[Math.floor((i * 17 + 3) % statuses.length)];
    const base = 100 + (i % 7) * 22.5;
    const limitPrice = type === 'limit' ? Math.round(base * 100) / 100 : undefined;
    // Realistic slippage: normally distributed around 0 with sigma ~4 bps
    const slipBps = ((((i * 1.618 + 2.71) % 1) - 0.5) * 2) * 8;
    const fillPrice =
      status === 'filled'
        ? Math.round((limitPrice ?? base) * (1 + slipBps / 10_000) * 100) / 100
        : undefined;
    return {
      id: `ord-${String(i + 1).padStart(3, '0')}`,
      symbol,
      side,
      type,
      quantity: 10 + ((i * 7) % 91),
      limitPrice,
      fillPrice,
      status,
      createdAt: now - (60 - i) * 8 * MS,
      filledAt: status === 'filled' ? now - (60 - i) * 8 * MS + 3_000 : undefined,
    };
  });
}

// ── Slippage histogram ────────────────────────────────────────────────────────

function normalPDF(x: number, mean: number, std: number): number {
  if (std === 0) return 0;
  return Math.exp(-0.5 * ((x - mean) / std) ** 2) / (std * Math.sqrt(2 * Math.PI));
}

function SlippageHistogram({ values }: { values: number[] }) {
  if (values.length < 3) {
    return (
      <div className="flex items-center justify-center h-full text-slate-500 text-xs">
        Not enough filled limit orders
      </div>
    );
  }

  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length;
  const std = Math.sqrt(variance) || 1;

  const numBins = Math.min(20, Math.max(8, Math.ceil(Math.sqrt(values.length))));
  const minVal = Math.min(...values);
  const maxVal = Math.max(...values);
  const range = maxVal - minVal || 1;
  const binWidth = range / numBins;

  const bins = Array.from({ length: numBins }, (_, i) => ({
    x0: minVal + i * binWidth,
    x1: minVal + (i + 1) * binWidth,
    count: 0,
  }));
  for (const v of values) {
    const idx = Math.min(Math.floor((v - minVal) / binWidth), numBins - 1);
    bins[idx].count++;
  }
  const maxCount = Math.max(...bins.map((b) => b.count), 1);

  const W = 480;
  const H = 140;
  const PAD = { top: 8, right: 16, bottom: 28, left: 32 };
  const iW = W - PAD.left - PAD.right;
  const iH = H - PAD.top - PAD.bottom;

  const xS = (v: number) => ((v - minVal) / range) * iW;
  const yS = (c: number) => iH - (c / maxCount) * iH;

  const nSteps = 80;
  const normalPts: [number, number][] = Array.from({ length: nSteps + 1 }, (_, i) => {
    const x = minVal + (i / nSteps) * range;
    const y = normalPDF(x, mean, std) * values.length * binWidth;
    return [xS(x), yS((y / maxCount) * maxCount)];
  });
  const pathD = normalPts.map(([x, y], i) => `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`).join(' ');

  // X-axis tick labels: min, 0 (if in range), max
  const xTicks: number[] = [minVal, maxVal];
  if (minVal < 0 && maxVal > 0) xTicks.splice(1, 0, 0);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-full" style={{ minHeight: 100 }}>
      <g transform={`translate(${PAD.left},${PAD.top})`}>
        {bins.map((bin, i) => {
          const x = xS(bin.x0);
          const bW = Math.max(iW / numBins - 1, 1);
          const bH = (bin.count / maxCount) * iH;
          const mid = (bin.x0 + bin.x1) / 2;
          const color = mid < 0 ? 'rgba(34,197,94,0.55)' : mid > 0 ? 'rgba(239,68,68,0.55)' : 'rgba(148,163,184,0.4)';
          return <rect key={i} x={x} y={iH - bH} width={bW} height={bH} fill={color} />;
        })}

        <path d={pathD} fill="none" stroke="#94a3b8" strokeWidth="1.5" />

        {minVal < 0 && maxVal > 0 && (
          <line x1={xS(0)} y1={0} x2={xS(0)} y2={iH} stroke="#475569" strokeWidth="1" strokeDasharray="3 3" />
        )}

        <line x1={0} y1={iH} x2={iW} y2={iH} stroke="#334155" />
        <line x1={0} y1={0} x2={0} y2={iH} stroke="#334155" />

        {xTicks.map((v) => (
          <text key={v} x={xS(v)} y={iH + 14} textAnchor="middle" fontSize="9" fill="#64748b">
            {v.toFixed(1)}
          </text>
        ))}

        <text x={-4} y={5} textAnchor="end" fontSize="9" fill="#64748b">{maxCount}</text>
        <text x={-4} y={iH} textAnchor="end" fontSize="9" fill="#64748b">0</text>

        <text x={iW / 2} y={iH + 24} textAnchor="middle" fontSize="9" fill="#64748b">Slippage (bps)</text>
      </g>
    </svg>
  );
}

// ── Metrics bar ───────────────────────────────────────────────────────────────

interface MetricProps {
  label: string;
  value: string;
  sub?: string;
  color?: string;
}

function Metric({ label, value, sub, color }: MetricProps) {
  return (
    <div className="flex flex-col gap-0.5 px-4 py-3 bg-surface-800 rounded-lg border border-slate-700/50 min-w-0">
      <span className="text-xs text-slate-500 uppercase tracking-wider truncate">{label}</span>
      <span className={`text-xl font-mono font-semibold ${color ?? 'text-slate-100'}`}>{value}</span>
      {sub && <span className="text-xs text-slate-500">{sub}</span>}
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

interface TradeBlotterPageProps {
  lastFill: Order | null;
}

type FilterSide = 'all' | 'buy' | 'sell';
type FilterStatus = 'all' | 'filled' | 'pending' | 'cancelled';

export function TradeBlotterPage({ lastFill }: TradeBlotterPageProps) {
  const [orders, setOrders] = useState<Order[]>([]);
  const [filterSymbol, setFilterSymbol] = useState('');
  const [filterSide, setFilterSide] = useState<FilterSide>('all');
  const [filterStatus, setFilterStatus] = useState<FilterStatus>('all');
  const [filterDateFrom, setFilterDateFrom] = useState('');
  const [filterDateTo, setFilterDateTo] = useState('');

  // Load historical orders
  useEffect(() => {
    async function load() {
      try {
        const res = await fetch('/api/orders');
        const live: Order[] = res.ok ? await res.json() : [];
        const mock = generateMockOrders();
        // Merge live on top of mock; dedup by id
        const map = new Map<string, Order>();
        for (const o of mock) map.set(o.id, o);
        for (const o of live) map.set(o.id, o);
        setOrders([...map.values()].sort((a, b) => b.createdAt - a.createdAt));
      } catch {
        setOrders(generateMockOrders());
      }
    }
    load();
  }, []);

  // Append live fills
  useEffect(() => {
    if (!lastFill) return;
    setOrders((prev) => {
      const idx = prev.findIndex((o) => o.id === lastFill.id);
      if (idx >= 0) {
        const next = [...prev];
        next[idx] = lastFill;
        return next;
      }
      return [lastFill, ...prev];
    });
  }, [lastFill]);

  // Filtered view
  const filtered = useMemo(() => {
    const sym = filterSymbol.trim().toUpperCase();
    const fromMs = filterDateFrom ? new Date(filterDateFrom).getTime() : 0;
    const toMs = filterDateTo ? new Date(filterDateTo + 'T23:59:59').getTime() : Infinity;
    return orders.filter((o) => {
      if (sym && !o.symbol.includes(sym)) return false;
      if (filterSide !== 'all' && o.side !== filterSide) return false;
      if (filterStatus !== 'all' && o.status !== filterStatus) return false;
      if (o.createdAt < fromMs || o.createdAt > toMs) return false;
      return true;
    });
  }, [orders, filterSymbol, filterSide, filterStatus, filterDateFrom, filterDateTo]);

  // Metrics
  const metrics = useMemo(() => {
    const filled = filtered.filter((o) => o.status === 'filled');
    const cancelled = filtered.filter((o) => o.status === 'cancelled');
    const slippages = filtered
      .map(computeSlippage)
      .filter((s): s is number => s !== null);

    const avgSlippage = slippages.length > 0 ? slippages.reduce((a, b) => a + b, 0) / slippages.length : null;
    const fillRate = filtered.length > 0 ? filled.length / filtered.length : null;
    const rejectionRate = filtered.length > 0 ? cancelled.length / filtered.length : null;
    const totalVolume = filled.reduce((s, o) => s + o.quantity, 0);

    return { avgSlippage, fillRate, rejectionRate, totalVolume, slippages };
  }, [filtered]);

  const slipColor =
    metrics.avgSlippage == null
      ? undefined
      : metrics.avgSlippage < 0
        ? 'text-gain'
        : metrics.avgSlippage > 0
          ? 'text-loss'
          : undefined;

  return (
    <div className="flex-1 flex flex-col overflow-hidden p-4 gap-4">
      {/* Page header */}
      <div>
        <h1 className="text-base font-semibold text-slate-200">Trade Blotter</h1>
        <p className="text-xs text-slate-500 mt-0.5">Fill quality · slippage analysis · order flow</p>
      </div>

      {/* Metrics bar */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <Metric
          label="Avg Slippage"
          value={metrics.avgSlippage != null ? `${metrics.avgSlippage.toFixed(2)} bps` : '—'}
          sub={metrics.slippages.length > 0 ? `${metrics.slippages.length} limit fills` : 'no limit fills'}
          color={slipColor}
        />
        <Metric
          label="Fill Rate"
          value={metrics.fillRate != null ? `${(metrics.fillRate * 100).toFixed(1)}%` : '—'}
          sub={`${filtered.filter((o) => o.status === 'filled').length} / ${filtered.length} orders`}
        />
        <Metric
          label="Rejection Rate"
          value={metrics.rejectionRate != null ? `${(metrics.rejectionRate * 100).toFixed(1)}%` : '—'}
          sub={`${filtered.filter((o) => o.status === 'cancelled').length} cancelled`}
          color={metrics.rejectionRate != null && metrics.rejectionRate > 0.1 ? 'text-loss' : undefined}
        />
        <Metric
          label="Total Volume"
          value={metrics.totalVolume.toLocaleString()}
          sub="shares / contracts filled"
        />
      </div>

      {/* Slippage distribution */}
      <div className="bg-surface-800 rounded-lg border border-slate-700/50 p-4">
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
            Slippage Distribution
          </span>
          <div className="flex items-center gap-3 text-xs text-slate-500">
            <span className="flex items-center gap-1">
              <span className="inline-block w-2.5 h-2.5 rounded-sm bg-gain/55" /> Negative (favourable)
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block w-2.5 h-2.5 rounded-sm bg-loss/55" /> Positive (adverse)
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block w-6 border-t border-slate-400" /> Normal fit
            </span>
          </div>
        </div>
        <div style={{ height: 140 }}>
          <SlippageHistogram values={metrics.slippages} />
        </div>
      </div>

      {/* Filters + Table */}
      <div className="flex-1 bg-surface-800 rounded-lg border border-slate-700/50 overflow-hidden flex flex-col min-h-0">
        {/* Filters */}
        <div className="flex flex-wrap items-center gap-2 px-4 py-2.5 border-b border-slate-700/50">
          <input
            type="text"
            placeholder="Symbol…"
            value={filterSymbol}
            onChange={(e) => setFilterSymbol(e.target.value)}
            className="bg-surface-900 border border-slate-700 rounded px-2.5 py-1 text-xs text-slate-200 placeholder-slate-600 w-24 focus:outline-none focus:border-sky-500"
          />
          <select
            value={filterSide}
            onChange={(e) => setFilterSide(e.target.value as FilterSide)}
            className="bg-surface-900 border border-slate-700 rounded px-2.5 py-1 text-xs text-slate-200 focus:outline-none focus:border-sky-500"
          >
            <option value="all">All sides</option>
            <option value="buy">Buy</option>
            <option value="sell">Sell</option>
          </select>
          <select
            value={filterStatus}
            onChange={(e) => setFilterStatus(e.target.value as FilterStatus)}
            className="bg-surface-900 border border-slate-700 rounded px-2.5 py-1 text-xs text-slate-200 focus:outline-none focus:border-sky-500"
          >
            <option value="all">All statuses</option>
            <option value="filled">Filled</option>
            <option value="pending">Pending</option>
            <option value="cancelled">Cancelled</option>
          </select>
          <input
            type="date"
            value={filterDateFrom}
            onChange={(e) => setFilterDateFrom(e.target.value)}
            className="bg-surface-900 border border-slate-700 rounded px-2.5 py-1 text-xs text-slate-200 focus:outline-none focus:border-sky-500"
          />
          <span className="text-slate-600 text-xs">→</span>
          <input
            type="date"
            value={filterDateTo}
            onChange={(e) => setFilterDateTo(e.target.value)}
            className="bg-surface-900 border border-slate-700 rounded px-2.5 py-1 text-xs text-slate-200 focus:outline-none focus:border-sky-500"
          />
          {(filterSymbol || filterSide !== 'all' || filterStatus !== 'all' || filterDateFrom || filterDateTo) && (
            <button
              onClick={() => {
                setFilterSymbol('');
                setFilterSide('all');
                setFilterStatus('all');
                setFilterDateFrom('');
                setFilterDateTo('');
              }}
              className="text-xs text-slate-500 hover:text-slate-300 transition-colors"
            >
              Clear
            </button>
          )}
          <span className="ml-auto text-xs text-slate-500">{filtered.length} orders</span>
        </div>

        {/* Table */}
        <div className="flex-1 overflow-auto">
          <table className="w-full text-xs min-w-[760px]">
            <thead className="sticky top-0 bg-surface-800 z-10">
              <tr className="text-xs text-slate-500 uppercase tracking-wider border-b border-slate-700/40">
                <th className="px-4 py-2 text-left font-medium">Time</th>
                <th className="px-4 py-2 text-left font-medium">Symbol</th>
                <th className="px-4 py-2 text-left font-medium">Side</th>
                <th className="px-4 py-2 text-left font-medium">Type</th>
                <th className="px-4 py-2 text-right font-medium">Qty</th>
                <th className="px-4 py-2 text-right font-medium">Limit Px</th>
                <th className="px-4 py-2 text-right font-medium">Fill Px</th>
                <th className="px-4 py-2 text-right font-medium">Slippage</th>
                <th className="px-4 py-2 text-center font-medium">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-700/20">
              {filtered.length === 0 && (
                <tr>
                  <td colSpan={9} className="px-4 py-10 text-center text-slate-500">
                    No orders match the current filters
                  </td>
                </tr>
              )}
              {filtered.map((o) => {
                const slip = computeSlippage(o);
                const isPending = o.status === 'pending';
                const rowCls = isPending ? 'opacity-50' : '';
                return (
                  <tr key={o.id} className={`hover:bg-surface-700/30 transition-colors ${rowCls}`}>
                    <td className="px-4 py-1.5 font-mono text-slate-400 whitespace-nowrap">
                      <span className="block text-slate-500">{fmtDate(o.createdAt)}</span>
                      <span>{fmtTime(o.createdAt)}</span>
                    </td>
                    <td className="px-4 py-1.5 font-semibold text-slate-100">{o.symbol}</td>
                    <td className={`px-4 py-1.5 font-medium ${o.side === 'buy' ? 'text-gain' : 'text-loss'}`}>
                      {o.side.toUpperCase()}
                    </td>
                    <td className="px-4 py-1.5 text-slate-400 capitalize">{o.type}</td>
                    <td className="px-4 py-1.5 text-right font-mono">{o.quantity.toLocaleString()}</td>
                    <td className="px-4 py-1.5 text-right font-mono text-slate-300">
                      {o.limitPrice != null ? `$${o.limitPrice.toFixed(2)}` : <span className="text-slate-600">—</span>}
                    </td>
                    <td className="px-4 py-1.5 text-right font-mono text-slate-100">
                      {o.fillPrice != null ? `$${o.fillPrice.toFixed(2)}` : <span className="text-slate-600">—</span>}
                    </td>
                    <td className="px-4 py-1.5 text-right font-mono">
                      {slip != null ? (
                        <span className={slip < 0 ? 'text-gain' : slip > 0 ? 'text-loss' : 'text-slate-400'}>
                          {slip > 0 ? '+' : ''}
                          {slip.toFixed(2)}
                        </span>
                      ) : (
                        <span className="text-slate-600">—</span>
                      )}
                    </td>
                    <td className="px-4 py-1.5 text-center">
                      <span
                        className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${
                          o.status === 'filled'
                            ? 'bg-gain/15 text-gain'
                            : o.status === 'pending'
                              ? 'bg-yellow-500/15 text-yellow-400'
                              : 'bg-slate-500/15 text-slate-400'
                        }`}
                      >
                        {o.status}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
