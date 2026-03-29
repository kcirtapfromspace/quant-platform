import { useEffect, useRef } from 'react';
import { createChart, IChartApi, ISeriesApi } from 'lightweight-charts';
import { usePortfolioHistory } from '../hooks/useApi';
import type { Portfolio } from '../types';

// ── Equity Sparkline ──────────────────────────────────────────────────────────

interface SparklineProps {
  data: Array<{ time: number; value: number }>;
}

function EquitySparkline({ data }: SparklineProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<'Area'> | null>(null);

  useEffect(() => {
    if (!containerRef.current || data.length === 0) return;

    const chart = createChart(containerRef.current, {
      layout: { background: { color: 'transparent' }, textColor: '#94a3b8' },
      grid: { vertLines: { color: 'transparent' }, horzLines: { color: 'rgba(51,65,85,0.3)' } },
      crosshair: { mode: 0 },
      rightPriceScale: { borderColor: 'rgba(51,65,85,0.4)', scaleMargins: { top: 0.1, bottom: 0.1 } },
      timeScale: { borderColor: 'rgba(51,65,85,0.4)', timeVisible: false },
      handleScroll: false,
      handleScale: false,
    });

    const lastVal = data[data.length - 1].value;
    const firstVal = data[0].value;
    const isUp = lastVal >= firstVal;

    const series = chart.addAreaSeries({
      lineColor: isUp ? '#22c55e' : '#ef4444',
      topColor: isUp ? 'rgba(34,197,94,0.25)' : 'rgba(239,68,68,0.25)',
      bottomColor: 'rgba(0,0,0,0)',
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    });

    series.setData(
      data.map((d) => ({ time: d.time as import('lightweight-charts').UTCTimestamp, value: d.value }))
    );

    chart.timeScale().fitContent();
    chartRef.current = chart;
    seriesRef.current = series;

    const ro = new ResizeObserver(() => {
      if (containerRef.current) chart.applyOptions({ width: containerRef.current.clientWidth });
    });
    ro.observe(containerRef.current);

    return () => {
      ro.disconnect();
      chart.remove();
    };
  }, [data]);

  return <div ref={containerRef} className="w-full h-full" />;
}

// ── Exposure Bar Chart ────────────────────────────────────────────────────────

const PALETTE = ['#3b82f6', '#22c55e', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316', '#14b8a6'];

function ExposureChart({ portfolio }: { portfolio: Portfolio }) {
  const positions = portfolio.positions;
  if (positions.length === 0) {
    return <div className="text-slate-500 text-sm text-center py-8">No positions</div>;
  }

  const total = positions.reduce((sum, p) => sum + p.marketValue, 0);
  const sorted = [...positions].sort((a, b) => b.marketValue - a.marketValue);

  return (
    <div className="space-y-2">
      {sorted.map((p, i) => {
        const pct = total > 0 ? (p.marketValue / total) * 100 : 0;
        return (
          <div key={p.symbol} className="flex items-center gap-3">
            <span className="w-12 text-xs font-semibold text-slate-300 text-right">{p.symbol}</span>
            <div className="flex-1 h-5 bg-slate-700/40 rounded overflow-hidden">
              <div
                className="h-full rounded transition-all duration-500"
                style={{ width: `${pct}%`, backgroundColor: PALETTE[i % PALETTE.length] }}
              />
            </div>
            <span className="w-12 text-xs font-mono text-slate-400 text-right">{pct.toFixed(1)}%</span>
          </div>
        );
      })}
    </div>
  );
}

// ── Summary Stat Card ─────────────────────────────────────────────────────────

function StatCard({ label, value, subValue, valueClass = 'text-slate-100' }: {
  label: string;
  value: string;
  subValue?: string;
  valueClass?: string;
}) {
  return (
    <div className="bg-surface-800 rounded-lg border border-slate-700/50 px-4 py-3">
      <div className="text-xs text-slate-400 uppercase tracking-wider mb-1">{label}</div>
      <div className={`text-xl font-mono font-semibold ${valueClass}`}>{value}</div>
      {subValue && <div className={`text-xs font-mono mt-0.5 ${valueClass}`}>{subValue}</div>}
    </div>
  );
}

// ── Portfolio Dashboard ───────────────────────────────────────────────────────

interface PortfolioDashboardProps {
  portfolio: Portfolio | null;
}

export function PortfolioDashboard({ portfolio }: PortfolioDashboardProps) {
  const { data: historyData, isLoading, isError } = usePortfolioHistory();

  const pnl = portfolio?.dailyPnl ?? 0;
  const pnlPct = portfolio?.dailyPnlPercent ?? 0;
  const pnlSign = pnl >= 0 ? '+' : '';
  const pnlClass = pnl >= 0 ? 'text-gain' : 'text-loss';

  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-4">
      <div className="grid grid-cols-2 xl:grid-cols-4 gap-3">
        <StatCard
          label="Total Value"
          value={`$${(portfolio?.totalValue ?? 0).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
        />
        <StatCard
          label="Cash"
          value={`$${(portfolio?.cash ?? 0).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
        />
        <StatCard
          label="Daily P&L"
          value={`${pnlSign}$${Math.abs(pnl).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
          subValue={`${pnlSign}${pnlPct.toFixed(2)}%`}
          valueClass={pnlClass}
        />
        <StatCard
          label="Equity"
          value={`$${(portfolio?.equity ?? 0).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
        />
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-3">
        <div className="xl:col-span-2 bg-surface-800 rounded-lg border border-slate-700/50 p-4">
          <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider mb-3">Equity Curve (Daily NAV)</h2>
          <div className="h-48">
            {isLoading && (
              <div className="h-full flex items-center justify-center text-slate-500 text-sm">Loading…</div>
            )}
            {isError && (
              <div className="h-full flex items-center justify-center text-red-400 text-sm">Failed to load history</div>
            )}
            {historyData && historyData.length > 0 && (
              <EquitySparkline data={historyData} />
            )}
          </div>
        </div>

        <div className="bg-surface-800 rounded-lg border border-slate-700/50 p-4">
          <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider mb-3">Exposure by Symbol</h2>
          {portfolio ? (
            <ExposureChart portfolio={portfolio} />
          ) : (
            <div className="text-slate-500 text-sm text-center py-8">Waiting for data…</div>
          )}
        </div>
      </div>

      <div className="bg-surface-800 rounded-lg border border-slate-700/50 overflow-hidden">
        <div className="px-4 py-2.5 border-b border-slate-700/50">
          <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">Positions</h2>
        </div>
        {!portfolio || portfolio.positions.length === 0 ? (
          <div className="px-4 py-8 text-center text-slate-500 text-sm">No open positions</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-xs text-slate-400 uppercase tracking-wider border-b border-slate-700/30">
                  <th className="px-4 py-2 text-left font-medium">Symbol</th>
                  <th className="px-4 py-2 text-right font-medium">Qty</th>
                  <th className="px-4 py-2 text-right font-medium">Avg Cost</th>
                  <th className="px-4 py-2 text-right font-medium">Price</th>
                  <th className="px-4 py-2 text-right font-medium">Mkt Value</th>
                  <th className="px-4 py-2 text-right font-medium">Unr. P&L</th>
                  <th className="px-4 py-2 text-right font-medium">P&L %</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-700/20">
                {portfolio.positions.map((p) => {
                  const sign = p.unrealizedPnl >= 0 ? '+' : '';
                  const cls = p.unrealizedPnl >= 0 ? 'text-gain' : 'text-loss';
                  return (
                    <tr key={p.symbol} className="hover:bg-surface-700/30 transition-colors">
                      <td className="px-4 py-2 font-semibold">{p.symbol}</td>
                      <td className="px-4 py-2 text-right font-mono">{p.quantity.toLocaleString()}</td>
                      <td className="px-4 py-2 text-right font-mono">${p.avgCost.toFixed(2)}</td>
                      <td className="px-4 py-2 text-right font-mono">${p.currentPrice.toFixed(2)}</td>
                      <td className="px-4 py-2 text-right font-mono">${p.marketValue.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
                      <td className={`px-4 py-2 text-right font-mono ${cls}`}>
                        {sign}${Math.abs(p.unrealizedPnl).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                      </td>
                      <td className={`px-4 py-2 text-right font-mono ${cls}`}>
                        {sign}{p.unrealizedPnlPercent.toFixed(2)}%
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
