import type { Portfolio } from '../types';

interface PositionsProps {
  portfolio: Portfolio | null;
}

export function Positions({ portfolio }: PositionsProps) {
  const positions = portfolio?.positions ?? [];

  return (
    <div className="bg-surface-800 rounded-lg border border-slate-700/50 overflow-hidden">
      <div className="px-4 py-2.5 border-b border-slate-700/50 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">Positions</h2>
        {portfolio && (
          <div className="flex items-center gap-4 text-xs">
            <span className="text-slate-400">
              Cash: <span className="font-mono text-slate-200">${portfolio.cash.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
            </span>
            <span className="text-slate-400">
              Equity: <span className="font-mono text-slate-200">${portfolio.equity.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
            </span>
          </div>
        )}
      </div>
      {positions.length === 0 ? (
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
                <th className="px-4 py-2 text-right font-medium">P&L</th>
                <th className="px-4 py-2 text-right font-medium">P&L %</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-700/20">
              {positions.map((p) => {
                const pnlColor = p.unrealizedPnl >= 0 ? 'text-gain' : 'text-loss';
                const sign = p.unrealizedPnl >= 0 ? '+' : '';
                return (
                  <tr key={p.symbol} className="hover:bg-surface-700/30 transition-colors">
                    <td className="px-4 py-2 font-semibold">{p.symbol}</td>
                    <td className="px-4 py-2 text-right font-mono">{p.quantity.toLocaleString()}</td>
                    <td className="px-4 py-2 text-right font-mono">${p.avgCost.toFixed(2)}</td>
                    <td className="px-4 py-2 text-right font-mono">${p.currentPrice.toFixed(2)}</td>
                    <td className="px-4 py-2 text-right font-mono">${p.marketValue.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
                    <td className={`px-4 py-2 text-right font-mono ${pnlColor}`}>
                      {sign}${Math.abs(p.unrealizedPnl).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </td>
                    <td className={`px-4 py-2 text-right font-mono ${pnlColor}`}>
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
  );
}
