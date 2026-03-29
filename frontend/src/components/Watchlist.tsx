import type { Quote } from '../types';

interface WatchlistProps {
  quotes: Map<string, Quote>;
  selectedSymbol: string;
  onSelect: (symbol: string) => void;
}

export function Watchlist({ quotes, selectedSymbol, onSelect }: WatchlistProps) {
  const sorted = Array.from(quotes.values()).sort((a, b) => a.symbol.localeCompare(b.symbol));

  return (
    <div className="bg-surface-800 rounded-lg border border-slate-700/50 overflow-hidden">
      <div className="px-4 py-2.5 border-b border-slate-700/50">
        <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">Watchlist</h2>
      </div>
      <div className="divide-y divide-slate-700/30 max-h-[calc(100vh-420px)] overflow-y-auto">
        {sorted.length === 0 && (
          <div className="px-4 py-8 text-center text-slate-500 text-sm">Connecting…</div>
        )}
        {sorted.map((q) => {
          const isSelected = q.symbol === selectedSymbol;
          const pnlColor = q.change >= 0 ? 'text-gain' : 'text-loss';
          const sign = q.change >= 0 ? '+' : '';

          return (
            <button
              key={q.symbol}
              onClick={() => onSelect(q.symbol)}
              className={`w-full px-4 py-2.5 flex items-center justify-between hover:bg-surface-700/50 transition-colors ${
                isSelected ? 'bg-accent/10 border-l-2 border-accent' : 'border-l-2 border-transparent'
              }`}
            >
              <div>
                <div className="font-semibold text-sm">{q.symbol}</div>
                <div className="text-xs text-slate-400">
                  Vol {(q.volume / 1_000_000).toFixed(1)}M
                </div>
              </div>
              <div className="text-right">
                <div className="font-mono text-sm font-medium">${q.price.toFixed(2)}</div>
                <div className={`font-mono text-xs ${pnlColor}`}>
                  {sign}{q.change.toFixed(2)} ({sign}{q.changePercent.toFixed(2)}%)
                </div>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
