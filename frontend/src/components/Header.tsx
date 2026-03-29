interface HeaderProps {
  connected: boolean;
  totalValue: number;
  dailyPnl: number;
  dailyPnlPercent: number;
}

export function Header({ connected, totalValue, dailyPnl, dailyPnlPercent }: HeaderProps) {
  const pnlColor = dailyPnl >= 0 ? 'text-gain' : 'text-loss';
  const pnlSign = dailyPnl >= 0 ? '+' : '';

  return (
    <header className="flex items-center justify-between px-6 py-3 bg-surface-800 border-b border-slate-700/50">
      <div className="flex items-center gap-4">
        <h1 className="text-lg font-semibold tracking-tight">
          <span className="text-accent">QUA</span> Trading
        </h1>
        <div className="flex items-center gap-1.5">
          <div className={`w-2 h-2 rounded-full ${connected ? 'bg-gain animate-pulse' : 'bg-loss'}`} />
          <span className="text-xs text-slate-400">{connected ? 'Live' : 'Disconnected'}</span>
        </div>
      </div>
      <div className="flex items-center gap-8">
        <div className="text-right">
          <div className="text-xs text-slate-400 uppercase tracking-wider">Portfolio Value</div>
          <div className="text-lg font-mono font-semibold">${totalValue.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</div>
        </div>
        <div className="text-right">
          <div className="text-xs text-slate-400 uppercase tracking-wider">Day P&L</div>
          <div className={`text-lg font-mono font-semibold ${pnlColor}`}>
            {pnlSign}${Math.abs(dailyPnl).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            <span className="text-sm ml-1">({pnlSign}{dailyPnlPercent.toFixed(2)}%)</span>
          </div>
        </div>
      </div>
    </header>
  );
}
