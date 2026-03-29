import { useState } from 'react';

export type View = 'trading' | 'portfolio' | 'market-data' | 'trade-blotter' | 'backtest' | 'analytics' | 'strategy-monitor' | 'risk';

interface NavItem {
  id: View;
  label: string;
  icon: React.ReactNode;
  disabled?: boolean;
}

interface SidebarProps {
  view: View;
  onViewChange: (v: View) => void;
}

function TradingIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="22 7 13.5 15.5 8.5 10.5 2 17" />
      <polyline points="16 7 22 7 22 13" />
    </svg>
  );
}

function PortfolioIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="2" y="7" width="20" height="14" rx="2" />
      <path d="M16 7V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v2" />
      <line x1="12" y1="12" x2="12" y2="16" />
      <line x1="10" y1="14" x2="14" y2="14" />
    </svg>
  );
}

function MarketDataIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="3" width="18" height="18" rx="2" />
      <path d="M8 12l3 3 5-5" />
    </svg>
  );
}

function BlotterIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M9 5H7a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2h-2" />
      <rect x="9" y="3" width="6" height="4" rx="1" />
      <line x1="9" y1="12" x2="15" y2="12" />
      <line x1="9" y1="16" x2="13" y2="16" />
    </svg>
  );
}

function BacktestIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="1 4 1 10 7 10" />
      <path d="M3.51 15a9 9 0 1 0 .49-3.51" />
    </svg>
  );
}

function AnalyticsIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="18" y1="20" x2="18" y2="10" />
      <line x1="12" y1="20" x2="12" y2="4" />
      <line x1="6" y1="20" x2="6" y2="14" />
    </svg>
  );
}

function StrategyIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="3" />
      <path d="M19.07 4.93a10 10 0 0 1 0 14.14M4.93 4.93a10 10 0 0 0 0 14.14" />
    </svg>
  );
}

function RiskIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
      <line x1="12" y1="9" x2="12" y2="13" />
      <line x1="12" y1="17" x2="12.01" y2="17" />
    </svg>
  );
}

const NAV_ITEMS: NavItem[] = [
  { id: 'trading', label: 'Live Trading', icon: <TradingIcon /> },
  { id: 'portfolio', label: 'Portfolio', icon: <PortfolioIcon /> },
  { id: 'market-data', label: 'Market Data', icon: <MarketDataIcon /> },
  { id: 'trade-blotter', label: 'Trade Blotter', icon: <BlotterIcon /> },
  { id: 'backtest', label: 'Backtest', icon: <BacktestIcon /> },
  { id: 'analytics', label: 'Analytics', icon: <AnalyticsIcon /> },
  { id: 'strategy-monitor', label: 'Strategy Monitor', icon: <StrategyIcon /> },
  { id: 'risk', label: 'Risk', icon: <RiskIcon /> },
];

export function Sidebar({ view, onViewChange }: SidebarProps) {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <aside
      className={`flex flex-col flex-shrink-0 bg-surface-800 border-r border-slate-700/50 transition-all duration-200 ${
        collapsed ? 'w-14' : 'w-52'
      }`}
    >
      <button
        onClick={() => setCollapsed((c) => !c)}
        className="flex items-center justify-center h-10 border-b border-slate-700/50 text-slate-500 hover:text-slate-300 transition-colors flex-shrink-0"
        title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
      >
        <svg
          width="16"
          height="16"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          className={`transition-transform duration-200 ${collapsed ? 'rotate-180' : ''}`}
        >
          <polyline points="15 18 9 12 15 6" />
        </svg>
      </button>

      <nav className="flex-1 py-2 overflow-y-auto">
        {NAV_ITEMS.map((item) => {
          const isActive = item.id === view;
          const isClickable = !item.disabled;

          return (
            <button
              key={item.id}
              onClick={() => isClickable && onViewChange(item.id as View)}
              disabled={item.disabled}
              title={collapsed ? item.label : undefined}
              className={`
                w-full flex items-center gap-3 px-4 py-2.5 text-sm transition-colors relative
                ${isActive ? 'text-sky-400 bg-sky-500/10' : ''}
                ${item.disabled ? 'text-slate-600 cursor-not-allowed' : isActive ? '' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700/30'}
              `}
            >
              {isActive && (
                <div className="absolute left-0 top-1 bottom-1 w-0.5 bg-sky-500 rounded-r" />
              )}
              <span className="flex-shrink-0">{item.icon}</span>
              {!collapsed && (
                <span className="flex-1 text-left truncate font-medium">{item.label}</span>
              )}
            </button>
          );
        })}
      </nav>
    </aside>
  );
}
