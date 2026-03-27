import { useState, useCallback } from 'react';
import { useWebSocket } from './hooks/useWebSocket';
import { Header } from './components/Header';
import { Watchlist } from './components/Watchlist';
import { PriceChart } from './components/PriceChart';
import { OrderPanel } from './components/OrderPanel';
import { Positions } from './components/Positions';
import { OrderHistory } from './components/OrderHistory';
import { BacktestPage } from './pages/BacktestPage';

type View = 'trading' | 'backtest';

export default function App() {
  const { quotes, portfolio, connected } = useWebSocket();
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
  const [orderRefresh, setOrderRefresh] = useState(0);
  const [view, setView] = useState<View>('trading');

  const handleOrderPlaced = useCallback(() => {
    setOrderRefresh((n) => n + 1);
  }, []);

  const currentQuote = quotes.get(selectedSymbol);

  return (
    <div className="flex flex-col h-screen">
      <Header
        connected={connected}
        totalValue={portfolio?.totalValue ?? 1_000_000}
        dailyPnl={portfolio?.dailyPnl ?? 0}
        dailyPnlPercent={portfolio?.dailyPnlPercent ?? 0}
      />

      {/* View tabs */}
      <div className="flex border-b border-slate-700/50 bg-slate-900 px-3 flex-shrink-0">
        {(['trading', 'backtest'] as View[]).map((v) => (
          <button
            key={v}
            onClick={() => setView(v)}
            className={`px-4 py-2 text-xs font-semibold uppercase tracking-wide transition-colors border-b-2 -mb-px ${
              view === v
                ? 'border-sky-500 text-sky-400'
                : 'border-transparent text-slate-500 hover:text-slate-300'
            }`}
          >
            {v === 'trading' ? 'Live Trading' : 'Backtest'}
          </button>
        ))}
      </div>

      {view === 'trading' ? (
        <div className="flex-1 flex overflow-hidden">
          {/* Left sidebar — Watchlist */}
          <div className="w-64 flex-shrink-0 p-3 overflow-y-auto border-r border-slate-700/50">
            <Watchlist quotes={quotes} selectedSymbol={selectedSymbol} onSelect={setSelectedSymbol} />
          </div>

          {/* Main content area */}
          <div className="flex-1 flex flex-col overflow-y-auto p-3 gap-3">
            <PriceChart symbol={selectedSymbol} currentQuote={currentQuote} />
            <div className="grid grid-cols-1 gap-3">
              <Positions portfolio={portfolio} />
              <OrderHistory refreshTrigger={orderRefresh} />
            </div>
          </div>

          {/* Right sidebar — Order entry */}
          <div className="w-72 flex-shrink-0 p-3 overflow-y-auto border-l border-slate-700/50">
            <OrderPanel
              symbol={selectedSymbol}
              currentQuote={currentQuote}
              cash={portfolio?.cash ?? 1_000_000}
              onOrderPlaced={handleOrderPlaced}
            />
          </div>
        </div>
      ) : (
        <div className="flex-1 overflow-y-auto">
          <BacktestPage />
        </div>
      )}
    </div>
  );
}
