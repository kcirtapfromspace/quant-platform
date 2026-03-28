import { useState, useCallback } from 'react';
import { useWebSocket } from './hooks/useWebSocket';
import { Header } from './components/Header';
import { Sidebar, type View } from './components/Sidebar';
import { Watchlist } from './components/Watchlist';
import { PriceChart } from './components/PriceChart';
import { OrderPanel } from './components/OrderPanel';
import { Positions } from './components/Positions';
import { OrderHistory } from './components/OrderHistory';
import { BacktestPage } from './pages/BacktestPage';
import { PortfolioDashboard } from './pages/PortfolioDashboard';
import { MarketDataPage } from './pages/MarketDataPage';

function PlaceholderPage({ title }: { title: string }) {
  return (
    <div className="flex-1 flex items-center justify-center text-slate-500">
      <div className="text-center">
        <div className="text-4xl mb-3">🚧</div>
        <div className="text-lg font-semibold text-slate-400">{title}</div>
        <div className="text-sm mt-1">Coming soon</div>
      </div>
    </div>
  );
}

export default function App() {
  const { quotes, portfolio, lastOhlcv, orderBooks, connected } = useWebSocket();
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
  const [orderRefresh, setOrderRefresh] = useState(0);
  const [view, setView] = useState<View>('trading');

  const handleOrderPlaced = useCallback(() => {
    setOrderRefresh((n) => n + 1);
  }, []);

  const currentQuote = quotes.get(selectedSymbol);

  return (
    <div className="flex flex-col h-screen bg-surface-900 text-slate-100">
      <Header
        connected={connected}
        totalValue={portfolio?.totalValue ?? 1_000_000}
        dailyPnl={portfolio?.dailyPnl ?? 0}
        dailyPnlPercent={portfolio?.dailyPnlPercent ?? 0}
      />

      <div className="flex flex-1 overflow-hidden">
        <Sidebar view={view} onViewChange={setView} />

        {/* Page content */}
        {view === 'trading' && (
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
        )}

        {view === 'portfolio' && (
          <PortfolioDashboard portfolio={portfolio} />
        )}

        {view === 'market-data' && (
          <MarketDataPage quotes={quotes} orderBooks={orderBooks} lastOhlcv={lastOhlcv} />
        )}

        {view === 'trade-blotter' && (
          <PlaceholderPage title="Trade Blotter" />
        )}

        {view === 'backtest' && (
          <div className="flex-1 overflow-y-auto">
            <BacktestPage />
          </div>
        )}

        {view === 'analytics' && (
          <PlaceholderPage title="Analytics" />
        )}
      </div>
    </div>
  );
}
