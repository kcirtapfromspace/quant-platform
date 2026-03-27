import { useState, useCallback } from 'react';
import { useWebSocket } from './hooks/useWebSocket';
import { Header } from './components/Header';
import { Watchlist } from './components/Watchlist';
import { PriceChart } from './components/PriceChart';
import { OrderPanel } from './components/OrderPanel';
import { Positions } from './components/Positions';
import { OrderHistory } from './components/OrderHistory';

export default function App() {
  const { quotes, portfolio, connected } = useWebSocket();
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
  const [orderRefresh, setOrderRefresh] = useState(0);

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
    </div>
  );
}
