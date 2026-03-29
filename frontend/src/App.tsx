import { useState, useCallback } from 'react';
import { useWebSocket } from './hooks/useWebSocket';
import { Header } from './components/Header';
import { Sidebar, type View } from './components/Sidebar';
import { Watchlist } from './components/Watchlist';
import { PriceChart } from './components/PriceChart';
import { OrderPanel } from './components/OrderPanel';
import { Positions } from './components/Positions';
import { OrderHistory } from './components/OrderHistory';
import { ErrorBoundary } from './components/ErrorBoundary';
import { BacktestPage } from './pages/BacktestPage';
import { PortfolioDashboard } from './pages/PortfolioDashboard';
import { MarketDataPage } from './pages/MarketDataPage';
import { TradeBlotterPage } from './pages/TradeBlotterPage';
import { AnalyticsPage } from './pages/AnalyticsPage';
import { StrategyMonitorPage } from './pages/StrategyMonitorPage';
import { RiskDashboardPage } from './pages/RiskDashboardPage';

export default function App() {
  const { quotes, portfolio, lastFill, lastOhlcv, orderBooks, connected } = useWebSocket();
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

        {view === 'trading' && (
          <div className="flex-1 flex overflow-hidden">
            <div className="w-64 flex-shrink-0 p-3 overflow-y-auto border-r border-slate-700/50">
              <Watchlist quotes={quotes} selectedSymbol={selectedSymbol} onSelect={setSelectedSymbol} />
            </div>

            <div className="flex-1 flex flex-col overflow-y-auto p-3 gap-3">
              <ErrorBoundary>
                <PriceChart symbol={selectedSymbol} currentQuote={currentQuote} />
              </ErrorBoundary>
              <div className="grid grid-cols-1 gap-3">
                <ErrorBoundary>
                  <Positions portfolio={portfolio} />
                </ErrorBoundary>
                <ErrorBoundary>
                  <OrderHistory refreshTrigger={orderRefresh} />
                </ErrorBoundary>
              </div>
            </div>

            <div className="w-72 flex-shrink-0 p-3 overflow-y-auto border-l border-slate-700/50">
              <ErrorBoundary>
                <OrderPanel
                  symbol={selectedSymbol}
                  currentQuote={currentQuote}
                  cash={portfolio?.cash ?? 1_000_000}
                  onOrderPlaced={handleOrderPlaced}
                />
              </ErrorBoundary>
            </div>
          </div>
        )}

        {view === 'portfolio' && (
          <ErrorBoundary>
            <PortfolioDashboard portfolio={portfolio} />
          </ErrorBoundary>
        )}

        {view === 'market-data' && (
          <ErrorBoundary>
            <MarketDataPage quotes={quotes} orderBooks={orderBooks} lastOhlcv={lastOhlcv} />
          </ErrorBoundary>
        )}

        {view === 'trade-blotter' && (
          <div className="flex-1 overflow-y-auto flex">
            <ErrorBoundary>
              <TradeBlotterPage lastFill={lastFill} />
            </ErrorBoundary>
          </div>
        )}

        {view === 'backtest' && (
          <div className="flex-1 overflow-y-auto">
            <ErrorBoundary>
              <BacktestPage />
            </ErrorBoundary>
          </div>
        )}

        {view === 'analytics' && (
          <div className="flex-1 overflow-y-auto">
            <ErrorBoundary>
              <AnalyticsPage />
            </ErrorBoundary>
          </div>
        )}

        {view === 'strategy-monitor' && (
          <div className="flex-1 overflow-y-auto">
            <ErrorBoundary>
              <StrategyMonitorPage />
            </ErrorBoundary>
          </div>
        )}

        {view === 'risk' && (
          <div className="flex-1 overflow-y-auto">
            <ErrorBoundary>
              <RiskDashboardPage />
            </ErrorBoundary>
          </div>
        )}
      </div>
    </div>
  );
}
