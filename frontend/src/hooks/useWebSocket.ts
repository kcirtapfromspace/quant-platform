import { useEffect, useRef, useCallback, useState } from 'react';
import type { Quote, Order, Portfolio, OhlcvBar, OrderBook, StrategyState, WsMessage } from '../types';

interface UseWebSocketReturn {
  quotes: Map<string, Quote>;
  portfolio: Portfolio | null;
  lastFill: Order | null;
  lastOhlcv: OhlcvBar | null;
  orderBooks: Map<string, OrderBook>;
  strategyStates: Map<string, StrategyState>;
  connected: boolean;
}

function resolveWsUrl(): string {
  const configured = import.meta.env.VITE_WS_URL as string | undefined;
  if (configured) return configured;
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}/ws`;
}

export function useWebSocket(): UseWebSocketReturn {
  const [quotes, setQuotes] = useState<Map<string, Quote>>(new Map());
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [lastFill, setLastFill] = useState<Order | null>(null);
  const [lastOhlcv, setLastOhlcv] = useState<OhlcvBar | null>(null);
  const [orderBooks, setOrderBooks] = useState<Map<string, OrderBook>>(new Map());
  const [strategyStates, setStrategyStates] = useState<Map<string, StrategyState>>(new Map());
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  const connect = useCallback(() => {
    const wsUrl = resolveWsUrl();
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => {
      setConnected(false);
      reconnectTimer.current = setTimeout(connect, 2000);
    };
    ws.onerror = () => ws.close();

    ws.onmessage = (event) => {
      const msg: WsMessage = JSON.parse(event.data as string);
      switch (msg.type) {
        case 'quote':
          setQuotes((prev) => {
            const next = new Map(prev);
            next.set((msg.data as Quote).symbol, msg.data as Quote);
            return next;
          });
          break;
        case 'portfolio':
          setPortfolio(msg.data as Portfolio);
          break;
        case 'fill':
          setLastFill(msg.data as Order);
          break;
        case 'ohlcv':
          setLastOhlcv(msg.data as OhlcvBar);
          break;
        case 'orderbook': {
          const book = msg.data as OrderBook;
          setOrderBooks((prev) => {
            const next = new Map(prev);
            next.set(book.symbol, book);
            return next;
          });
          break;
        }
        case 'strategy_state': {
          const state = msg.data as StrategyState;
          setStrategyStates((prev) => {
            const next = new Map(prev);
            next.set(state.strategy_key, state);
            return next;
          });
          break;
        }
      }
    };
  }, []);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  return { quotes, portfolio, lastFill, lastOhlcv, orderBooks, strategyStates, connected };
}
