import { useEffect, useRef, useCallback, useState } from 'react';
import type { Quote, Order, Portfolio, OhlcvBar, OrderBook, WsMessage } from '../types';

interface UseWebSocketReturn {
  quotes: Map<string, Quote>;
  portfolio: Portfolio | null;
  lastFill: Order | null;
  lastOhlcv: OhlcvBar | null;
  orderBooks: Map<string, OrderBook>;
  connected: boolean;
}

export function useWebSocket(): UseWebSocketReturn {
  const [quotes, setQuotes] = useState<Map<string, Quote>>(new Map());
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [lastFill, setLastFill] = useState<Order | null>(null);
  const [lastOhlcv, setLastOhlcv] = useState<OhlcvBar | null>(null);
  const [orderBooks, setOrderBooks] = useState<Map<string, OrderBook>>(new Map());
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  const connect = useCallback(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => {
      setConnected(false);
      reconnectTimer.current = setTimeout(connect, 2000);
    };
    ws.onerror = () => ws.close();

    ws.onmessage = (event) => {
      const msg: WsMessage = JSON.parse(event.data);
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

  return { quotes, portfolio, lastFill, lastOhlcv, orderBooks, connected };
}
