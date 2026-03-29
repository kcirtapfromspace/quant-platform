export interface Quote {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  high: number;
  low: number;
  open: number;
  previousClose: number;
  volume: number;
  timestamp: number;
}

export interface OhlcvBar {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export type OrderSide = 'buy' | 'sell';
export type OrderType = 'market' | 'limit';
export type OrderStatus = 'pending' | 'filled' | 'cancelled';

export interface Order {
  id: string;
  symbol: string;
  side: OrderSide;
  type: OrderType;
  quantity: number;
  limitPrice?: number;
  fillPrice?: number;
  status: OrderStatus;
  createdAt: number;
  filledAt?: number;
}

export interface Position {
  symbol: string;
  quantity: number;
  avgCost: number;
  currentPrice: number;
  marketValue: number;
  unrealizedPnl: number;
  unrealizedPnlPercent: number;
}

export interface Portfolio {
  cash: number;
  equity: number;
  totalValue: number;
  dailyPnl: number;
  dailyPnlPercent: number;
  positions: Position[];
}

export interface OrderBookEntry {
  price: number;
  size: number;
}

export interface OrderBook {
  symbol: string;
  bids: OrderBookEntry[];
  asks: OrderBookEntry[];
}

export type StrategyStatus = 'active' | 'paper' | 'halted' | 'backtesting';
export type Regime = 'bull' | 'bear' | 'sideways';
export type StrategyCategory = 'Time-series' | 'Factor' | 'Cross-sectional';

export interface StrategyState {
  strategy_key: string;
  name: string;
  status: StrategyStatus;
  regime: Regime;
  signal_confidence: number; // 0.0–1.0
  daily_pnl: number;
  positions: number;
  category: StrategyCategory;
}

export interface WsMessage {
  type: 'quote' | 'fill' | 'portfolio' | 'ohlcv' | 'orderbook' | 'strategy_state';
  data: Quote | Order | Portfolio | OhlcvBar | OrderBook | StrategyState;
}
