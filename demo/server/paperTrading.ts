import { QuoteData } from './marketData.js';

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

export interface PositionState {
  symbol: string;
  quantity: number;
  avgCost: number;
}

export interface PortfolioSnapshot {
  cash: number;
  equity: number;
  totalValue: number;
  dailyPnl: number;
  dailyPnlPercent: number;
  positions: Array<{
    symbol: string;
    quantity: number;
    avgCost: number;
    currentPrice: number;
    marketValue: number;
    unrealizedPnl: number;
    unrealizedPnlPercent: number;
  }>;
}

const INITIAL_CASH = 1_000_000;
let nextOrderId = 1;

export class PaperTradingEngine {
  private cash: number = INITIAL_CASH;
  private initialCash: number = INITIAL_CASH;
  private positions: Map<string, PositionState> = new Map();
  private orders: Order[] = [];
  private onFill?: (order: Order) => void;

  constructor(opts?: { initialCash?: number; onFill?: (order: Order) => void }) {
    if (opts?.initialCash) {
      this.cash = opts.initialCash;
      this.initialCash = opts.initialCash;
    }
    this.onFill = opts?.onFill;
  }

  placeOrder(symbol: string, side: OrderSide, type: OrderType, quantity: number, limitPrice?: number): Order {
    const order: Order = {
      id: `ORD-${String(nextOrderId++).padStart(6, '0')}`,
      symbol: symbol.toUpperCase(),
      side,
      type,
      quantity,
      limitPrice: type === 'limit' ? limitPrice : undefined,
      status: 'pending',
      createdAt: Date.now(),
    };
    this.orders.push(order);
    return order;
  }

  processQuotes(quotes: Map<string, QuoteData>): Order[] {
    const filled: Order[] = [];
    for (const order of this.orders) {
      if (order.status !== 'pending') continue;

      const quote = quotes.get(order.symbol);
      if (!quote) continue;

      let shouldFill = false;
      let fillPrice = quote.price;

      if (order.type === 'market') {
        shouldFill = true;
      } else if (order.type === 'limit' && order.limitPrice != null) {
        if (order.side === 'buy' && quote.price <= order.limitPrice) {
          shouldFill = true;
          fillPrice = order.limitPrice;
        } else if (order.side === 'sell' && quote.price >= order.limitPrice) {
          shouldFill = true;
          fillPrice = order.limitPrice;
        }
      }

      if (shouldFill) {
        const cost = fillPrice * order.quantity;
        if (order.side === 'buy') {
          if (cost > this.cash) continue;
          this.cash -= cost;
          this.addToPosition(order.symbol, order.quantity, fillPrice);
        } else {
          const pos = this.positions.get(order.symbol);
          if (!pos || pos.quantity < order.quantity) continue;
          this.cash += cost;
          this.removeFromPosition(order.symbol, order.quantity);
        }

        order.status = 'filled';
        order.fillPrice = fillPrice;
        order.filledAt = Date.now();
        filled.push(order);
        this.onFill?.(order);
      }
    }
    return filled;
  }

  getPortfolio(currentPrices: Map<string, QuoteData>): PortfolioSnapshot {
    const positions = Array.from(this.positions.values())
      .filter((p) => p.quantity > 0)
      .map((p) => {
        const currentPrice = currentPrices.get(p.symbol)?.price ?? p.avgCost;
        const marketValue = currentPrice * p.quantity;
        const costBasis = p.avgCost * p.quantity;
        const unrealizedPnl = marketValue - costBasis;
        const unrealizedPnlPercent = costBasis !== 0 ? (unrealizedPnl / costBasis) * 100 : 0;
        return {
          symbol: p.symbol,
          quantity: p.quantity,
          avgCost: p.avgCost,
          currentPrice,
          marketValue,
          unrealizedPnl,
          unrealizedPnlPercent,
        };
      });

    const equity = positions.reduce((sum, p) => sum + p.marketValue, 0);
    const totalValue = this.cash + equity;
    const dailyPnl = totalValue - this.initialCash;
    const dailyPnlPercent = (dailyPnl / this.initialCash) * 100;

    return { cash: this.cash, equity, totalValue, dailyPnl, dailyPnlPercent, positions };
  }

  getOrders(): Order[] {
    return [...this.orders].reverse();
  }

  cancelOrder(orderId: string): boolean {
    const order = this.orders.find((o) => o.id === orderId);
    if (!order || order.status !== 'pending') return false;
    order.status = 'cancelled';
    return true;
  }

  private addToPosition(symbol: string, quantity: number, price: number) {
    const existing = this.positions.get(symbol);
    if (existing) {
      const totalCost = existing.avgCost * existing.quantity + price * quantity;
      existing.quantity += quantity;
      existing.avgCost = totalCost / existing.quantity;
    } else {
      this.positions.set(symbol, { symbol, quantity, avgCost: price });
    }
  }

  private removeFromPosition(symbol: string, quantity: number) {
    const existing = this.positions.get(symbol);
    if (!existing) return;
    existing.quantity -= quantity;
    if (existing.quantity <= 0) {
      this.positions.delete(symbol);
    }
  }
}
