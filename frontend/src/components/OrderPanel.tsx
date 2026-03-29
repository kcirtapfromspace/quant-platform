import { useState } from 'react';
import { usePlaceOrder } from '../hooks/useApi';
import type { Quote } from '../types';

interface OrderPanelProps {
  symbol: string;
  currentQuote?: Quote;
  cash: number;
  onOrderPlaced: () => void;
}

export function OrderPanel({ symbol, currentQuote, cash, onOrderPlaced }: OrderPanelProps) {
  const [side, setSide] = useState<'buy' | 'sell'>('buy');
  const [orderType, setOrderType] = useState<'market' | 'limit'>('market');
  const [quantity, setQuantity] = useState('100');
  const [limitPrice, setLimitPrice] = useState('');
  const [error, setError] = useState('');

  const placeOrderMutation = usePlaceOrder();

  const price = orderType === 'limit' && limitPrice ? parseFloat(limitPrice) : (currentQuote?.price ?? 0);
  const qty = parseInt(quantity) || 0;
  const estimatedCost = price * qty;

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError('');
    if (!qty || qty <= 0) return setError('Invalid quantity');
    if (orderType === 'limit' && (!limitPrice || parseFloat(limitPrice) <= 0)) return setError('Invalid limit price');
    if (side === 'buy' && estimatedCost > cash) return setError('Insufficient cash');

    try {
      await placeOrderMutation.mutateAsync({
        symbol,
        side,
        type: orderType,
        quantity: qty,
        limitPrice: orderType === 'limit' ? parseFloat(limitPrice) : undefined,
      });
      onOrderPlaced();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Order failed');
    }
  }

  return (
    <div className="bg-surface-800 rounded-lg border border-slate-700/50">
      <div className="px-4 py-2.5 border-b border-slate-700/50">
        <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">Order</h2>
      </div>
      <form onSubmit={handleSubmit} className="p-4 space-y-3">
        <div className="grid grid-cols-2 gap-1 bg-surface-900 rounded-md p-0.5">
          <button
            type="button"
            onClick={() => setSide('buy')}
            className={`py-1.5 text-sm font-semibold rounded transition-colors ${
              side === 'buy' ? 'bg-gain text-white' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Buy
          </button>
          <button
            type="button"
            onClick={() => setSide('sell')}
            className={`py-1.5 text-sm font-semibold rounded transition-colors ${
              side === 'sell' ? 'bg-loss text-white' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Sell
          </button>
        </div>

        <div className="grid grid-cols-2 gap-1 bg-surface-900 rounded-md p-0.5">
          <button
            type="button"
            onClick={() => setOrderType('market')}
            className={`py-1.5 text-xs font-medium rounded transition-colors ${
              orderType === 'market' ? 'bg-surface-700 text-white' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Market
          </button>
          <button
            type="button"
            onClick={() => setOrderType('limit')}
            className={`py-1.5 text-xs font-medium rounded transition-colors ${
              orderType === 'limit' ? 'bg-surface-700 text-white' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Limit
          </button>
        </div>

        <div>
          <label className="block text-xs text-slate-400 mb-1">Symbol</label>
          <div className="bg-surface-900 rounded px-3 py-2 text-sm font-mono font-semibold">{symbol}</div>
        </div>

        <div>
          <label className="block text-xs text-slate-400 mb-1">Quantity</label>
          <input
            type="number"
            min="1"
            value={quantity}
            onChange={(e) => setQuantity(e.target.value)}
            className="w-full bg-surface-900 border border-slate-700 rounded px-3 py-2 text-sm font-mono focus:outline-none focus:border-accent"
          />
        </div>

        {orderType === 'limit' && (
          <div>
            <label className="block text-xs text-slate-400 mb-1">Limit Price</label>
            <input
              type="number"
              step="0.01"
              min="0.01"
              value={limitPrice}
              onChange={(e) => setLimitPrice(e.target.value)}
              placeholder={currentQuote?.price.toFixed(2)}
              className="w-full bg-surface-900 border border-slate-700 rounded px-3 py-2 text-sm font-mono focus:outline-none focus:border-accent"
            />
          </div>
        )}

        <div className="flex justify-between text-xs text-slate-400 py-1">
          <span>Est. {side === 'buy' ? 'Cost' : 'Proceeds'}</span>
          <span className="font-mono">${estimatedCost.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
        </div>
        <div className="flex justify-between text-xs text-slate-400">
          <span>Available Cash</span>
          <span className="font-mono">${cash.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
        </div>

        {error && <div className="text-xs text-loss bg-loss/10 rounded px-3 py-1.5">{error}</div>}

        <button
          type="submit"
          disabled={placeOrderMutation.isPending}
          className={`w-full py-2.5 rounded font-semibold text-sm transition-colors ${
            side === 'buy'
              ? 'bg-gain hover:bg-gain/90 text-white'
              : 'bg-loss hover:bg-loss/90 text-white'
          } disabled:opacity-50`}
        >
          {placeOrderMutation.isPending ? 'Submitting…' : `${side === 'buy' ? 'Buy' : 'Sell'} ${symbol}`}
        </button>
      </form>
    </div>
  );
}
