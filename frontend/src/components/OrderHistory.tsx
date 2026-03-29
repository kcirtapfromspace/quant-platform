import { useOrders, useCancelOrder } from '../hooks/useApi';

interface OrderHistoryProps {
  refreshTrigger: number;
}

export function OrderHistory({ refreshTrigger }: OrderHistoryProps) {
  const { data: orders, isLoading, isError, refetch } = useOrders();
  const cancelMutation = useCancelOrder();

  // Re-fetch when a new order is placed
  if (refreshTrigger) void refetch();

  if (isLoading) {
    return (
      <div className="bg-surface-800 rounded-lg border border-slate-700/50 overflow-hidden">
        <div className="px-4 py-2.5 border-b border-slate-700/50">
          <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">Orders</h2>
        </div>
        <div className="px-4 py-8 text-center text-slate-500 text-sm">Loading orders…</div>
      </div>
    );
  }

  if (isError) {
    return (
      <div className="bg-surface-800 rounded-lg border border-slate-700/50 overflow-hidden">
        <div className="px-4 py-2.5 border-b border-slate-700/50">
          <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">Orders</h2>
        </div>
        <div className="px-4 py-8 text-center text-red-400 text-sm">Failed to load orders</div>
      </div>
    );
  }

  if (!orders || orders.length === 0) {
    return (
      <div className="bg-surface-800 rounded-lg border border-slate-700/50 overflow-hidden">
        <div className="px-4 py-2.5 border-b border-slate-700/50">
          <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">Orders</h2>
        </div>
        <div className="px-4 py-8 text-center text-slate-500 text-sm">No orders yet</div>
      </div>
    );
  }

  return (
    <div className="bg-surface-800 rounded-lg border border-slate-700/50 overflow-hidden">
      <div className="px-4 py-2.5 border-b border-slate-700/50">
        <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">Orders</h2>
      </div>
      <div className="overflow-x-auto max-h-60 overflow-y-auto">
        <table className="w-full text-sm">
          <thead className="sticky top-0 bg-surface-800">
            <tr className="text-xs text-slate-400 uppercase tracking-wider border-b border-slate-700/30">
              <th className="px-4 py-2 text-left font-medium">ID</th>
              <th className="px-4 py-2 text-left font-medium">Symbol</th>
              <th className="px-4 py-2 text-left font-medium">Side</th>
              <th className="px-4 py-2 text-left font-medium">Type</th>
              <th className="px-4 py-2 text-right font-medium">Qty</th>
              <th className="px-4 py-2 text-right font-medium">Price</th>
              <th className="px-4 py-2 text-center font-medium">Status</th>
              <th className="px-4 py-2 text-center font-medium"></th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-700/20">
            {orders.map((o) => (
              <tr key={o.id} className="hover:bg-surface-700/30 transition-colors">
                <td className="px-4 py-1.5 font-mono text-xs text-slate-400">{o.id}</td>
                <td className="px-4 py-1.5 font-semibold">{o.symbol}</td>
                <td className={`px-4 py-1.5 font-medium ${o.side === 'buy' ? 'text-gain' : 'text-loss'}`}>
                  {o.side.toUpperCase()}
                </td>
                <td className="px-4 py-1.5 text-slate-300">{o.type}</td>
                <td className="px-4 py-1.5 text-right font-mono">{o.quantity}</td>
                <td className="px-4 py-1.5 text-right font-mono">
                  {o.fillPrice ? `$${o.fillPrice.toFixed(2)}` : o.limitPrice ? `$${o.limitPrice.toFixed(2)}` : '-'}
                </td>
                <td className="px-4 py-1.5 text-center">
                  <span
                    className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${
                      o.status === 'filled'
                        ? 'bg-gain/20 text-gain'
                        : o.status === 'pending'
                          ? 'bg-yellow-500/20 text-yellow-400'
                          : 'bg-slate-500/20 text-slate-400'
                    }`}
                  >
                    {o.status}
                  </span>
                </td>
                <td className="px-4 py-1.5 text-center">
                  {o.status === 'pending' && (
                    <button
                      onClick={() => cancelMutation.mutate(o.id)}
                      disabled={cancelMutation.isPending}
                      className="text-xs text-loss hover:underline disabled:opacity-50"
                    >
                      Cancel
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
