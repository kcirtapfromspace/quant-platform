import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiFetch, apiDelete } from '../lib/apiClient';
import type { Order, OhlcvBar, StrategyState } from '../types';

// ── Types ────────────────────────────────────────────────────────────────────

export interface RiskSnapshot {
  var95: number;
  var99: number;
  drawdown: number;
  maxDrawdown: number;
  circuitBreakerArmed: boolean;
  positionLimitUtilization: Array<{ symbol: string; utilization: number }>;
}

export interface PlaceOrderParams {
  symbol: string;
  side: 'buy' | 'sell';
  type: 'market' | 'limit';
  quantity: number;
  limitPrice?: number;
}

// ── React Query hooks ─────────────────────────────────────────────────────────

export function useOrders() {
  return useQuery<Order[]>({
    queryKey: ['orders'],
    queryFn: () => apiFetch<Order[]>('/orders'),
    refetchInterval: 5_000,
  });
}

export function useStrategies() {
  return useQuery<StrategyState[]>({
    queryKey: ['strategies'],
    queryFn: () => apiFetch<StrategyState[]>('/strategies'),
    refetchInterval: 10_000,
  });
}

export function usePortfolioHistory() {
  return useQuery<Array<{ time: number; value: number }>>({
    queryKey: ['portfolio', 'history'],
    queryFn: () => apiFetch<Array<{ time: number; value: number }>>('/portfolio/history'),
    staleTime: 60_000,
  });
}

export function useRiskSnapshot() {
  return useQuery<RiskSnapshot>({
    queryKey: ['risk', 'snapshot'],
    queryFn: () => apiFetch<RiskSnapshot>('/risk/snapshot'),
    refetchInterval: 15_000,
  });
}

// ── Mutations ─────────────────────────────────────────────────────────────────

export function usePlaceOrder() {
  const qc = useQueryClient();
  return useMutation<Order, Error, PlaceOrderParams>({
    mutationFn: (params) =>
      apiFetch<Order>('/orders', { method: 'POST', body: JSON.stringify(params) }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['orders'] }),
  });
}

export function useCancelOrder() {
  const qc = useQueryClient();
  return useMutation<void, Error, string>({
    mutationFn: (id) => apiDelete(`/orders/${id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['orders'] }),
  });
}

// ── Imperative helpers (for components using useEffect fetch patterns) ─────────

export function fetchHistory(symbol: string, range = '6mo'): Promise<OhlcvBar[]> {
  return apiFetch<OhlcvBar[]>(`/history/${symbol}?range=${range}`);
}

export function fetchOhlcv(symbol: string, interval = '5m'): Promise<OhlcvBar[]> {
  return apiFetch<OhlcvBar[]>(`/market/ohlcv?symbol=${encodeURIComponent(symbol)}&interval=${interval}`);
}

export function fetchWatchlist(): Promise<string[]> {
  return apiFetch<string[]>('/watchlist');
}

export function placeOrder(params: PlaceOrderParams): Promise<Order> {
  return apiFetch<Order>('/orders', { method: 'POST', body: JSON.stringify(params) });
}

export function cancelOrder(id: string): Promise<void> {
  return apiDelete(`/orders/${id}`);
}
