//! Threshold-based portfolio rebalancing with trade generation.
//!
//! Mirrors `quant.portfolio.rebalancer`.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ── Trade ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub symbol: String,
    /// "BUY" or "SELL".
    pub side: String,
    /// Target portfolio weight (post-trade).
    pub target_weight: f64,
    /// Change in weight (can be negative for sells).
    pub trade_weight: f64,
    /// Absolute dollar amount of the trade.
    pub dollar_amount: f64,
}

// ── RebalanceResult ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RebalanceResult {
    pub trades: Vec<Trade>,
    /// Total portfolio turnover (sum of absolute weight changes / 2).
    pub turnover: f64,
    pub rebalance_triggered: bool,
}

// ── Rebalancer ────────────────────────────────────────────────────────────────

pub struct Rebalancer {
    /// Minimum total turnover to trigger a rebalance.
    pub threshold: f64,
    /// Dead band for individual trades (trades smaller than this are skipped).
    pub min_trade_weight: f64,
}

impl Default for Rebalancer {
    fn default() -> Self {
        Self {
            threshold: 0.01,
            min_trade_weight: 0.001,
        }
    }
}

impl Rebalancer {
    /// Generate trades to move `current_weights` toward `target_weights`.
    ///
    /// # Arguments
    /// * `target_weights`  — {symbol: weight} from the optimizer.
    /// * `current_weights` — {symbol: weight} from the OMS positions.
    /// * `portfolio_value` — total portfolio value in dollars.
    pub fn rebalance(
        &self,
        target_weights: &HashMap<String, f64>,
        current_weights: &HashMap<String, f64>,
        portfolio_value: f64,
    ) -> RebalanceResult {
        // Compute raw weight deltas (target - current) for each symbol.
        let mut all_symbols: Vec<String> = target_weights
            .keys()
            .chain(current_weights.keys())
            .cloned()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        all_symbols.sort();

        let deltas: Vec<(String, f64)> = all_symbols
            .iter()
            .map(|sym| {
                let target = target_weights.get(sym).copied().unwrap_or(0.0);
                let current = current_weights.get(sym).copied().unwrap_or(0.0);
                (sym.clone(), target - current)
            })
            .collect();

        // Total turnover = half of sum of absolute deltas.
        let total_turnover: f64 = deltas.iter().map(|(_, d)| d.abs()).sum::<f64>() / 2.0;

        // Check threshold.
        if total_turnover < self.threshold {
            return RebalanceResult {
                trades: vec![],
                turnover: total_turnover,
                rebalance_triggered: false,
            };
        }

        // Build trades, skipping dead-band.
        let trades: Vec<Trade> = deltas
            .into_iter()
            .filter(|(_, delta)| delta.abs() >= self.min_trade_weight)
            .map(|(symbol, delta)| {
                let side = if delta > 0.0 { "BUY" } else { "SELL" }.to_string();
                let target = target_weights.get(&symbol).copied().unwrap_or(0.0);
                Trade {
                    symbol,
                    side,
                    target_weight: target,
                    trade_weight: delta,
                    dollar_amount: delta.abs() * portfolio_value,
                }
            })
            .collect();

        RebalanceResult {
            trades,
            turnover: total_turnover,
            rebalance_triggered: true,
        }
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_rebalance_below_threshold() {
        let rebalancer = Rebalancer {
            threshold: 0.05,
            min_trade_weight: 0.001,
        };
        let target: HashMap<String, f64> = [("AAPL".to_string(), 0.51)].into_iter().collect();
        let current: HashMap<String, f64> = [("AAPL".to_string(), 0.50)].into_iter().collect();
        let result = rebalancer.rebalance(&target, &current, 100_000.0);
        assert!(!result.rebalance_triggered);
        assert!(result.trades.is_empty());
    }

    #[test]
    fn test_rebalance_above_threshold() {
        let rebalancer = Rebalancer::default();
        let target: HashMap<String, f64> = [("AAPL".to_string(), 0.6), ("GOOG".to_string(), 0.4)]
            .into_iter()
            .collect();
        let current: HashMap<String, f64> = [("AAPL".to_string(), 0.3), ("GOOG".to_string(), 0.7)]
            .into_iter()
            .collect();
        let result = rebalancer.rebalance(&target, &current, 100_000.0);
        assert!(result.rebalance_triggered);
        assert_eq!(result.trades.len(), 2);
    }

    #[test]
    fn test_turnover_calculation() {
        let rebalancer = Rebalancer::default();
        let target: HashMap<String, f64> = [("A".to_string(), 1.0)].into_iter().collect();
        let current: HashMap<String, f64> = [("A".to_string(), 0.5), ("B".to_string(), 0.5)]
            .into_iter()
            .collect();
        let result = rebalancer.rebalance(&target, &current, 100_000.0);
        // |1.0-0.5| + |0.0-0.5| = 1.0 → turnover = 0.5
        assert!((result.turnover - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_buy_sell_classification() {
        let rebalancer = Rebalancer {
            threshold: 0.0,
            min_trade_weight: 0.0,
        };
        let target: HashMap<String, f64> =
            [("BUY_ME".to_string(), 0.6), ("SELL_ME".to_string(), 0.2)]
                .into_iter()
                .collect();
        let current: HashMap<String, f64> =
            [("BUY_ME".to_string(), 0.3), ("SELL_ME".to_string(), 0.5)]
                .into_iter()
                .collect();
        let result = rebalancer.rebalance(&target, &current, 1_000.0);
        let buy = result.trades.iter().find(|t| t.symbol == "BUY_ME").unwrap();
        let sell = result
            .trades
            .iter()
            .find(|t| t.symbol == "SELL_ME")
            .unwrap();
        assert_eq!(buy.side, "BUY");
        assert_eq!(sell.side, "SELL");
    }
}
