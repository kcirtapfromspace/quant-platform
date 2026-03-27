//! Drawdown-based circuit breaker.

/// Trips if drawdown from peak exceeds `max_drawdown_fraction`.
///
/// Phase 1 scaffold — full state machine in Phase 3.
#[derive(Debug, Clone)]
pub struct DrawdownCircuitBreaker {
    pub max_drawdown_fraction: f64,
}

impl DrawdownCircuitBreaker {
    pub fn new(max_drawdown_fraction: f64) -> Self {
        assert!(
            max_drawdown_fraction > 0.0 && max_drawdown_fraction < 1.0,
            "max_drawdown_fraction must be in (0, 1)"
        );
        Self {
            max_drawdown_fraction,
        }
    }

    /// Returns `true` if trading should be halted.
    ///
    /// `peak`: all-time high portfolio value.
    /// `current`: current portfolio value.
    pub fn is_tripped(&self, peak: f64, current: f64) -> bool {
        if peak <= 0.0 {
            return false;
        }
        let drawdown = (peak - current) / peak;
        drawdown >= self.max_drawdown_fraction
    }

    /// Current drawdown fraction (0.0 if peak ≤ 0).
    pub fn drawdown(&self, peak: f64, current: f64) -> f64 {
        if peak <= 0.0 {
            return 0.0;
        }
        ((peak - current) / peak).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_not_tripped_within_limit() {
        let cb = DrawdownCircuitBreaker::new(0.10);
        assert!(!cb.is_tripped(1_000_000.0, 950_000.0)); // 5% drawdown < 10%
    }

    #[test]
    fn test_tripped_at_limit() {
        let cb = DrawdownCircuitBreaker::new(0.10);
        assert!(cb.is_tripped(1_000_000.0, 900_000.0)); // 10% drawdown = limit
    }

    #[test]
    fn test_tripped_beyond_limit() {
        let cb = DrawdownCircuitBreaker::new(0.10);
        assert!(cb.is_tripped(1_000_000.0, 800_000.0)); // 20% > 10%
    }

    #[test]
    fn test_drawdown_calculation() {
        let cb = DrawdownCircuitBreaker::new(0.20);
        let dd = cb.drawdown(1_000_000.0, 900_000.0);
        assert!((dd - 0.10).abs() < 1e-9);
    }

    #[test]
    fn test_zero_peak_not_tripped() {
        let cb = DrawdownCircuitBreaker::new(0.10);
        // peak <= 0 should never trip
        assert!(!cb.is_tripped(0.0, 0.0));
        assert!(!cb.is_tripped(-1.0, -2.0));
    }

    #[test]
    fn test_drawdown_zero_peak_returns_zero() {
        let cb = DrawdownCircuitBreaker::new(0.10);
        assert_eq!(cb.drawdown(0.0, 500_000.0), 0.0);
    }

    #[test]
    fn test_drawdown_current_above_peak_clamps_to_zero() {
        let cb = DrawdownCircuitBreaker::new(0.10);
        // current > peak — no drawdown, should clamp to 0
        assert_eq!(cb.drawdown(900_000.0, 1_000_000.0), 0.0);
    }
}
