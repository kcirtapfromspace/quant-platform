//! Exposure limit checks.

/// Checks whether adding a position would breach exposure limits.
///
/// Phase 1 scaffold — full policy enforcement in Phase 3.
#[derive(Debug, Clone)]
pub struct ExposureLimits {
    /// Maximum gross exposure as a fraction of capital (e.g. 1.5 = 150%).
    pub max_gross_fraction: f64,
    /// Maximum single-position size as a fraction of capital (e.g. 0.1 = 10%).
    pub max_position_fraction: f64,
    /// Maximum net exposure as a fraction of capital (e.g. 1.0 = 100%).
    pub max_net_fraction: f64,
}

impl Default for ExposureLimits {
    fn default() -> Self {
        Self {
            max_gross_fraction: 1.5,
            max_position_fraction: 0.10,
            max_net_fraction: 1.0,
        }
    }
}

impl ExposureLimits {
    pub fn new(max_gross_fraction: f64, max_position_fraction: f64, max_net_fraction: f64) -> Self {
        Self {
            max_gross_fraction,
            max_position_fraction,
            max_net_fraction,
        }
    }

    /// Returns `None` (approved) or `Some(reason)` (rejected).
    pub fn check(
        &self,
        capital: f64,
        current_gross: f64,
        current_net: f64,
        order_value: f64,
    ) -> Option<String> {
        if capital <= 0.0 {
            return Some("capital must be positive".into());
        }
        let new_gross = current_gross + order_value.abs();
        if new_gross / capital > self.max_gross_fraction {
            return Some(format!(
                "gross exposure {:.1}% exceeds limit {:.1}%",
                new_gross / capital * 100.0,
                self.max_gross_fraction * 100.0,
            ));
        }
        let new_net = current_net + order_value;
        if (new_net / capital).abs() > self.max_net_fraction {
            return Some(format!(
                "net exposure {:.1}% exceeds limit {:.1}%",
                new_net / capital * 100.0,
                self.max_net_fraction * 100.0,
            ));
        }
        if order_value.abs() / capital > self.max_position_fraction {
            return Some(format!(
                "position size {:.1}% exceeds limit {:.1}%",
                order_value.abs() / capital * 100.0,
                self.max_position_fraction * 100.0,
            ));
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_within_limits_approved() {
        let limits = ExposureLimits::default();
        let result = limits.check(1_000_000.0, 500_000.0, 100_000.0, 50_000.0);
        assert!(result.is_none());
    }

    #[test]
    fn test_gross_breach_rejected() {
        let limits = ExposureLimits::default();
        // Adding 600k to 1M gross on 1M capital → 160% > 150%
        let result = limits.check(1_000_000.0, 1_000_000.0, 0.0, 600_000.0);
        assert!(result.is_some());
    }

    #[test]
    fn test_position_size_breach_rejected() {
        let limits = ExposureLimits::default();
        // Single position of 150k on 1M capital → 15% > 10%
        let result = limits.check(1_000_000.0, 0.0, 0.0, 150_000.0);
        assert!(result.is_some());
    }

    #[test]
    fn test_zero_capital_rejected() {
        let limits = ExposureLimits::default();
        let result = limits.check(0.0, 0.0, 0.0, 50_000.0);
        assert!(result.is_some());
    }

    #[test]
    fn test_net_exposure_long_breach() {
        let limits = ExposureLimits::default(); // max_net_fraction = 1.0
                                                // current_net = 800k, add 300k long → net = 1.1M on 1M capital = 110% > 100%
        let result = limits.check(1_000_000.0, 800_000.0, 800_000.0, 300_000.0);
        assert!(result.is_some());
    }

    #[test]
    fn test_net_exposure_short_breach() {
        let limits = ExposureLimits::default(); // max_net_fraction = 1.0
                                                // current_net = -800k, add -300k short → net = -1.1M → abs = 110% > 100%
        let result = limits.check(1_000_000.0, 800_000.0, -800_000.0, -300_000.0);
        assert!(result.is_some());
    }

    #[test]
    fn test_short_order_gross_uses_abs() {
        let limits = ExposureLimits::default(); // max_gross_fraction = 1.5
                                                // Short order of -600k; gross goes from 1M to 1.6M on 1M capital = 160% > 150%
        let result = limits.check(1_000_000.0, 1_000_000.0, 0.0, -600_000.0);
        assert!(result.is_some());
    }

    #[test]
    fn test_custom_limits_constructor() {
        let limits = ExposureLimits::new(2.0, 0.20, 1.5);
        assert_eq!(limits.max_gross_fraction, 2.0);
        assert_eq!(limits.max_position_fraction, 0.20);
        assert_eq!(limits.max_net_fraction, 1.5);
        // Should pass: 50k on 1M capital = 5% < 20%
        assert!(limits.check(1_000_000.0, 0.0, 0.0, 50_000.0).is_none());
    }
}
