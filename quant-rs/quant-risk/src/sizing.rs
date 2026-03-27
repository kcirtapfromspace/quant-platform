//! Position sizing methods.

/// How position size is calculated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SizingMethod {
    FixedFraction,
    Kelly,
    VolatilityTarget,
}

/// Fixed-fraction sizing: buy `fraction` of `capital` at `price`.
///
/// Returns the quantity (may be fractional; caller rounds as needed).
/// Returns 0.0 if price ≤ 0 or fraction ≤ 0.
pub fn position_size_fixed_fraction(capital: f64, price: f64, fraction: f64) -> f64 {
    if price <= 0.0 || fraction <= 0.0 || capital <= 0.0 {
        return 0.0;
    }
    (capital * fraction) / price
}

/// Kelly fraction: `win_rate - (1 - win_rate) / win_loss_ratio`.
///
/// Clamps to `[0, 1]`. Returns 0.0 if inputs are invalid.
pub fn kelly_fraction(win_rate: f64, win_loss_ratio: f64) -> f64 {
    if win_rate <= 0.0 || win_rate >= 1.0 || win_loss_ratio <= 0.0 {
        return 0.0;
    }
    let f = win_rate - (1.0 - win_rate) / win_loss_ratio;
    f.clamp(0.0, 1.0)
}

/// Volatility-target sizing: size such that position volatility equals `target_vol * capital`.
///
/// `quantity = (target_vol * capital) / (volatility * price)`
/// Returns 0.0 if any input is non-positive.
pub fn position_size_vol_target(capital: f64, price: f64, volatility: f64, target_vol: f64) -> f64 {
    if capital <= 0.0 || price <= 0.0 || volatility <= 0.0 || target_vol <= 0.0 {
        return 0.0;
    }
    (target_vol * capital) / (volatility * price)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_fraction_basic() {
        let qty = position_size_fixed_fraction(100_000.0, 50.0, 0.02);
        // 100_000 * 0.02 / 50 = 40
        assert!((qty - 40.0).abs() < 1e-9);
    }

    #[test]
    fn test_fixed_fraction_zero_price() {
        assert_eq!(position_size_fixed_fraction(100_000.0, 0.0, 0.02), 0.0);
    }

    #[test]
    fn test_kelly_basic() {
        // 60% win rate, 2:1 win/loss
        let f = kelly_fraction(0.6, 2.0);
        // f = 0.6 - 0.4/2 = 0.4
        assert!((f - 0.4).abs() < 1e-9);
    }

    #[test]
    fn test_kelly_negative_clamps_to_zero() {
        // 30% win rate, 1:1 — negative Kelly
        let f = kelly_fraction(0.3, 1.0);
        assert_eq!(f, 0.0);
    }

    #[test]
    fn test_vol_target_basic() {
        let qty = position_size_vol_target(1_000_000.0, 100.0, 0.2, 0.1);
        // (0.1 * 1_000_000) / (0.2 * 100) = 100_000 / 20 = 5000
        assert!((qty - 5_000.0).abs() < 1e-9);
    }

    #[test]
    fn test_fixed_fraction_zero_fraction() {
        assert_eq!(position_size_fixed_fraction(100_000.0, 50.0, 0.0), 0.0);
    }

    #[test]
    fn test_fixed_fraction_zero_capital() {
        assert_eq!(position_size_fixed_fraction(0.0, 50.0, 0.05), 0.0);
    }

    #[test]
    fn test_kelly_win_rate_zero() {
        // win_rate <= 0 is invalid
        assert_eq!(kelly_fraction(0.0, 2.0), 0.0);
    }

    #[test]
    fn test_kelly_win_rate_one() {
        // win_rate >= 1.0 is invalid
        assert_eq!(kelly_fraction(1.0, 2.0), 0.0);
    }

    #[test]
    fn test_kelly_win_loss_ratio_zero() {
        // win_loss_ratio <= 0 is invalid
        assert_eq!(kelly_fraction(0.6, 0.0), 0.0);
    }

    #[test]
    fn test_vol_target_zero_volatility() {
        assert_eq!(position_size_vol_target(1_000_000.0, 100.0, 0.0, 0.1), 0.0);
    }

    #[test]
    fn test_vol_target_zero_target() {
        assert_eq!(position_size_vol_target(1_000_000.0, 100.0, 0.2, 0.0), 0.0);
    }
}
