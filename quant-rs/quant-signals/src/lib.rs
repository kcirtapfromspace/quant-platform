//! Pure-Rust signal framework.
//!
//! Phase 1 scaffold — trait definitions only. Full port of signal strategies in Phase 4.

/// Signal direction emitted by a strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalDirection {
    Long,
    Short,
    Flat,
}

/// A signal emitted at a given bar.
#[derive(Debug, Clone)]
pub struct Signal {
    pub symbol: String,
    pub direction: SignalDirection,
    /// Strength in [0, 1]; 0.0 means no conviction.
    pub strength: f64,
}

impl Signal {
    pub fn new(symbol: impl Into<String>, direction: SignalDirection, strength: f64) -> Self {
        Self {
            symbol: symbol.into(),
            direction,
            strength: strength.clamp(0.0, 1.0),
        }
    }

    pub fn is_flat(&self) -> bool {
        self.direction == SignalDirection::Flat
    }
}

/// Trait implemented by all signal generators.
///
/// Full implementations (MomentumSignal, MeanReversionSignal, BreakoutSignal) come in Phase 4.
pub trait BaseSignal: Send + Sync {
    /// Unique name identifying this signal generator.
    fn name(&self) -> &str;

    /// Generate a signal from a slice of close prices (ascending chronological order).
    /// Returns `None` if there is insufficient data (warm-up).
    fn generate(&self, symbol: &str, closes: &[f64]) -> Option<Signal>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct AlwaysLong;
    impl BaseSignal for AlwaysLong {
        fn name(&self) -> &str {
            "always_long"
        }
        fn generate(&self, symbol: &str, _closes: &[f64]) -> Option<Signal> {
            Some(Signal::new(symbol, SignalDirection::Long, 1.0))
        }
    }

    #[test]
    fn test_signal_strength_clamped() {
        let s = Signal::new("AAPL", SignalDirection::Long, 2.5);
        assert_eq!(s.strength, 1.0);
    }

    #[test]
    fn test_always_long_signal() {
        let gen = AlwaysLong;
        let sig = gen.generate("AAPL", &[100.0, 101.0, 102.0]).unwrap();
        assert_eq!(sig.direction, SignalDirection::Long);
        assert_eq!(sig.symbol, "AAPL");
    }

    #[test]
    fn test_flat_detection() {
        let s = Signal::new("TSLA", SignalDirection::Flat, 0.0);
        assert!(s.is_flat());
    }
}
