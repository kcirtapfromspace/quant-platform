# QUA-92: Vol Regime Sleeve — PF Gap Closure Attempt

**Date:** 2026-03-28
**CTO:** 927b53f6
**Priority:** HIGH (CRO-endorsed next lever after QUA-88 HMM closed negative)
**Status:** PLANNING

---

## Objective

Push Profit Factor from 1.237 (QUA-85) toward 1.30 aspirational target by adding a
dedicated **volatility regime sleeve** — a 5th sleeve based on the low-volatility anomaly.

---

## Hypothesis

The low-volatility anomaly is one of the most robust documented factor premia. Low-vol
stocks historically deliver better risk-adjusted returns due to:
1. **Institutional constraints** — managers benchmark-hugging → neglect low-vol names
2. **Lottery bias** — retail over-weights high-vol speculative names → mispricing
3. **PF channel** — fewer catastrophic drawdowns → fewer large losses → improved PF ratio

By dedicating a sleeve to this factor (separate from using it as one input in the adaptive
sleeve), we give it full portfolio-construction weight and allow the optimizer to select the
purest low-vol names without dilution.

---

## Design

### Sleeve Architecture (QUA-92 treatment)

| Sleeve | Signals | Capital Weight | Type |
|--------|---------|----------------|------|
| momentum_us_equity | RSI momentum | 30% | momentum |
| trend_following_us_equity | MACD + SMA | 25% | trend |
| mean_reversion_us_equity | BB mean reversion | 20% | mean_reversion |
| vol_regime_us_equity | VolatilitySignal + ReturnQualitySignal | 15% | low_vol |
| adaptive_combined | All 4 signals IC-weighted | 10% | momentum |

vs QUA-85 control (4 sleeves):
- momentum 35% / trend 30% / mean_reversion 15% / adaptive 20%

### Vol Regime Sleeve Signal Design

- **Primary signal:** `VolatilitySignal(period=20, low_vol=0.12, high_vol=0.40)`
  - Score +1 for low-vol stocks (below 12% annualised) → long
  - Score -1 for high-vol stocks (above 40% annualised) → avoid
  - Long-only constraint → high-vol names simply go to cash
- **Secondary signal:** `ReturnQualitySignal(period=60, sharpe_cap=3.0)`
  - Quality filter: only hold low-vol names that also have acceptable Sharpe
- **Combination:** `CombinationMethod.EQUAL_WEIGHT` (simple average)
- **Portfolio constraints:** long_only=True, max_weight=0.05, max_gross_exposure=0.6

---

## Backtest Script: `run_vol_regime_backtest.py`

### Runs

| Run | Config | Purpose |
|-----|--------|---------|
| Run A | signal_expansion_ensemble (QUA-85) | Control / baseline |
| Run B | vol_regime_ensemble (5-sleeve) | Treatment |
| Run C | vol_regime_standalone | Isolate vol sleeve alpha |

### Walk-Forward Parameters

Same as QUA-85 for comparability:
- IS=90 bars, OOS=30 bars, step_size=30, expanding=True
- Commission: 10 bps one-way
- min_history=100
- Optimizer: MVO, risk_aversion=5.0

---

## CRO Gate Targets

| Gate | Threshold | QUA-85 result | QUA-92 target |
|------|-----------|---------------|---------------|
| OOS Sharpe | >= 0.60 | 1.034 | >= 1.03 (maintain) |
| Profit Factor | >= 1.10 | 1.237 | >= 1.28 (improvement) |
| Max Drawdown | < 20% | 19.83% | < 19% (improvement) |
| WFE | >= 0.20 | 0.899 | >= 0.85 (maintain) |

---

## Risk Considerations

1. **MaxDD proximity:** QUA-85 is 17bps from gate. Vol sleeve should HELP MaxDD (low-vol
   stocks have lower drawdowns). Watch closely.
2. **Correlation with momentum:** Low-vol and momentum are historically negatively correlated.
   If vol regime sleeve reduces momentum exposure proportionally, net effect depends on how
   well the optimizer balances them.
3. **Adaptive sleeve shrinkage:** Reducing adaptive to 10% from 20% reduces the IC-weight
   dynamic allocation benefit. May need to tune.

---

## Implementation Tasks

1. Write `quant/scripts/run_vol_regime_backtest.py` — copy structure from
   `run_signal_expansion_backtest.py`, add:
   - `VolRegimeSleeveSignal` wrapper (combines VolatilitySignal + ReturnQualitySignal)
   - `run_vol_regime()` function (5-sleeve config)
   - `run_vol_regime_standalone()` function (1-sleeve isolation)
   - Reuse `run_signal_expansion()` as Run A control

2. No Rust changes needed — all signals exist in Python.

3. Results directory: `backtest-results/vol-regime/`

---

## Owner

- **Backtest script:** BackendEngineer (delegate from CTO)
- **CRO submission:** CTO
- **CRO gate decision:** CRO

---

**CTO sign-off on plan:** 927b53f6 — 2026-03-28
