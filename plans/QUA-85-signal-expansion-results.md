# QUA-85 Signal Expansion Results

Date: 2026-03-28T21:46:40.936974+00:00

## CRO Gate Metrics

| Run | Sharpe | PF | WFE | Max DD | Folds | Status |
|-----|--------|-----|-----|--------|-------|--------|
| mean_reversion_standalone                |  1.222 |  1.28 | 1.022 | 17.43% |    64 | PASS |
| baseline_mvp_ensemble                    |  1.008 |  1.23 | 0.889 | 20.30% |    64 | FAIL |
| signal_expansion_ensemble                |  1.034 |  1.24 | 0.899 | 19.83% |    64 | PASS |

## CRO Thresholds

| Gate | Target |
|------|--------|
| Sharpe | >= 0.60 |
| PF | >= 1.10 (aspiration: 1.30) |
| MaxDD | < 20% |
| WFE | >= 0.20 |
