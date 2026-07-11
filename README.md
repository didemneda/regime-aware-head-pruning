# Regime-Aware Attention Head Pruning

This project investigates regime-conditional attention head selection for time series Transformers.

## Initial Scope

- Model: PatchTST
- Dataset: ETTh1
- Prediction length: 96
- Regimes:
  - Trend-dominant
  - Seasonality-dominant
  - Noise-dominant
## Baseline Selection

We evaluated several PatchTST configurations on ETTh1 with a prediction horizon of 96.

| Experiment | Seq Len | d_model | Heads | Val Loss | Test MSE | Test MAE |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 336 | 128 | 4 | 0.6732 | 0.3739 | 0.3989 |
| B1 | 512 | 128 | 16 | 0.7382 | 0.3764 | 0.4053 |
| B2 | 512 | 256 | 16 | 0.7811 | 0.3947 | 0.4234 |
| B3 | 336 | 128 | 16 | 0.6781 | 0.3735 | 0.3994 |
| B4 | 336 | 128 | 8 | 0.6735 | 0.3725 | 0.3982 |

B4 was selected as the main pruning baseline because it provides 24 total attention heads while maintaining competitive forecasting performance.

## Regime Detection

Validation windows were labeled using STL decomposition on the OT variable of ETTh1. Each input window of length 336 was decomposed into trend, seasonal, and residual components. The dominant regime was assigned based on the component with the highest normalized variance.

| Regime | Windows | Percentage |
|---|---:|---:|
| Trend | 2134 | 76.62% |
| Residual | 359 | 12.89% |
| Seasonal | 292 | 10.48% |

A confidence margin was computed as the difference between the highest and second-highest component scores. Using a threshold of 0.05, 2490 out of 2785 windows were considered confidently labeled.

### Regime Detection Outputs

- `etth1_validation_regimes_ot_seq336.csv`: regime label and STL scores for all 2785 validation windows.
- `etth1_validation_regime_summary.csv`: regime count and percentage summary.
- `etth1_validation_confidence_summary.csv`: confidence statistics per regime.
- `etth1_validation_regimes_confident_seq336.csv`: subset of windows with confidence margin >= 0.05.
## Head Importance Analysis

Head importance was measured by masking one attention head at a time and computing the change in validation MSE. Importance was computed globally and separately for trend, seasonal, and residual regimes.

The analysis shows that several heads behave differently across regimes. For example, Layer 1 Head 1 has negative seasonal importance but positive residual importance, indicating that it is harmful for seasonal windows but useful for residual-dominant windows. This supports the motivation for regime-aware head selection instead of relying only on a single global pruning mask.

A 25% static pruning candidate set was created by selecting the six heads with the lowest overall importance.

Static 25% head pruning improved validation MSE from 0.6781 to 0.6537, corresponding to a 3.60% reduction. However, on the held-out test set, MSE increased from 0.3725 to 0.3783, indicating a 1.55% degradation. Regime-level validation analysis showed that static pruning improved trend and residual regimes but degraded seasonal windows by 4.45%. This suggests that a single global pruning mask may not be optimal across all temporal regimes and motivates the proposed dynamic regime-aware head selection strategy.
## Project Members

- Didem Neda Aksaç
- Betül Aydeğer
