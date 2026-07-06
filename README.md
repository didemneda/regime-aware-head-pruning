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
## Project Members

- Didem Neda Aksaç
- Betül Aydeğer
