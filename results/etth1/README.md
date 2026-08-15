# ETTh1 Reproducibility and Efficiency Results

This directory contains the five-seed ETTh1 results produced for the
regime-aware attention-head pruning experiments.

## Experimental setup

- Model: PatchTST
- Forecast horizon: 96
- Input length: 336
- Seeds: 7, 42, 1234, 2026 and 3407
- Pruning ratio: 25%
- Active attention heads after pruning: 18/24
- External Time-Series-Library commit:
  `4e938a1767106324dd753b2a44832bf870a0252e`

Validation and test loaders were evaluated with `shuffle=False` to
preserve temporal alignment between forecast windows and regime labels.

## Five-seed forecasting results

| Method | Test MSE | Test MAE |
|---|---:|---:|
| Unpruned baseline | 0.374411 ± 0.001660 | 0.400383 ± 0.001407 |
| Static 25% pruning | 0.381510 ± 0.003080 | 0.406426 ± 0.003095 |
| Dynamic joint 25% pruning | 0.380710 ± 0.001576 | 0.404949 ± 0.001474 |

Dynamic pruning achieved a lower mean error than static pruning and
obtained a lower test MSE in four of the five seeds. The unpruned
baseline remained the most accurate method.

## Efficiency interpretation

Structural pruning reduced active parameters by approximately 5.40%
and estimated FLOPs by approximately 11.96%.

Static structural pruning is the most practical deployment
configuration among the evaluated pruned methods. The three-model
dynamic structural configuration requires additional checkpoint
storage and regime-detection time.

The five-seed mask-shape experiment produced:

| Method | Mean latency (ms) | Standard deviation (ms) |
|---|---:|---:|
| Dynamic expected | 9.101987 | 0.073073 |
| Static | 9.115961 | 0.009634 |

The approximately 0.15% mean difference is not practically meaningful.
Static masks showed more stable latency across seeds.

Absolute latency measurements may vary between GPU sessions.
The mask-shape table should be interpreted as a within-session
sensitivity comparison.
