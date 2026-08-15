# ETTh1 Hierarchical Moving-Block Bootstrap Summary

Differences are `left method - right method`; negative values favor the left method.
The primary analysis resamples training seeds and circular blocks of 96 windows.

| Metric | Left | Right | Difference | 95% CI | Conclusion |
|---|---|---|---:|---:|---|
| MAE | dynamic_joint_25 | static_25 | -0.00147776 | [-0.00350206, 0.00094822] | inconclusive |
| MAE | dynamic_joint_25 | unpruned_baseline | 0.00456545 | [0.00338673, 0.00572893] | right_better |
| MAE | static_25 | unpruned_baseline | 0.00604321 | [0.00381860, 0.00822147] | right_better |
| MSE | dynamic_joint_25 | static_25 | -0.00080054 | [-0.00350895, 0.00238994] | inconclusive |
| MSE | dynamic_joint_25 | unpruned_baseline | 0.00629976 | [0.00455772, 0.00811119] | right_better |
| MSE | static_25 | unpruned_baseline | 0.00710030 | [0.00426160, 0.01006463] | right_better |
