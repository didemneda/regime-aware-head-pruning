# Regime-Aware Attention Head Pruning

This project investigates regime-conditional attention head selection for time series Transformers. See [ReCAHS.pdf](ReCAHS.pdf) for the original project proposal.

## Table of Contents

- [Executive Summary](#executive-summary)
- [Initial Scope](#initial-scope)
- [Baseline Selection](#baseline-selection)
- [Regime Detection](#regime-detection)
- [Head Importance Analysis](#head-importance-analysis)
- [Pruning and Regime-Aware Evaluation Experiments](#pruning-and-regime-aware-evaluation-experiments)
  - [Experiment 04: Static 25% Head Pruning](#experiment-04-static-25-head-pruning)
  - [Experiment 05: Dynamic Regime-Aware 50% Keep](#experiment-05-dynamic-regime-aware-50-keep)
  - [Experiment 06: Dynamic Regime-Aware 75% Keep](#experiment-06-dynamic-regime-aware-75-keep)
  - [Experiment 07: Test Regime Detection and Dynamic 75% Test Evaluation](#experiment-07-test-regime-detection-and-dynamic-75-test-evaluation)
  - [Experiment 08: Random and Magnitude-Based Baseline Pruning](#experiment-08-random-and-magnitude-based-baseline-pruning)
  - [Experiment 09: Fine-Tuning the Static 25% Pruned Model (Policy A)](#experiment-09-fine-tuning-the-static-25-pruned-model-policy-a)
  - [Experiment 10: Joint (Greedy) Regime-Aware Head Selection and Confidence-Margin Robustness Check](#experiment-10-joint-greedy-regime-aware-head-selection-and-confidence-margin-robustness-check)
  - [Experiment 11: Efficiency Metrics (Parameter Count, FLOPs, Inference Latency)](#experiment-11-efficiency-metrics-parameter-count-flops-inference-latency)
  - [Experiment 12: Extension to ETTm1](#experiment-12-extension-to-ettm1)
- [Summary of Findings](#summary-of-findings)
- [Project Members](#project-members)

## Executive Summary

**Hypothesis:** attention head importance in PatchTST varies by the temporal regime (trend/seasonal/residual-dominant) of the input window, so a regime-conditional pruning mask should outperform a single global pruning mask. Tested on ETTh1 (PatchTST, `seq_len=336, pred_len=96`, B4 baseline: `d_model=128, n_heads=8, e_layers=3`, 24 total attention heads).

**All methods on ETTh1, 25% pruning ratio, ranked by test MSE (lower is better):**

| Rank | Setting | Val MSE | Test MSE | Test vs. baseline | Parameters | FLOPs |
|---:|---|---:|---:|---:|---:|---:|
| — | B4 no pruning (baseline) | 0.6781 | **0.3725** | — | 915,936 | 120,873,984 |
| 1 | **Dynamic 75% keep — joint/greedy selection (Exp. 10)** | 0.6536 | **0.3774** | +1.31% | 866,496 (-5.4%) | 106,423,296 (-12.0%) |
| 2 | Static 25% pruning — global importance (Exp. 04) | 0.6537 | 0.3783 | +1.55% | 866,496 (-5.4%) | 106,423,296 (-12.0%) |
| 3 | Dynamic 75% keep — independent scoring (Exp. 06/07) | 0.6597 | 0.3794 | +1.83% | 866,496 (-5.4%) | 106,423,296 (-12.0%) |
| 4 | Random 25% pruning, mean of 5 seeds (Exp. 08) | 0.6867 | 0.3799 | +1.97% | 866,496 (-5.4%) | 106,423,296 (-12.0%) |
| 5 | Magnitude-based 25% pruning (Exp. 08) | 0.6684 | 0.3810 | +2.26% | 866,496 (-5.4%) | 106,423,296 (-12.0%) |
| = | Static 25% pruning, fine-tuned (Exp. 09) | 0.6537 | 0.3783 | +1.55% | 866,496 (-5.4%) | 106,423,296 (-12.0%) |

**Replication on ETTm1 (Exp. 12), same architecture and 25% pruning ratio:**

| Setting | Val MSE | Test MSE | Test vs. baseline |
|---|---:|---:|---:|
| ETTm1 no pruning (baseline) | 0.3789 | **0.2876** | — |
| **ETTm1 dynamic 75% keep — joint/greedy selection** | **0.3582** | **0.2889** | **+0.45%** |
| ETTm1 static 25% pruning | 0.3607 | 0.2980 | +3.60% |

**Key findings:**
1. **Regime-specific head importance is real**: some heads are harmful for one regime and helpful for another (e.g. Layer 1 / Head 1: negative for seasonal windows, positive for residual windows).
2. **Every importance-driven pruning method beats random pruning on validation** — the pruning signal is meaningful, not noise. But **only the unpruned baseline is best on the test set**; every pruning method costs some test accuracy, and the ranking on validation does not match the ranking on test for any method.
3. **The mask-construction criterion matters more than fine-tuning.** Naive fine-tuning after pruning (Exp. 09, as proposed) did not help — it consistently overfit at every learning rate tried. Switching from independent per-head scoring to **joint/greedy selection that accounts for head interactions** (Exp. 10) did help, and produced the best pruning result of the project.
4. **Regime-labeling noise contributes to the generalization gap**: pruning-induced degradation on the test set is ~2x larger on windows with ambiguous STL regime labels than on confidently-labeled ones, across every method (Exp. 10).
5. **FLOPs/parameter savings from pruning are real but modest (~5-12%) and do not automatically translate into GPU speedup** at this model scale; more importantly, STL-based regime detection itself (~83 ms/window on CPU) is ~300x slower per sample than the pruned model's forward pass, making it the actual bottleneck for any real deployment of the dynamic method (Exp. 11).
6. **The core finding replicates on a second dataset (ETTm1, Exp. 12), even more strongly.** Joint/greedy selection again beats static pruning on both validation and test; its test-side regression (+0.45%) is ~8x smaller than static pruning's (+3.60%), a much larger gap than on ETTh1 (+1.31% vs. +1.55%) — plausibly because ETTm1's test set has a far more balanced regime mix (33%/47%/20% trend/seasonal/residual) than ETTh1's trend-dominated one (93.6%), giving the dynamic method more real opportunity to matter.

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

## Pruning and Regime-Aware Evaluation Experiments

After selecting the B4 PatchTST baseline and completing the STL-based regime detection and attention head importance analysis, we evaluated several pruning and head selection strategies. The goal of these experiments was to test whether attention heads in the selected PatchTST model contain redundancy and whether regime-specific head selection can improve forecasting performance.

> **Note on validation MSE values:** The "B4 no pruning" validation MSE reported in Experiments 04-06 below (0.6781) is recomputed independently in those notebooks via `compute_regime_losses_with_mask` on a rebuilt validation loader, whereas the Baseline Selection table above reports the training-time validation loss (0.6735) from `Time-Series-Library`'s `vali()` function. Both use the same B4 checkpoint, but the two loaders/aggregation methods are not identical, so the two numbers differ slightly (~0.7%). All within-notebook comparisons (baseline vs. pruned) are still valid since they use the same evaluation code path.

## Experiment 04: Static 25% Head Pruning

In this experiment, we evaluated a static pruning strategy using the head importance scores computed on the validation set. The six heads with the lowest overall importance scores were pruned from the B4 model, corresponding to 25% of all attention heads.

The selected static pruning mask removed 6 out of 24 heads and kept 18 active heads. Validation results showed that static pruning improved the B4 baseline:

| Setting | Active Heads | Pruned Heads | Validation MSE | Validation MAE |
|---|---:|---:|---:|---:|
| B4 no pruning | 24 | 0 | 0.6781 | 0.5551 |
| Static 25% pruning | 18 | 6 | 0.6537 | 0.5507 |

This corresponds to a 3.60% reduction in validation MSE. However, on the test set, static pruning increased the test MSE from 0.3725 to 0.3783. This suggests that the static pruning mask improves validation performance but does not generalize perfectly to unseen test data.

## Experiment 05: Dynamic Regime-Aware 50% Keep

The first dynamic regime-aware experiment used a more aggressive setting where each input window used only 50% of the attention heads. For each regime, trend, seasonal, and residual, the top 12 heads were selected based on regime-specific importance scores.

Unlike static pruning, this approach applies a different mask depending on the regime label of each input window:


trend window     -> trend-specific top 12 heads
seasonal window  -> seasonal-specific top 12 heads
residual window  -> residual-specific top 12 heads

Validation results showed that dynamic 50% keep improved the baseline but underperformed static pruning:

| Setting | Active Heads | Pruned Heads | Validation MSE | Validation MAE |
|---|---:|---:|---:|---:|
| B4 no pruning | 24 | 0 | 0.6781 | 0.5551 |
| Dynamic 50% keep | 12 | 12 | 0.6678 | 0.5553 |

Dynamic 50% keep reduced validation MSE by 1.51% compared to the unpruned baseline. However, regime-level analysis showed that this setting degraded seasonal and residual windows. Since it pruned 50% of the heads, this setting was considered too aggressive for a fair comparison against static 25% pruning.

## Experiment 06: Dynamic Regime-Aware 75% Keep

To make the dynamic method comparable with static 25% pruning, we evaluated a second dynamic setting with 75% keep ratio. In this setting, each regime-specific mask keeps 18 heads and prunes 6 heads, matching the same pruning ratio as the static 25% pruning experiment.

| Setting | Active Heads | Pruned Heads | Validation MSE | Validation MAE |
|---|---:|---:|---:|---:|
| B4 no pruning | 24 | 0 | 0.6781 | 0.5551 |
| Static 25% pruning | 18 | 6 | 0.6537 | 0.5507 |
| Dynamic 50% keep | 12 | 12 | 0.6678 | 0.5553 |
| Dynamic 75% keep | 18 | 6 | 0.6597 | 0.5512 |

Dynamic 75% keep improved validation MSE by 2.71% compared to the unpruned B4 baseline. It also performed better than the dynamic 50% setting, confirming that the 50% keep ratio was too aggressive. However, static 25% pruning still achieved the best validation result.

Regime-level validation analysis showed that dynamic 75% keep mainly improved trend-dominant windows while slightly degrading seasonal and residual windows. This indicates that regime-aware selection provides useful signals, but the current head selection strategy is still not optimal across all regimes.

## Experiment 07: Test Regime Detection and Dynamic 75% Test Evaluation

Since dynamic selection requires a regime label for each input window, we generated STL-based regime labels for the ETTh1 test windows. The test set was highly imbalanced toward the trend regime:

| Regime | Count | Percentage |
|---|---:|---:|
| Trend | 2606 | 93.57% |
| Seasonal | 125 | 4.49% |
| Residual | 54 | 1.94% |

After generating test regime labels, we evaluated the dynamic 75% keep strategy on the test set and compared it with the unpruned B4 baseline and static 25% pruning.

| Setting | Active Heads | Pruned Heads | Test MSE | Test MAE |
|---|---:|---:|---:|---:|
| B4 no pruning | 24 | 0 | 0.3725 | 0.3982 |
| Static 25% pruning | 18 | 6 | 0.3783 | 0.4044 |
| Dynamic 75% keep | 18 | 6 | 0.3794 | 0.4053 |

On the test set, the unpruned B4 model achieved the best performance. Static 25% pruning increased test MSE by 1.55%, while dynamic 75% keep increased test MSE by 1.83%. Regime-level test analysis showed that dynamic 75% keep improved residual-dominant windows but degraded trend and seasonal windows. Since the test set is overwhelmingly trend-dominant, the trend degradation dominated the overall test result.

## Experiment 08: Random and Magnitude-Based Baseline Pruning

The project proposal specifies four baselines: no pruning, random head pruning, global importance pruning, and magnitude-based pruning. Experiments 04-07 covered no pruning and global (loss-based) importance pruning. [notebooks/08_random_and_magnitude_baseline_pruning.ipynb](notebooks/08_random_and_magnitude_baseline_pruning.ipynb) adds the two missing baselines, reusing the same B4 checkpoint, `HeadMaskController`, and evaluation functions as notebook 04:

- **Random head pruning**: 6 of 24 heads are pruned at random, repeated over 5 seeds (0-4) and averaged (mean ± std) to avoid reporting a lucky/unlucky draw.
- **Magnitude-based pruning**: each head's importance is approximated by the L2 norm of its `query_projection`/`key_projection`/`value_projection` weight rows and `out_projection` weight columns; the 6 lowest-norm heads are pruned.

| Setting | Active Heads | Pruned Heads | Validation MSE | Test MSE |
|---|---:|---:|---:|---:|
| B4 no pruning | 24 | 0 | 0.6781 | 0.3725 |
| Static 25% pruning (global importance) | 18 | 6 | 0.6537 | 0.3783 |
| Dynamic 75% keep (regime-aware) | 18 | 6 | 0.6597 | 0.3794 |
| Magnitude-based 25% pruning | 18 | 6 | 0.6684 | 0.3810 |
| Random 25% pruning (mean of 5 seeds) | 18 | 6 | 0.6867 | 0.3799 |

Random pruning increased validation MSE by 1.27% and test MSE by 1.97% relative to the unpruned baseline, confirming that the 25% pruning ratio itself is not "free" — removing heads without any importance signal reliably hurts performance in both directions (unlike static/dynamic pruning, which improved validation MSE).

Magnitude-based pruning improved validation MSE by 1.43% (better than random, but weaker than the loss-based static and dynamic methods), yet it produced the **worst test MSE of any method evaluated** (+2.26%), even behind random pruning (+1.97%). This suggests that weight-norm is a poor proxy for functional head importance in this model: the heads with the smallest projection weights are not reliably the least useful ones at inference time, and optimizing against a weak proxy can generalize worse than not optimizing at all. This strengthens the case for the project's core approach of measuring importance directly from validation loss (and, further, per regime) rather than from a static weight-magnitude heuristic.

Across all five settings, the unpruned B4 baseline remains the best performer on the test set, and the ranking by test MSE (baseline < static < dynamic < random < magnitude) does not match the ranking by validation MSE (static < dynamic < magnitude < baseline < random) for any pruned method. Closing this validation/test gap — most likely by fine-tuning the pruned models rather than evaluating the pruned mask zero-shot — is the main open problem going forward (see Summary of Findings and Future Work below).

## Experiment 09: Fine-Tuning the Static 25% Pruned Model (Policy A)

The project proposal's Policy A calls for fine-tuning after pruning ("Heads with low scores will be pruned, and the model will be fine-tuned"), a step Experiment 04 did not include — it evaluated the static 25% pruned mask zero-shot. [notebooks/09_static_pruning_finetune.ipynb](notebooks/09_static_pruning_finetune.ipynb) adds this missing step: the same 6 heads stay pruned (mask always active), and the model is fine-tuned for a few epochs at a reduced learning rate (1e-5, up to 5 epochs, early stopping patience 2) so the 18 active heads and the rest of the network can adapt to the missing heads, before re-evaluating on validation and test.

| Setting | Validation MSE | Test MSE | Test MSE vs. baseline |
|---|---:|---:|---:|
| B4 no pruning | 0.6781 | 0.3725 | — |
| Static 25% pruning (zero-shot, Experiment 04) | 0.6537 | 0.3783 | +1.55% |
| Static 25% pruning (fine-tuned, Policy A complete) | 0.6537 | 0.3783 | +1.55% |

**Fine-tuning did not improve on the zero-shot pruned model, and early stopping correctly reverted to the zero-shot checkpoint** — the numbers above are identical because the fine-tuning run never found a checkpoint better than the starting point. This was tested at two learning rates (1e-5 with 5 max epochs/patience 2, and a 10x lower 1e-6 with 10 max epochs/patience 4), and both showed the same qualitative pattern:

| Epoch | Train MSE | Masked Validation MSE (lr=1e-6 run) |
|---:|---:|---:|
| 0 (zero-shot) | — | 0.6537 |
| 1 | 0.3556 | 0.6605 |
| 2 | 0.3532 | 0.6680 |
| 3 | 0.3521 | 0.6725 |
| 4 | 0.3517 | 0.6737 |

Train MSE decreases monotonically while masked validation MSE increases monotonically from the very first epoch, at both learning rates tried. This is a classic overfitting signature, not a learning-rate-tuning problem: B4 was already trained to a validation-loss optimum with all 24 heads active, and *any* further gradient steps on the training set — even at a very low learning rate — pull the remaining active heads away from that optimum and toward the (comparatively small) ETTh1 training set, rather than finding a better shared optimum for the pruned architecture. Practical implications for Policy A: naive fine-tuning of the already-converged baseline is not the right lever here; more promising directions (not yet tried) include fine-tuning with a validation-loss-aware regularizer or a much smaller subset of parameters (e.g. only the layers immediately downstream of the pruned heads), or restarting training with the mask active from an earlier, less-converged checkpoint rather than fine-tuning the fully-converged B4 model.

## Experiment 10: Joint (Greedy) Regime-Aware Head Selection and Confidence-Margin Robustness Check

Experiments 05-07 built Policy B by scoring each head *independently* per regime (leave-one-head-out) and keeping the top-k by that score — an approach that cannot see interactions between heads (removing heads A and B together can hurt more, or less, than the sum of their individual effects). [notebooks/10_joint_regime_head_selection.ipynb](notebooks/10_joint_regime_head_selection.ipynb) replaces this with **greedy backward elimination**: per regime, starting from all 24 heads active, it repeatedly removes whichever single head least harms masked validation MSE — with prior removals already applied — until only 18 remain (matching the 75% keep ratio of Experiment 06 for a fair comparison). It also regenerates the test-set STL regime labels (produced in Experiment 07 but never committed to this repository) and uses each window's `confidence_margin` to check whether dynamic selection performs better on confidently-labeled windows than on ambiguous ones — a direct test of whether regime-labeling noise is part of why the validation/test generalization gap has been so persistent throughout this project.

| Setting | Validation MSE | Test MSE | Test MSE vs. baseline |
|---|---:|---:|---:|
| B4 no pruning | 0.6781 | 0.3725 | — |
| Dynamic 75% keep (independent scoring, Experiment 06/07) | 0.6597 | 0.3794 | +1.83% |
| Static 25% pruning | 0.6537 | 0.3783 | +1.55% |
| **Dynamic 75% keep (joint/greedy selection)** | **0.6536** | **0.3774** | **+1.31%** |

**Joint (greedy) selection is the best pruning method evaluated in this project so far**, on both validation and test. It essentially ties static pruning on validation (0.6536 vs 0.6537) and clearly beats it on test (+1.31% vs +1.55%), and it improves substantially over the independent-scoring dynamic method it replaces (+1.31% vs +1.83% on test — roughly a **29% reduction in the test-side regression** caused by pruning). Broken down by regime on the test set, joint selection wins on trend (0.3789 vs 0.3810, the dominant regime at 93.6% of test windows) and seasonal (0.3486 vs 0.3556), while independent scoring is marginally better on the small residual regime (0.3569 vs 0.3719, only 54 windows). Since trend dominates the test set so heavily, the trend-regime improvement drives the overall win. This confirms the hypothesis motivating this experiment: independent per-head scoring misses interactions between heads that a joint (interaction-aware) selection can recover.

**Confidence-margin breakdown.** On the test set, every pruning method's degradation relative to the unpruned baseline is roughly twice as large on ambiguous windows (`confidence_margin < 0.05`) as on confidently-labeled ones (static: +1.47% confident vs +2.78% ambiguous; independent dynamic: +1.76% vs +3.09%; joint dynamic: +1.28% vs +1.74%). This supports the idea that regime-labeling noise is part of why pruning does not generalize as well as it does on validation: windows where the model's regime is itself unclear are also the windows where losing any heads hurts the most, regardless of which pruning method is used. Joint selection has the smallest gap between the two subsets of any method, consistent with it being the most robust option overall. On validation the picture is noisier (the ambiguous subset is small, 295 windows): static and joint pruning *improve more* on ambiguous validation windows than confident ones, while independent-scoring dynamic pruning is the only method that is actually worse than baseline on ambiguous validation windows (+3.24%) despite improving on confident ones (-3.40%) — a concrete example of independent scoring choosing a mask that is not robust to label uncertainty, which joint selection avoids.

## Experiment 11: Efficiency Metrics (Parameter Count, FLOPs, Inference Latency)

The proposal asks for parameter count, FLOPs, and inference latency alongside accuracy for every method, but Experiments 04-10 all implement pruning as *soft/functional* masking (heads are zeroed out after being computed, not removed) — measuring efficiency on those models directly would show no difference from the baseline, which would misrepresent what pruning actually buys. [notebooks/11_efficiency_metrics.ipynb](notebooks/11_efficiency_metrics.ipynb) performs **structural pruning** instead: it physically removes the pruned heads' rows/columns from each layer's projection matrices, verifies the resulting smaller model produces the same output (up to floating-point rounding) as the corresponding soft-masked model, and then measures real parameter count, FLOPs (via `thop`), and GPU inference latency (batch of 32, averaged over 50 timed runs after warm-up).

Static pruning (Experiment 04) becomes one smaller model. The joint dynamic method (Experiment 10, the best-performing method overall) becomes **three** smaller models — one per regime — since a deployment would route each window to the sub-model matching its detected regime; the notebook reports both the three individual costs and a regime-frequency-weighted "expected" cost using the test-set regime distribution (~93.6% trend / ~4.5% seasonal / ~1.9% residual, from Experiment 07/10). It also separately benchmarks the wall-clock cost of the STL regime-detection step itself, a real per-window cost that Policy B (dynamic) pays and Policy A (static)/the unpruned baseline do not.

All structurally pruned models passed the correctness check (max output difference vs. the corresponding soft-masked model was below the 1e-3 floating-point tolerance).

| Setting | Parameters | vs. baseline | FLOPs | vs. baseline | Latency (ms/batch of 32) | vs. baseline |
|---|---:|---:|---:|---:|---:|---:|
| B4 no pruning | 915,936 | — | 120,873,984 | — | 8.98 | — |
| Static 25% pruning | 866,496 | -5.40% | 106,423,296 | -11.96% | 8.91 | -0.81% |
| Joint dynamic — trend sub-model | 866,496 | -5.40% | 106,423,296 | -11.96% | 9.36 | +4.21% |
| Joint dynamic — seasonal sub-model | 866,496 | -5.40% | 106,423,296 | -11.96% | 12.43 | +38.40% |
| Joint dynamic — residual sub-model | 866,496 | -5.40% | 106,423,296 | -11.96% | 13.72 | +52.78% |
| Joint dynamic — expected (regime-frequency-weighted) | 866,496 | -5.40% | 106,423,296 | -11.96% | 9.58 | +6.69% |

STL regime-detection overhead: **83.3 ± 51.8 ms/window** (CPU, `statsmodels` STL, averaged over 100 validation windows) — a cost unique to the dynamic method that the static/unpruned settings do not pay.

**Parameters and FLOPs are trustworthy and consistent**: pruning 25% of heads (6/24) removes 5.40% of total parameters and 11.96% of FLOPs for *every* pruned variant, regardless of which specific heads are removed — the gap between the head-level pruning ratio and the whole-model savings is expected, since attention heads are only one part of PatchTST (patch embedding, feed-forward layers, and the output head are all untouched by head pruning).

**Latency numbers should be treated with caution.** Static pruning and the joint-trend sub-model have the *exact same* per-layer head-count distribution (6/5/7 active heads across the three layers) and therefore identical tensor shapes and FLOPs, yet their measured latencies differ by 5 percentage points (-0.81% vs. +4.21%) — architecturally identical models should not show a real difference this large. The latency measurements were taken sequentially within a single Colab GPU session in the order (baseline, static, trend, seasonal, residual), and the measured latency increases monotonically in that same order (-0.81% → +4.21% → +38.40% → +52.78%), which is a strong indicator of a **sequential-benchmark artifact** (GPU clock/thermal drift, memory allocator fragmentation, or background contention building up over the run) rather than a genuine property of the pruned architectures. A more reliable measurement would benchmark each model in a freshly restarted runtime, or interleave/randomize the benchmarking order and average over multiple repetitions — noted here as a limitation rather than re-run, given time constraints. What *is* reliable is that structural pruning at this scale (a handful of attention heads removed from an already compact `d_model=128` model) does not guarantee a GPU speedup even though it reliably reduces FLOPs — small matrix multiplications are often bottlenecked by kernel-launch overhead and memory bandwidth rather than raw compute, so a "12% fewer FLOPs" result does not automatically translate into "12% faster."

**The STL overhead is the real headline number for Policy B's practical cost.** At ~83 ms/window on CPU, STL-based regime detection alone is roughly **9x slower than the entire unpruned model's forward pass on a batch of 32 windows** (83 ms vs. 8.98 ms), and per-sample it dwarfs the ~0.28 ms/sample forward-pass cost by nearly 300x. If regime detection must run per-window at serving time, Policy B's end-to-end latency would be completely dominated by this STL step, not by any savings (or losses) from the pruned forward pass — any practical deployment of the dynamic method would need a much faster regime-detection proxy (e.g. a lightweight learned classifier, or caching/batching STL across windows) for the approach to be worth adopting on efficiency grounds, independent of its accuracy benefits.

## Experiment 12: Extension to ETTm1

The proposal states the method would be extended to other datasets such as ETTm1 and Weather if regime-specific head importance patterns were clear on ETTh1 — which Experiments 03-10 confirmed (e.g. heads with opposite-signed importance across regimes, and joint selection beating every other pruning criterion tried). [notebooks/12_ettm1_extension.ipynb](notebooks/12_ettm1_extension.ipynb) tests whether this transfers to **ETTm1**, the same electricity-transformer sensor at 15-minute instead of hourly resolution (same 7 columns, ~4x more windows for the same real-world span). It trains the same B4 architecture directly on ETTm1 (no re-sweep of B0-B4), adapts STL's seasonal period from 24 to 96 steps (still one day), and re-tests only the two strongest ETTh1 methods — static 25% pruning and joint/greedy dynamic 75% keep — against the unpruned baseline; the other ETTh1 baselines (random, magnitude, independent-scoring dynamic, fine-tuning, efficiency metrics) already answered their respective methodological questions and are not repeated here.

| Setting | Validation MSE | Test MSE | Test MSE vs. baseline |
|---|---:|---:|---:|
| ETTm1 no pruning | 0.3789 | **0.2876** | — |
| ETTm1 static 25% pruning | 0.3607 | 0.2980 | +3.60% |
| **ETTm1 dynamic 75% keep (joint/greedy)** | **0.3582** | **0.2889** | **+0.45%** |

**The core finding replicates on ETTm1, and more strongly than on ETTh1.** Joint/greedy dynamic selection again beats static pruning on both validation (-5.46% vs. -4.80% relative to baseline) and test (+0.45% vs. +3.60%) — and here the margin between the two methods is much larger: joint selection's test-side regression is roughly **8x smaller** than static pruning's (+0.45% vs. +3.60%, compared to a 1.2x difference on ETTh1: +1.31% vs. +1.55%). Static pruning's zero-shot mask, tuned only for validation loss, degrades test performance substantially more on ETTm1 than it did on ETTh1, while joint selection stays close to the unpruned baseline on both splits. One plausible contributor: ETTm1's test set has a much more balanced regime mix (33.2% trend / 47.0% seasonal / 19.8% residual, from `ettm1_joint_test_regime_summary.csv`) than ETTh1's test set (93.6% trend), so a *dynamic* method that genuinely switches masks per regime has more opportunity to matter — and more to lose from a poor (independent-scoring or static) mask — than on a test set dominated by a single regime. This is consistent with the project's central hypothesis: regime-conditional head selection matters most exactly when a dataset's temporal regimes are themselves well-represented and varied.

## Summary of Findings

With Experiment 08, all four baselines requested in the project proposal (no pruning, random pruning, global importance pruning, magnitude-based pruning) plus the proposed dynamic regime-aware method have now been evaluated at a matched 25% pruning ratio on B4. The results show that attention head redundancy exists in the selected PatchTST baseline, especially on the validation set: static (loss-based) pruning, dynamic regime-aware selection, and even magnitude-based pruning all improved validation MSE compared to the unpruned baseline, while random pruning made it worse. This confirms the pruning signal itself is meaningful — an importance-agnostic 25% cut reliably hurts (random), while every criterion that uses *some* importance signal helps on validation.

However, test-set evaluation showed that none of these validation-side improvements generalize completely: the unpruned B4 model remained the best on the test set, and every pruning method increased test MSE to some degree (independent-scoring dynamic +1.83%, random +1.97%, magnitude +2.26%). Magnitude-based pruning is notable for being the *worst* method on the test set despite being a deterministic, importance-driven heuristic — worse even than random pruning — indicating that weight-norm is a particularly poor proxy for head importance in this model.

Experiment 09 attempted to close this gap via Policy A's missing fine-tuning step. The attempt was unsuccessful in a specifically informative way: at two different learning rates, further training of the pruned model consistently overfit the training set (falling train loss, rising masked validation loss from epoch 1 onward) rather than finding a better optimum for the reduced head set. This showed that the validation/test generalization gap is not simply a matter of "not having fine-tuned yet" — naive fine-tuning of an already-converged checkpoint is not sufficient to fix it.

Experiment 10 then closed most of the gap through a different lever: replacing independent per-head scoring with **joint (greedy) head selection** — which accounts for interactions between heads rather than scoring them in isolation. This produced the best pruning result of the entire project: test MSE +1.31% (vs. +1.55% for static and +1.83% for independent-scoring dynamic), while matching static pruning's validation-side improvement almost exactly. The confidence-margin analysis further showed that pruning-induced degradation on the test set concentrates on windows with ambiguous regime labels (roughly 2x larger than on confidently-labeled windows, across every method), supporting the idea that STL regime-labeling noise — not just the choice of pruning criterion — is a real contributor to the validation/test gap, and that joint selection is comparatively more robust to that noise than independent scoring.

Experiment 11 added the efficiency side of the comparison the proposal asked for. Structural pruning confirmed that a 25% head cut reliably removes 5.4% of parameters and 11.96% of FLOPs regardless of which specific heads are chosen — but that this does not automatically translate into a measured GPU speedup at this model scale, and exposed a much larger practical cost specific to Policy B: STL-based regime detection (~83 ms/window on CPU) is nearly 300x slower per sample than the pruned model's own forward pass, meaning the dynamic method's real-world efficiency case depends entirely on replacing that detection step with something far cheaper, independent of its accuracy advantage.

Overall, the results across Experiments 04-11 (ETTh1) show that regime-aware head selection is a promising direction that clearly outperforms naive baselines (random, magnitude pruning) and, once head interactions are accounted for via joint selection, comes closer to matching the unpruned baseline on test accuracy than any other pruning method tried, while also removing a real (if modest) share of parameters and FLOPs. The mask-construction criterion (independent vs. joint) mattered far more than fine-tuning did for closing the generalization gap, and the accuracy story (Experiments 04-10) turned out to be more tractable than the efficiency story (Experiment 11), where regime detection overhead — not the pruned forward pass — is the actual bottleneck.

Experiment 12 confirmed this is not an ETTh1-specific artifact: on ETTm1, joint/greedy selection again beat static pruning on both validation and test, and by a substantially *larger* margin than on ETTh1 (test regression 8x smaller than static's, vs. 1.2x smaller on ETTh1) — plausibly because ETTm1's more balanced regime distribution gives a genuinely dynamic method more opportunity to add value than ETTh1's trend-dominated test set does. This strengthens the project's central claim: attention head importance is regime-dependent, accounting for that dependence with an interaction-aware (joint) selection method measurably improves generalization over both a single global mask and a naively regime-conditional one, and the benefit appears to scale with how much regime diversity a dataset actually has.

Future work should extend joint selection beyond the greedy/backward-elimination heuristic (e.g. a more global combinatorial search), pursue more stable and cheaper regime labeling (both for accuracy — the confidence-margin results suggest ambiguous windows are a concrete target — and to make Policy B's latency viable), re-measure latency with a methodology that isolates GPU state per model (fresh runtimes or randomized/repeated ordering) rather than a single sequential pass, measure efficiency metrics on ETTm1 as well (Experiment 11 was ETTh1-only), and evaluate the method on additional prediction lengths and the Weather dataset.

## Project Members

- Didem Neda Aksaç
- Betül Aydeğer
