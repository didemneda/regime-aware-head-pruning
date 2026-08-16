# ReCAHS — Cluster Experiment Results (v2, August 2026)

Full-scale follow-up to the notebook experiments (01–17): 4 datasets ×
4 horizons × 5 seeds = 80 trained PatchTST models, sparsity sweep at
6 ratios × 6 methods, run on 2×H100 under Slurm in a single pinned
environment (Time-Series-Library `4e938a17`, torch 2.13.0+cu130,
seeds 7/42/1234/2026/3407).

Code: [`cluster/`](cluster/) · Raw results incl. per-window losses:
`cluster_results/runs/` · Aggregated stats & figures:
`cluster_results/runs/analysis/`

## Findings

1. **Global importance ranking (the standard static method) is fragile.**
   It collapses at real sparsity on ETTm1 (+15–46% test MSE) and long
   horizons, where it is beaten even by random pruning. The Week-7
   "everything within noise" picture was an artifact of testing only 25%
   sparsity on ETTh1.

2. **Joint greedy selection fixes it.** Greedy backward elimination
   (pooled or per-regime) stays within ~1% of the unpruned baseline at 25%
   sparsity and degrades gracefully to 75%. Dynamic (greedy, regime-routed)
   vs static (independent ranking): 14/16–15/16 cell wins at 50–75%
   sparsity, sign-test p = 0.004 / 0.0005.

3. **Controlled attribution: the criterion, not the routing.** A pooled
   regime-agnostic greedy mask (static_greedy control) beats the
   regime-routed masks in 12–16 of 16 cells at every ratio. The dynamic
   method's advantage over the standard baseline is explained by joint
   selection, not by regime conditioning.

4. **No deployable router beats one good mask** at 25% sparsity: STL
   routing (1/16), learned regime detector (≈STL, but 45× faster:
   4.3 ms vs 194.5 ms/window), learned oracle-router (2/16),
   confidence-gated router (2/16), confidence fallback (no gain).

5. **Oracle headroom is real.** Routing each test window to its
   loss-minimizing regime mask beats the *unpruned* model in 15/16 cells
   (up to −12%, ETTh2@96: 0.2753 vs 0.3140). Regime masks are genuine
   specialists (matched-mask advantage in 12/16 cells) but per-window
   best-mask identity is not predictable from input features — the open
   problem this work poses.

6. **With retraining, 25% pruning is free.** Scratch-training with the
   mask active: ETTm1@96 0.2905 vs baseline 0.2920 (better), ETTh1@96
   0.3770 vs 0.3745 (+0.7%), at −5.4% parameters / −12% FLOPs.
   Regime-conditional training did not improve on this (tested negative).

7. **Mechanism.** Within a model, 10.9/24 heads on average flip importance
   sign across regimes (permutation-significant on ETTh2); across seeds,
   head rankings barely correlate (τ≈0.05) — head roles are per-model,
   so masks must be selected per model.

## The router study (post-oracle follow-up)

Can *any* inference-time-legal method capture the oracle headroom? Twelve
deployable variants, all compared against pooled static-greedy over the
16 cells (oracle-gain recovery in parentheses):

| Family | Best variant | Wins /16 | Recovery |
|---|---|---:|---:|
| STL / learned regime labels | detector (45× faster) | 1 | <0 |
| Argmin classification | margin-weighted HGB | 2 | −31% |
| Loss / regret regression | centered delta_reg | 1 | −33% |
| Delayed-feedback temporal | rolling regret | 3 | −21% |
| Slow-timescale bandits | block-96 | 5 | −14% |
| Regret-gated deviation | gate@q95 | 6 | −3% |
| **Prediction ensembling** | **top2 / invmse** | **top2: 4/4 datasets** | **+6 to +9%** |

Three falsification tests rule out fixable causes of router failure:
in-sample fits reach 96% of oracle (capacity is sufficient); a 10× router
learning curve is flat on held-out regret (not data-limited); a
train-on-test-half diagnostic and model-representation features also fail
(not distribution shift, not feature poverty). Per-window best-mask
identity is aleatoric at inference time.

**Ensembling is the constructive answer**: averaging the masked
specialists' *predictions* (uniform, val-weighted, or top-2) beats pooled
static-greedy on 3–4 of 4 datasets at horizon 96, recovers +6–9% of the
oracle gain, ties the unpruned baseline on ETTm1 and **beats it on ETTh2
(0.3070 vs 0.3140)** — don't select, combine. Its one failure (weather) is
mechanistically informative: 97%-seasonal labels make the regime masks
nearly identical, leaving no diversity to average.

**Soft regime-conditioned ensembles close the question.** Giving the
STL-detected regime's mask weight α (others share 1−α): recovery falls
monotonically in α — +9.2% (α=0.4) → +7.4% (0.6) → +1.8% (0.8) → negative
(1.0 = hard routing) — and the α=0.4 / regime-inverse-MSE variants merely
tie the regime-blind weighted ensemble (2/4 datasets each way). Regime
information adds no measurable value at inference time, even softly; its
real contribution is *offline*, in constructing the diverse specialist
masks the ensemble averages.

**Diversity-source ablation (closed):** pools of greedy masks selected on
random validation subsamples (same criterion, size, keep-count, no regime
information) ensemble about as well as regime pools on average (regime
wins 2/4 datasets, random 2/4 — random even beats the unpruned baseline
on ETTm1: 0.2909 vs 0.2920), but regime pools have systematically deeper
oracles (4/4 datasets, e.g. ETTh2 0.2730 vs 0.2978). Regime partitioning
is the strongest diversity *source* — its extra oracle depth is exactly
the aleatoric part no method exploits — yet not necessary for the
deployable ensemble gains: the ensemble effect is a diversity effect.

## Statistical toolkit

Per-seed paired tests + cross-cell sign tests; hierarchical circular
moving-block bootstrap (seeds × 96-window blocks, 10k replicates) per
cell; circular-rotation permutation tests for regime dependence;
per-window losses stored for every configuration
(`pruning/window_losses.npz`) so all statistics are recomputable without
re-running models.

## Paper framing

*"Joint selection, not regime routing: a controlled study of adaptive
attention-head pruning for time-series Transformers."* Target venues:
ECML-PKDD / IJCNN / PAKDD main tracks or NeurIPS/ICLR
efficiency/time-series workshops. The one reviewer-facing addition worth
new compute: replicating on a second architecture (iTransformer).
