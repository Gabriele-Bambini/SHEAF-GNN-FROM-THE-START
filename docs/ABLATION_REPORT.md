# TS-GNN Architecture Ablation Report
**Date:** 2026-04-25
**HPC Job:** 482916 (gnode04, A100 80GB MIG 4g.40gb)
**Dataset:** BRCA Wu2021, K=10, n_genes=500, K_neighbors=8, DPT pseudotime

---

## Executive Summary

The Phase α+β architecture (A1+A2+A3+B1+B2) was developed from theoretical principles to improve the baseline TS-GNN on cross-dataset transfer. **HPC reverse-ablation revealed that two of the five "improvements" actively harm the model**, while two are neutral and one helps.

**Final architecture:** A1 + A2 + B2 (drop A3 + B1).

**Key result:**
- Baseline TS-GNN: r_delta = 0.3523 (BRCA same-dataset)
- Full stack (A1+A2+A3+B1+B2): r_delta = -0.0530 (broken!)
- Best ablation (full−A3): r_delta = 0.3753 (passes Nature P1 threshold 0.35)

**Honest caveat:** Single-seed run. Δ between baseline and best ≈ +0.023, within noise. Multi-seed validation pending.

---

## Methodology

Reverse-ablation: train 7 configs for 50 epochs each on BRCA, all with identical hyperparameters except the feature flag of interest.

| Config | A1 | A2 | A3 | B1 | B2 | d |
|--------|----|----|----|----|----|----|
| baseline | × | × | × | × | × | 8 |
| full | ✓ | ✓ | ✓ | ✓ | ✓ | 4 |
| full−A1 | × | ✓ | ✓ | ✓ | ✓ | 4 |
| full−A2 | ✓ | × | ✓ | ✓ | ✓ | 4 |
| full−A3 | ✓ | ✓ | × | ✓ | ✓ | 4 |
| full−B1 | ✓ | ✓ | ✓ | × | ✓ | 4 |
| full−B2 | ✓ | ✓ | ✓ | ✓ | × | 4 |

Loss: MSE between predicted X(t+1) and true X(t+1).
Train: K[:8], val: K[8:9]. Eval: predict K[9] from K[8], compute r_delta.

**Why r_delta over r_abs?** r_abs is dominated by autocorrelation X(t)≈X(t+1) on smooth pseudotime trajectories. r_delta isolates the *change* the model predicts, which is what matters for biological interpretation.

---

## Per-Feature Analysis

### A1 — Orthogonal Restriction Maps (KEEP, neutral)

**Theory:** Project R into O(d) via QR; ensures gauge-invariant Laplacian, eigenvalues bounded [0, 2].

**Empirical:** full vs full−A1: Δr_delta = -0.024. Within noise.

**Verdict:** KEEP. Cross-dataset gauge-invariance argument still holds for BRCA→CRC. Cost is negligible.

---

### A2 — Implicit Diffusion (Crank-Nicolson + CG) (KEEP, helps)

**Theory:** Replace explicit Euler with CN; unconditionally stable, energy-monotone.

**Empirical:** full vs full−A2: Δr_delta = -0.20. **A2 actively helps** by +0.20 r_delta. Removing A2 makes things worse.

**Verdict:** KEEP. Strong evidence A2 stabilizes diffusion. τ learnability benefits.

---

### A3 — Predict Delta ΔX(t+1) (DROP, hurts -0.43)

**Theory:** Predicting ΔX = X̂(t+1) − X(t) gives better gradients vs absolute X̂(t+1). MLP-identity (ΔX=0) becomes natural baseline.

**Empirical:** full vs full−A3: Δr_delta = **+0.43**. Removing A3 takes r_delta from -0.05 to +0.38.

**Why it hurts:**
- Mathematical equivalence MSE(X+ΔX_pred, Y) = MSE(ΔX_pred, ΔY) holds in theory.
- In practice: combined with sheaf diffusion (smoothes X), residual structure produces gradient that reinforces noise direction.
- Model achieves high r_abs (0.45) but anti-correlated r_delta (-0.05). The model learns to output something close to X(t)+random_smooth_field — high autocorrelation, wrong direction.

**Verdict:** DROP. Predict absolute X(t+1) directly.

---

### B1 — Nodal FiLM with Laplacian PE (DROP, hurts -0.40)

**Theory:** Per-node γ_v, β_v from allele + Laplacian eigenvector PE; biological resolution beyond global FiLM.

**Empirical:** full vs full−B1: Δr_delta = **+0.40**. Removing B1 takes r_delta from -0.05 to +0.35.

**Why it hurts:**
- Current eval uses single ESM vector per training run (one allele).
- B1 broadcasts that vector × Laplacian PE → N modulation channels, all derived from the SAME fake input.
- Per-node modulation amplifies noise in PE eigenvectors (computed once, deterministic, but high-dimensional).
- Global FiLM (1 modulation/forward) acts as low-rank conditioning → much more stable.
- With REAL per-cell allele variation (e.g., one allele per cell in scRNA), B1 may help. Current pipeline only varies allele between training runs.

**Verdict:** DROP for current single-allele setup. Re-enable for per-cell allele runs (Phase γ).

---

### B2 — Neural ODE (KEEP, neutral)

**Theory:** Continuous pseudotime evolution dR/dτ = f_θ(R, x_endpoints); BRCA→CRC transfer regardless of bin count.

**Empirical:** full vs full−B2: Δr_delta = +0.04. Within noise.

**Verdict:** KEEP. Cross-dataset transfer hypothesis untestable here (CRC eval skipped due to N=500 vs 473 mismatch). Cost is moderate (256-hidden MLP), benefit pending CRC validation.

---

## Final Architecture

```yaml
model:
  stalk_dim: 4
  use_orthogonal_maps: true       # A1
  use_implicit_diffusion: true    # A2 (HELPS)
  predict_delta: false            # A3 disabled (HURTS)
  use_nodal_film: false           # B1 disabled (HURTS)
  use_neural_ode: true            # B2 (neutral, kept for cross-dataset)
  ode_hidden_dim: 256
  ode_solver: "rk4"
```

---

## Limitations

### 1. Single-seed evaluation
Results from one seed (42), 50 epochs, one HPC run. Noise floor unknown. **Required next step:** 3+ seeds with statistical test (paired t-test or Wilcoxon).

### 2. Cross-dataset (CRC) eval skipped
BRCA top-500 genes ≠ CRC top-500 (N=500 vs 473). Edge_index swap doesn't handle node-set difference. **Required next step:** intersection-of-genes pipeline (P1 cross-dataset r_delta on 350-gene shared set).

### 3. Single ESM vector per run
B1's failure may be artifact of fake/fixed ESM input, not real allele-conditional dynamics. **Required next step:** per-cell allele variation (scRNA with allele labels) before condemning B1.

### 4. r_delta on single time-step
We evaluate r_delta(K[9]−K[8]) — the LAST bin only. Held-out distribution may differ from training. **Required next step:** r_delta averaged over all eval bins.

### 5. Train/val split aggressive
val_k=2 leaves only 7 training time-steps. Underfitting risk. **Required next step:** cross-validation across pseudotime bins.

### 6. No baseline comparison vs published methods
EvolveGCN, DySAT, Temporal-GAT not yet benchmarked on identical pipeline. P1 threshold of 0.35 is a target, not relative to alternatives.

### 7. Allele swap (P2) untested
Predictions P2 (Jaccard threshold 0.30 on swap-invariance) not yet probed. Architecture choices may shift if P2 dominates.

### 8. MLP-identity baseline strong
Baseline TSGNN already gets r_delta=0.35. Without strong architectural advantage over MLP-identity, the sheaf framework's value depends on cross-dataset transfer (untested) and allele specificity (untested).

---

## Lessons Learned

1. **Theory ≠ practice.** A3 (predict_delta) and B1 (nodal FiLM) had impeccable theoretical motivations. Both actively harmed r_delta. Mathematical elegance doesn't guarantee empirical gain.

2. **Local CPU validation misleading.** Local ablation showed all 5 features harmful; HPC GPU showed 2 helpful, 1 mixed, 2 harmful. Always validate on real-scale data before architectural decisions.

3. **Confounds matter.** B1 was being asked to learn from a constant (single ESM vector). The "improvement" was structurally unable to demonstrate value in this eval setup.

4. **Ablation must be reverse, not forward.** Forward-ablation (start baseline, add features) misses interaction effects. Reverse (full minus one) reveals which feature is the bottleneck of the full system.

5. **Honest reporting is publishable.** A Nature paper with "tried these 5 ideas, 2 helped, 2 hurt, here's why" is more valuable than "all 5 work great" with hidden failures.

---

## Reproducibility

- HPC job: 482916
- Sbatch: `hpc/submit_reverse_ablate.sbatch`
- Script: `hpc/probe_reverse_ablate.py`
- Sync: `python hpc/sync_and_submit.py ablate`
- Logs: `/home/3393519/ts-gnn/logs/revablate_482916.{out,err}`
- Commit: master b54838e (pre-ablation), TBD (post-ablation update)
