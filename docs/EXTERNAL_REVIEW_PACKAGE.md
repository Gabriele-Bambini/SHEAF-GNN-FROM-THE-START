# TSGNN — Mathematical Review Package for External LLM

**Purpose**: Self-contained document for external LLM review of the mathematical soundness of a Temporal Sheaf Graph Neural Network (TSGNN) architecture. The model trains successfully but post-hoc bias probes reveal that three architectural claims (allele-specificity via FiLM, sheaf restriction maps, temporal dynamics) are **empirically unsupported**. We need an independent math review to confirm or refute our root-cause hypothesis.

**Author**: Gabriele Bambini, Bocconi University
**Date**: 2026-04-26
**Project**: Cellular sheaf neural networks for cross-dataset gene expression dynamics

---

## 1. PROJECT OVERVIEW

### 1.1 Goal
Predict gene expression at time t+1 from t, transferring across cancer datasets (BRCA → CRC), with allele-specific conditioning from ESM-2 protein embeddings of TP53 mutants.

### 1.2 Pipeline
```
single-cell RNA-seq (BRCA or CRC)
  → DPT pseudotime → K=10 bins
  → top-500 HVG (combined-variance ranked)
  → KNN graph (k=8) over genes
  → TSGNN: input X(t) ∈ ℝ^{N×1}, ESM ∈ ℝ^{1280} → predict X(t+1) ∈ ℝ^{N×1}
  → metric: r_delta = Pearson(Ŷ - X(t), X(t+1) - X(t))
```

### 1.3 Architecture (5 components)
| ID | Component | Math |
|----|-----------|------|
| A1 | Orthogonal restriction maps | F ∈ O(d) via QR(F̃) |
| A2 | Crank-Nicolson implicit diffusion | (I + τ/2 L̂) x_{k+1} = (I - τ/2 L̂) x_k |
| A3 | residual prediction | X̂(t+1) = X(t) + Δ (REJECTED) |
| B1 | Nodal FiLM with Laplacian PE | per-node γ,β (REJECTED) |
| B2 | Neural ODE for restriction evolution | dR/dτ = f_θ(R, x_endpoints) |

Final config: **A1 + A2 + B2** (predict_delta=False, use_nodal_film=False).

---

## 2. MATHEMATICAL FORMULATION

### 2.1 Cellular Sheaf
- Graph G = (V, E), |V|=N nodes (genes), |E|=M edges (KNN).
- Stalk space: ℝ^d at each node (d=4).
- Restriction maps: for each oriented edge e=(u,v), F_{u,e}, F_{v,e} ∈ ℝ^{d×d}.
- Projected to O(d): F_{u,e}, F_{v,e} ← QR(·)[0] (Stiefel/orthogonal manifold).

### 2.2 Connection Laplacian
For each edge e=(u,v):
- Edge weight matrix: W_e = F_{v,e}^T F_{u,e} ∈ ℝ^{d×d}
- Block Laplacian:
  - L_F[u,u] += F_{u,e}^T F_{u,e} = I (since F orthogonal) — diagonal
  - L_F[v,v] += F_{v,e}^T F_{v,e} = I — diagonal
  - L_F[u,v] -= W_e^T = -F_{u,e}^T F_{v,e}
  - L_F[v,u] -= W_e = -F_{v,e}^T F_{u,e}

L_F ∈ ℝ^{Nd × Nd}, symmetric PSD.

### 2.3 Normalized Laplacian
L̂ = D^{-1/2} L_F D^{-1/2}, where D is block-diagonal with D[v,v] = deg(v) · I.

Spectral property after normalization: σ(L̂) ⊆ [0, 2].

### 2.4 Implicit Diffusion (A2: Crank-Nicolson)
One step:
```
(I + τ/2 · L̂) x_{k+1} = (I - τ/2 · L̂) x_k
```
Solved via Conjugate Gradient (max 20 iter, tol 1e-4). τ ∈ [0.01, 1.0] learnable.

Energy E(x) = ½ x^T L̂ x is non-increasing — A-stable.

### 2.5 FiLM Conditioning
ESM ∈ ℝ^{1280} → MLP → z ∈ ℝ^{128} → MLP → (γ, β) ∈ ℝ^{2 · 2d²}

Modulation applied to FLATTENED restriction maps before QR:
```
maps_flat ∈ ℝ^{E × 2d²}        # raw GRU/ODE output
maps_modulated = γ * maps_flat + β    # FiLM
maps_modulated reshape → (E, 2, d, d) → QR → Laplacian
```

Initialization: γ ≈ 1, β ≈ 0 (last layer std=0.01, bias=0).

### 2.6 Forward Pass (per pseudotime step t)
```python
def step(x_t, R_t, esm):
    x_latent = input_proj(x_t)              # (N, d)
    R_new = neural_ode(R_t, x_latent)       # (E, 2, d, d)
    R_modulated = film(esm, R_new.flatten()) # (E, 2d²)
    R_orth = QR(R_modulated.reshape(...))   # (E, 2, d, d) ∈ O(d)
    L_F = compute_laplacian(R_orth)
    L_norm = normalize(L_F)
    x_diffused = implicit_diffusion(x_latent, L_norm)  # CG solve
    x_pred = output_proj(x_diffused)        # (N, 1)
    return x_pred
```

---

## 3. EMPIRICAL OBSERVATIONS (the puzzle)

### 3.1 Cross-dataset transfer WORKS
- Train BRCA, evaluate CRC zero-shot: r_delta = 0.699 ± 0.004 (3 seeds)
- Identity baseline on CRC: r_delta = -0.005
- Mean predictor: r_delta = 0.707
- **Verdict**: model learns real, transferable directional signal.

### 3.2 Bias Probe (job 483174) — 3 critical failures

| Test | Setup | Result | Verdict |
|------|-------|--------|---------|
| **T5** FiLM dead | Train, then eval CRC with ESM=zero vs ESM=randn×0.1 | swing = 0.001 | **FAIL** — ESM has no effect |
| **T6** Pseudotime artifact | Shuffle bin order on CRC | shuffled r_d = 0.7435 > original 0.70 | **FAIL** — shuffling helps |
| **T7** Graph unused | Replace KNN with random edges | random r_d = 0.6987 ≈ 0.70 | **FAIL** — graph irrelevant |

### 3.3 Diagnostic Probe v2 (job 483305) — SMOKING GUN

Capture intermediate activations:
```
Stalks (post-diffusion, just before output projection):
  zero ESM:  mean=-0.0179, std=0.3454
  rand ESM:  mean=-0.0179, std=0.3454
  L2 distance = 0.000000  ← IDENTICAL
```

But:
```
FiLM γ outputs:
  zero ESM: mean=1.0155, std=0.0501
  rand ESM: mean=1.0163, std=0.0526
  L2 distance = 0.0506      ← responsive
```

So: **FiLM produces measurably different γ/β, but post-diffusion stalks are bit-identical.**

---

## 4. OUR ROOT-CAUSE HYPOTHESIS

The combination of:
- **A1 (QR projection to O(d))** — destroys magnitude info
- **A2 (Crank-Nicolson with normalized Laplacian)** — normalizes edge weight magnitudes again

…fundamentally erases all FiLM scaling information AND makes the diffusion fixed point independent of edge structure.

### 4.1 Why FiLM dies (T5)

Pipeline: `R_raw → γ*R + β → reshape → QR → R_orth`

For any matrix M ∈ ℝ^{d×d}, QR(M)[0] = orthogonal Q with QᵀQ = I. The norm/magnitude of M is encoded in R (upper triangular part of QR), which is **discarded**. Therefore:

```
QR(γ * M + β) ≠ γ * QR(M) + β
```

Specifically: **scaling by γ is annihilated**. Only the rotational direction of M survives. β only shifts the matrix before QR, which is a non-linear projection, so its effect partially survives but is non-monotonic.

After QR:
- Diagonal blocks of L_F = F^T F = I (independent of γ, β)
- Off-diagonal blocks = F_u^T F_v (orthogonal × orthogonal = orthogonal, magnitude bounded)

Then normalized Laplacian: L̂ = D^{-1/2} L_F D^{-1/2}. With D = deg(v) · I (deterministic from graph structure), normalization further removes any remaining magnitude dependence.

### 4.2 Why graph dies (T7)

After QR + normalization, L̂ has:
- Eigenvalues in [0, 2] regardless of edge structure
- Block-diagonal entries = I (uniform across nodes after normalization)
- Off-diagonal = orthogonal matrices (always norm-bounded)

The Crank-Nicolson solver `(I + τ/2 L̂) x = (I - τ/2 L̂) x_0` converges to the kernel of L̂ as τ → ∞. With small fixed τ, the result is a smoothed version of x_0.

**Conjecture**: with σ(L̂) ⊆ [0, 2] and CG finding minimum-residual solution, the diffused state is dominated by the largest-magnitude eigencomponents of x_0 projected onto ker(L̂) — which is approximately constant per connected component, regardless of edge weighting.

### 4.3 Why pseudotime dies (T6)

If diffusion converges near the kernel, temporal adjacency in the graph becomes irrelevant. Shuffling pseudotime bins doesn't change the graph or the diffusion dynamics meaningfully because:
- Each timestep, the model sees X(t) → KNN graph → diffusion
- KNN graph is built from gene-gene similarity, NOT from temporal ordering
- Temporal info is implicit in input X(t) sequence, not in graph structure

Once diffusion smooths X(t) to near-kernel, sequence order is destroyed.

---

## 5. KEY CODE EXCERPTS

### 5.1 Forward pass (`tsgnn.py`)
```python
def _step(current_maps, node_feat_t, allele_emb):
    # 1. Project input to stalk space
    x_latent = self.input_proj(node_feat_t)  # (N, d)

    # 2. Evolve restriction maps via Neural ODE
    new_maps = self.temporal_evolution(current_maps, x_latent, self.edge_index)

    # 3. FiLM modulation (BEFORE QR projection)
    E, d = new_maps.shape[0], self.d
    maps_flat = new_maps.reshape(E, -1)              # (E, 2*d*d)
    maps_modulated = self.allele_film(allele_emb, maps_flat)  # (E, 2*d*d)
    cmaps = maps_modulated.reshape(E, 2, d, d)

    # 4. Compute Laplacian (this internally QR-projects to O(d) THEN normalizes)
    L_F = self.sheaf_layer.compute_connection_laplacian(cmaps)

    # 5. Implicit diffusion (CG solve)
    x = x_latent
    for _ in range(self.num_diffusion_steps):
        x = self.sheaf_layer.diffuse(x, L_F)

    # 6. Output projection
    delta = self.output_proj(x_interpreted)  # (N, 1)
    return delta  # predict_delta=False
```

### 5.2 QR projection (`sheaf_orthogonal.py`)
```python
def project_to_orthogonal(maps):
    """maps: (..., d, d) → returns Q ∈ O(d)."""
    Q, _ = torch.linalg.qr(maps)
    return Q

class OrthogonalSheafDiffusion(NormalizedVectorizedSheafDiffusion):
    def compute_connection_laplacian(self, restriction_maps):
        # PROJECT TO O(d) — this discards magnitude
        Q_maps = project_maps_bundle(restriction_maps)
        # Then build normalized Laplacian (parent class)
        return super().compute_connection_laplacian(Q_maps)
```

### 5.3 Implicit Crank-Nicolson (`diffusion_implicit.py`)
```python
def diffuse(self, x, L_F):
    """One CN step: (I + τ/2 L_F) x_next = (I - τ/2 L_F) x_curr"""
    Nd = x.numel()
    tau = self.tau  # learnable, clamped [0.01, 1.0]

    A = torch.eye(Nd) + (tau * 0.5) * L_F
    b = x.flatten() - (tau * 0.5) * (L_F @ x.flatten())

    x_next = conjugate_gradient(A, b, x0=x.flatten(), max_iter=20, tol=1e-4)
    return x_next.reshape(N, d)
```

### 5.4 FiLM (`film.py`)
```python
class AlleleFiLM(nn.Module):
    def forward(self, esm_embedding, gru_output, edge_index=None):
        z = self.allele_encoder(esm_embedding)        # (1, 128)
        film_params = self.film_generator(z)          # (1, 2*modulation_dim)
        gamma, beta = film_params.chunk(2, dim=-1)
        gamma = gamma + 1.0                           # gamma ≈ 1 init
        modulated = gamma * gru_output + beta          # (E, modulation_dim)
        return modulated
```

Initialization (last layer): `nn.init.normal_(W, std=0.01); nn.init.zeros_(b)`.

---

## 6. QUESTIONS FOR REVIEWER

### 6.1 Confirmation of root cause
1. **Q1**: Is our analysis correct that QR projection to O(d) annihilates the γ scaling from FiLM? Specifically:
   ```
   Is QR(γ * M + β) ≠ γ * QR(M) + β  for γ > 0, β ∈ ℝ^{d×d}?
   ```
   And does the resulting orthogonal matrix lose all γ-magnitude information?

2. **Q2**: With L_F built from F ∈ O(d) (orthogonal-only restriction maps), is it true that:
   - L_F[u,u] = deg(u) · I (block-diagonal, identical to scaled identity)
   - L_F[u,v] is bounded in operator norm by 1 (since F^T F has σ ≤ 1)
   - σ(L_F) ⊆ [0, 2 deg_max]
   
   And after D^{-1/2} L D^{-1/2} normalization, σ(L̂) ⊆ [0, 2]?

3. **Q3**: Does Crank-Nicolson `(I + τ/2 L̂) x = (I - τ/2 L̂) x_0` with τ ≤ 1 and L̂ in [0, 2] converge to a fixed state that depends *primarily* on the graph topology (kernel of L̂) and *minimally* on the specific edge orientations / restriction maps?

4. **Q4**: Is there a way to keep:
   - Gauge invariance (orthogonal maps)
   - Stability (implicit diffusion)
   - **Edge-weight dependence** (FiLM scaling matters)
   simultaneously? Or are these fundamentally incompatible?

### 6.2 Architectural alternatives
5. **Q5**: Should FiLM modulate **after** orthogonalization, e.g., `γ * R_orth + β` (treat O(d) maps as features)? Does this preserve any meaningful property?

6. **Q6**: Should we drop Laplacian normalization (use unnormalized L_F)? The eigenvalues become [0, 2 · deg_max] but edge weights matter.

7. **Q7**: Should we drop QR projection (let restriction maps be unconstrained)? Loses gauge invariance for cross-dataset transfer, but FiLM and graph would matter.

8. **Q8**: Is there a known sheaf-theoretic construction that:
   - Allows edge-weight-dependent diffusion
   - Maintains spectral guarantees
   - Compatible with implicit solvers?
   References would help.

### 6.3 Reframing
9. **Q9**: If the diagnosis is correct and unfixable, the model is empirically learning **node-feature manifold structure** (since graph+FiLM are irrelevant). What is the model actually learning? Possibilities:
   - Just a per-cell autoregressive predictor (X(t+1) ≈ X(t) baseline)?
   - Cell-distribution density estimator?
   - Bottleneck-based denoiser?
   
   How can we characterize the actual learned function?

10. **Q10**: For paper revision: what is the minimal honest claim consistent with our results? 
    - "Cross-dataset transferable directional signal in scRNA-seq via stalk-bottleneck representation" — does this hold mathematically?

---

## 7. SUPPORTING EXPERIMENTAL DATA

### 7.1 Bias Probe (Job 483174)
```
T1 identity baseline:    r_d = -0.005   PASS (chosen 0.70 crushes baseline)
T2 mean predictor:       r_d = +0.707
T3 BRCA-only HVG:        r_d = +0.694   PASS (combined was 0.70, drop -0.55%)
T4 CUDA non-det std:     0.0010         PASS (vs seed-std 0.004)
T5 FiLM swing:           +0.001         FAIL (<0.01 = dead)
T6 pseudotime shuffle:   +0.7435        FAIL (>original 0.70)
T7 random graph:         +0.6987        FAIL (≈original 0.70)
```

### 7.2 Diagnostic v2 (Job 483305)
```
FiLM γ outputs:
  zero ESM: mean=1.0155, std=0.0501
  rand ESM: mean=1.0163, std=0.0526
  Δ L2 dist = 0.0506   (FiLM responsive)

Stalks post-diffusion:
  zero ESM: mean=-0.0179, std=0.3454
  rand ESM: mean=-0.0179, std=0.3454
  Δ L2 dist = 0.000000   (IDENTICAL → diffusion washes signal)

r_delta:
  zero: +0.6977
  rand: +0.6967
  Δ: +0.001021   (predictions effectively identical)
```

### 7.3 3-seed in-domain validation (Job 482926)
```
chosen r_delta:    0.349 ± 0.005   (3/3 P1 ≥ 0.35)
baseline r_delta:  0.275 ± 0.058
Δ chosen-baseline: +0.074

chosen r_abs:      0.064            (very low — stalk bottleneck artifact)
baseline r_abs:    0.164
```

### 7.4 Reverse ablation (Job 482916, single seed, 50 epochs)
```
config            r_delta   Δ vs baseline
baseline          +0.352    —
full (all 5)      -0.053    -0.405  (collapse)
full−A1           +0.328    -0.025
full−A2           -0.252    -0.604  (A2 most positive)
full−A3           +0.375    +0.023  (A3 hurts)
full−B1           +0.346    -0.006  (B1 hurts)
full−B2           -0.011    -0.363  (B2 neutral)
```

---

## 8. WHAT WE NEED FROM YOU

1. **Confirm or refute** the QR-annihilates-FiLM hypothesis with proof.
2. **Confirm or refute** the implicit-diffusion-converges-independent-of-edges hypothesis.
3. **Suggest the minimal architectural change** that would make FiLM/graph/temporal claims valid.
4. **Suggest the honest paper narrative** if the architecture is unfixable: what IS this model learning?
5. **Cite relevant literature** on sheaf neural networks, gauge-invariant GNNs, edge-weighted spectral methods that handle this trade-off.

---

## 9. APPENDIX — Full file paths for code review

If reviewer wants original source:
- `src/tsgnn/model/tsgnn.py` — main model (450 lines)
- `src/tsgnn/model/sheaf_orthogonal.py` — A1 QR projection (240 lines)
- `src/tsgnn/model/diffusion_implicit.py` — A2 Crank-Nicolson (255 lines)
- `src/tsgnn/model/sheaf_vectorized.py` — base normalized Laplacian (700 lines)
- `src/tsgnn/model/film.py` — FiLM conditioning (160 lines)
- `src/tsgnn/model/film_nodal.py` — B1 nodal FiLM (290 lines, REJECTED)
- `src/tsgnn/model/temporal_ode.py` — B2 Neural ODE (230 lines)
- `src/tsgnn/model/temporal.py` — legacy GRU (140 lines)
- `hpc/probe_bias.py` — bias test suite (240 lines)
- `hpc/probe_diagnostic_v2.py` — signal trace (200 lines)
- `docs/TEST_LOG.md` — full chronological experiments
- `docs/ARCHITECTURE_REPORT.md` — full theory + intuition (~2000 lines)

---

**END OF REVIEW PACKAGE**

Total length: ~12k tokens. Self-contained for math review without external lookups.
