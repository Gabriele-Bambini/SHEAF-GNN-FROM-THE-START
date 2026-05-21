# Chain of Verification Assessment - 2026-05-02

Scope: full `ts-gnn` repository, legacy code, HPC scripts, results/logs, and
the `FROM THE START` rewrite prototype.

Method:
- 10 focused read-only sub-agent audits: mathematics, architecture, code, data,
  biology, metrics/statistics, training, HPC/reproducibility, shortcut controls,
  and rebuild plan.
- Direct local backtests by coordinator.
- No destructive git operation. Worktree was already dirty/untracked-heavy.

## Executive Verdict

The project does not have one isolated bug. The current failure is a stack of:

1. Runtime code not matching the mathematical claims.
2. Architecture that lets graph, ESM and time become inert.
3. Data pipeline problems that can invalidate allele-specific and zero-shot
   claims before the model is even considered.
4. Metrics that can be high under identity or mean-drift shortcuts.
5. HPC/reproducibility drift that makes successful runs hard to audit.

The sheaf mathematics is not fundamentally invalid. The base formula
`L = delta^T W delta` is sound when the runtime actually builds that operator
with non-negative weights and valid restriction maps. The problem is that the
runtime and pipeline often do not satisfy the conditions needed for the claims
being made.

## Local Backtests Performed

### Backtest 1 - Legacy EdgeWeightNet ESM path

Command: instantiate `EdgeWeightNet`, compare two ESM vectors at init.

Result:

```text
edge_weight_std 0.0
esm_output_delta_max 0.0
esm_grad_max_at_init 0.0
head_grad_max_at_init 1.9925405979156494
```

Interpretation: at init, all edge weights are constant and predictions are
ESM-insensitive. The head itself can learn after updates, but the ESM path is
dead at step 0.

Relevant file: `src/tsgnn/model/edge_weight_net.py`.

### Backtest 2 - Legacy Neural ODE identity-flow claim

Command: instantiate `SheafNeuralODE` with zero-initialized velocity head.

Result:

```text
R1_minus_layernorm_max 0.0
R1_minus_R0_max 0.7361503839492798
velocity_last_weight_norm 0.0
```

Interpretation: with zero velocity, the ODE output is exactly `LayerNorm(R0)`,
not `R0`. The ODE is not identity-like at init in the legacy runtime.

Relevant file: `src/tsgnn/model/temporal_ode.py`.

### Backtest 3 - Test suite status

`python -m compileall -q src 'FROM THE START\src'` passed.

`pytest 'FROM THE START\tests' -q` passed:

```text
11 passed
```

Targeted legacy tests failed:

```text
pytest tests/test_sheaf.py tests/test_vectorized.py tests/test_tsgnn.py \
  tests/test_loss.py tests/test_metrics.py tests/test_predict_delta.py \
  tests/test_implicit_diffusion.py tests/test_neural_ode.py -q --maxfail=3

17 passed, 3 failed
```

Failures:
- `test_no_allele_forward`: shape mismatch when `use_allele_conditioning=False`.
- `test_decompose_rewiring_delta_nonzero`: allele delta Laplacians are zero.
- `test_decompose_rewiring_delta_zero_without_allele`: same no-allele shape bug.

Full `pytest -q` timed out after 244 seconds, so full-suite status is not
verified.

## Agent Consensus

### 1. Mathematics

Findings:
- The Hansen-Ghrist sheaf Laplacian formula is basically correct when built as
  `delta^T W delta`.
- The legacy v2 runtime does not always satisfy its own stated assumptions.
- In the legacy path, temporal ODE maps are raw `(E,2,d,d)` matrices and then
  `get_orthogonal_maps()` uses QR, so Cayley is bypassed for the temporal path.
- The weighted runtime Laplacian is not the normalized Laplacian claimed by the
  docs/reports. Spectrum in `[0,2]` is not guaranteed in that path.
- `eps * I` is fine as numerical regularization, but then zero eigenvalues and
  cohomological kernel claims are no longer exact.

Verdict: the mathematical object is defensible, but the implementation and
reporting overclaim what the runtime actually does.

### 2. Architecture

Findings:
- Per-edge learned maps make zero-shot transfer to a different graph invalid:
  learned map index `e` is tied to an arbitrary edge position.
- L5 in `FROM THE START` still evolves raw maps and QR-projects them; it does
  not evolve skew/tangent parameters and does not pass real `delta_t`.
- ESM is no longer FiLM in the main path; it is edge-weight conditioning.
- Legacy ESM edge-weight head is zero-initialized; `FROM THE START` partly
  fixes this, but the real-data runners still use dummy/random ESM in places.
- `predict_delta` is semantically inconsistent across configs and reports.

Verdict: the architecture lets sophisticated modules exist while being
functionally unused.

### 3. Code

Concrete bugs:
- No-allele ablation crashes because `nn.Identity()` emits raw `esm_dim`, while
  `EdgeWeightNet` expects `conditioning_dim`.
- `torchdiffeq` is used but not consistently declared in env files.
- `edge_weights_seq` is computed and stored but not passed into the model.
- VIPER/regulon loss returns zero for centered nonzero matrices when sum is 0.
- Pipeline can set loss components to zero or mask non-finite values.
- Some evaluation helpers do not match `TSGNN.forward` input/output API.
- `run_nature.py` has import-time environment patching behavior.

Verdict: there are real implementation bugs that can invalidate ablations and
baselines independently of model quality.

### 4. Data Pipeline

Findings:
- TP53 allele labels appear to be proxy/imputed from external frequency/subtype
  logic rather than reliable patient/cell ground truth in the main BRCA path.
- At least one local allele CSV conflict is silently dropped by duplicate
  handling.
- Some pipelines build sequences/graphs/pseudotime before train/val splitting.
- CRC zero-shot can use target-domain feature selection before alignment.
- Batch correction is not necessarily the tensor used by the model.
- Pseudotime root and binning are fragile; interpolation can leak future bins.

Verdict: allele-specific and zero-shot claims are not safe until the data
contract is rebuilt.

### 5. Biological Rationale

Defensible:
- TP53 hotspot biology is a plausible scientific motivation.
- ESM-style protein embeddings are defensible if embeddings and allele labels
  are real and audited.
- Sheaf diffusion is a reasonable model class for heterogeneous gene-gene
  transport.

Weak in current pipeline:
- Pearson/correlation KNN is co-expression, not a causal GRN.
- `abs(correlation)` removes activation/repression direction.
- Stalk dimension labels are interpretive unless anchored by explicit data or
  losses.
- DPT is an exploratory ordering, not biological time.
- CRC with unknown or weak allele labels cannot validate allele specificity.

Verdict: the project is a plausible exploratory framework, not yet evidence of
TP53 allele-specific rewiring.

### 6. Metrics and Statistics

Findings:
- `r_d` can be explained by mean drift; it is not enough.
- Pipeline `r_delta_perp` logic is not consistently used as a hard gate.
- Existing controls T5/T6/T7 are diagnostic but not always fail-closed.
- Single-seed or small-n results are not enough for claims.
- Permutation tests must respect temporal/patient structure.

Required primary metric:
- `model_r_delta_perp - mean_predictor_r_delta_perp`, computed across all
  transitions, with train-only mean drift and bootstrap/seed CI.

### 7. Training and Optimization

Findings:
- Current selection pressure favors mean-drift/identity-like solutions.
- ESM signal is weak or zero at init in legacy.
- Training loops/configs are fragmented (`epochs` vs `max_epochs`, runner vs
  trainer, config flags ignored).
- NaN masking can convert failure into apparent stability.
- Baseline budgets are not consistently fair or parameter matched.

Verdict: model comparison is currently not clean enough to support a mechanism
claim.

### 8. HPC and Reproducibility

Findings:
- Environment definitions drift across `environment.yml`, Dockerfile, sbatch
  scripts and ad-hoc `pip install` inside jobs.
- There are multiple competing HPC entrypoints.
- Run manifests lack enough data for audit: resolved config, git SHA/diff,
  package list, Slurm metadata, data hashes, command, device.
- Determinism is incomplete; Python `hash()` based fallback seeds are unstable
  across processes.
- Output persistence is fragile; some scripts delete global checkpoint folders.
- Some SSH/HPC helper scripts contain risky credential/host-key/job-cancel
  patterns. Do not quote or reuse embedded secrets.

Verdict: standardize one HPC path before trusting new results.

### 9. Shortcut Inventory

The model can look good without using the intended mechanism via:

- Identity/autocorrelation.
- Mean drift.
- KNN graph redundancy with expression features.
- Random/constant graph doing the same work.
- Constant edge weights.
- ESM fallback/random/same embedding.
- Label/subtype leakage.
- Train/val/test leakage through preprocessing or temporal binning.
- Batch/depth captured by DPT root.
- NaN/missing metrics hidden by permissive reporting.
- Baselines trained with different budgets.

### 10. Rebuild Plan

Consensus: rebuild is justified. Do not keep the monolithic `TSGNN` as the
canonical implementation. Keep the data loaders and the `FROM THE START`
L0-L5 idea, but migrate it into a clean package path without spaces.

## External Sheaf Implementations

Additional review was added after inspecting GitHub implementations connected
to Cristian Bodnar and/or Pietro Lio:

- `twitter-research/neural-sheaf-diffusion`
- `IuliaDuta/sheaf_HNN`
- `hakanaktas0/DSNN`

Full notes: `docs/EXTERNAL_SHEAF_IMPLEMENTATIONS_REVIEW_2026-05-02.md`.

Main impact on this assessment:
- Use Neural Sheaf Diffusion as the engineering template for the sheaf core:
  separate orthogonal parametrization, Laplacian builders, dense reference
  tests, and model layers.
- Do not copy the node-classification experiment logic; the task here is
  time-series gene dynamics and has different leakage/shortcut risks.
- Treat hypergraph sheaves as a later biological upgrade for pathways,
  regulons, TF modules or protein complexes.
- Treat directed/complex sheaves as a later option only after signed/directed
  biological edges are audited.
- The immediate rebuild should import the NSD-style verification discipline:
  sparse builder output must match a dense `delta.T @ delta` reference, and
  PSD/symmetry/spectrum claims must be tested where claimed.

## What Is Wrong: Categorized

### Mathematics

Not wrong in principle:
- Sheaf Laplacian formula.
- Orthogonal transport maps.
- CN diffusion under PSD symmetric `L`.

Wrong or overstated in runtime:
- Cayley is not used in all paths that claim Cayley.
- Runtime weighted Laplacian may be unnormalized while reports assume spectral
  normalization.
- LayerNorm after zero ODE velocity makes the flow non-identity.
- `eps * I` breaks exact kernel/cohomology interpretation.

### Architecture

Core issue: the model is over-parameterized and under-constrained relative to
the signal. It can minimize MSE through node-local or drift paths while graph,
ESM and temporal mechanisms are inactive.

Highest-impact fixes:
- Per-edge maps must be generated from edge attributes/context if zero-shot
  cross-graph is required.
- Evolve skew/tangent parameters for L5; remove QR raw-map projection.
- Pass real `delta_t` or remove temporal claim.
- Rename FiLM diagnostics to ESM edge-weight sensitivity unless FiLM is truly
  reintroduced.
- Make edge prior weights part of the Laplacian or delete them from the data
  contract.

### Code

Immediate failures:
- Fix no-allele conditioning shape.
- Declare and lock `torchdiffeq`.
- Stop ignoring `edge_weights_seq`.
- Fail hard on non-finite losses in scientific runs.
- Align evaluation functions with `TSGNN.forward`.
- Remove import-time monkey patches.

### Data

Immediate blockers:
- Audit TP53 labels and remove proxy/imputed labels from allele-specific
  claims.
- Freeze gene set/order/stats from train before processing holdout/CRC.
- Split by patient before preprocessing decisions that learn from data.
- Store and validate data hashes/manifests.
- Treat DPT as a hypothesis, not time ground truth.

### Biology

Claims to avoid now:
- "learned allele-specific TP53 rewiring"
- "uses temporal dynamics"
- "uses biological GRN"
- "sheaf mechanism explains transfer"

Defensible wording now:
- "The framework tests whether sheaf, graph, ESM and pseudotime mechanisms add
  value beyond mean-drift and MLP baselines."

## Fail-Closed Acceptance Gates

Every artifact must include:
- `identity`
- `mean_delta`
- `linear_ar1`
- `mlp`
- `gcn_static`
- `zero_esm`
- `same_esm`
- `shuffled_esm`
- `random_graph`
- `degree_preserved_random_graph`
- `constant_graph`
- `shuffled_time`
- `reversed_time`
- `patient_overlap`
- `nan_count`
- `r_delta`
- `r_delta_perp`
- confidence intervals across seeds

Hard gates:
- `nan_count == 0`
- `patient_overlap == 0`
- no fallback/random ESM in scientific mode
- full model beats mean-drift on `r_delta_perp` by predeclared margin
- graph model beats random/constant graph
- L4+ beats zero/same/shuffled ESM
- L5 beats shuffled/reversed time
- all gates reported in JSON and table

Suggested exploratory thresholds:
- `full_r_delta_perp - mean_r_delta_perp >= 0.03`
- graph/time/mechanism destructive ablation drop >= 0.02
- ESM swing smoke >= 0.01

Suggested claim thresholds:
- CI95 lower bound above 0 for every relevant margin
- primary margin >= 0.05 for mean-drift, graph and time
- allele-swap/top-k mechanism overlap with FDR correction

## Rebuild Roadmap

### Phase 0 - Source of Truth

Goal: stop drift.

Actions:
- Create clean package path, e.g. `src/tsgnn/rebuild` or `src/sheaf_ts`.
- Move `FROM THE START` ideas into that path.
- Define one model API: `forward(X_seq, esm, graph_bundle) -> predictions`.
- Define one data bundle schema with gene order, stats, graph, edge weights,
  pseudotime bins, patient IDs, allele labels and manifest hashes.
- Define one HPC entrypoint and one run manifest.

Gate:
- import has no writes
- config validates before run
- CPU smoke passes
- run manifest complete

### Phase 1 - Baselines First

Implement before sheaf:
- identity
- mean drift
- AR(1)
- MLP
- GCN static
- no-edge
- random graph
- shuffled time
- zero/same/shuffled ESM

Gate:
- all baselines produce same JSON schema
- no missing metrics
- no NaN

Implementation started:
- `src/sheaf_ts/eval/artifacts.py`
- `src/sheaf_ts/eval/metrics.py`
- `src/sheaf_ts/eval/gates.py`
- `src/sheaf_ts/eval/cross_validation.py`
- `src/sheaf_ts/eval/stats.py`
- `src/sheaf_ts/hpo/optuna_runner.py`
- `tests/test_rebuild_eval_hpo.py`

Verified controls/infrastructure in the second implementation slice:
- experiment artifacts reject missing controls and non-finite metrics
- `r_delta_perp` projects out train-only mean drift
- scientific gates reject controls too close to the model
- repeated group K-fold keeps patient/group IDs disjoint
- temporal blocked splits never train on future bins
- Optuna adapter runs on injected CV objectives without importing the legacy
  biological pipeline
- bootstrap CI and Benjamini-Hochberg FDR helpers are deterministic

### Phase 2 - Data Contract

Actions:
- validate real TP53 allele labels
- reject duplicate/conflicting patient labels
- patient-level split before fitting normalization/gene selection/pseudotime
- train-frozen gene order and normalization for holdout/CRC
- data hash manifest

Gate:
- no proxy labels in scientific mode
- no target-domain feature selection in zero-shot
- patient overlap is zero

### Phase 3 - Sheaf Core

Actions:
- adopt the Neural Sheaf Diffusion separation pattern:
  `orthogonal.py`, `coboundary.py`, `laplacian_builders.py`,
  `spectral_checks.py`
- Cayley/exponential/Householder maps in tangent/skew space
- diagonal, general and normalized connection Laplacian builders
- vectorized weighted normalized Laplacian
- CN solver with residual logging
- no silent clamps in scientific mode
- explicit sparse-vs-dense `delta^T W delta` verification tests

Gate:
- PSD, symmetry, expected kernel when `eps=0`
- normalized spectrum test where claimed
- energy monotonicity under exact/accepted residual

Implementation started:
- `src/sheaf_ts/core/orthogonal.py`
- `src/sheaf_ts/core/coboundary.py`
- `src/sheaf_ts/core/laplacian_builders.py`
- `src/sheaf_ts/core/spectral_checks.py`
- `tests/test_rebuild_sheaf_core.py`

Verified invariants in the first implementation slice:
- Cayley maps are orthogonal and differentiable.
- General sheaf Laplacian matches dense coboundary `B.T @ B` for arbitrary
  non-orthogonal maps.
- Connection Laplacian matches the general builder when maps are orthogonal.
- Connection builder rejects non-orthogonal maps when verification is enabled.
- Normalized connection Laplacian spectrum is checked in `[0, 2]`.
- Symmetry and PSD checks fail closed.

### Phase 4 - L0-L5 Ladder

Actions:
- L0 MLP
- L1 graph
- L2 static sheaf
- L3 learnable sheaf maps
- L4 ESM edge-weight or matrix-weight conditioning
- L5 temporal tangent-space ODE

Gate:
- each level must beat previous accepted level
- each added mechanism must pass its destructive-control gate
- if L1 fails graph gate, stop sheaf claims
- if L4 fails ESM gate, stop allele claims
- if L5 fails time gate, stop temporal claims

### Phase 5 - Real Backtest

Seeds:
- `42, 123, 456, 789, 2026`

Splits:
- patient-level BRCA holdout
- BIOKEY holdout
- CRC zero-shot after model freeze
- METABRIC bulk only after primary freeze

Gate:
- `r_delta_perp` margin vs mean baseline with CI
- random graph degrades
- shuffled/reversed time degrades
- zero/same/shuffled ESM degrades
- allele swap/top-k biologically interpretable with FDR correction

Synthetic precursor implemented before real data:
- `src/sheaf_ts/experiments/synthetic_sheaf.py`
- `src/sheaf_ts/experiments/synthetic_runner.py`
- `scripts/run_synthetic_sheaf_backtest.py`
- `tests/test_rebuild_synthetic_backtest.py`

The synthetic generator creates transitions from a known sheaf diffusion
operator. The oracle sheaf predictor must beat:
- train-only mean drift
- random graph after removing the current-state/autocorrelation shortcut

Local smoke manifest:
- `scratch/synthetic_sheaf_manifest.json`
- accepted strict gates
- oracle `r_delta_perp = 1.0`
- random graph residual control around `0.054`

This is now the first safe candidate for Slurm smoke testing once the HPC
manifest writer records git status, environment, command, seeds and output
hashes.

HPC-safe manifest layer implemented:
- `src/sheaf_ts/hpc/manifest.py`
- `hpc/submit_synthetic_sheaf_smoke.sbatch`
- `tests/test_rebuild_hpc_manifest.py`

Local runtime smoke:
- `scratch/synthetic_sheaf_runtime_manifest.json`
- records command, Python/platform, Slurm env if present, git SHA/status,
  package versions and output SHA256

Slurm submission status:
- not submitted yet
- non-interactive SSH check to `bocconi` failed with public-key permission
  denial
- once SSH key/alias is fixed, submit the safe smoke with:
  `sbatch hpc/submit_synthetic_sheaf_smoke.sbatch`

### Phase 6 - Reporting

Report must show:
- accepted/rejected per level
- all negative controls
- CIs and seed list
- effective config
- git/diff/env/data hashes
- no hidden NaNs

## Final Decision

Do not keep pushing the current monolith as the main scientific result.

Use `FROM THE START` as the conceptual seed, but harden it around:
- clean data contract
- baseline-first evaluation
- fail-closed gates
- true runtime math
- reproducible HPC manifest

Only after those gates pass can the project claim sheaf, graph, ESM, temporal
or biological rewiring mechanisms.
