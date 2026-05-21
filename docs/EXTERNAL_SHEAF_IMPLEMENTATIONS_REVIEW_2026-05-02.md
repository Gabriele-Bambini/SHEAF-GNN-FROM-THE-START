# External Sheaf Implementations Review - 2026-05-02

Scope: GitHub implementations connected to Cristian Bodnar and/or Pietro Lio,
reviewed as implementation references for the `ts-gnn` rebuild.

Primary references:
- Neural Sheaf Diffusion official code:
  https://github.com/twitter-research/neural-sheaf-diffusion
- Sheaf Hypergraph Networks official code:
  https://github.com/IuliaDuta/sheaf_HNN
- Directed Sheaf Neural Networks code:
  https://github.com/hakanaktas0/DSNN

Local read-only clones used for inspection:
- `C:\Users\02gab\AppData\Local\Temp\codex-sheaf-references\nsd`
- `C:\Users\02gab\AppData\Local\Temp\codex-sheaf-references\sheaf_HNN`
- `C:\Users\02gab\AppData\Local\Temp\codex-sheaf-references\DSNN`

## Main Takeaway

The useful pattern is not "copy the model". The useful pattern is how the
official sheaf code separates the mathematical core from the experiment:

1. Orthogonal parametrization is isolated.
2. Laplacian construction is isolated.
3. Sparse builder output is tested against a dense `delta.T @ delta`
   reference.
4. PSD, symmetry and normalized-spectrum claims are tested explicitly.
5. Model layers consume the Laplacian after these invariants are enforced.

This is the opposite of the current `ts-gnn` monolith, where sheaf math,
ODE maps, ESM conditioning, graph construction, metrics and HPC runner logic
are coupled.

## Reference 1 - Neural Sheaf Diffusion

Repository: https://github.com/twitter-research/neural-sheaf-diffusion

Relevant files:
- `models/orthogonal.py`
- `models/laplacian_builders.py`
- `lib/laplace.py`
- `models/test_builders.py`
- `models/test_orthogonal.py`
- `models/disc_models.py`

### What To Reuse Conceptually

Use separate Laplacian builders:
- `DiagLaplacianBuilder`
- `GeneralLaplacianBuilder`
- `NormConnectionLaplacianBuilder`

This matters because each map family has a different correct diagonal:
- diagonal maps: `sum maps**2`
- general maps: `sum Q.T @ Q`
- connection/orthogonal maps: degree or weighted degree only because
  `Q.T @ Q = I`

This directly addresses one of our observed bugs: using `w * I` on the diagonal
is only valid for truly orthogonal maps. For general or LayerNorm-mutated maps,
the diagonal must be built from `F_v<e.T @ F_v<e`.

Use an explicit orthogonal parametrization module:
- matrix exponential
- Cayley
- Householder
- Euler for small dimensions

The important design rule: raw matrices should not silently enter a connection
Laplacian. The API should accept either tangent/skew parameters or already
validated maps with a declared map family.

Use dense-reference tests:
- build a sparse Laplacian with the production builder
- build a dense reference by constructing the coboundary matrix
- assert equality within tolerance
- assert symmetry
- assert PSD
- assert spectrum `<= 2` only when the normalized operator is actually used

This is the single most important test pattern to import into `ts-gnn`.

### What Not To Copy Blindly

The NSD code is for node classification on heterophilic graph benchmarks, not
time-series gene dynamics. Its experimental logic is not a drop-in fit.

The code assumes graph conventions that must be made explicit in our pipeline:
- undirected edges are represented by directed pairs
- edge maps are tied to the current edge index
- normalized spectrum claims apply only to the normalized builder

Some numerical compromises must be documented if reused:
- SVD/eigendecomposition normalization perturbations
- clamp operations
- detached normalizers

These are acceptable engineering tools only if the run manifest says they are
enabled and the metrics report does not overclaim exact cohomology/kernel
interpretations.

## Reference 2 - Sheaf Hypergraph Networks

Repository: https://github.com/IuliaDuta/sheaf_HNN

Relevant files:
- `sheaf_builder.py`
- `hgcn_sheaf_laplacians.py`
- `layers.py`
- `models.py`

### What To Reuse Conceptually

This repo is useful for biological structure because it treats node-hyperedge
incidence as first-class structure. For `ts-gnn`, that suggests a better
biological graph path than KNN:

- genes as nodes
- pathways, regulons, TF modules, protein complexes or perturbation programs
  as hyperedges
- sheaf maps on `(gene, biological module)` incidences

This is more biologically defensible than Pearson KNN when the claim is TP53
rewiring.

The repo also keeps several sheaf families side by side:
- diagonal
- orthogonal
- general
- low-rank

For our rebuild this supports an L0-L5 ladder where each map family is a
controlled level, not a hidden config switch.

### What Not To Copy Blindly

The hypergraph code is heavier and not required for the first rebuild. It should
not be introduced before:

- the current graph sheaf core passes dense-reference tests
- baselines and shortcut controls are fail-closed
- real biological hyperedges are available and audited

In other words: hypergraphs are a Phase 2 biological upgrade, not the first
debugging step.

## Reference 3 - Directed Sheaf Neural Networks

Repository: https://github.com/hakanaktas0/DSNN

Relevant files:
- `models/disc_models.py`
- `models/sheaf_models.py`
- `utils/magnetic_laplacian.py`
- `models/laplacian_builders.py`

### What To Reuse Conceptually

The useful idea is to make directionality explicit. DSNN introduces complex
phase factors over directed edges and multiplies them into the sheaf Laplacian
path.

For `ts-gnn`, this is relevant only if the biological graph has signed or
directed edges:

- TF activates gene
- TF represses gene
- pathway edge has direction
- perturbation effect has source/target orientation

This would be more biologically meaningful than `abs(correlation)`, which
destroys activation/repression sign.

### What Not To Copy Blindly

Do not add complex-valued directed diffusion to the current project yet.
It expands the failure surface:

- complex activations
- complex dropout
- magnetic phase hyperparameters
- extra numerical tests
- harder biological interpretation

Directionality should first enter as a clean signed/directed graph contract and
destructive controls. Complex DSNN-style machinery is a later option.

## Concrete Changes For Our Rebuild

### Sheaf Core API

Create a core with these modules:

- `orthogonal.py`: Cayley/matrix-exp/Householder parametrizations.
- `coboundary.py`: dense and sparse coboundary construction.
- `laplacian_builders.py`: diagonal, general, normalized connection builders.
- `spectral_checks.py`: symmetry, PSD, normalized-spectrum and energy checks.
- `tests/test_laplacian_builders.py`: sparse-vs-dense reference tests.

The production model should never construct a Laplacian in a training loop by
hand. It should call a builder that is tested independently.

### Runtime Gates

For every scientific run, save:

- map family: diagonal, general, orthogonal, low-rank
- parametrization: Cayley, matrix_exp, QR, raw
- whether maps passed orthogonality check
- whether `diag == sum F.T @ F` was verified on a sample
- whether normalized spectrum was checked when claimed
- whether edge index was doubled/undirected or directed/signed

### Biological Upgrade Path

Do not start with KNN as the canonical biological graph. The upgrade order
should be:

1. expression KNN only as a weak baseline graph
2. signed correlation graph with sign-preserving controls
3. curated GRN/regulon graph
4. pathway/regulon hypergraph if data supports it
5. directed sheaf only if signed/directed biological edges are audited

### Red Flags To Avoid

Do not claim a connection Laplacian if:

- maps are not orthogonal
- QR projection happens after a non-biological normalization that destroys the
  intended parametrization
- diagonal blocks are assumed instead of computed
- edge maps are learned per edge index and then transferred zero-shot to a
  different graph

Do not claim biological rewiring if:

- random graph is equivalent
- shuffled ESM is equivalent
- reversed/shuffled pseudotime is equivalent
- mean drift baseline is equivalent
- labels are proxy or conflict-resolved silently

## Decision

Use Neural Sheaf Diffusion as the engineering template for the sheaf core.
Use Sheaf Hypergraph Networks and DSNN as later biological-structure ideas.

For the immediate `ts-gnn` rebuild, the priority is:

1. NSD-style builder/test separation.
2. Dense-reference verification of `delta.T @ delta`.
3. Fail-closed shortcut controls.
4. Only then GRN/hypergraph/directed extensions.

## Implementation Status

First slice implemented under the clean package path `src/sheaf_ts/core`:

- `orthogonal.py`: compact skew parameters, Cayley maps, matrix exponential,
  orthogonality error.
- `coboundary.py`: dense coboundary builder used as a mathematical reference.
- `laplacian_builders.py`: general and connection Laplacian builders with
  block normalization.
- `spectral_checks.py`: fail-closed symmetry, PSD and normalized-spectrum
  checks.

The associated test file is `tests/test_rebuild_sheaf_core.py`. It encodes the
NSD-style invariant that production Laplacian builders must agree with a dense
`delta.T @ delta` reference, and that the connection shortcut `diag = w * I`
is valid only for orthogonal maps.
