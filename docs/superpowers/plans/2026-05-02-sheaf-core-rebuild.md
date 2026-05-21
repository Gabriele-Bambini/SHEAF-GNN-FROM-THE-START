# Sheaf Core Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a clean, independently testable sheaf-math core that can become the canonical base for the TS-GNN rebuild.

**Architecture:** Add a new top-level package at `src/sheaf_ts` instead of extending the legacy `tsgnn.model` monolith. The core separates orthogonal parametrization, coboundary construction, Laplacian builders, and spectral checks, following the implementation discipline seen in Neural Sheaf Diffusion.

**Tech Stack:** Python 3.10+, PyTorch, pytest. No `torch_sparse`, `torch_scatter`, scanpy, ESM, or HPC dependencies in the core.

---

### Task 1: Core Math Package Skeleton

**Files:**
- Create: `src/sheaf_ts/__init__.py`
- Create: `src/sheaf_ts/core/__init__.py`
- Create: `src/sheaf_ts/core/orthogonal.py`
- Create: `src/sheaf_ts/core/coboundary.py`
- Create: `src/sheaf_ts/core/laplacian_builders.py`
- Create: `src/sheaf_ts/core/spectral_checks.py`
- Test: `tests/test_rebuild_sheaf_core.py`

- [ ] **Step 1: Write failing tests**

Add tests for:
- Cayley maps are orthogonal and differentiable.
- General Laplacian matches dense coboundary `B.T @ B` for non-orthogonal maps.
- Connection Laplacian matches general Laplacian only for orthogonal maps.
- Connection builder rejects non-orthogonal maps when verification is enabled.
- Normalized connection Laplacian has spectrum in `[0, 2]`.
- Spectral checks fail closed on bad matrices.

- [ ] **Step 2: Verify tests fail before implementation**

Run:

```powershell
pytest tests/test_rebuild_sheaf_core.py -q
```

Expected: FAIL because `sheaf_ts.core` does not exist.

- [ ] **Step 3: Implement minimal core**

Implement:
- `skew_from_params(params, d)`
- `cayley_map(params, d)`
- `orthogonality_error(maps)`
- `build_dense_coboundary(num_nodes, edge_index, maps, edge_weights=None)`
- `build_general_laplacian(...)`
- `build_connection_laplacian(...)`
- `normalize_block_laplacian(...)`
- `SpectralReport`
- `check_symmetric`, `check_psd`, `check_normalized_spectrum`

- [ ] **Step 4: Verify tests pass**

Run:

```powershell
pytest tests/test_rebuild_sheaf_core.py -q
```

Expected: all new tests pass.

### Task 2: Integrate Documentation And Gates

**Files:**
- Modify: `docs/CHAIN_OF_VERIFICATION_ASSESSMENT_2026-05-02.md`
- Modify: `docs/EXTERNAL_SHEAF_IMPLEMENTATIONS_REVIEW_2026-05-02.md`

- [ ] **Step 1: Document core API**

Add a short implementation note listing the new canonical files and the invariants they enforce.

- [ ] **Step 2: Verify import and compile**

Run:

```powershell
python -m compileall -q src/sheaf_ts
pytest tests/test_rebuild_sheaf_core.py -q
```

Expected: compile succeeds and tests pass.

### Task 3: Next Phase After Core

Only after Task 1 and Task 2 pass:
- Add baseline artifact schema.
- Add fail-closed JSON gates around mean drift, random graph, shuffled ESM, and shuffled time.
- Add a synthetic backtest where the correct sheaf mechanism is known.

This task is intentionally not implemented in the first pass; the first pass must make the sheaf core mathematically auditable.
