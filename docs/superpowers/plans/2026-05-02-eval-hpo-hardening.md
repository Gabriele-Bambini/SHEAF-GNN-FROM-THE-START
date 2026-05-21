# Evaluation And HPO Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add modular, fail-closed evaluation infrastructure for cross-validation, shortcut controls, experiment artifacts, and Optuna-ready hyperparameter search.

**Architecture:** Extend the clean `src/sheaf_ts` package instead of the legacy `tsgnn` monolith. `sheaf_ts.eval` owns metrics, artifact schema, cross-validation splitters, and scientific gates. `sheaf_ts.hpo` owns Optuna adapters that accept a caller-provided objective, so HPO is reusable without importing the heavy biological pipeline.

**Tech Stack:** Python 3.10+, NumPy, PyTorch, pytest, optional Optuna. The core must still import without Optuna installed.

---

### Task 1: Experiment Artifact Schema

**Files:**
- Create: `src/sheaf_ts/eval/artifacts.py`
- Test: `tests/test_rebuild_eval_hpo.py`

- [ ] Write failing tests for finite metrics, no NaNs, required controls and JSON roundtrip.
- [ ] Implement `ExperimentResult`, `ControlResult`, `validate_experiment_result`, `to_json_dict`.
- [ ] Run `pytest tests/test_rebuild_eval_hpo.py -q`.

### Task 2: Robust Metrics And Gates

**Files:**
- Create: `src/sheaf_ts/eval/metrics.py`
- Create: `src/sheaf_ts/eval/gates.py`
- Modify: `src/sheaf_ts/eval/__init__.py`
- Test: `tests/test_rebuild_eval_hpo.py`

- [ ] Write failing tests for `r_delta`, train-only mean drift, `r_delta_perp`, and primary margin vs mean baseline.
- [ ] Implement fail-closed mechanism gates: graph beats random/constant, ESM beats zero/same/shuffled, time beats shuffled/reversed.
- [ ] Run `pytest tests/test_rebuild_eval_hpo.py -q`.

### Task 3: Cross-Validation Splitters

**Files:**
- Create: `src/sheaf_ts/eval/cross_validation.py`
- Test: `tests/test_rebuild_eval_hpo.py`

- [ ] Write failing tests for patient-disjoint repeated K-fold and temporal blocked splits.
- [ ] Implement deterministic splitters that raise on impossible split requests.
- [ ] Run `pytest tests/test_rebuild_eval_hpo.py -q`.

### Task 4: Optuna Adapter

**Files:**
- Create: `src/sheaf_ts/hpo/__init__.py`
- Create: `src/sheaf_ts/hpo/optuna_runner.py`
- Test: `tests/test_rebuild_eval_hpo.py`

- [ ] Write failing tests for deterministic parameter suggestions and missing-Optuna behavior.
- [ ] Implement `SearchSpace`, `TrialResult`, and `run_optuna_cv` using an injected objective callable.
- [ ] Ensure module import works even if Optuna is absent; only running search requires Optuna.
- [ ] Run `pytest tests/test_rebuild_eval_hpo.py -q`.

### Task 5: Verification

Run:

```powershell
python -m compileall -q src/sheaf_ts
pytest tests/test_rebuild_sheaf_core.py tests/test_rebuild_eval_hpo.py -q
pytest tests/test_sheaf_math.py tests/test_sheaf.py -q
```

Expected:
- new clean package compiles
- new rebuild tests pass
- legacy sheaf math tests still pass

## Implementation Status

Implemented files:
- `src/sheaf_ts/eval/artifacts.py`
- `src/sheaf_ts/eval/metrics.py`
- `src/sheaf_ts/eval/gates.py`
- `src/sheaf_ts/eval/cross_validation.py`
- `src/sheaf_ts/eval/stats.py`
- `src/sheaf_ts/hpo/optuna_runner.py`
- `tests/test_rebuild_eval_hpo.py`

Fresh verification:
- `python -m compileall -q src/sheaf_ts`
- `pytest tests/test_rebuild_sheaf_core.py tests/test_rebuild_eval_hpo.py -q`
- `pytest tests/test_sheaf_math.py tests/test_sheaf.py -q`
- Optuna import smoke with `PYTHONPATH=src`, Optuna version `4.7.0`

## HPC Policy

Slurm jobs should be launched only after local smoke tests pass and the job can
write a complete manifest. Do not use HPC as a debugging substitute for missing
local invariants.

Minimum conditions before submitting Slurm:
- clean package compiles
- rebuild tests pass locally
- synthetic backtest runner exists
- resolved config, git status, command, package versions, seed list and data
  hashes are written to the run directory
- failed gates make the job exit non-zero in strict scientific mode

Synthetic runner status:
- `scripts/run_synthetic_sheaf_backtest.py` exists.
- Local strict smoke wrote `scratch/synthetic_sheaf_manifest.json`.
- The manifest accepted mean-drift and random-graph controls.
- Before Slurm submission, add an HPC manifest wrapper that records Slurm job
  metadata, git status, package versions and command line.
