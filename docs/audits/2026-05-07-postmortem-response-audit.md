# Postmortem Response Audit - TS-GNN

Date: 2026-05-07
Scope: response to the TP53-BRCA TS-GNN postmortem findings.
Mode: verified where possible from the current workspace.

## Executive Verdict

The postmortem is directionally right, but the current workspace has moved since that snapshot.

Main correction: the visible root currently contains one active `ts-gnn/` directory, not three sibling `ts-gnn` variants. The shared graph still contains historical references to `ts-gnn-chatgpt`, and `ts-gnn/FROM THE START/` contains a later parallel code tree, but the current top-level source of truth should be `ts-gnn/`.

Current highest-risk blockers:

1. Git/workspace state is abnormal: many files appear both deleted and untracked. Freeze cleanup until this is reconciled.
2. Documentation truth is split: `PROJECT_CONTEXT.md` and `PROJECT_STATUS.md` disagree on score, phase, and next action.
3. Real-data validation remains under-controlled: `tests/test_e2e_real.py` is slow and skipped when data is absent, and no `.github/workflows` CI was found.
4. `ultimate/` is still a shadow package outside `src/tsgnn/`.
5. Manuscript narrative still needs an explicit BRCA-first vs CRC-zero-shot decision.

## Verified Evidence

| Area | Current evidence | Assessment |
|---|---|---|
| Canonical folder | Top-level root currently shows `ts-gnn/` only; no sibling `ts-gnn local/` or `ts-gnn-chatgpt/` visible. | Treat `ts-gnn/` as canonical unless old copies are recovered elsewhere. |
| GRN path fixes | `src/tsgnn/data/grn_construction.py` autodiscovers `human_regulatory.txt` and `human.source`; TargetGeneReg candidates include `.xlsx`, `p53_targets_benchmark.csv`, and `p53_targets.csv`. | Fix landed in canonical. |
| Synthetic fallback | `construct_base_grn()` raises `RuntimeError` if no prior regulatory edges are found. | Silent synthetic fallback mostly closed in canonical GRN construction. |
| Real data | `data/external/string/9606.protein.links.v12.0.txt`, `data/external/string/9606.protein.aliases.v12.0.txt.gz`, `data/external/targetgenereg/p53_targets_benchmark.csv`, `data/external/regnetwork/human.source` exist. CRC H5 also exists at `data/raw/crc/GSE178341_crc10x_full_c295v4_submit.h5`. | Data availability is better than old memory claimed. |
| E2E | `tests/test_e2e_real.py` is marked `@pytest.mark.slow` and `skipif(not BRCA_PATH.exists())`. | Real-data check can disappear silently unless deliberately run. |
| CI | No `.github/workflows` directory found in current tree; only `ultimate/ci.yml`. | CI claim needs rebuild or relocation. |
| Docs conflict | `PROJECT_CONTEXT.md` says 88.2/100, 137/137 passing, 2026-04-22 audit. `PROJECT_STATUS.md` says 41.3/100 regression and user should upload to Drive. | `PROJECT_STATUS.md` is stale or branch-specific. |
| `ultimate/` | Contains `base_model.py`, `config_schema.py`, `model_registry.py`, `sheaf_vectorized.py`, `test_properties.py`, docs, and its own `ci.yml`. | Refactor residue. Decide merge or archive. |
| CRC narrative | `configs/phase0_crc.yaml` exists; `src/tsgnn/data/crc_loader.py` exists and expects GSE178341 H5. CRC H5 is present. | F10 should be softened: CRC is no longer missing, but paper/code framing still needs explicit alignment. |
| Cleanup artifacts | `data/*-DESKTOP-*` directories exist. `.gitignore` already ignores `*.zip` and archives but does not ignore `debug_output.txt`, `diag_output.txt`, `output_geo.*`, `.gemini/`, `desktop.ini`, or `*-DESKTOP-*`. | Low-risk cleanup after Git state is fixed. |

## Finding Responses

### F1 - Three divergent TS-GNN folders

Status: partially verified, partially stale.

Current root does not expose three sibling folders. However, graph memory references historical `ts-gnn-chatgpt`, and `FROM THE START/` contains another source tree. The real issue remains: no written canonical policy.

Response:

1. Declare `ts-gnn/` canonical.
2. Add `CANONICAL.md` at repo root after Git state is reconciled.
3. Inventory `FROM THE START/` separately before deleting or merging anything.
4. Do not delete old trees until unique files are listed.

Priority: HIGH.

### F2 - `PROJECT_STATUS.md` stale

Status: verified.

`PROJECT_STATUS.md` conflicts with `PROJECT_CONTEXT.md`. It claims a 41.3/100 regression while `PROJECT_CONTEXT.md` claims 88.2/100 and 137/137 passing.

Response:

1. Keep `PROJECT_CONTEXT.md` as single source of truth.
2. Replace `PROJECT_STATUS.md` with a short redirect, or move it to `docs/archive/`.
3. Add a README note: "Start from `PROJECT_CONTEXT.md`; ignore archived status docs unless doing history reconstruction."

Priority: HIGH.

### F3 - `ultimate/` shadow package

Status: verified.

`ultimate/` is a separate implementation/documentation island. `src/tsgnn/model/sheaf_vectorized.py` has newer hardening not present in `ultimate/sheaf_vectorized.py`, including normalized export, device-buffer movement, epsilon regularization, and NaN guard.

Response:

1. Treat `src/tsgnn/` as source of truth.
2. Diff `ultimate/{base_model,config_schema,model_registry,sheaf_vectorized}.py` against current `src`.
3. Port only missing, tested value.
4. Archive/delete `ultimate/` after tests pass.

Priority: MEDIUM.

### F4 - Real-data E2E validation gap

Status: verified.

The current E2E test is useful but too weak as a release gate because it skips by path and does not exercise full training.

Response:

1. Add a mini real-data smoke fixture or a deterministic subset manifest.
2. Add `pytest -m slow tests/test_e2e_real.py -v` as a manual/on-demand CI job.
3. Add a stronger check that GRN construction uses real prior edges and never synthetic fallback.
4. Record one clean run artifact in `results/validation/`.

Priority: HIGH.

### F5 - OneDrive conflict folders

Status: verified.

`data/external-DESKTOP-0HEPV91`, `data/processed-DESKTOP-0HEPV91`, and `data/raw-DESKTOP-0HEPV91` exist.

Response:

1. First fix Git/index state.
2. Then delete only confirmed duplicate/conflict directories.
3. Extend `.gitignore` with `*-DESKTOP-*` and `desktop.ini`.

Priority: LOW.

### F6 - Hardcoded-path/silent-fallback bug class

Status: mostly verified, still worth hardening.

Canonical GRN loader has autodiscovery and `RuntimeError`. But the error message still mentions old candidate names such as `network.csv` and `TargetGeneReg.csv`, while actual candidates include `human.source` and `p53_targets_benchmark.csv`.

Response:

1. Update error messages to list actual candidate paths checked.
2. Add tests for extension drift: TargetGeneReg `.csv` when `.xlsx` absent; RegNetwork `human.source` when canonical name absent.
3. Add `src/tsgnn/data/README.md` documenting rule: caller passes directory/root or nothing; loader owns autodiscovery.

Priority: HIGH.

### F7 - Old memory stale

Status: verified.

STRING and CRC data are now present. Shared notes only contain a thin `index.md`; project memory in `graphify-out/memory/` is historical.

Response:

1. Update shared notes after cleanup decisions.
2. Record current data availability and canonical status.

Priority: LOW.

### F8 - Orphans to clean

Status: partially verified.

`.gitignore` already ignores archives like `*.zip`, but debug logs and OS/sync artifacts are not fully covered.

Response:

1. Add ignore patterns: `debug_output.txt`, `diag_output.txt`, `output_geo.*`, `.gemini/`, `desktop.ini`, `*-DESKTOP-*`.
2. Delete only after Git status is sane.

Priority: LOW.

### F9 - Review-driven article generation pattern

Status: verified from root docs.

`review.txt` and `article.txt` exist and look like a reusable critique-to-paper loop.

Response:

1. Codify as `docs/research-writing-review-loop.md`.
2. Make quality gates explicit: weakness map, section coverage, evidence status, claim/data alignment.

Priority: LOW.

### F10 - Article/code claim divergence

Status: updated.

The old claim "no CRC dataset downloaded" is now false in the current workspace: CRC loader and GSE178341 H5 are present. The remaining issue is framing: BRCA is the core training path, CRC appears to be zero-shot/cross-cancer validation.

Response:

Decision needed:

1. BRCA-first paper: train on BRCA, use CRC only as external zero-shot validation.
2. CRC-first thesis/paper: move CRC into Phase 0 and make BRCA secondary.

Recommended: BRCA-first, CRC-zero-shot. It matches proposal pivot, existing data pipeline, and `crc_loader.py` language.

Priority: HIGH.

## Immediate Triage Order

1. Freeze destructive cleanup until Git/index state is fixed.
2. Declare canonical source: `ts-gnn/`.
3. Resolve docs conflict: `PROJECT_CONTEXT.md` wins, `PROJECT_STATUS.md` becomes redirect/archive.
4. Harden real-data validation: slow E2E job plus loader autodiscovery tests.
5. Decide narrative: BRCA-first with CRC-zero-shot.
6. Archive or merge `ultimate/`.
7. Cleanup OneDrive/log/archive artifacts.
8. Update Graphify shared notes and record actions.

## Proposed One-Sentence Response to the Postmortem

The diagnosis is basically correct, but current evidence changes the remediation: `ts-gnn/` should be frozen as canonical, stale docs and shadow trees should be archived, real-data E2E must become a deliberate release gate, and the manuscript should be reframed as BRCA-first with CRC zero-shot validation because CRC data now exists but is not the main training narrative.

