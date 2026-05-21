# TS-GNN Project Post-Mortem And Cleanup - 2026-05-21

Scope: `G:\Other computers\My PC\Desktop\Sheaf Neural Networks`

Active repo: `G:\Other computers\My PC\Desktop\Sheaf Neural Networks\ts-gnn`

Mode: Graphify-first, 10-agent audit, conservative cleanup. No aggressive data,
checkpoint, result, worktree, or `.git` object deletion.

## Executive Verdict

This project is a strong engineering and evaluation artifact, but it is not
ready to be presented as a positive Nature-style TS-GNN mechanism story.

The defensible scientific story today is negative/control-driven:

- You built a serious fail-closed evaluation harness.
- The harness repeatedly falsified broad TS-GNN mechanism claims.
- BRCA TP53 source-clamped signal has some narrow support.
- CRC transfer is negative.
- Ovarian/HGSOC signal is sample-sensitive and not general.
- Allele-specific rewiring, graph-specific mechanism, FiLM/ESM mechanism,
  temporal mechanism, and broad cancer transfer are not supported.

Do not call this "Nature-ready" unless the claim is reframed as a
reproducibility/control paper or all fail-closed gates are repaired and rerun.

## What Was Done This Session

### Audit

Ten audit scopes were launched or executed:

1. Repo topology, git state, worktrees, large folders.
2. Graphify shared/local context and project identity conflicts.
3. Python/source architecture and stale imports.
4. Notebooks, experiments, outputs, datasets, duplicate artifacts.
5. Documentation and publication narrative drift.
6. Dependencies, environments, build/test commands.
7. Tests and verification coverage.
8. Secrets, credentials, and HPC operational risk.
9. Conservative cleanup candidates.
10. Scientific validity and publication readiness.

Claude Opus planning delegate was attempted through `.agent/scripts/claude-opus-plan.ps1`.
It exited with code 1 and no useful output, so the audit continued with local
Codex subagents plus direct verification.

### Cleanup Applied

Deleted only generated/rebuildable artifacts:

- Root `node_modules`.
- `ts-gnn/.pytest_cache`.
- `ts-gnn/src/tsgnn.egg-info`.
- Main-repo `__pycache__` folders under `hpc`, `scripts`, `src`, and `tests`.
- Three hash-identical duplicate Graphify converted markdown files:
  - `graphify-out/converted/critical_review_tsgnn_aa4b1aef.md`
  - `graphify-out/converted/tp53_brca_proposal_c02a5aba.md`
  - `graphify-out/converted/tsgnn_article_final_3d1b3e00.md`

Archived root loose artifacts into:

`G:\Other computers\My PC\Desktop\Sheaf Neural Networks\archive\2026-05-21-root-loose-artifacts`

Moved there:

- Root Node/doc generation mini-project:
  - `package.json`
  - `package-lock.json`
  - `generate_article.js`
- Old article/review/prompt outputs:
  - `article.txt`
  - `review.txt`
  - `TSGNN_ClaudeCode_Prompt.md`
  - `tsgnn_article_final.docx`
  - `critical_review_tsgnn.docx`
- Loose reference PDFs:
  - `tommaso_ravasio_bemacs_25.pdf`
  - `Claude Code Prompt.pdf`
  - `tp53_grn_ultimate.pdf`

Small source/doc fixes applied:

- Repaired `requirements.txt`, removing NUL corruption and restoring
  `torchdiffeq==0.2.5`.
- Aligned `pyproject.toml` version to `0.2.0`, matching `src/tsgnn/__init__.py`.
- Fixed README reference from missing `hpc/ABLATION_REPORT.md` to
  `docs/ABLATION_REPORT.md`.

### Not Touched

Intentionally left untouched:

- `ts-gnn/data`
- `ts-gnn/checkpoints`
- `ts-gnn/results`
- `ts-gnn/FROM THE START`
- `ts-gnn/FROM THE START/.claude/worktrees`
- `.git` internals and loose Git objects
- core Graphify directories
- source, tests, configs, docs, HPC scripts

Reason: these contain primary data, results, worktree state, or project context.
They need explicit archival decisions, not blind deletion.

## Chain Of Verification

## Session Verification Results

Commands run after cleanup:

```powershell
python -c "from pathlib import Path; b=Path('ts-gnn/requirements.txt').read_bytes(); print(b.count(b'\x00'))"
git -C ts-gnn diff --check
python -m pytest --collect-only -q -p no:cacheprovider
python -m ruff check src tests
python -m pytest tests -m 'not slow' -q -p no:cacheprovider --tb=short
git -C ts-gnn status --short --branch
```

Observed results:

- `requirements.txt` NUL count: `0`.
- `git diff --check`: exit 0.
- pytest collect-only: `258 tests collected in 2.75s`.
- fast pytest suite: `251 passed, 6 failed, 1 deselected, 3 warnings in 69.50s`.
- failing tests:
  - `tests/test_fts_hpc_rebuild_cache.py::test_sync_password_quotes_remote_paths_with_spaces`
  - `tests/test_hardening_verification.py::test_implicit_gradient_flow`
  - `tests/test_hardening_verification.py::test_neural_ode_step_invariance`
  - `tests/test_hardening_verification.py::test_diffusion_method_parity`
  - `tests/test_implicit_diffusion.py::test_config_factory_no_implicit`
  - `tests/test_predict_delta.py::test_gradient_flows_predict_delta`
- ruff: failed with `1371 errors`; first classes are import ordering,
  Python 3.13 style upgrades, unused imports/locals, naming rules in tests, and
  an invalid `# noqa` directive in `src/tsgnn/model/edge_weight_net.py`.
- Graphify refresh: first run failed on Windows `charmap` encoding for a unicode
  arrow; rerun with `PYTHONUTF8=1` succeeded and rebuilt `graphify-out` with
  `3568 nodes`, `5533 edges`, and `264 communities`.

Interpretation:

The package is importable enough to collect tests and run most of the active
suite, but hardening/mechanism tests currently fail. This independently supports
the scientific post-mortem: the remaining failures are not cosmetic; they point
at HPC path quoting, gradient flow, ODE sensitivity, diffusion parity, config
routing, and residual-gradient behavior.

### V0 - Context Attachment

Evidence:

- Shared map contains `Sheaf Neural Networks -> sheaf-neural-networks`.
- Nested mappings also exist:
  - `ts-gnn -> ts-gnn`
  - `ts-gnn/FROM THE START -> from-the-start`

Conclusion:

The project has three overlapping Graphify identities. For repo-specific facts,
prefer the deepest relevant context, but this should be consolidated.

### V1 - Topology Verification

Evidence:

- Workspace root is not a Git repo.
- `ts-gnn` is the active Git repo.
- Branch: `codex/sheaf-blindato-v3`.
- Remote: `https://github.com/GabrieleBambini00/ts-gnn.git`.
- `ts-gnn/FROM THE START/.claude/worktrees` contains registered worktrees.

Conclusion:

The workspace is not one clean repository. It is a parent folder containing a
real repo, loose manuscript/reference artifacts, Graphify context, and archived
worktrees.

### V2 - Git Health Verification

Evidence:

- `.git` is about 9.1 GB.
- `git count-objects -vH` reported about 8.91 GiB loose objects.
- Git warns about broken refs from `desktop.ini`:
  - `.git/refs/desktop.ini`
  - `.git/refs/heads/desktop.ini`
  - `.git/refs/heads/codex/desktop.ini`
  - `.git/refs/heads/claude/desktop.ini`
  - `.git/refs/tags/desktop.ini`

Conclusion:

Git metadata is dirty and slow. Do not manually prune `.git/objects`. Make a
backup, remove/repair the bogus `desktop.ini` refs, then run a controlled
`git gc`/`git fsck` sequence.

### V3 - Dependency Verification

Evidence:

- `requirements.txt` had 20 NUL bytes near `mygene==3.2.2` and
  `torchdiffeq==0.2.5`.
- `pyproject.toml`, `requirements.txt`, `environment.yml`, Docker, and HPC
  setup disagree on Python/CUDA/Torch versions.
- Missing/optional imports include `torchdiffeq`, `paramiko`, `scikit-learn`,
  `h5py`, `mygene`, `igraph`, `leidenalg`, `pyarrow`.

Conclusion:

The dependency story is not reproducible yet. This session repaired the most
concrete corruption, but the project still needs one canonical dependency source
and generated env files.

### V4 - Test Verification

Static inventory:

- Active pytest suite: `ts-gnn/tests`
- 35 `test_*.py` files
- 239 static `test_` function definitions
- `pytest.ini` restricts test discovery to `tests`
- `ultimate/test_properties.py` is a shadow suite, not collected by default
- Root `test_*.py` files are network/manual probes, not active pytest suite

Risks:

- No active `.github/workflows`.
- `CHANGELOG.md` claims CI exists, but filesystem contradicts that.
- Real-data test can skip silently if BRCA raw path is absent.
- Coverage gate is not enforced by default.
- `Makefile` lint target can swallow ruff failure.

Required commands:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest --collect-only -q -p no:cacheprovider
python -m pytest tests -m "not slow" -q -p no:cacheprovider --tb=short
python -m pytest ultimate/test_properties.py -q --tb=short
python -m pytest tests ultimate/test_properties.py --cov=tsgnn --cov=sheaf_ts --cov-report=term-missing --cov-fail-under=50 --tb=short
python -m pytest -m slow tests/test_e2e_real.py -v --tb=short
```

Conclusion:

There is broad unit coverage, but release verification is not enforceable until
CI and real-data gates are explicit.

### V5 - Source Architecture Verification

Findings:

- Current source and `FROM THE START` duplicate core classes:
  - `src/tsgnn/model/tsgnn.py`
  - `FROM THE START/src/tsgnn/model/tsgnn.py`
  - `src/sheaf_ts/core/laplacian_builders.py`
  - `FROM THE START/src/sheaf_ts/core/laplacian_builders.py`
- `hpc/smoke_test.py` imports stale `tsgnn.model.film_nodal`, absent from current package.
- CLI entrypoints depend on repo-local `scripts/run_pipeline.py`.
- `tsgnn-evaluate` is a stub.

Conclusion:

The active package exists, but it is still entangled with archive/source-shadow
trees and repo-local scripts.

### V6 - Artifact Verification

Evidence:

- Total scanned size about 24.2 GB.
- `ts-gnn/data` about 12.7 GB.
- `ts-gnn/FROM THE START/.claude/worktrees` about 10.9 GB.
- Worktree duplicate tarballs/data can recover roughly 9-10 GB, but only after
  unmerged edits are audited.
- Primary raw/processed data and checkpoints should be preserved.

Conclusion:

Most space is not disposable cache. The safe delete was small by design.
The big cleanup is a worktree/data archival decision.

### V7 - Security/HPC Verification

Findings:

- No obvious plaintext private keys, API tokens, `.env`, or `.env.local` in scope.
- HPC scripts expose operational risk:
  - password-capable Paramiko flow
  - `AutoAddPolicy()` host-key trust bypass
  - hard-coded personal/HPC account paths
  - unpassphrased SSH key generation encouraged
  - broad `scancel` helpers
  - remote shell injection surfaces in sync/submit flows

Conclusion:

This is not a leaked-secret emergency, but HPC automation should be hardened
before sharing or publishing.

### V8 - Scientific Verification

Core evidence:

- `docs/ABLATION_REPORT.md`: full stack underperforms; A3/B1 harmful in single seed.
- `docs/TEST_LOG.md`: FiLM swing dead, shuffled pseudotime better, random graph same as real graph.
- `docs/CHAIN_OF_VERIFICATION_ASSESSMENT_2026-05-02.md`: math sound in principle, but runtime/data/metrics do not support broad graph/ESM/temporal/allele claims.
- `FROM THE START/results/hpc_nature_multiseed_496333/interpretation.md`: CRC zero-shot fails; baselines/shuffle beat TS-GNN on completed seeds.
- `FROM THE START/results/probe_alpha/...summary.json`: full-stack v2 Cayley does not beat MLP/legacy.
- `FROM THE START/results/tp53_regulon_hpc_cached_488452_source_clamped_certified.json`: narrow BRCA TP53 source-clamped signal passes.
- `FROM THE START/results/tp53_real_brca_crc_source_clamped_500g_k10_metadata_aware_489308.json`: CRC source-clamped fails destructive control.
- `FROM THE START/results/ovarian_final_cache_candidate/source_signature_failure_autopsy.md`: random-source controls beat context sources in all 9 folds.
- `FROM THE START/results/final_training_contract.json`: broad ovarian, allele-specific, and final TS-GNN training claims are explicitly blocked.

Conclusion:

Positive mechanism claims are not currently defensible. Negative/control-driven
claim is defensible.

## Highest Priority Findings

### P0 - Scientific Story Drift

Problem:

Docs and root artifacts still imply positive mechanism/Nature readiness, while
latest evidence blocks those claims.

Action:

- Make `docs/CHAIN_OF_VERIFICATION_ASSESSMENT_2026-05-02.md` or a new
  `docs/CURRENT_SCIENTIFIC_STATUS.md` the claim source of truth.
- Stamp `PROJECT_STATUS.md`, `CRITICAL_PATH.md`, `NATURE_PLAN.md`,
  `ARCHITECTURE_REPORT.md`, and old manuscript files as historical unless updated.

### P0 - Git Metadata Bloat And Broken Refs

Problem:

`.git` is about 9.1 GB with broken `desktop.ini` refs and about 8.91 GiB loose
objects.

Action:

1. Backup `ts-gnn/.git`.
2. Remove only bogus `desktop.ini` ref files after backup.
3. Run:

```powershell
git -C ts-gnn fsck --full
git -C ts-gnn gc --prune=now
git -C ts-gnn count-objects -vH
```

Do this as a separate git-maintenance task.

### P0 - Worktree/Data Shadow Explosion

Problem:

`FROM THE START/.claude/worktrees` is about 10.9 GB and contains duplicate data,
tarballs, notebooks, logs, and code.

Action:

- For each worktree, run:

```powershell
git -C "<worktree>" status --short --branch
git -C "<worktree>" diff --stat
git -C "<worktree>" ls-files --others --exclude-standard
```

- Preserve any meaningful diffs.
- Archive or remove stale worktrees through `git worktree remove`, not manual delete.

### P1 - Reproducibility Split

Problem:

`pyproject.toml`, `requirements.txt`, `requirements-dev.txt`, `environment.yml`,
Docker, and HPC scripts disagree.

Action:

- Pick one source of truth, preferably `pyproject.toml` plus lock generation.
- Split extras: `dev`, `hpc`, `ode`, `viz`, `real-data`.
- Generate `requirements*.txt` from the canonical source.

### P1 - No Active CI

Problem:

Tests exist, but no active GitHub Actions workflow exists. `ultimate/ci.yml` is
not active and would swallow lint failure.

Action:

- Add `.github/workflows/ci.yml`.
- Run fast unit tests, lint, import smoke, and optional scheduled real-data gate.

### P1 - HPC Security Hardening

Problem:

Password-capable automation, AutoAddPolicy, broad scancel, and unquoted remote
commands are not publish/share safe.

Action:

- Move password flow to local-only ignored scripts or remove it.
- Use key-only BatchMode by default.
- Reject unknown host keys.
- Scope cancellation by job id/name.
- Quote every remote path.

## Cleanup Backlog

### Safe Next Deletes

Only after another quick verification:

- Remaining generated cache directories:
  - `.ruff_cache`
  - `.pytest_cache`
  - `__pycache__`
  - `*.egg-info`
- `desktop.ini` outside `.git` and outside registered worktrees.

### Archive Candidates

- `ts-gnn/ultimate`
- root loose docs not yet archived:
  - `tp53_brca_proposal.docx`
  - `tp53_gnn_project.md`
  - old proposal/report files, after checking whether they are cited
- root probe scripts:
  - `test_urls.py`
  - `test_url_status.py`
  - `test_api.py`
  - `test_geo.py`
  - `test_fixes.py`

### Review Required

- `ts-gnn/checkpoints`
- `ts-gnn/results`
- `ts-gnn/FROM THE START`
- `ts-gnn/FROM THE START/.claude/worktrees`
- duplicate ovarian raw data under both `ts-gnn/data` and `FROM THE START/data`

## Post-Mortem: What Went Wrong

### 1. Project Identity Fragmented

The same physical tree became:

- a parent workspace,
- a Git repo,
- a nested Graphify project,
- an archive/source-shadow project,
- and a worktree host.

Result: search, graph context, and scientific state disagree.

### 2. Positive Narrative Outlived Negative Evidence

Early docs kept optimistic labels like Nature-ready, validated, final, or
mechanistic after controls had already falsified the broad claims.

Result: readers cannot tell whether a document is current status, aspiration, or
historical artifact.

### 3. Verification Was Strong But Not Gated

The project has many good tests, contracts, probes, and audits. The problem is
not lack of verification effort. The problem is lack of one enforced release
gate that makes stale claims impossible.

### 4. HPC Became Product Code

Local HPC convenience scripts grew into real automation with account-specific
paths, password support, broad cancellation, and host-key shortcuts.

Result: useful for solo iteration, risky for sharing.

### 5. Data And Worktrees Became Storage

Worktrees stored data copies, tarballs, zips, logs, and notebook clones.

Result: 10+ GB of ambiguous state that cannot be safely deleted without
worktree-by-worktree review.

## Post-Mortem: What Worked

- Fail-closed contracts are the best part of the project.
- Graphify helped reveal core abstractions and duplicate/shadow structure.
- Tests are numerous enough to support serious cleanup once CI exists.
- The project already contains strong internal evidence against overclaiming.
- Conservative cleanup avoided data loss and kept reproducibility artifacts.

## Recommended Next Sprint

1. Create `docs/CURRENT_SCIENTIFIC_STATUS.md`.
2. Add active `.github/workflows/ci.yml`.
3. Fix `hpc/smoke_test.py` stale `film_nodal` import.
4. Harden HPC auth and cancellation.
5. Audit and remove stale Claude worktrees using `git worktree remove`.
6. Consolidate dependency sources.
7. Move `FROM THE START` outside the active repo or mark it as immutable archive.
8. Run the chain:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest --collect-only -q -p no:cacheprovider
python -m pytest tests -m "not slow" -q -p no:cacheprovider --tb=short
python -m pytest ultimate/test_properties.py -q --tb=short
python -m ruff check src tests
git -C . status --short --branch
```

## Final Classification

Repository health:

- Codebase: promising but split by archives/shadows.
- Git health: poor until `.git` bloat and broken refs are fixed.
- Test base: broad, not CI-enforced.
- Dependency reproducibility: weak but repairable.
- HPC safety: solo-use acceptable, share/publication unsafe.
- Scientific readiness: negative/control paper possible; positive mechanism
  paper not ready.

The next high-leverage move is not more modeling. It is claim hygiene plus
reproducibility hardening.
