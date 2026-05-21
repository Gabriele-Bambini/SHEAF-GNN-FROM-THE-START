# Postmortem Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the TS-GNN postmortem into a clean canonical repo state, real-data validation gate, and aligned BRCA/CRC manuscript story.

**Architecture:** Keep `ts-gnn/` as canonical. Make docs point to one source of truth, add tests around real-data loader contracts, then archive shadow/refactor residue only after unique content is inventoried.

**Tech Stack:** Python, pytest, GitHub Actions or manual CI shell jobs, Markdown docs, Graphify shared ledger.

---

## File Structure

- Modify: `PROJECT_STATUS.md` - replace stale status with redirect to `PROJECT_CONTEXT.md`.
- Modify: `README.md` - document canonical status and startup order.
- Create: `CANONICAL.md` - define canonical tree and quarantine rules.
- Create: `src/tsgnn/data/README.md` - document loader autodiscovery convention.
- Modify: `src/tsgnn/data/grn_construction.py` - improve missing-data errors with actual candidate paths.
- Modify: `tests/test_grn_construction.py` - add extension-drift tests.
- Create: `.github/workflows/slow-real-data.yml` - manual slow E2E validation job, if GitHub CI is desired.
- Create: `docs/research-writing-review-loop.md` - codify review-to-paper workflow.
- Modify: `.gitignore` - add debug/sync artifact patterns.

## Task 0: Freeze And Inspect Workspace

- [ ] **Step 1: Capture Git status**

Run:

```powershell
rtk git status --short
```

Expected: current anomaly is visible, with many paths duplicated as deleted and untracked.

- [ ] **Step 2: Do not delete files**

No command. Confirm in notes that cleanup waits until Git/index state is fixed.

Expected: no file removal in this task.

## Task 1: Declare Canonical Source

- [ ] **Step 1: Create `CANONICAL.md`**

Content:

```markdown
# Canonical Repository Policy

Canonical source tree: `ts-gnn/`.

Read order:
1. `PROJECT_CONTEXT.md`
2. `README.md`
3. `docs/audits/`
4. source under `src/tsgnn/`

Historical or quarantine trees:
- `ultimate/`
- `FROM THE START/`
- any recovered `ts-gnn-chatgpt/` or `ts-gnn local/`

Rules:
- Do not edit quarantine trees unless the task explicitly says so.
- Port unique value into `src/tsgnn/` with tests.
- Delete/archive quarantine trees only after an inventory document lists unique files.
```

- [ ] **Step 2: Update `README.md`**

Add near top:

```markdown
> Canonical status: this repository treats `PROJECT_CONTEXT.md` and `src/tsgnn/` as the source of truth. Historical trees such as `ultimate/` and `FROM THE START/` are quarantine material until inventoried.
```

- [ ] **Step 3: Run doc check**

Run:

```powershell
rtk powershell -NoProfile -Command "Select-String -LiteralPath README.md,CANONICAL.md -Pattern 'Canonical|PROJECT_CONTEXT|quarantine'"
```

Expected: both files show canonical policy.

## Task 2: Retire Stale Project Status

- [ ] **Step 1: Replace `PROJECT_STATUS.md`**

Content:

```markdown
# Project Status Redirect

This file is archived/stale.

Current source of truth:
- `PROJECT_CONTEXT.md`
- latest audit under `docs/audits/`

Reason: this file contained branch/session-specific claims that conflict with `PROJECT_CONTEXT.md`.
```

- [ ] **Step 2: Verify no active doc points to stale status**

Run:

```powershell
rtk powershell -NoProfile -Command "Select-String -Path *.md,docs/**/*.md -Pattern 'PROJECT_STATUS' -CaseSensitive:$false"
```

Expected: only redirect/archive references remain.

## Task 3: Harden GRN Loader Contract

- [ ] **Step 1: Add tests for TargetGeneReg `.csv` autodiscovery**

In `tests/test_grn_construction.py`, add:

```python
def test_load_targetgenereg_discovers_csv(tmp_path, monkeypatch):
    from tsgnn.data import grn_construction as grn

    external = tmp_path / "external"
    tgr = external / "targetgenereg"
    tgr.mkdir(parents=True)
    (tgr / "p53_targets_benchmark.csv").write_text("gene_name,regulation_direction,evidence_level\nCDKN1A,activated,A\n")
    monkeypatch.setattr(grn, "EXTERNAL_DIR", external)

    df = grn.load_targetgenereg()

    assert list(df["gene_name"]) == ["CDKN1A"]
```

- [ ] **Step 2: Add tests for RegNetwork `human.source` autodiscovery**

In `tests/test_grn_construction.py`, add:

```python
def test_load_regnetwork_discovers_human_source(tmp_path, monkeypatch):
    from tsgnn.data import grn_construction as grn

    external = tmp_path / "external"
    rn = external / "regnetwork"
    rn.mkdir(parents=True)
    (rn / "human.source").write_text("TF1\tTARGET1\tactivation\n")
    monkeypatch.setattr(grn, "EXTERNAL_DIR", external)

    df = grn.load_regnetwork()

    assert not df.empty
```

- [ ] **Step 3: Run tests to verify behavior**

Run:

```powershell
rtk pytest tests/test_grn_construction.py -q
```

Expected: tests pass or fail only due to parser assumptions that must then be fixed.

- [ ] **Step 4: Improve error message in `construct_base_grn()`**

Replace static missing-data names with actual supported candidates:

```python
"  - RegNetwork database files (human_regulatory.txt or human.source)\n"
"  - TargetGeneReg database (p53_targets_benchmark.xlsx, p53_targets_benchmark.csv, or p53_targets.csv)\n"
```

- [ ] **Step 5: Run tests again**

Run:

```powershell
rtk pytest tests/test_grn_construction.py -q
```

Expected: pass.

## Task 4: Add Loader Convention Docs

- [ ] **Step 1: Create `src/tsgnn/data/README.md`**

Content:

```markdown
# Data Loader Contract

Loader rule: callers should pass no file path unless they are testing a specific file. Production callers pass `TSGNN_DATA_DIR` or rely on default discovery.

Each loader owns filename autodiscovery:
- RegNetwork: `human_regulatory.txt`, `human.source`
- TargetGeneReg: `p53_targets_benchmark.xlsx`, `p53_targets_benchmark.csv`, `p53_targets.csv`
- STRING: `9606.protein.links.v12.0.txt.gz`, `9606.protein.links.v12.0.txt`

Missing real prior data must raise a clear error. Synthetic GRNs are allowed only through `_generate_synthetic_grn()` in explicit tests/development code.
```

- [ ] **Step 2: Verify doc mentions current candidate names**

Run:

```powershell
rtk powershell -NoProfile -Command "Select-String -LiteralPath src/tsgnn/data/README.md -Pattern 'human.source|p53_targets_benchmark.csv|Synthetic'"
```

Expected: all patterns found.

## Task 5: Add Real-Data Validation Gate

- [ ] **Step 1: Create manual slow workflow**

Create `.github/workflows/slow-real-data.yml`:

```yaml
name: slow-real-data

on:
  workflow_dispatch:
    inputs:
      data_dir:
        description: "Path to mounted TSGNN data directory"
        required: true
        default: "data"

jobs:
  slow-real-data:
    runs-on: self-hosted
    env:
      TSGNN_DATA_DIR: ${{ inputs.data_dir }}
    steps:
      - uses: actions/checkout@v4
      - name: Install test dependencies
        run: |
          python -m pip install -U pip
          python -m pip install -e ".[dev]"
      - name: Run slow real-data tests
        run: pytest -m slow tests/test_e2e_real.py -v
```

- [ ] **Step 2: Run local slow test when data is mounted**

Run:

```powershell
rtk pytest -m slow tests/test_e2e_real.py -v
```

Expected: either pass with real BRCA data or skip with explicit "Dati BRCA non scaricati".

## Task 6: Resolve BRCA/CRC Narrative

- [ ] **Step 1: Create `docs/brca-crc-position.md`**

Content:

```markdown
# BRCA/CRC Position

Primary story: BRCA-first TP53 allele-specific rewiring.

Training/data core:
- GSE176078
- GSE158508
- TCGA BRCA TP53 annotations

External validation:
- CRC GSE178341 zero-shot/cross-cancer transfer

Manuscript rule:
- Claims about model development and training should be BRCA-first.
- CRC claims must be phrased as external validation unless the training config is changed to CRC-first.
```

- [ ] **Step 2: Search manuscript inputs for CRC-first claims**

Run:

```powershell
rtk powershell -NoProfile -Command "Select-String -LiteralPath ..\\article.txt,..\\review.txt,*.md -Pattern 'CRC|colorectal|BRCA|breast' -CaseSensitive:$false"
```

Expected: list of claims to update manually.

## Task 7: Inventory `ultimate/` Before Archiving

- [ ] **Step 1: Create inventory**

Run:

```powershell
rtk powershell -NoProfile -Command "Compare-Object (Get-Content src/tsgnn/model/sheaf_vectorized.py) (Get-Content ultimate/sheaf_vectorized.py) | Out-File docs/ultimate_sheaf_vectorized_diff.txt"
```

Expected: diff file created for review.

- [ ] **Step 2: Decide merge/archive**

No deletion yet. Write decision in `docs/ultimate_inventory.md`:

```markdown
# ultimate/ Inventory

Decision: archive unless a listed item is not present in `src/tsgnn/`.

Files:
- `ultimate/sheaf_vectorized.py`
- `ultimate/base_model.py`
- `ultimate/config_schema.py`
- `ultimate/model_registry.py`
- `ultimate/test_properties.py`
```

## Task 8: Cleanup Ignore Rules

- [ ] **Step 1: Extend `.gitignore`**

Add:

```gitignore
# Debug/session artifacts
debug_output.txt
diag_output.txt
output_geo.*
.gemini/
desktop.ini
*-DESKTOP-*/
```

- [ ] **Step 2: Verify ignore patterns**

Run:

```powershell
rtk git check-ignore debug_output.txt diag_output.txt output_geo.txt .gemini/ desktop.ini data/raw-DESKTOP-0HEPV91/
```

Expected: all paths are ignored.

## Task 9: Record Shared Graphify Action

- [ ] **Step 1: Record action**

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "G:\Other computers\My PC\Desktop\graphify total context\shared\scripts\record-act.ps1" -ProjectPath "G:\Other computers\My PC\Desktop\Sheaf Neural Networks\ts-gnn" -Action "Created postmortem response audit and remediation plan; canonical policy pending Git-state reconciliation."
```

Expected: ledger append succeeds.

- [ ] **Step 2: Refresh graph after implementation**

Run only after code/doc changes are complete:

```powershell
rtk python -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('.'))"
```

Expected: graph rebuild succeeds, or missing graphify is recorded.

