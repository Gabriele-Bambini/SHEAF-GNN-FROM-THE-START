# Anti-Shortcut Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent TS-GNN from reporting scientifically positive results when performance is explained by shortcut baselines rather than real graph, allele, or temporal signal.

**Architecture:** Add a strict scientific gate layer to the Nature pipeline, then fix the highest-leverage information-flow bugs that make graph/ESM/time inert, and finally rerun the full L0→L5 ladder under stop/go acceptance criteria. The pipeline must fail closed: if a model does not beat shortcut controls, the run is marked scientifically invalid even if raw headline metrics look good.

**Tech Stack:** Python, PyTorch, pytest, existing `scripts/run_nature.py` pipeline, existing HPC probe scripts, YAML configs, JSON artifacts.

---

## File Map

**New files**
- `G:/Other computers/My PC/Desktop/Sheaf Neural Networks/ts-gnn/src/tsgnn/evaluation/shortcut_gates.py`
- `G:/Other computers/My PC/Desktop/Sheaf Neural Networks/ts-gnn/tests/test_shortcut_gates.py`

**Modified files**
- `G:/Other computers/My PC/Desktop/Sheaf Neural Networks/ts-gnn/scripts/run_nature.py`
- `G:/Other computers/My PC/Desktop/Sheaf Neural Networks/ts-gnn/src/tsgnn/data/nature_adapter.py`
- `G:/Other computers/My PC/Desktop/Sheaf Neural Networks/ts-gnn/src/tsgnn/model/temporal_ode.py`
- `G:/Other computers/My PC/Desktop/Sheaf Neural Networks/ts-gnn/src/tsgnn/model/edge_weight_net.py`
- `G:/Other computers/My PC/Desktop/Sheaf Neural Networks/ts-gnn/src/tsgnn/model/tsgnn.py`
- `G:/Other computers/My PC/Desktop/Sheaf Neural Networks/ts-gnn/configs/nature.yaml`
- `G:/Other computers/My PC/Desktop/Sheaf Neural Networks/ts-gnn/configs/nature_v2.yaml`
- `G:/Other computers/My PC/Desktop/Sheaf Neural Networks/ts-gnn/tests/test_nature_infra.py`
- `G:/Other computers/My PC/Desktop/Sheaf Neural Networks/ts-gnn/tests/test_neural_ode.py`
- `G:/Other computers/My PC/Desktop/Sheaf Neural Networks/ts-gnn/tests/test_tsgnn.py`

**Runtime artifacts**
- `G:/Other computers/My PC/Desktop/Sheaf Neural Networks/ts-gnn/results/runs/*/benchmarks.json`
- `G:/Other computers/My PC/Desktop/Sheaf Neural Networks/ts-gnn/results/runs/*/nature_run_summary.json`

---

## Phase 1 — Fail Closed on Shortcut Wins

**Objective:** Make the Nature pipeline scientifically incapable of declaring success when the model only matches or loses to shortcut controls.

**Stop/Go rule for Phase 1**
- **Go** only if the pipeline can emit explicit gate verdicts and mark a run invalid when any critical gate fails.
- **Stop** if a run can still produce a “positive” summary without proving superiority over shortcut baselines.

### Task 1: Add shortcut-gate evaluation primitives

**Files:**
- Create: `G:/Other computers/My PC/Desktop/Sheaf Neural Networks/ts-gnn/src/tsgnn/evaluation/shortcut_gates.py`
- Test: `G:/Other computers/My PC/Desktop/Sheaf Neural Networks/ts-gnn/tests/test_shortcut_gates.py`

- [ ] **Step 1: Write the failing tests**

```python
from tsgnn.evaluation.shortcut_gates import (
    ShortcutGateConfig,
    evaluate_shortcut_gates,
)


def test_evaluate_shortcut_gates_fails_when_model_does_not_beat_mean_predictor():
    cfg = ShortcutGateConfig(
        min_delta_over_mean=0.01,
        min_shuffle_drop=0.05,
        min_random_graph_drop=0.05,
        min_esm_swing=0.01,
    )
    result = evaluate_shortcut_gates(
        cfg,
        model_r_perp=0.6972,
        mean_r_perp=0.7084,
        shuffled_r_perp=0.7435,
        random_graph_r_perp=0.6987,
        esm_swing=0.0010,
    )
    assert result["scientific_pass"] is False
    assert result["gates"]["beats_mean_predictor"]["pass"] is False


def test_evaluate_shortcut_gates_passes_only_when_all_critical_gates_pass():
    cfg = ShortcutGateConfig(
        min_delta_over_mean=0.01,
        min_shuffle_drop=0.05,
        min_random_graph_drop=0.05,
        min_esm_swing=0.01,
    )
    result = evaluate_shortcut_gates(
        cfg,
        model_r_perp=0.82,
        mean_r_perp=0.70,
        shuffled_r_perp=0.60,
        random_graph_r_perp=0.58,
        esm_swing=0.03,
    )
    assert result["scientific_pass"] is True
```

- [ ] **Step 2: Run the tests to verify RED**

Run:

```powershell
pytest tests/test_shortcut_gates.py -q
```

Expected: import failure because `tsgnn.evaluation.shortcut_gates` does not exist yet.

- [ ] **Step 3: Write the minimal gate module**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ShortcutGateConfig:
    min_delta_over_mean: float = 0.01
    min_shuffle_drop: float = 0.05
    min_random_graph_drop: float = 0.05
    min_esm_swing: float = 0.01


def evaluate_shortcut_gates(
    cfg: ShortcutGateConfig,
    *,
    model_r_perp: float,
    mean_r_perp: float,
    shuffled_r_perp: float,
    random_graph_r_perp: float,
    esm_swing: float,
) -> dict:
    gates = {
        "beats_mean_predictor": {
            "pass": (model_r_perp - mean_r_perp) >= cfg.min_delta_over_mean,
            "margin": model_r_perp - mean_r_perp,
        },
        "uses_pseudotime": {
            "pass": (model_r_perp - shuffled_r_perp) >= cfg.min_shuffle_drop,
            "margin": model_r_perp - shuffled_r_perp,
        },
        "uses_graph": {
            "pass": (model_r_perp - random_graph_r_perp) >= cfg.min_random_graph_drop,
            "margin": model_r_perp - random_graph_r_perp,
        },
        "uses_esm": {
            "pass": esm_swing >= cfg.min_esm_swing,
            "margin": esm_swing,
        },
    }
    return {
        "scientific_pass": all(g["pass"] for g in gates.values()),
        "gates": gates,
    }
```

- [ ] **Step 4: Run the tests to verify GREEN**

Run:

```powershell
pytest tests/test_shortcut_gates.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/tsgnn/evaluation/shortcut_gates.py tests/test_shortcut_gates.py
git commit -m "test: add shortcut gate primitives"
```

### Task 2: Thread the gate verdicts into the Nature pipeline

**Files:**
- Modify: `G:/Other computers/My PC/Desktop/Sheaf Neural Networks/ts-gnn/scripts/run_nature.py`
- Modify: `G:/Other computers/My PC/Desktop/Sheaf Neural Networks/ts-gnn/configs/nature.yaml`
- Modify: `G:/Other computers/My PC/Desktop/Sheaf Neural Networks/ts-gnn/configs/nature_v2.yaml`
- Test: `G:/Other computers/My PC/Desktop/Sheaf Neural Networks/ts-gnn/tests/test_nature_infra.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_phase_zeroshot_summary_includes_scientific_gate_block():
    result = {
        "tsgnn": {
            "crc_delta_pearson": 0.70,
            "crc_shuffled_delta_pearson": 0.74,
            "crc_random_graph_delta_pearson": 0.70,
            "crc_mean_predictor_delta_pearson": 0.71,
            "crc_esm_swing": 0.001,
        }
    }
    gate_block = run_nature._build_shortcut_gate_block(
        result,
        {
            "predictions": {},
            "science": {"strict_shortcut_gates": True},
            "shortcut_gates": {
                "min_delta_over_mean": 0.01,
                "min_shuffle_drop": 0.05,
                "min_random_graph_drop": 0.05,
                "min_esm_swing": 0.01,
            },
        },
    )
    assert gate_block["scientific_pass"] is False
    assert gate_block["gates"]["uses_pseudotime"]["pass"] is False


def test_nature_summary_marks_run_invalid_when_shortcut_gate_fails():
    summary = {"zeroshot": {"scientific_gate": {"scientific_pass": False}}}
    verdict = run_nature._derive_scientific_verdict(summary)
    assert verdict["scientific_pass"] is False
```

- [ ] **Step 2: Run the tests to verify RED**

Run:

```powershell
pytest tests/test_nature_infra.py -q
```

Expected: missing helper functions in `run_nature.py`.

- [ ] **Step 3: Implement gate integration**

```python
from tsgnn.evaluation.shortcut_gates import ShortcutGateConfig, evaluate_shortcut_gates


def _build_shortcut_gate_block(results: dict, cfg: dict) -> dict:
    gate_cfg = ShortcutGateConfig(**cfg.get("shortcut_gates", {}))
    model = results["tsgnn"]
    return evaluate_shortcut_gates(
        gate_cfg,
        model_r_perp=model["crc_delta_perp"],
        mean_r_perp=model["crc_mean_predictor_delta_perp"],
        shuffled_r_perp=model["crc_shuffled_delta_perp"],
        random_graph_r_perp=model["crc_random_graph_delta_perp"],
        esm_swing=model["crc_esm_swing"],
    )


def _derive_scientific_verdict(summary: dict) -> dict:
    zeroshot_gate = summary.get("zeroshot", {}).get("scientific_gate")
    return {
        "scientific_pass": bool(zeroshot_gate and zeroshot_gate["scientific_pass"]),
    }
```

Config block to add in both YAML files:

```yaml
science:
  strict_shortcut_gates: true

shortcut_gates:
  min_delta_over_mean: 0.01
  min_shuffle_drop: 0.05
  min_random_graph_drop: 0.05
  min_esm_swing: 0.01
```

- [ ] **Step 4: Run the tests to verify GREEN**

Run:

```powershell
pytest tests/test_nature_infra.py -q
```

Expected: all new tests pass and existing infra tests stay green.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_nature.py configs/nature.yaml configs/nature_v2.yaml tests/test_nature_infra.py
git commit -m "feat: fail nature pipeline on shortcut wins"
```

---

## Phase 2 — Restore Real Signal Paths

**Objective:** Fix the highest-probability architectural blockers that keep time and allele signals inert even when the Laplacian math is corrected.

**Stop/Go rule for Phase 2**
- **Go** only if the code path can physically carry nontrivial time and ESM signal into the model and the related tests prove the signal is not normalized away at initialization.
- **Stop** if the pipeline still computes nominal controls while the model entry points cannot consume the underlying signal.

### Task 3: Pass real temporal boundaries through adapter → pipeline → model

**Files:**
- Modify: `G:/Other computers/My PC/Desktop/Sheaf Neural Networks/ts-gnn/src/tsgnn/data/nature_adapter.py`
- Modify: `G:/Other computers/My PC/Desktop/Sheaf Neural Networks/ts-gnn/scripts/run_nature.py`
- Modify: `G:/Other computers/My PC/Desktop/Sheaf Neural Networks/ts-gnn/src/tsgnn/model/tsgnn.py`
- Test: `G:/Other computers/My PC/Desktop/Sheaf Neural Networks/ts-gnn/tests/test_nature_infra.py`
- Test: `G:/Other computers/My PC/Desktop/Sheaf Neural Networks/ts-gnn/tests/test_tsgnn.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_adapter_returns_bin_boundaries_for_dpt_binning():
    import anndata as ad
    import numpy as np
    import pandas as pd

    X = np.random.default_rng(0).normal(size=(24, 12)).astype(np.float32)
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(24)]),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(12)]),
    )
    bundle = adata_to_tsgnn_tensors(
        adata,
        K=5,
        n_genes=10,
        pseudotime_method="dpt",
    )
    assert "bin_boundaries" in bundle
    assert len(bundle["bin_boundaries"]) == 6


def test_run_nature_passes_bin_boundaries_to_model(monkeypatch):
    called = {}

    class DummyModel:
        def __call__(self, x, esm, edge_weights_seq=None, bin_boundaries=None, return_stalks=False):
            called["bin_boundaries"] = bin_boundaries
            return [x[0]], [], []

    bundle = {"X_seq": torch.randn(4, 3, 1), "bin_boundaries": torch.linspace(0, 1, 5)}
    run_nature._predict_forecast(DummyModel(), bundle, torch.zeros(1, 8), torch.device("cpu"))
    assert called["bin_boundaries"] is not None
```

- [ ] **Step 2: Run the tests to verify RED**

Run:

```powershell
pytest tests/test_nature_infra.py tests/test_tsgnn.py -q
```

Expected: bundle lacks `bin_boundaries` and helpers do not pass them.

- [ ] **Step 3: Implement time-boundary plumbing**

```python
def _pseudotime_bins(...):
    ...
    edges = np.linspace(0, n, K + 1, dtype=int)
    bins = [order[edges[i]:edges[i + 1]] for i in range(K)]
    boundaries = np.zeros(K + 1, dtype=np.float32)
    boundaries[0] = float(pt[order[0]])
    boundaries[-1] = float(pt[order[-1]])
    for i in range(1, K):
        boundaries[i] = float(pt[order[edges[i]]])
    return bins, boundaries


bundle["bin_boundaries"] = torch.tensor(boundaries, dtype=torch.float32)
```

Then update all relevant `model(...)` calls in `run_nature.py` to pass:

```python
out = model(X_in, esm, bin_boundaries=bundle.get("bin_boundaries"))
```

- [ ] **Step 4: Run the tests to verify GREEN**

Run:

```powershell
pytest tests/test_nature_infra.py tests/test_tsgnn.py -q
```

Expected: all new tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/tsgnn/data/nature_adapter.py scripts/run_nature.py src/tsgnn/model/tsgnn.py tests/test_nature_infra.py tests/test_tsgnn.py
git commit -m "feat: propagate pseudotime boundaries through nature pipeline"
```

### Task 4: Remove ODE output LayerNorm and make the ESM path non-silent at init

**Files:**
- Modify: `G:/Other computers/My PC/Desktop/Sheaf Neural Networks/ts-gnn/src/tsgnn/model/temporal_ode.py`
- Modify: `G:/Other computers/My PC/Desktop/Sheaf Neural Networks/ts-gnn/src/tsgnn/model/edge_weight_net.py`
- Test: `G:/Other computers/My PC/Desktop/Sheaf Neural Networks/ts-gnn/tests/test_neural_ode.py`
- Test: `G:/Other computers/My PC/Desktop/Sheaf Neural Networks/ts-gnn/tests/test_tsgnn.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_identity_at_init_preserves_maps_without_layernorm(edge_index):
    ode = _make_ode(d=4)
    R0 = torch.randn(edge_index.shape[1], 2, 4, 4)
    x = torch.randn(5, 4)
    R1 = ode(R0, x, edge_index, delta_t=torch.tensor(0.1))
    diff = (R1 - R0).abs().max().item()
    assert diff < 1e-5


def test_edge_weight_head_is_not_exactly_uniform_after_init():
    net = EdgeWeightNet(stalk_dim=4, esm_latent_dim=8, hidden_dim=16)
    x = torch.randn(6, 4)
    z = torch.randn(8)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    w = net(x, z, edge_index)
    assert float(w.std()) > 0.0
```

- [ ] **Step 2: Run the tests to verify RED**

Run:

```powershell
pytest tests/test_neural_ode.py tests/test_tsgnn.py -q
```

Expected: current tests assume `LayerNorm(R0)` identity and `EdgeWeightNet` is uniform at init.

- [ ] **Step 3: Implement the minimal architectural cleanup**

In `temporal_ode.py`, replace:

```python
R_final = self.layer_norm(R_final)
```

with:

```python
# Do not normalize R_final here; preserve the ODE state geometry.
```

and then remove `self.layer_norm` from the module entirely once the tests pass.

In `edge_weight_net.py`, replace zero init:

```python
nn.init.zeros_(self.head.weight)
nn.init.zeros_(self.head.bias)
```

with a small nonzero init:

```python
nn.init.normal_(self.head.weight, mean=0.0, std=1e-3)
nn.init.zeros_(self.head.bias)
```

- [ ] **Step 4: Run the tests to verify GREEN**

Run:

```powershell
pytest tests/test_neural_ode.py tests/test_tsgnn.py -q
```

Expected: identity-at-init now means “preserve `R0`”, and edge weights are no longer exactly constant.

- [ ] **Step 5: Commit**

```bash
git add src/tsgnn/model/temporal_ode.py src/tsgnn/model/edge_weight_net.py tests/test_neural_ode.py tests/test_tsgnn.py
git commit -m "fix: unblock temporal and ESM signal paths"
```

---

## Phase 3 — Rebuild the Ladder with Hard Acceptance Gates

**Objective:** Re-establish the L0→L5 ladder under a protocol where each added block must prove it is both useful and causally active.

**Stop/Go rule for Phase 3**
- **Go** from one level to the next only if the new level beats the previous accepted level on the target metric *and* activates its own control gate.
- **Stop** immediately if a level improves the headline metric without activating the signal it claims to use.

### Task 5: Encode the acceptance protocol in the benchmark artifacts

**Files:**
- Modify: `G:/Other computers/My PC/Desktop/Sheaf Neural Networks/ts-gnn/scripts/run_nature.py`
- Test: `G:/Other computers/My PC/Desktop/Sheaf Neural Networks/ts-gnn/tests/test_nature_infra.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_scientific_verdict_rejects_level_when_controls_are_inert():
    verdict = run_nature._acceptance_decision(
        candidate_name="L5 temporal",
        baseline_name="L4 ESM",
        candidate_r_perp=0.6990,
        baseline_r_perp=0.7075,
        gate_block={
            "gates": {
                "uses_pseudotime": {"pass": False},
                "uses_graph": {"pass": False},
                "uses_esm": {"pass": False},
                "beats_mean_predictor": {"pass": True},
            },
            "scientific_pass": False,
        },
    )
    assert verdict["accepted"] is False
    assert verdict["reason"] == "control_gate_failed"
```

- [ ] **Step 2: Run the tests to verify RED**

Run:

```powershell
pytest tests/test_nature_infra.py -q
```

Expected: acceptance helper missing.

- [ ] **Step 3: Implement the acceptance helper**

```python
def _acceptance_decision(*, candidate_name, baseline_name, candidate_r_perp, baseline_r_perp, gate_block):
    if not gate_block["scientific_pass"]:
        return {"accepted": False, "reason": "control_gate_failed"}
    if candidate_r_perp <= baseline_r_perp:
        return {"accepted": False, "reason": "does_not_beat_previous_level"}
    return {"accepted": True, "reason": "accepted"}
```

- [ ] **Step 4: Run the tests to verify GREEN**

Run:

```powershell
pytest tests/test_nature_infra.py -q
```

Expected: new acceptance tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_nature.py tests/test_nature_infra.py
git commit -m "feat: encode stop-go acceptance decisions"
```

### Task 6: Operational rerun protocol for L0→L5

**Files:**
- Modify: `G:/Other computers/My PC/Desktop/Sheaf Neural Networks/ts-gnn/PROJECT_STATUS.md`
- Modify: `G:/Other computers/My PC/Desktop/Sheaf Neural Networks/ts-gnn/configs/nature_v2.yaml`
- Test: `G:/Other computers/My PC/Desktop/Sheaf Neural Networks/ts-gnn/tests/test_nature_infra.py`

- [ ] **Step 1: Write the failing documentation/status assertion**

```python
def test_project_status_mentions_shortcut_gates_as_blocking_criteria():
    text = Path("PROJECT_STATUS.md").read_text(encoding="utf-8")
    assert "beats_mean_predictor" in text
    assert "uses_pseudotime" in text
    assert "uses_graph" in text
    assert "uses_esm" in text
```

- [ ] **Step 2: Run the test to verify RED**

Run:

```powershell
pytest tests/test_nature_infra.py -q
```

Expected: status file does not mention the new blocking criteria.

- [ ] **Step 3: Update the operational protocol**

Add a section to `PROJECT_STATUS.md` that says:

```markdown
## Scientific Stop/Go Gates

Every Nature/HPC run is invalid unless:
- `beats_mean_predictor`
- `uses_pseudotime`
- `uses_graph`
- `uses_esm`

Ladder policy:
- L2 is the static sheaf reference.
- L3 is accepted only if it improves over L2 and keeps gates green.
- L4 is accepted only if ESM gate turns green.
- L5 is accepted only if pseudotime gate turns green and it beats L4.
```

Also pin the strict mode in `nature_v2.yaml`:

```yaml
science:
  strict_shortcut_gates: true
```

- [ ] **Step 4: Run the test to verify GREEN**

Run:

```powershell
pytest tests/test_nature_infra.py -q
```

Expected: operational-status test passes.

- [ ] **Step 5: Commit**

```bash
git add PROJECT_STATUS.md configs/nature_v2.yaml tests/test_nature_infra.py
git commit -m "docs: codify stop-go protocol for ladder reruns"
```

---

## Acceptance Matrix

Use this matrix after implementation and before any new claim:

| Level | Must Beat | Must Prove |
|---|---|---|
| L2 sheaf-static | `L0 MLP`, `L1 GCN` on `r_Δ⊥` | Nothing beyond `beats_mean_predictor` |
| L3 Cayley | L2 | Math-correct sheaf path stays green |
| L4 ESM | L3 | `uses_esm = PASS` |
| L5 temporal | L4 | `uses_pseudotime = PASS` |

Hard reject rules:
- `random graph ≈ real graph` → reject graph-based claim
- `shuffled pseudotime ≥ real pseudotime` → reject temporal claim
- `ESM swing < threshold` → reject allele-conditioning claim
- `model ≤ mean predictor on r_Δ⊥` → reject zero-shot claim

---

## Verification Checklist

- `pytest tests/test_shortcut_gates.py -q`
- `pytest tests/test_nature_infra.py -q`
- `pytest tests/test_tsgnn.py tests/test_neural_ode.py -q`
- One local dry-run of `scripts/run_nature.py` on a smoke configuration
- One HPC rerun of the ladder/probe with strict gates enabled

Expected outcomes before declaring success:
- artifacts contain explicit gate verdicts
- failing gates mark `scientific_pass: false`
- L4 cannot pass with dead ESM
- L5 cannot pass with dead pseudotime
