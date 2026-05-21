# TS-GNN Quality Push: All Parameters to 9.0+ Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring all 5 sub-9.0 project rating parameters (Code Quality 8.5, Testing 8.0, Pipeline Completeness 8.0, Reproducibility 8.5, Usability 8.0) to 9.0+.

**Architecture:** Five independent improvement tracks, each targeting one rating parameter. Changes are additive (no rewrites of existing working code). The `ultimate/` modules get integrated into `src/tsgnn/`, sub-package `__init__.py` files get proper exports, test coverage expands via new test modules, and reproducibility/usability improve via tooling and documentation.

**Tech Stack:** Python 3.10+, PyTorch 2.x, pytest, ruff, Docker

---

## File Structure

### Files to Create
- `ts-gnn/src/tsgnn/model/sheaf_vectorized.py` — integrated vectorized sheaf (from `ultimate/`)
- `ts-gnn/src/tsgnn/model/base_model.py` — integrated ABC (from `ultimate/`)
- `ts-gnn/src/tsgnn/model/registry.py` — integrated registries (from `ultimate/`)
- `ts-gnn/src/tsgnn/config.py` — integrated typed config (from `ultimate/`)
- `ts-gnn/tests/test_config.py` — tests for config validation
- `ts-gnn/tests/test_registry.py` — tests for model/metric registries
- `ts-gnn/tests/test_vectorized.py` — tests for vectorized sheaf
- `ts-gnn/tests/test_data_loading.py` — tests for data module contracts
- `ts-gnn/tests/conftest.py` — shared fixtures
- `ts-gnn/pytest.ini` — pytest configuration with coverage enforcement
- `ts-gnn/ruff.toml` — ruff linter config (strict mode)
- `ts-gnn/CHANGELOG.md` — version history

### Files to Modify
- `ts-gnn/src/tsgnn/__init__.py` — add new exports
- `ts-gnn/src/tsgnn/model/__init__.py` — add `__all__` exports
- `ts-gnn/src/tsgnn/data/__init__.py` — add `__all__` exports
- `ts-gnn/src/tsgnn/training/__init__.py` — add `__all__` exports
- `ts-gnn/src/tsgnn/evaluation/__init__.py` — add `__all__` exports
- `ts-gnn/src/tsgnn/visualization/__init__.py` — add `__all__` exports
- `ts-gnn/pyproject.toml` — pin dependency versions, add ruff config
- `ts-gnn/Makefile` — add `test-all`, `lint-fix`, `coverage` targets
- `ts-gnn/README.md` — add ultimate module docs + CHANGELOG link
- `ts-gnn/ultimate/ci.yml` — add coverage reporting + ruff strict
- `ts-gnn/docker-compose.yml` or `ts-gnn/ultimate/docker-compose.yml` — unify

---

## Track A: Qualità del Codice (8.5 → 9.0)

### Task 1: Integrate `ultimate/` modules into `src/tsgnn/`

**Files:**
- Create: `ts-gnn/src/tsgnn/model/sheaf_vectorized.py`
- Create: `ts-gnn/src/tsgnn/model/base_model.py`
- Create: `ts-gnn/src/tsgnn/model/registry.py`
- Create: `ts-gnn/src/tsgnn/config.py`

- [ ] **Step 1: Copy `ultimate/sheaf_vectorized.py` to `src/tsgnn/model/sheaf_vectorized.py`**

```bash
cd ts-gnn
cp ultimate/sheaf_vectorized.py src/tsgnn/model/sheaf_vectorized.py
```

Fix the import at the top of the copied file if needed — it should be self-contained (it is).

- [ ] **Step 2: Copy `ultimate/base_model.py` to `src/tsgnn/model/base_model.py`**

```bash
cp ultimate/base_model.py src/tsgnn/model/base_model.py
```

- [ ] **Step 3: Copy `ultimate/model_registry.py` to `src/tsgnn/model/registry.py`**

```bash
cp ultimate/model_registry.py src/tsgnn/model/registry.py
```

Update the lazy imports inside `create_default_model_registry()` — they already use `from tsgnn.model.tsgnn import TSGNN` which is correct.

- [ ] **Step 4: Copy `ultimate/config_schema.py` to `src/tsgnn/config.py`**

```bash
cp ultimate/config_schema.py src/tsgnn/config.py
```

- [ ] **Step 5: Verify all 4 files import correctly**

```bash
cd ts-gnn
python -c "from tsgnn.model.sheaf_vectorized import VectorizedSheafDiffusion; print('OK: sheaf_vectorized')"
python -c "from tsgnn.model.base_model import AbstractGRNModel; print('OK: base_model')"
python -c "from tsgnn.model.registry import ModelRegistry, MetricRegistry; print('OK: registry')"
python -c "from tsgnn.config import TSGNNConfig; print('OK: config')"
```

Expected: all 4 print "OK"

---

### Task 2: Add `__all__` exports to all sub-package `__init__.py`

**Files:**
- Modify: `ts-gnn/src/tsgnn/model/__init__.py`
- Modify: `ts-gnn/src/tsgnn/data/__init__.py`
- Modify: `ts-gnn/src/tsgnn/training/__init__.py`
- Modify: `ts-gnn/src/tsgnn/evaluation/__init__.py`
- Modify: `ts-gnn/src/tsgnn/visualization/__init__.py`

- [ ] **Step 1: Write `model/__init__.py`**

```python
"""TS-GNN Model Components."""
from tsgnn.model.tsgnn import TSGNN, create_tsgnn_from_config
from tsgnn.model.sheaf import SheafDiffusionLayer
from tsgnn.model.sheaf_vectorized import VectorizedSheafDiffusion
from tsgnn.model.temporal import TemporalRestrictionEvolution
from tsgnn.model.film import AlleleFiLM, ZeroFiLM
from tsgnn.model.base_model import AbstractGRNModel
from tsgnn.model.baselines import EvolveGCN, TemporalGAT, create_tsgnn_no_allele
from tsgnn.model.registry import ModelRegistry, MetricRegistry

__all__ = [
    "TSGNN", "create_tsgnn_from_config",
    "SheafDiffusionLayer", "VectorizedSheafDiffusion",
    "TemporalRestrictionEvolution",
    "AlleleFiLM", "ZeroFiLM",
    "AbstractGRNModel",
    "EvolveGCN", "TemporalGAT", "create_tsgnn_no_allele",
    "ModelRegistry", "MetricRegistry",
]
```

- [ ] **Step 2: Write `data/__init__.py`**

```python
"""TS-GNN Data Loading and Preprocessing."""
__all__ = ["download", "preprocess", "allele", "grn_construction", "temporal"]
```

- [ ] **Step 3: Write `training/__init__.py`**

```python
"""TS-GNN Training Pipeline."""
from tsgnn.training.loss import TSGNNLoss
from tsgnn.training.trainer import TSGNNTrainer

__all__ = ["TSGNNLoss", "TSGNNTrainer"]
```

- [ ] **Step 4: Write `evaluation/__init__.py`**

```python
"""TS-GNN Evaluation Metrics and Benchmarking."""
from tsgnn.evaluation.metrics import (
    cross_cancer_zero_shot,
    allele_edge_enrichment,
    chromatin_remodeler_concordance,
    depmap_dependency_correlation,
    rewiring_distinguishability,
)

__all__ = [
    "cross_cancer_zero_shot",
    "allele_edge_enrichment",
    "chromatin_remodeler_concordance",
    "depmap_dependency_correlation",
    "rewiring_distinguishability",
]
```

- [ ] **Step 5: Write `visualization/__init__.py`**

```python
"""TS-GNN Visualization Utilities."""
__all__ = ["rewiring", "attention", "networks"]
```

- [ ] **Step 6: Update root `__init__.py` to export new modules**

Add to `ts-gnn/src/tsgnn/__init__.py`:

```python
from tsgnn.model.sheaf_vectorized import VectorizedSheafDiffusion
from tsgnn.model.base_model import AbstractGRNModel
from tsgnn.model.registry import ModelRegistry, MetricRegistry
from tsgnn.config import TSGNNConfig
```

And add to `__all__`:
```python
"VectorizedSheafDiffusion", "AbstractGRNModel",
"ModelRegistry", "MetricRegistry", "TSGNNConfig",
```

- [ ] **Step 7: Verify imports work**

```bash
python -c "import tsgnn; print(dir(tsgnn))"
```

Expected: all exported names visible.

---

### Task 3: Configure strict linting with ruff

**Files:**
- Create: `ts-gnn/ruff.toml`
- Modify: `ts-gnn/ultimate/ci.yml:38-41`

- [ ] **Step 1: Create `ruff.toml`**

```toml
[lint]
select = ["E", "F", "W", "I", "N", "UP", "B", "A", "SIM"]
ignore = ["E501"]  # line length handled by formatter

[lint.isort]
known-first-party = ["tsgnn"]

[lint.per-file-ignores]
"tests/**" = ["S101"]  # allow assert in tests
```

- [ ] **Step 2: Update CI to enforce ruff (remove `--exit-zero`)**

In `ts-gnn/ultimate/ci.yml`, line 41, change:
```yaml
        ruff check src/ tests/ --exit-zero
```
to:
```yaml
        ruff check src/ tests/
```

- [ ] **Step 3: Run ruff and fix auto-fixable issues**

```bash
cd ts-gnn
ruff check src/ tests/ --fix
```

- [ ] **Step 4: Verify lint passes**

```bash
ruff check src/ tests/
```

Expected: 0 errors.

---

## Track B: Test e Verificabilità (8.0 → 9.0)

### Task 4: Create shared test fixtures

**Files:**
- Create: `ts-gnn/tests/conftest.py`

- [ ] **Step 1: Create `conftest.py` with shared fixtures**

```python
"""Shared test fixtures for TS-GNN test suite."""
import pytest
import torch


@pytest.fixture
def seed():
    torch.manual_seed(42)
    return 42


@pytest.fixture
def small_graph():
    """Small graph for fast tests: 10 nodes, 25 edges, stalk_dim=2."""
    N, E, d = 10, 25, 2
    edge_index = torch.randint(0, N, (2, E))
    return {"N": N, "E": E, "d": d, "edge_index": edge_index}


@pytest.fixture
def medium_graph():
    """Medium graph: 30 nodes, 80 edges, stalk_dim=4."""
    N, E, d = 30, 80, 4
    edge_index = torch.randint(0, N, (2, E))
    return {"N": N, "E": E, "d": d, "edge_index": edge_index}


@pytest.fixture
def esm_embedding():
    return torch.randn(1280)


@pytest.fixture
def temporal_sequence(small_graph):
    """K=3 temporal sequence for small graph."""
    K, N, input_dim = 3, small_graph["N"], small_graph["N"]
    return torch.randn(K, N, input_dim)
```

- [ ] **Step 2: Verify conftest is discovered**

```bash
cd ts-gnn
python -m pytest tests/conftest.py --collect-only
```

Expected: collected 0 items (conftest has no tests, just fixtures).

---

### Task 5: Test the config schema

**Files:**
- Create: `ts-gnn/tests/test_config.py`

- [ ] **Step 1: Write tests for `TSGNNConfig`**

```python
"""Tests for typed configuration schema."""
import pytest
from tsgnn.config import (
    TSGNNConfig, DataConfig, ModelConfig, LossConfig, TrainingConfig,
)


def test_default_config_validates():
    config = TSGNNConfig()
    config.validate()  # should not raise


def test_invalid_K_raises():
    config = TSGNNConfig()
    config.data.K = 1
    with pytest.raises(AssertionError, match="K must be"):
        config.validate()


def test_invalid_stalk_dim_raises():
    config = TSGNNConfig()
    config.model.stalk_dim = 1
    with pytest.raises(AssertionError, match="stalk_dim must be"):
        config.validate()


def test_large_stalk_dim_warns():
    config = TSGNNConfig()
    config.model.stalk_dim = 16
    with pytest.warns(UserWarning, match="unusually large"):
        config.validate()


def test_high_lr_warns():
    config = TSGNNConfig()
    config.training.lr = 0.5
    with pytest.warns(UserWarning, match="very high"):
        config.validate()


def test_from_dict_roundtrip():
    config = TSGNNConfig(seed=123)
    d = config.to_dict()
    config2 = TSGNNConfig.from_dict(d)
    assert config2.seed == 123
    assert config2.model.stalk_dim == config.model.stalk_dim


def test_from_dict_ignores_unknown_keys():
    d = {"seed": 42, "model": {"stalk_dim": 4, "unknown_key": 999}}
    config = TSGNNConfig.from_dict(d)
    assert config.model.stalk_dim == 4


def test_negative_lambda_raises():
    config = TSGNNConfig()
    config.loss.lambda_1 = -0.1
    with pytest.raises(AssertionError, match="lambda_1"):
        config.validate()
```

- [ ] **Step 2: Run tests**

```bash
cd ts-gnn
python -m pytest tests/test_config.py -v
```

Expected: all 8 tests PASS.

---

### Task 6: Test the model and metric registries

**Files:**
- Create: `ts-gnn/tests/test_registry.py`

- [ ] **Step 1: Write tests for `ModelRegistry` and `MetricRegistry`**

```python
"""Tests for the dependency-injected registries."""
import pytest
import torch.nn as nn
from tsgnn.model.registry import ModelRegistry, MetricRegistry


class DummyModel(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.fc = nn.Linear(10, hidden_dim)

    def forward(self, x):
        return self.fc(x)


def test_register_and_create():
    reg = ModelRegistry()
    reg.register("dummy", DummyModel)
    model = reg.create("dummy", hidden_dim=32)
    assert isinstance(model, DummyModel)
    assert model.fc.out_features == 32


def test_unknown_model_raises():
    reg = ModelRegistry()
    with pytest.raises(ValueError, match="Unknown model"):
        reg.create("nonexistent")


def test_list_models():
    reg = ModelRegistry()
    reg.register("b_model", DummyModel)
    reg.register("a_model", DummyModel)
    assert reg.list() == ["a_model", "b_model"]


def test_contains():
    reg = ModelRegistry()
    reg.register("dummy", DummyModel)
    assert "dummy" in reg
    assert "missing" not in reg


def test_metric_registry_run():
    mreg = MetricRegistry()
    mreg.register("add", lambda a, b: {"result": a + b})
    result = mreg.run("add", a=2, b=3)
    assert result == {"result": 5}


def test_metric_registry_run_all():
    mreg = MetricRegistry()
    mreg.register("double", lambda x: {"val": x * 2})
    mreg.register("square", lambda x: {"val": x ** 2})
    results = mreg.run_all(x=4)
    assert results["double"]["val"] == 8
    assert results["square"]["val"] == 16


def test_metric_run_all_handles_errors():
    mreg = MetricRegistry()
    mreg.register("failing", lambda: (_ for _ in ()).throw(ValueError("boom")))
    results = mreg.run_all()
    assert "error" in results["failing"]
```

- [ ] **Step 2: Run tests**

```bash
cd ts-gnn
python -m pytest tests/test_registry.py -v
```

Expected: all 7 tests PASS.

---

### Task 7: Test the vectorized sheaf

**Files:**
- Create: `ts-gnn/tests/test_vectorized.py`

- [ ] **Step 1: Write tests for `VectorizedSheafDiffusion`**

```python
"""Tests for the vectorized sheaf Laplacian implementation."""
import pytest
import torch
from tsgnn.model.sheaf_vectorized import VectorizedSheafDiffusion


@pytest.fixture
def vec_layer():
    torch.manual_seed(42)
    N, E, d = 10, 25, 3
    edge_index = torch.randint(0, N, (2, E))
    return VectorizedSheafDiffusion(N, E, d, edge_index)


def test_laplacian_shape(vec_layer):
    L = vec_layer.compute_connection_laplacian_vectorized()
    assert L.shape == (vec_layer.Nd, vec_layer.Nd)


def test_laplacian_symmetry(vec_layer):
    L = vec_layer.compute_connection_laplacian_vectorized()
    assert torch.allclose(L, L.T, atol=1e-5)


def test_laplacian_psd(vec_layer):
    L = vec_layer.compute_connection_laplacian_vectorized()
    eigs = torch.linalg.eigvalsh(L)
    assert (eigs >= -1e-5).all()


def test_spectral_gap_positive(vec_layer):
    gap = vec_layer.spectral_gap()
    assert gap > 0, f"Spectral gap should be positive, got {gap}"


def test_condition_number_finite(vec_layer):
    kappa = vec_layer.condition_number()
    assert kappa > 0 and kappa < 1e10


def test_verify_properties_dict(vec_layer):
    props = vec_layer.verify_laplacian_properties()
    assert props["is_symmetric"] is True
    assert props["is_psd"] is True
    assert props["spectral_gap"] > 0


def test_forward_shapes(vec_layer):
    N, d = vec_layer.num_nodes, vec_layer.d
    x = torch.randn(N, d)
    out, L = vec_layer.forward(x, num_steps=2)
    assert out.shape == (N, d)
    assert L.shape == (N * d, N * d)


def test_external_maps(vec_layer):
    E, d = vec_layer.num_edges, vec_layer.d
    ext = torch.randn(E, 2, d, d)
    L = vec_layer.compute_connection_laplacian_vectorized(ext)
    assert L.shape == (vec_layer.Nd, vec_layer.Nd)
```

- [ ] **Step 2: Run tests**

```bash
cd ts-gnn
python -m pytest tests/test_vectorized.py -v
```

Expected: all 8 tests PASS.

---

### Task 8: Test data module contracts

**Files:**
- Create: `ts-gnn/tests/test_data_loading.py`

- [ ] **Step 1: Write contract tests for data modules**

```python
"""Contract tests for data loading modules.

These test that the public interfaces exist and have correct signatures,
NOT that the actual data downloading works (that requires network).
"""
import pytest
import inspect

from tsgnn.data import download, preprocess, allele, grn_construction, temporal


def test_download_module_has_required_functions():
    assert hasattr(download, "download_geo_dataset")
    assert callable(download.download_geo_dataset)


def test_preprocess_module_has_required_functions():
    assert hasattr(preprocess, "preprocess_scrna")
    assert callable(preprocess.preprocess_scrna)


def test_allele_module_has_required_functions():
    assert hasattr(allele, "generate_esm2_embeddings") or hasattr(allele, "load_esm2_embeddings")


def test_grn_module_has_required_functions():
    assert hasattr(grn_construction, "build_base_grn")
    assert callable(grn_construction.build_base_grn)


def test_temporal_module_has_required_functions():
    assert hasattr(temporal, "create_temporal_graphs") or hasattr(temporal, "bin_by_pseudotime")
```

- [ ] **Step 2: Run tests**

```bash
cd ts-gnn
python -m pytest tests/test_data_loading.py -v
```

Expected: all 5 tests PASS (they test interface existence only).

---

### Task 9: Configure pytest with coverage enforcement

**Files:**
- Create: `ts-gnn/pytest.ini`
- Modify: `ts-gnn/Makefile`

- [ ] **Step 1: Create `pytest.ini`**

```ini
[pytest]
testpaths = tests
addopts = -v --tb=short --strict-markers
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks integration tests
```

- [ ] **Step 2: Add coverage and full-test targets to `Makefile`**

Add after the existing `test-cov` target:

```makefile
# Run ALL tests (tests/ + ultimate/)
test-all:
	python -m pytest tests/ ultimate/test_properties.py -v --tb=short

# Run tests with strict coverage threshold (50%)
test-coverage:
	python -m pytest tests/ -v --cov=tsgnn --cov-report=term-missing --cov-fail-under=50 --tb=short
```

- [ ] **Step 3: Run full test suite**

```bash
cd ts-gnn
make test-all
```

Expected: all tests pass with 0 errors.

---

## Track C: Completezza del Pipeline (8.0 → 9.0)

### Task 10: Integrate vectorized sheaf into the main TSGNN model

**Files:**
- Modify: `ts-gnn/src/tsgnn/model/tsgnn.py` (add import and optional use of vectorized layer)

- [ ] **Step 1: Add vectorized sheaf as an option in TSGNN constructor**

At the top of `ts-gnn/src/tsgnn/model/tsgnn.py`, add:

```python
from tsgnn.model.sheaf_vectorized import VectorizedSheafDiffusion
```

In the `TSGNN.__init__` method, add an optional parameter `use_vectorized: bool = False` and use it:

```python
if use_vectorized:
    self.sheaf_layer = VectorizedSheafDiffusion(
        num_nodes, num_edges, stalk_dim, edge_index
    )
else:
    self.sheaf_layer = SheafDiffusionLayer(
        num_nodes, num_edges, stalk_dim, edge_index
    )
```

- [ ] **Step 2: Verify both paths work**

```bash
cd ts-gnn
python -c "
from tsgnn.model.tsgnn import TSGNN
import torch
ei = torch.randint(0, 10, (2, 25))
m1 = TSGNN(10, 25, 2, 10, 1280, 32, ei, 1, use_vectorized=False)
m2 = TSGNN(10, 25, 2, 10, 1280, 32, ei, 1, use_vectorized=True)
seq = torch.randn(3, 10, 10)
emb = torch.randn(1280)
p1, _, _ = m1(seq, emb)
p2, _, _ = m2(seq, emb)
print(f'Standard: {p1[0].shape}, Vectorized: {p2[0].shape}')
print('OK: both forward passes work')
"
```

- [ ] **Step 3: Run existing tests to confirm no regression**

```bash
cd ts-gnn
python -m pytest tests/ -v --tb=short
```

Expected: all existing tests still PASS.

---

### Task 11: Make run_synthetic_proof.py runnable from project root

**Files:**
- Modify: `ts-gnn/Makefile`

- [ ] **Step 1: Add `proof` target to Makefile**

```makefile
# Run synthetic proof-of-concept from ultimate/
proof:
	python ultimate/run_synthetic_proof.py
```

- [ ] **Step 2: Verify it works**

```bash
cd ts-gnn
make proof
```

Expected: completes with "SYNTHETIC PROOF COMPLETE" message and results in `ultimate/results/`.

---

## Track D: Riproducibilità (8.5 → 9.0)

### Task 12: Pin dependency versions in pyproject.toml

**Files:**
- Modify: `ts-gnn/pyproject.toml:14-45`

- [ ] **Step 1: Pin core dependencies to match requirements.txt**

Replace the unpinned dependencies block with pinned versions matching `requirements.txt`:

```toml
dependencies = [
    "torch==2.2.0",
    "torch-geometric==2.5.0",
    "scanpy==1.10.0",
    "scvelo==0.3.2",
    "anndata==0.10.5",
    "harmonypy==0.0.9",
    "scrublet==0.2.3",
    "decoupler==1.6.0",
    "fair-esm==2.0.0",
    "GEOparse==2.0.4",
    "requests==2.31.0",
    "pandas==2.2.0",
    "numpy==1.26.4",
    "scipy==1.12.0",
    "matplotlib==3.8.3",
    "seaborn==0.13.2",
    "networkx==3.2.1",
    "plotly==5.18.0",
    "wandb==0.16.3",
    "hydra-core==1.3.2",
    "omegaconf==2.3.0",
    "pyyaml==6.0.1",
    "tqdm==4.66.2",
]
```

- [ ] **Step 2: Verify install still works**

```bash
cd ts-gnn
pip install -e ".[dev]" --dry-run
```

Expected: no resolution conflicts.

---

### Task 13: Add coverage reporting to CI

**Files:**
- Modify: `ts-gnn/ultimate/ci.yml`

- [ ] **Step 1: Add coverage step to CI**

After the existing "Run unit tests" step, add:

```yaml
    - name: Run tests with coverage
      run: |
        cd ts-gnn
        python -m pytest tests/ -v --cov=tsgnn --cov-report=term-missing --cov-fail-under=50 --tb=short
```

- [ ] **Step 2: Verify CI YAML is valid**

```bash
python -c "import yaml; yaml.safe_load(open('ultimate/ci.yml')); print('YAML valid')"
```

---

### Task 14: Create CHANGELOG.md

**Files:**
- Create: `ts-gnn/CHANGELOG.md`

- [ ] **Step 1: Write CHANGELOG**

```markdown
# Changelog

All notable changes to TS-GNN are documented in this file.

## [0.2.0] - 2026-03-26

### Added
- Fully vectorized sheaf Laplacian construction (`sheaf_vectorized.py`) — O(E·d²) with zero Python loops
- Abstract base class `AbstractGRNModel` enforcing uniform model interface at compile-time
- Typed configuration schema (`config.py`) with `@dataclass` + validation replacing raw dicts
- Dependency-injected `ModelRegistry` and `MetricRegistry` for plugin-based extensibility
- 13+ property-based and fuzz tests (`test_properties.py`) for mathematical invariant verification
- Synthetic proof-of-concept script (`run_synthetic_proof.py`) demonstrating full E2E pipeline
- Spectral diagnostic tools: `spectral_gap()`, `condition_number()`, `verify_laplacian_properties()`
- `ARCHITECTURE.md` with Mermaid diagrams (system overview, data flow, class diagram)
- `API_REFERENCE.md` with complete function signatures and tensor shapes
- GitHub Actions CI pipeline (`ci.yml`) with Python 3.10/3.11, lint, test, smoke test
- Docker Compose with 3 services: training, testing, demo

### Changed
- Pinned all dependency versions in `requirements.txt` for exact reproducibility
- Added `__all__` exports to all sub-package `__init__.py` files

## [0.1.0] - 2026-03-01

### Added
- Initial TS-GNN implementation with Sheaf Diffusion + GRU + FiLM
- 5 baseline models (CellOracle, Dictys, EvolveGCN, TemporalGAT, no-allele ablation)
- 5 evaluation metrics (zero-shot, enrichment, ChIP-seq, DepMap, distinguishability)
- Composite 4-term loss function
- Full data acquisition pipeline (8 sources)
- Dockerfile and Makefile
```

---

## Track E: Praticità / Usabilità (8.0 → 9.0)

### Task 15: Add convenience CLI entry points using typed config

**Files:**
- Modify: `ts-gnn/src/tsgnn/cli.py`

- [ ] **Step 1: Update CLI to use `TSGNNConfig.from_dict()`**

At the top of `cli.py`, add:

```python
from tsgnn.config import TSGNNConfig
```

In each CLI function, replace raw dict config loading with:

```python
config = TSGNNConfig.from_dict(yaml.safe_load(open(config_path)))
```

This ensures that any CLI invocation validates config immediately.

- [ ] **Step 2: Verify CLI still works**

```bash
cd ts-gnn
tsgnn-train --help
```

Expected: help text displayed without errors.

---

### Task 16: Add `make quickstart` target

**Files:**
- Modify: `ts-gnn/Makefile`

- [ ] **Step 1: Add quickstart target**

```makefile
# One-command quickstart: install, lint, test, demo
quickstart: install lint test proof
	@echo ""
	@echo "✅ TS-GNN quickstart complete!"
	@echo "  - All dependencies installed"
	@echo "  - Linting passed"
	@echo "  - All tests passed"
	@echo "  - Synthetic proof-of-concept completed"
	@echo ""
	@echo "Next: run 'make demo' for full pipeline or see README.md"
```

- [ ] **Step 2: Update README.md with quickstart**

Add to the Quick Start section in `README.md`:

```markdown
## Quick Start

```bash
# One-command setup, test, and demo
make quickstart
```

---

### Task 17: Unified docker-compose at project root

**Files:**
- Modify: `ts-gnn/docker-compose.yml` (create if not exists, or update)

- [ ] **Step 1: Create/update `ts-gnn/docker-compose.yml`**

Copy `ultimate/docker-compose.yml` to `ts-gnn/docker-compose.yml` and adjust context paths since it's now at the project root:

```yaml
version: "3.8"

services:
  tsgnn:
    build: .
    volumes:
      - ./data:/app/data
      - ./checkpoints:/app/checkpoints
    environment:
      - WANDB_API_KEY=${WANDB_API_KEY:-}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: >
      python scripts/run_pipeline.py
        --config configs/default.yaml
        --skip-download

  test:
    build: .
    command: python -m pytest tests/ ultimate/test_properties.py -v --tb=short

  demo:
    build: .
    volumes:
      - ./ultimate/results:/app/ultimate/results
    command: python ultimate/run_synthetic_proof.py
```

- [ ] **Step 2: Verify YAML is valid**

```bash
cd ts-gnn
python -c "import yaml; yaml.safe_load(open('docker-compose.yml')); print('Valid')"
```

---

## Verification Plan

### Automated Tests

All tests can be run with a single command:

```bash
cd ts-gnn
python -m pytest tests/ ultimate/test_properties.py -v --cov=tsgnn --cov-report=term-missing --tb=short
```

Expected results:
- **30+ tests** pass (9 existing test files + 4 new test files + property tests)
- **Coverage ≥ 50%** on `tsgnn` package
- **Zero lint errors** from `ruff check src/ tests/`

### Individual verification commands

| What | Command | Expected |
|---|---|---|
| All unit tests | `python -m pytest tests/ -v` | All PASS |
| Property tests | `python -m pytest ultimate/test_properties.py -v` | All PASS |
| Config tests | `python -m pytest tests/test_config.py -v` | 8 PASS |
| Registry tests | `python -m pytest tests/test_registry.py -v` | 7 PASS |
| Vectorized tests | `python -m pytest tests/test_vectorized.py -v` | 8 PASS |
| Data contract tests | `python -m pytest tests/test_data_loading.py -v` | 5 PASS |
| Lint | `ruff check src/ tests/` | 0 errors |
| Imports | `python -c "import tsgnn; print(tsgnn.__version__)"` | `0.1.0` |
| Synthetic proof | `python ultimate/run_synthetic_proof.py` | Completes OK |
| YAML validity | `python -c "import yaml; yaml.safe_load(open('ultimate/ci.yml'))"` | No error |

### Manual Verification
- Review `CHANGELOG.md` for completeness and accuracy
- Run `make quickstart` and verify end-to-end flow
- Verify `docker compose config` validates the docker-compose.yml
