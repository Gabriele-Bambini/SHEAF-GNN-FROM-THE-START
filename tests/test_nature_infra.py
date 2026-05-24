"""Tests for the Nature-publication infrastructure added in Phase A."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

RUN_NATURE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "run_nature.py"
_SPEC = importlib.util.spec_from_file_location("run_nature_test_module", RUN_NATURE_PATH)
run_nature = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(run_nature)


# ─── stats ───────────────────────────────────────────────────────────────────

def test_bootstrap_ci_basic():
    from tsgnn.evaluation.stats import bootstrap_ci
    out = bootstrap_ci([0.1, 0.2, 0.3, 0.4, 0.5], n_boot=500, seed=1)
    assert out["lo"] < out["mean"] < out["hi"]
    assert out["n"] == 5


def test_bh_correction_monotone():
    from tsgnn.evaluation.stats import benjamini_hochberg
    p = np.array([0.001, 0.01, 0.04, 0.2, 0.8])
    q = benjamini_hochberg(p)
    assert (q >= p).all()
    assert (np.diff(np.sort(q)) >= -1e-12).all()


def test_wilcoxon_paired_identical():
    from tsgnn.evaluation.stats import wilcoxon_paired
    out = wilcoxon_paired([1, 2, 3], [1, 2, 3])
    assert out["p"] == 1.0


# ─── allele swap ─────────────────────────────────────────────────────────────

class _DummyFiLMModel(torch.nn.Module):
    def forward(self, x, emb):
        return x + emb.mean(dim=-1, keepdim=True)


def test_swap_jaccard_changes_with_embedding():
    from tsgnn.evaluation.allele_swap import swap_jaccard
    torch.manual_seed(0)
    m = _DummyFiLMModel()
    X = torch.randn(30, 8)
    a = torch.randn(1, 1280)
    b = torch.randn(1, 1280) * 5
    j = swap_jaccard(m, X, a, b, k=5)
    assert 0.0 <= j <= 1.0


# ─── bulk enrichment ─────────────────────────────────────────────────────────

def test_enrichment_per_allele_shape():
    pytest.importorskip("anndata")
    pytest.importorskip("scipy")
    import anndata as ad
    import pandas as pd

    rng = np.random.default_rng(0)
    n_samples, n_genes = 60, 120
    X = rng.normal(size=(n_samples, n_genes)).astype(np.float32)
    alleles = ["WT"] * 40 + ["R175H"] * 10 + ["R273H"] * 10
    # Inject signal in R175H for first 30 genes
    X[40:50, :30] += 2.0
    obs = pd.DataFrame({"tp53_allele_canonical": alleles})
    var = pd.DataFrame(index=[f"G{i}" for i in range(n_genes)])
    a = ad.AnnData(X=X, obs=obs, var=var)

    from tsgnn.evaluation.bulk_enrichment import enrichment_per_allele
    pred_sets = {"R175H": [f"G{i}" for i in range(30)],
                 "R273H": [f"G{i}" for i in range(30)]}
    df = enrichment_per_allele(a, pred_sets)
    assert set(df["allele"]) == {"R175H", "R273H"}
    r175 = df.set_index("allele").loc["R175H"]
    r273 = df.set_index("allele").loc["R273H"]
    # Only R175H has injected signal → smaller p
    assert r175["raw_p"] <= r273["raw_p"]


# ─── registry ────────────────────────────────────────────────────────────────

def test_registry_lists_four_datasets():
    from tsgnn.data.registry import list_datasets
    assert {"brca_wu2021", "biokey", "crc_pelka2021", "metabric"}.issubset(set(list_datasets()))


def test_registry_unknown_raises():
    from tsgnn.data.registry import load_dataset
    with pytest.raises(KeyError):
        load_dataset("not_a_real_dataset")


# ─── baselines (forward pass + interface) ────────────────────────────────────

@pytest.mark.parametrize("name", ["mlp", "gcn_static", "dysat"])
def test_new_baselines_forward(name):
    from tsgnn.model.baselines import create_baseline
    N, E, K, d = 20, 40, 5, 8
    edge_index = torch.randint(0, N, (2, E))
    X = torch.randn(K, N, d)
    esm = torch.randn(1280)
    model = create_baseline(
        name,
        num_nodes=N, input_dim=d, hidden_dim=16, edge_index=edge_index,
    )
    out = model(X, esm)
    # All new baselines return (preds, maps, laps) for K>1 inputs
    assert isinstance(out, tuple)
    preds, _, _ = out
    assert len(preds) == K
    assert preds[0].shape == (N, d)


def test_holdout_eval_returns_finite_metric_without_training():
    class IdentityForecastModel(torch.nn.Module):
        def forward(self, x, emb):
            return x.clone()

    model = IdentityForecastModel()
    bundle = {"X_seq": torch.arange(24, dtype=torch.float32).reshape(4, 3, 2)}
    esm = torch.zeros(1, 8)

    metrics = run_nature._evaluate_one(model, bundle, esm, torch.device("cpu"))

    assert metrics["eval_loss"] >= 0.0
    assert metrics["val_loss"] == metrics["eval_loss"]
    assert metrics["train_loss"] is None


def test_prepare_result_dirs_isolates_runs_and_resets_latest(tmp_path):
    first = run_nature._prepare_result_dirs(
        tmp_path,
        run_id="run-a",
        config_name="nature",
        phases=["train", "zeroshot"],
        seed=42,
        smoke=False,
    )
    stale = first["latest_dir"] / "stale.txt"
    stale.write_text("old")
    assert stale.exists()

    second = run_nature._prepare_result_dirs(
        tmp_path,
        run_id="run-b",
        config_name="nature",
        phases=["train"],
        seed=7,
        smoke=True,
    )

    assert first["run_dir"].name == "run-a"
    assert second["run_dir"].name == "run-b"
    assert not (second["latest_dir"] / "stale.txt").exists()
    manifest = second["manifest_path"].read_text()
    assert '"run_id": "run-b"' in manifest
