from __future__ import annotations

import json
from pathlib import Path

import torch


def _write_fake_real_cache(tmp_path: Path) -> tuple[Path, Path]:
    processed = tmp_path / "processed"
    processed.mkdir()
    k, n = 6, 5
    base = torch.linspace(0.0, 1.0, steps=k).view(k, 1, 1)
    gene = torch.linspace(0.2, 1.0, steps=n).view(1, n, 1)
    x_seq = base * gene
    temporal = {
        "train_data": {"WT": {"node_features_seq": x_seq}},
        "val_data": {},
        "alleles": ["WT"],
        "real_esm": {"WT": torch.zeros(1280)},
        "K": k,
        "N": n,
        "E": 8,
    }
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 0, 1, 2],
            [1, 2, 3, 4, 0, 2, 3, 4],
        ],
        dtype=torch.long,
    )
    grn = {
        "gene_list": [f"G{i}" for i in range(n)],
        "edge_index": edge_index,
        "edge_weight": torch.ones(edge_index.shape[1]),
        "N": n,
        "E": edge_index.shape[1],
    }
    temporal_path = processed / "temporal_sequences.pt"
    grn_path = processed / "grn_artifacts.pt"
    torch.save(temporal, temporal_path)
    torch.save(grn, grn_path)
    return temporal_path, grn_path


def test_runtime_manifest_has_source_snapshot_when_git_unavailable(monkeypatch, tmp_path):
    from sheaf_ts.hpc import manifest as manifest_mod

    source = tmp_path / "src" / "sheaf_ts" / "example.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    out = tmp_path / "out.json"
    out.write_text("{}", encoding="utf-8")

    def fake_run_text(command, *, cwd):
        if command[:2] == ["git", "rev-parse"]:
            return "exit=128; stderr=fatal: not a git repository"
        if command[:2] == ["git", "status"]:
            return "exit=128; stderr=fatal: not a git repository"
        return ""

    monkeypatch.setattr(manifest_mod, "_run_text", fake_run_text)
    result = manifest_mod.build_runtime_manifest(
        run_id="no-git",
        command=["python", "x.py"],
        output_paths=[out],
        cwd=tmp_path,
        extra={"kind": "test"},
    )

    assert result["code_snapshot"]["available"] is True
    assert result["code_snapshot"]["sha256"]
    assert result["code_snapshot"]["git_unavailable"] is True


def test_real_data_smoke_writes_gated_manifest(tmp_path):
    from sheaf_ts.experiments.real_data_smoke import run_real_data_smoke

    temporal_path, grn_path = _write_fake_real_cache(tmp_path)
    out = tmp_path / "real_smoke.json"

    manifest = run_real_data_smoke(
        temporal_path=temporal_path,
        grn_path=grn_path,
        out_path=out,
        run_id="pytest-real-smoke",
        train_steps=3,
        strict=False,
    )

    assert out.exists()
    encoded = json.loads(out.read_text(encoding="utf-8"))
    assert encoded["run_id"] == "pytest-real-smoke"
    assert "sheaf" in encoded["result"]["model_metrics"]
    assert "mlp" in encoded["result"]["baseline_metrics"]
    assert "gcn" in encoded["result"]["baseline_metrics"]
    assert encoded["gate_verdict"]["status"] in {"ACCEPT", "REJECT"}
    assert manifest == encoded


def test_real_hpo_smoke_writes_fold_results(tmp_path):
    from sheaf_ts.experiments.real_hpo_smoke import run_real_hpo_smoke

    temporal_path, grn_path = _write_fake_real_cache(tmp_path)
    out = tmp_path / "hpo.json"

    manifest = run_real_hpo_smoke(
        temporal_path=temporal_path,
        grn_path=grn_path,
        out_path=out,
        run_id="pytest-real-hpo",
        n_trials=2,
        n_folds=2,
        seed=7,
    )

    assert out.exists()
    assert manifest["run_id"] == "pytest-real-hpo"
    assert len(manifest["trial_results"]) >= 1
    assert len(manifest["trial_results"][0]["fold_scores"]) == 2
    assert "best_value" in manifest
