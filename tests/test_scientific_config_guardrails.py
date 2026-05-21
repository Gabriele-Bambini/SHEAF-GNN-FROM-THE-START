from pathlib import Path

import yaml


def test_nature_v2_requires_baselines_and_original_p1_threshold():
    cfg = yaml.safe_load(Path("configs/nature_v2.yaml").read_text(encoding="utf-8"))

    assert cfg["benchmarks"]["models"]
    assert {"tsgnn", "gcn_static", "mlp"}.issubset(set(cfg["benchmarks"]["models"]))
    assert cfg["predictions"]["P1_zeroshot_r_threshold"] >= 0.35
    assert cfg["science"]["strict_real_data"] is True


def test_run_nature_baselines_use_same_epoch_budget_and_fail_closed():
    text = Path("scripts/run_nature.py").read_text(encoding="utf-8")

    assert 'cfg.get("benchmarks", {}).get("models", [])' not in text
    assert 'max(5, cfg["training"]["epochs"] // 5)' not in text
    assert 'raise ValueError("scientific zeroshot requires at least one baseline control")' in text


def test_run_nature_writes_runtime_manifest():
    text = Path("scripts/run_nature.py").read_text(encoding="utf-8")

    assert "build_runtime_manifest" in text
    assert "runtime_manifest.json" in text


def test_run_nature_does_not_patch_installed_packages():
    text = Path("scripts/run_nature.py").read_text(encoding="utf-8")

    assert "_patch_matplotlib_fontconfig" not in text
    assert 'open(fp, "w"' not in text
