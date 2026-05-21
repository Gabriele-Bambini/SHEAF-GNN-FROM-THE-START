import json
from pathlib import Path
import subprocess
import sys

import pytest


def test_runtime_manifest_contains_audit_fields(tmp_path):
    from sheaf_ts.hpc.manifest import build_runtime_manifest, validate_runtime_manifest

    output_file = tmp_path / "result.json"
    output_file.write_text('{"status": "ok"}', encoding="utf-8")

    manifest = build_runtime_manifest(
        run_id="pytest-runtime",
        command=["python", "scripts/run_synthetic_sheaf_backtest.py"],
        output_paths=[output_file],
        cwd=Path.cwd(),
        extra={"kind": "synthetic_smoke"},
    )

    validate_runtime_manifest(manifest)
    encoded = json.dumps(manifest, sort_keys=True)

    assert "pytest-runtime" in encoded
    assert manifest["command"] == ["python", "scripts/run_synthetic_sheaf_backtest.py"]
    assert manifest["git"]["status_short"] is not None
    assert manifest["environment"]["python"]
    assert manifest["outputs"][0]["sha256"]
    assert manifest["extra"]["kind"] == "synthetic_smoke"


def test_runtime_manifest_fails_closed_when_required_fields_missing():
    from sheaf_ts.hpc.manifest import validate_runtime_manifest

    with pytest.raises(ValueError, match="run_id"):
        validate_runtime_manifest({"command": ["python"]})


def test_synthetic_slurm_script_targets_strict_synthetic_runner_only():
    script = Path("hpc/submit_synthetic_sheaf_smoke.sbatch")

    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert "run_synthetic_sheaf_backtest.py" in text
    assert "run_nature.py" not in text
    assert "--no-strict" not in text
    assert "synthetic_sheaf_manifest" in text


def test_real_smoke_slurm_script_targets_gated_runner():
    script = Path("hpc/submit_real_smoke.sbatch")

    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert "run_real_data_smoke.py" in text
    assert "--runtime-manifest" in text
    assert "--strict" not in text
    assert "real_smoke_manifest" in text


def test_real_hpo_slurm_script_targets_optuna_cv_runner():
    script = Path("hpc/submit_real_hpo_smoke.sbatch")

    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert "run_real_hpo_smoke.py" in text
    assert "pip install" not in text
    assert "pip\", \"install\"" not in text
    assert "--study-db" in text
    assert "--artifact-dir \"$OUT_DIR\"" in text
    assert "--n-trials 3" in text


def test_real_hpo_mega_slurm_script_targets_resume_safe_optuna_runner():
    script = Path("hpc/submit_real_hpo_mega.sbatch")

    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert "run_real_hpo_smoke.py" in text
    assert "--study-db \"$STUDY_DB\"" in text
    assert "--artifact-dir \"$OUT_DIR\"" in text
    assert "N_TRIALS=\"${N_TRIALS:-128}\"" in text
    assert "HPO_TIMEOUT=\"${HPO_TIMEOUT:-14400}\"" in text


def test_hpc_ensure_optuna_script_is_available():
    script = Path("hpc/ensure_optuna.sh")

    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert "conda activate tsgnn" in text
    assert "pip install" not in text
    assert "pip\", \"install\"" not in text


def test_slurm_scripts_do_not_mutate_python_environment():
    offenders = []
    for script in Path("hpc").glob("*.sbatch"):
        text = script.read_text(encoding="utf-8", errors="ignore")
        if "pip install" in text or '"pip", "install"' in text:
            offenders.append(script.as_posix())

    assert offenders == []


def test_synthetic_cli_can_write_runtime_manifest(tmp_path):
    out_path = tmp_path / "synthetic.json"
    runtime_path = tmp_path / "runtime.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_synthetic_sheaf_backtest.py",
            "--out",
            str(out_path),
            "--runtime-manifest",
            str(runtime_path),
            "--run-id",
            "pytest-cli-runtime",
            "--num-nodes",
            "6",
            "--stalk-dim",
            "2",
            "--num-steps",
            "5",
            "--tau",
            "0.05",
            "--seed",
            "31",
            "--train-steps",
            "3",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    assert runtime["run_id"] == "pytest-cli-runtime"
    assert runtime["outputs"][0]["sha256"]
