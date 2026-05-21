from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
HPC_DIR = ROOT / "hpc"
if str(HPC_DIR) not in sys.path:
    sys.path.insert(0, str(HPC_DIR))

import sync_and_submit  # noqa: E402


def test_main_accepts_paramiko_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sync_and_submit, "main_paramiko", lambda args: 23, raising=False)

    assert sync_and_submit.main(["status", "--auth", "paramiko"]) == 23


def test_main_accepts_real_smoke_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    actions: list[str] = []

    def fake_main_paramiko(args):  # noqa: ANN001
        actions.append(args.action)
        return 0

    monkeypatch.setattr(sync_and_submit, "main_paramiko", fake_main_paramiko, raising=False)

    assert sync_and_submit.main(["real_smoke", "--auth", "paramiko"]) == 0
    assert sync_and_submit.main(["hpo_smoke", "--auth", "paramiko"]) == 0
    assert actions == ["real_smoke", "hpo_smoke"]


def test_main_accepts_ensure_optuna_action(monkeypatch: pytest.MonkeyPatch) -> None:
    actions: list[str] = []

    def fake_main_paramiko(args):  # noqa: ANN001
        actions.append(args.action)
        return 0

    monkeypatch.setattr(sync_and_submit, "main_paramiko", fake_main_paramiko, raising=False)

    assert sync_and_submit.main(["ensure_optuna", "--auth", "paramiko"]) == 0
    assert actions == ["ensure_optuna"]


def test_submit_sbatch_parses_job_id_and_queries_queue(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    commands: list[str] = []

    def fake_run(ssh, command: str, *, check: bool = False):  # noqa: ANN001
        commands.append(command)
        if "sbatch" in command:
            return 0, "Submitted batch job 12345\n", ""
        return 0, "queue\n", ""

    monkeypatch.setattr(sync_and_submit, "paramiko_run", fake_run, raising=False)

    result = sync_and_submit.submit_sbatch(object(), "~/ts-gnn", "hpc/submit_synthetic_sheaf_smoke.sbatch")

    assert result == 0
    assert commands == [
        "mkdir -p ~/ts-gnn/logs",
        "cd ~/ts-gnn && sbatch hpc/submit_synthetic_sheaf_smoke.sbatch",
        "squeue -j 12345",
    ]
    assert "Submitted Slurm job: 12345" in capsys.readouterr().out


def test_submit_sbatch_stops_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[str] = []

    def fake_run(ssh, command: str, *, check: bool = False):  # noqa: ANN001
        commands.append(command)
        return 1, "", "sbatch failed"

    monkeypatch.setattr(sync_and_submit, "paramiko_run", fake_run, raising=False)

    assert sync_and_submit.submit_sbatch(object(), "~/ts-gnn", "hpc/submit_synthetic_sheaf_smoke.sbatch") == 1
    assert commands == ["mkdir -p ~/ts-gnn/logs"]


def test_submit_sbatch_fails_closed_when_job_id_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(ssh, command: str, *, check: bool = False):  # noqa: ANN001
        return 0, "submitted but no id\n", ""

    monkeypatch.setattr(sync_and_submit, "paramiko_run", fake_run, raising=False)

    assert sync_and_submit.submit_sbatch(object(), "~/ts-gnn", "hpc/submit_synthetic_sheaf_smoke.sbatch") != 0
