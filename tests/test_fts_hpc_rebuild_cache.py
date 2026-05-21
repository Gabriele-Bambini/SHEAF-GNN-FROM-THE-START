from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
HPC_DIR = ROOT / "hpc"
if str(HPC_DIR) not in sys.path:
    sys.path.insert(0, str(HPC_DIR))

import sync_and_submit  # noqa: E402


def test_sync_includes_from_the_start_rebuild_files() -> None:
    rel_files = {path.relative_to(ROOT).as_posix() for path in sync_and_submit.iter_sync_files()}

    assert "FROM THE START/scripts/rebuild_real_cache.py" in rel_files
    assert "FROM THE START/scripts/submit_ablation.sbatch" in rel_files
    assert "FROM THE START/src/sheaf_ts/data/rebuild_cache.py" in rel_files
    assert "FROM THE START/src/sheaf_ts/data/cache_manifest.py" in rel_files


def test_main_accepts_fts_rebuild_cache_action(monkeypatch: pytest.MonkeyPatch) -> None:
    actions: list[str] = []

    def fake_main_paramiko(args):  # noqa: ANN001
        actions.append(args.action)
        return 0

    monkeypatch.setattr(sync_and_submit, "main_paramiko", fake_main_paramiko, raising=False)

    assert sync_and_submit.main(["fts_rebuild_cache", "--auth", "paramiko"]) == 0
    assert sync_and_submit.main(["rebuild_real_cache", "--auth", "paramiko"]) == 0
    assert actions == ["fts_rebuild_cache", "rebuild_real_cache"]


def test_sync_password_quotes_remote_paths_with_spaces(monkeypatch: pytest.MonkeyPatch) -> None:
    local_file = ROOT / "FROM THE START" / "scripts" / "rebuild_real_cache.py"
    commands: list[str] = []
    puts: list[tuple[str, str]] = []

    class FakeSftp:
        def put(self, local: str, remote: str) -> None:
            puts.append((local, remote))

        def close(self) -> None:
            pass

    class FakeSsh:
        def open_sftp(self) -> FakeSftp:
            return FakeSftp()

    def fake_run(ssh, command: str, *, check: bool = False):  # noqa: ANN001
        commands.append(command)
        if 'printf "%s" "$HOME"' in command:
            return 0, "/home/tester", ""
        return 0, "", ""

    monkeypatch.setattr(sync_and_submit, "iter_sync_files", lambda: [local_file], raising=False)
    monkeypatch.setattr(sync_and_submit, "paramiko_run", fake_run, raising=False)
    monkeypatch.setattr(
        sync_and_submit,
        "expand_remote_root_password",
        lambda ssh, remote_root: "/home/tester/ts-gnn",
        raising=False,
    )

    sync_and_submit.sync_files_password(FakeSsh(), "~/ts-gnn")

    assert "mkdir -p /home/tester/ts-gnn/'FROM THE START'/scripts" in commands
    assert puts[0][1] == "/home/tester/ts-gnn/FROM THE START/scripts/rebuild_real_cache.py"


def test_remote_shell_quote_preserves_tilde_and_quotes_space_segment() -> None:
    assert sync_and_submit.shq("~/ts-gnn/FROM THE START/scripts") == "~/ts-gnn/'FROM THE START'/scripts"


def test_fts_rebuild_cache_slurm_script_is_fail_closed() -> None:
    script = ROOT / "hpc" / "submit_fts_rebuild_cache.sbatch"

    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert "FROM THE START" in text
    assert "scripts/rebuild_real_cache.py" in text
    assert "scripts/audit_real_cache.py" in text
    assert "TSGNN_DATA_DIR" in text
    assert "--synthetic-smoke" not in text
    assert "pip install" not in text
    assert "set -euo pipefail" in text
