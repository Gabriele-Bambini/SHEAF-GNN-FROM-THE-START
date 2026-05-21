"""Runtime manifests for local and Slurm runs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Sequence

__all__ = [
    "build_runtime_manifest",
    "sha256_file",
    "validate_runtime_manifest",
    "write_runtime_manifest",
]


def _run_text(command: Sequence[str], *, cwd: Path) -> str:
    try:
        completed = subprocess.run(
            list(command),
            cwd=str(cwd),
            text=True,
            capture_output=True,
            check=False,
            timeout=20,
        )
    except Exception as exc:
        return f"unavailable: {exc.__class__.__name__}"
    output = completed.stdout.strip()
    error = completed.stderr.strip()
    if completed.returncode != 0:
        return f"exit={completed.returncode}; stdout={output}; stderr={error}"
    return output


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _output_record(path: Path) -> dict[str, Any]:
    exists = path.exists()
    record: dict[str, Any] = {
        "path": str(path),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else None,
        "sha256": sha256_file(path) if exists and path.is_file() else None,
    }
    return record


def _iter_snapshot_files(root: Path) -> list[Path]:
    include_roots = ("src/sheaf_ts", "src/tsgnn", "scripts", "hpc", "tests", "configs")
    include_files = ("requirements.txt", "requirements-dev.txt", "environment.yml", "pyproject.toml")
    include_suffixes = {".py", ".sbatch", ".sh", ".ps1", ".yaml", ".yml", ".toml", ".txt"}
    files: list[Path] = []
    for rel_root in include_roots:
        base = root / rel_root
        if not base.exists():
            continue
        try:
            for path in base.rglob("*"):
                if path.is_file() and path.suffix.lower() in include_suffixes:
                    files.append(path)
        except Exception as e:
            # Fallback for Windows shortcut path issues
            continue
    for rel_file in include_files:
        path = root / rel_file
        if path.exists() and path.is_file():
            files.append(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def _source_snapshot(root: Path, *, git_sha: str, git_status: str) -> dict[str, Any]:
    files = _iter_snapshot_files(root)
    digest = hashlib.sha256()
    records = []
    for path in files:
        rel = path.relative_to(root).as_posix()
        file_hash = sha256_file(path)
        digest.update(rel.encode("utf-8"))
        digest.update(file_hash.encode("ascii"))
        records.append(
            {
                "path": rel,
                "sha256": file_hash,
                "size_bytes": path.stat().st_size,
            }
        )
    git_unavailable = "not a git repository" in git_sha or "not a git repository" in git_status
    return {
        "available": bool(records),
        "git_unavailable": git_unavailable,
        "file_count": len(records),
        "sha256": digest.hexdigest() if records else None,
        "files": records[:200],
    }


def build_runtime_manifest(
    *,
    run_id: str,
    command: Sequence[str],
    output_paths: Sequence[str | Path],
    cwd: str | Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-safe manifest for one local/HPC run."""
    root = Path(cwd)
    outputs = [_output_record(Path(path)) for path in output_paths]
    git_sha = _run_text(["git", "rev-parse", "HEAD"], cwd=root)
    git_status = _run_text(["git", "status", "--short"], cwd=root)
    git_diff = _run_text(
        ["git", "diff", "--", "src/sheaf_ts", "src/tsgnn", "scripts", "hpc", "tests", "configs"],
        cwd=root,
    )
    manifest = {
        "run_id": run_id,
        "command": list(command),
        "cwd": str(root),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "executable": sys.executable,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
            "slurm_node": os.environ.get("SLURMD_NODENAME"),
        },
        "git": {
            "sha": git_sha,
            "status_short": git_status,
            "diff_hash": hashlib.sha256(git_diff.encode("utf-8")).hexdigest(),
        },
        "code_snapshot": _source_snapshot(root, git_sha=git_sha, git_status=git_status),
        "packages": {
            "torch": _run_text(
                [sys.executable, "-c", "import torch; print(torch.__version__)"],
                cwd=root,
            ),
            "optuna": _run_text(
                [sys.executable, "-c", "import optuna; print(optuna.__version__)"],
                cwd=root,
            ),
        },
        "outputs": outputs,
        "extra": extra or {},
    }
    validate_runtime_manifest(manifest)
    return manifest


def validate_runtime_manifest(manifest: dict[str, Any]) -> None:
    required = ("run_id", "command", "environment", "git", "outputs")
    for key in required:
        if key not in manifest:
            raise ValueError(f"runtime manifest missing required field: {key}")
    if not manifest["run_id"]:
        raise ValueError("run_id is required")
    if not manifest["command"]:
        raise ValueError("command is required")
    if "python" not in manifest["environment"]:
        raise ValueError("environment.python is required")
    if "status_short" not in manifest["git"]:
        raise ValueError("git.status_short is required")
    for output in manifest["outputs"]:
        if "path" not in output or "sha256" not in output:
            raise ValueError("each output record requires path and sha256")


def write_runtime_manifest(
    manifest: dict[str, Any],
    path: str | Path,
) -> None:
    validate_runtime_manifest(manifest)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
