"""Small OpenSSH helpers for Bocconi HPC scripts.

These helpers intentionally do not support password-based automation. Configure
the local OpenSSH alias `bocconi-stud` and install the public key interactively
once, then run all jobs through the normal `ssh`/`scp` clients.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable


DEFAULT_HOST_ALIAS = os.environ.get("TSGNN_HPC_HOST", "bocconi")
DEFAULT_REMOTE_ROOT = os.environ.get("TSGNN_HPC_REMOTE_ROOT", "~/ts-gnn")


def run_ssh(command: str, *, host: str = DEFAULT_HOST_ALIAS, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ssh", host, command],
        check=check,
        text=True,
        capture_output=True,
    )


def run_print(command: str, *, host: str = DEFAULT_HOST_ALIAS, check: bool = False) -> int:
    print(f"$ ssh {host} {command}")
    proc = run_ssh(command, host=host, check=False)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="")
    if check and proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return proc.returncode


def scp_to_remote(
    paths: Iterable[str | Path],
    remote_dir: str,
    *,
    host: str = DEFAULT_HOST_ALIAS,
    recursive: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    cmd = ["scp"]
    if recursive:
        cmd.append("-r")
    cmd.extend(str(Path(path)) for path in paths)
    cmd.append(f"{host}:{remote_dir}")
    print("$ " + " ".join(cmd))
    return subprocess.run(cmd, check=check, text=True)
