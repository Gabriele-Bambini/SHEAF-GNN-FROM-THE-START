from __future__ import annotations

import argparse
import glob
import os
import re
import shlex
import tempfile
import time
from pathlib import Path

import zipfile
from hpc_ssh import DEFAULT_HOST_ALIAS, DEFAULT_REMOTE_ROOT, run_print, scp_to_remote

from hpc_password import run as paramiko_run

SYNC_PATTERNS = [
    "src/sheaf_ts/**/*.py",
    "src/tsgnn/**/*.py",
    "FROM THE START/src/sheaf_ts/**/*.py",
    "FROM THE START/scripts/*.py",
    "FROM THE START/scripts/*.sbatch",
    "FROM THE START/tests/*.py",
    "FROM THE START/configs/*.yaml",
    "FROM THE START/configs/*.yml",
    "FROM THE START/README.md",
    "hpc/*.py",
    "hpc/*.ps1",
    "hpc/*.sh",
    "hpc/*.sbatch",
    "scripts/*.py",
    "tests/*.py",
    "configs/*.yaml",
    "configs/*.yml",
    "pyproject.toml",
    "requirements.txt",
]

SUBMIT_ACTIONS = {
    "smoke",
    "probe",
    "train",
    "synthetic",
    "real_smoke",
    "hpo_smoke",
    "hpo_mega",
    "fts_rebuild_cache",
    "rebuild_real_cache",
    "ovca_gse154600_context_audit",
    "ovca_gse154600_context_gate",
}
SYNC_ACTIONS = {"sync", "ensure_optuna", *SUBMIT_ACTIONS}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def shq(path: str) -> str:
    """Quote a remote POSIX path segment-wise while preserving leading `~`."""
    parts = path.split("/")
    quoted: list[str] = []
    for idx, part in enumerate(parts):
        if idx == 0 and part == "":
            quoted.append("")
        elif idx == 0 and part.startswith("~"):
            quoted.append(part)
        elif part == "":
            quoted.append(part)
        else:
            quoted.append(shlex.quote(part))
    return "/".join(quoted)


def iter_sync_files() -> list[Path]:
    base = project_root()
    files: list[Path] = []
    for pattern in SYNC_PATTERNS:
        files.extend(Path(p) for p in glob.glob(str(base / pattern), recursive=True) if Path(p).is_file())
    return sorted(set(files))


def _sync_zip_path(base: Path) -> Path:
    """Return a writable temporary zip path even when the repo checkout is read-only."""
    root = os.environ.get("TSGNN_SYNC_TMP_DIR")
    tmp_root = Path(root) if root else Path(tempfile.gettempdir())
    stamp = int(time.time())
    out_dir = tmp_root / "tsgnn_sync" / f"{os.getpid()}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Keep a stable basename so the remote unzip command can find it.
    return out_dir / "hpc_sync_temp.zip"


def sync_files(host: str, remote_root: str) -> None:
    base = project_root()
    files = iter_sync_files()
    print(f"[sync] {len(files)} files -> {host}:{remote_root}")

    zip_path = _sync_zip_path(base)
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for local in files:
                rel = local.relative_to(base).as_posix()
                archive.write(local, rel)

        # 1. Ensure remote root exists
        run_print(f"mkdir -p {shq(remote_root)}", host=host, check=True)
        # 2. Send zip
        scp_to_remote([zip_path], remote_root, host=host, check=True)
        # 3. Unzip and cleanup on remote
        run_print(f"cd {shq(remote_root)} && unzip -o hpc_sync_temp.zip && rm hpc_sync_temp.zip", host=host, check=True)
    finally:
        if zip_path.exists():
            zip_path.unlink()


def expand_remote_root_password(ssh, remote_root: str) -> str:
    if not remote_root.startswith("~"):
        return remote_root
    from hpc_password import run

    _, stdout, _ = run(ssh, 'printf "%s" "$HOME"', check=True)
    home = stdout.strip()
    if remote_root == "~":
        return home
    if remote_root.startswith("~/"):
        return f"{home}/{remote_root[2:]}"
    return remote_root


def sync_files_password(ssh, remote_root: str) -> None:
    base = project_root()
    files = iter_sync_files()
    sftp_root = expand_remote_root_password(ssh, remote_root).rstrip("/")
    print(f"[sync] {len(files)} files -> <hpc>:{sftp_root}")

    zip_path = _sync_zip_path(base)
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for local in files:
                rel = local.relative_to(base).as_posix()
                archive.write(local, rel)

        paramiko_run(ssh, f"mkdir -p {shq(sftp_root)}", check=True)
        sftp = ssh.open_sftp()
        try:
            sftp.put(str(zip_path), f"{sftp_root}/hpc_sync_temp.zip")
        finally:
            sftp.close()

        paramiko_run(ssh, f"cd {shq(sftp_root)} && unzip -o hpc_sync_temp.zip && rm hpc_sync_temp.zip", check=True)
    finally:
        if zip_path.exists():
            zip_path.unlink()


def read_logs(host: str, remote_root: str, job_id: str) -> None:
    run_print(f"cat {remote_root}/logs/*_{job_id}.out 2>/dev/null || true", host=host, check=False)
    run_print(f"cat {remote_root}/logs/*_{job_id}.err 2>/dev/null || true", host=host, check=False)


SBATCH_JOB_RE = re.compile(r"Submitted batch job\s+(\d+)")


def fix_remote_line_endings(ssh, remote_root: str) -> int:
    command = (
        f"cd {remote_root} && "
        "find hpc -type f \\( -name '*.sh' -o -name '*.sbatch' \\) "
        "-exec sed -i 's/\\r$//' {} +"
    )
    return paramiko_run(ssh, command, check=False)[0]


def submit_sbatch(ssh, remote_root: str, sbatch_path: str, *, env_prefix: str | None = None) -> int:
    prefix = f"{env_prefix} " if env_prefix else ""
    logs_dir = f"{remote_root.rstrip('/')}/logs"
    code, _stdout, _stderr = paramiko_run(ssh, f"mkdir -p {shq(logs_dir)}")
    if code != 0:
        return code
    code, stdout, _stderr = paramiko_run(ssh, f"cd {shq(remote_root)} && {prefix}sbatch {sbatch_path}")
    if code != 0:
        return code

    match = SBATCH_JOB_RE.search(stdout)
    if not match:
        print("Slurm submission returned success but no job id was parsed.")
        return 2

    job_id = match.group(1)
    print(f"Submitted Slurm job: {job_id}")
    paramiko_run(ssh, f"squeue -j {job_id}", check=False)
    return 0


def main_paramiko(args: argparse.Namespace) -> int:
    from hpc_password import HpcAuthenticationError, connect

    try:
        ssh = connect()
    except HpcAuthenticationError as exc:
        print(exc)
        return 1
    try:
        if args.action in SYNC_ACTIONS:
            sync_files_password(ssh, args.remote_root)
            fix_remote_line_endings(ssh, args.remote_root)

        if args.action == "sync":
            return 0
        if args.action == "ensure_optuna":
            return paramiko_run(ssh, f"cd {args.remote_root} && bash hpc/ensure_optuna.sh")[0]
        if args.action == "smoke":
            return submit_sbatch(ssh, args.remote_root, "hpc/submit_smoke.sbatch")
        if args.action == "probe":
            return submit_sbatch(ssh, args.remote_root, "hpc/submit_probe_alpha.sbatch", env_prefix=f"TSGNN_CONFIG={args.config}")
        if args.action == "train":
            return submit_sbatch(ssh, args.remote_root, "hpc/submit_train_v2.sbatch", env_prefix=f"TSGNN_CONFIG={args.config}")
        if args.action == "synthetic":
            return submit_sbatch(ssh, args.remote_root, "hpc/submit_synthetic_sheaf_smoke.sbatch")
        if args.action == "real_smoke":
            return submit_sbatch(ssh, args.remote_root, "hpc/submit_real_smoke.sbatch")
        if args.action == "hpo_smoke":
            return submit_sbatch(ssh, args.remote_root, "hpc/submit_real_hpo_smoke.sbatch")
        if args.action == "hpo_mega":
            return submit_sbatch(ssh, args.remote_root, "hpc/submit_real_hpo_mega.sbatch")
        if args.action in {"fts_rebuild_cache", "rebuild_real_cache"}:
            return submit_sbatch(ssh, args.remote_root, "hpc/submit_fts_rebuild_cache.sbatch")
        if args.action == "ovca_gse154600_context_audit":
            return submit_sbatch(ssh, args.remote_root, "hpc/submit_ovca_gse154600_context_audit.sbatch")
        if args.action == "ovca_gse154600_context_gate":
            return submit_sbatch(ssh, args.remote_root, "hpc/submit_ovca_gse154600_context_gate.sbatch")
        if args.action == "status":
            return paramiko_run(ssh, "squeue --me 2>/dev/null || squeue -u $(whoami)")[0]
        if args.action == "log":
            if not args.extra:
                raise SystemExit("log requires a job id")
            paramiko_run(ssh, f"cat {args.remote_root}/logs/*_{args.extra}.out 2>/dev/null || true")
            paramiko_run(ssh, f"cat {args.remote_root}/logs/*_{args.extra}.err 2>/dev/null || true")
            return 0
        if args.action == "nuke":
            if not args.confirm:
                raise SystemExit("Refusing to cancel jobs without --confirm")
            return paramiko_run(ssh, "scancel -u $(whoami)")[0]
        if args.action == "ls":
            target = args.extra or args.remote_root
            return paramiko_run(ssh, f"ls -F {target}")[0]
        if args.action == "read":
            if not args.extra:
                raise SystemExit("read requires a remote path")
            return paramiko_run(ssh, f"cat {args.extra}")[0]
        return 2
    finally:
        ssh.close()


def main_password(args: argparse.Namespace) -> int:
    return main_paramiko(args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HPC sync/submit helper with SSH-key or Paramiko auth.")
    parser.add_argument(
        "action",
        choices=[
            "sync",
            "ensure_optuna",
            "smoke",
            "probe",
            "train",
            "synthetic",
            "real_smoke",
            "hpo_smoke",
            "hpo_mega",
            "fts_rebuild_cache",
            "rebuild_real_cache",
            "ovca_gse154600_context_audit",
            "ovca_gse154600_context_gate",
            "status",
            "log",
            "nuke",
            "ls",
            "read",
        ],
    )
    parser.add_argument("extra", nargs="?", help="job id or remote path")
    parser.add_argument("--host", default=DEFAULT_HOST_ALIAS)
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--config", default="configs/nature_v2.yaml")
    parser.add_argument("--auth", choices=["ssh", "password", "paramiko"], default=os.environ.get("TSGNN_HPC_AUTH", "ssh"))
    parser.add_argument("--confirm", action="store_true", help="Required for nuke.")
    args = parser.parse_args(argv)

    if args.auth in {"password", "paramiko"}:
        return main_paramiko(args)

    if args.action in SYNC_ACTIONS:
        sync_files(args.host, args.remote_root)

    if args.action == "sync":
        return 0
    if args.action == "ensure_optuna":
        return run_print(f"cd {args.remote_root} && bash hpc/ensure_optuna.sh", host=args.host, check=False)
    if args.action == "smoke":
        return run_print(f"cd {args.remote_root} && sbatch hpc/submit_smoke.sbatch", host=args.host, check=False)
    if args.action == "probe":
        return run_print(f"cd {args.remote_root} && TSGNN_CONFIG={args.config} sbatch hpc/submit_probe_alpha.sbatch", host=args.host, check=False)
    if args.action == "train":
        return run_print(f"cd {args.remote_root} && TSGNN_CONFIG={args.config} sbatch hpc/submit_train_v2.sbatch", host=args.host, check=False)
    if args.action == "synthetic":
        return run_print(f"cd {args.remote_root} && sbatch hpc/submit_synthetic_sheaf_smoke.sbatch", host=args.host, check=False)
    if args.action == "real_smoke":
        return run_print(f"cd {args.remote_root} && sbatch hpc/submit_real_smoke.sbatch", host=args.host, check=False)
    if args.action == "hpo_smoke":
        return run_print(f"cd {args.remote_root} && sbatch hpc/submit_real_hpo_smoke.sbatch", host=args.host, check=False)
    if args.action == "hpo_mega":
        return run_print(f"cd {args.remote_root} && sbatch hpc/submit_real_hpo_mega.sbatch", host=args.host, check=False)
    if args.action in {"fts_rebuild_cache", "rebuild_real_cache"}:
        return run_print(
            f"mkdir -p {shq(args.remote_root.rstrip('/') + '/logs')} && "
            f"cd {shq(args.remote_root)} && sbatch hpc/submit_fts_rebuild_cache.sbatch",
            host=args.host,
            check=False,
        )
    if args.action == "ovca_gse154600_context_audit":
        return run_print(
            f"mkdir -p {shq(args.remote_root.rstrip('/') + '/logs')} && "
            f"cd {shq(args.remote_root)} && sbatch hpc/submit_ovca_gse154600_context_audit.sbatch",
            host=args.host,
            check=False,
        )
    if args.action == "ovca_gse154600_context_gate":
        return run_print(
            f"mkdir -p {shq(args.remote_root.rstrip('/') + '/logs')} && "
            f"cd {shq(args.remote_root)} && sbatch hpc/submit_ovca_gse154600_context_gate.sbatch",
            host=args.host,
            check=False,
        )
    if args.action == "status":
        return run_print("squeue --me 2>/dev/null || squeue -u $(whoami)", host=args.host, check=False)
    if args.action == "log":
        if not args.extra:
            raise SystemExit("log requires a job id")
        read_logs(args.host, args.remote_root, args.extra)
        return 0
    if args.action == "nuke":
        if not args.confirm:
            raise SystemExit("Refusing to cancel jobs without --confirm")
        return run_print("scancel -u $(whoami)", host=args.host, check=False)
    if args.action == "ls":
        target = args.extra or args.remote_root
        return run_print(f"ls -F {target}", host=args.host, check=False)
    if args.action == "read":
        if not args.extra:
            raise SystemExit("read requires a remote path")
        return run_print(f"cat {args.extra}", host=args.host, check=False)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
