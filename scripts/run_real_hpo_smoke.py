"""Run a tiny Optuna/CV sweep only after real-data smoke passes."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sheaf_ts.experiments.real_hpo_smoke import run_real_hpo_smoke
from sheaf_ts.hpc.manifest import build_runtime_manifest, write_runtime_manifest


MANDATORY_HPO_ARTIFACTS = {
    "study_db": "optuna_study.db",
    "trials_csv": "trials.csv",
    "best_params_json": "best_params.json",
    "best_config_yaml": "best_config.yaml",
    "summary_markdown": "summary.md",
    "env_info": "env_info.json",
    "git_status": "git_status.txt",
    "git_diff": "git_diff.patch",
    "command_line": "command_line.txt",
    "seed_list": "seed_list.json",
    "split_manifest": "split_manifest.json",
    "top_k_candidate_configs": "top_k_configs.json",
}


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_git_text(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return proc.stdout


def _write_best_config_yaml(path: Path, manifest: dict[str, object], seed: int) -> None:
    best_params = manifest.get("best_params", {})
    lines = [
        "run:",
        f"  run_id: {manifest.get('run_id')}",
        f"  seed: {seed}",
        f"  objective: {manifest.get('objective')}",
        "dataset:",
    ]
    dataset = manifest.get("dataset", {})
    if isinstance(dataset, dict):
        for key, value in sorted(dataset.items()):
            lines.append(f"  {key}: {value}")
    lines.append("model:")
    if isinstance(best_params, dict):
        for key, value in sorted(best_params.items()):
            lines.append(f"  {key}: {value}")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_trials_csv(path: Path, manifest: dict[str, object]) -> None:
    trials = manifest.get("trial_results", [])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["trial", "mean_score", "params_json", "fold_scores_json"])
        writer.writeheader()
        if isinstance(trials, list):
            for index, trial in enumerate(trials):
                if not isinstance(trial, dict):
                    continue
                writer.writerow(
                    {
                        "trial": index,
                        "mean_score": trial.get("mean_score"),
                        "params_json": json.dumps(trial.get("params", {}), sort_keys=True),
                        "fold_scores_json": json.dumps(trial.get("fold_scores", [])),
                    }
                )


def _write_summary(path: Path, manifest: dict[str, object], artifacts: dict[str, str]) -> None:
    lines = [
        "# Real HPO Smoke Summary",
        "",
        f"- run_id: {manifest.get('run_id')}",
        f"- objective: {manifest.get('objective')}",
        f"- best_value: {manifest.get('best_value')}",
        f"- best_params: `{json.dumps(manifest.get('best_params', {}), sort_keys=True)}`",
        f"- trials: {len(manifest.get('trial_results', [])) if isinstance(manifest.get('trial_results'), list) else 0}",
        "",
        "## Mandatory Artifacts",
        "",
    ]
    for key in MANDATORY_HPO_ARTIFACTS:
        lines.append(f"- {key}: `{artifacts[key]}`")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_hpo_artifacts(
    *,
    manifest: dict[str, object],
    artifact_dir: Path,
    command: list[str],
    seed: int,
    study_db: str | None,
    top_k: int = 10,
) -> dict[str, str]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    paths = {key: artifact_dir / filename for key, filename in MANDATORY_HPO_ARTIFACTS.items()}
    _write_trials_csv(paths["trials_csv"], manifest)
    _write_json(paths["best_params_json"], manifest.get("best_params", {}))
    _write_best_config_yaml(paths["best_config_yaml"], manifest, seed)
    _write_json(
        paths["env_info"],
        {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "cwd": str(ROOT),
        },
    )
    paths["git_status"].write_text(_run_git_text("status", "--short"), encoding="utf-8")
    paths["git_diff"].write_text(_run_git_text("diff", "--no-ext-diff"), encoding="utf-8")
    paths["command_line"].write_text(" ".join(command) + "\n", encoding="utf-8")
    _write_json(paths["seed_list"], [seed])
    _write_json(paths["split_manifest"], manifest.get("cv", {}))
    trials = manifest.get("trial_results", [])
    sorted_trials = []
    if isinstance(trials, list):
        sorted_trials = sorted(
            (trial for trial in trials if isinstance(trial, dict)),
            key=lambda trial: float(trial.get("mean_score", float("-inf"))),
            reverse=True,
        )[:top_k]
    _write_json(paths["top_k_candidate_configs"], sorted_trials)
    if study_db:
        paths["study_db"] = Path(study_db)
    else:
        _write_json(paths["study_db"], {"storage": None, "reason": "sqlite study db not requested"})
    artifact_paths = {key: str(path) for key, path in paths.items()}
    _write_summary(paths["summary_markdown"], manifest, artifact_paths)
    return artifact_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporal-path", default="data/processed/temporal_sequences.pt")
    parser.add_argument("--grn-path", default="data/processed/grn_artifacts.pt")
    parser.add_argument("--out", required=True, help="Path to output HPO manifest JSON")
    parser.add_argument("--run-id", default="real-hpo-smoke")
    parser.add_argument("--runtime-manifest", default=None, help="Optional runtime manifest JSON path")
    parser.add_argument("--study-db", default=None, help="Optional SQLite Optuna study file path")
    parser.add_argument("--artifact-dir", default=None, help="Directory for mandatory HPO smoke artifacts")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--n-folds", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=None, help="Optional Optuna timeout in seconds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allele", default="WT")
    args = parser.parse_args()

    manifest = run_real_hpo_smoke(
        temporal_path=args.temporal_path,
        grn_path=args.grn_path,
        out_path=args.out,
        run_id=args.run_id,
        n_trials=args.n_trials,
        n_folds=args.n_folds,
        seed=args.seed,
        allele=args.allele,
        study_storage=args.study_db,
        timeout=args.timeout,
    )
    print(
        "real_hpo_smoke",
        "best_value",
        f"{manifest['best_value']:.6f}",
        "trials",
        len(manifest["trial_results"]),
        "manifest",
        args.out,
    )
    if args.study_db:
        print("optuna_study", args.study_db)

    if args.runtime_manifest:
        runtime = build_runtime_manifest(
            run_id=args.run_id,
            command=sys.argv,
            output_paths=[Path(args.out), Path(args.study_db)] if args.study_db else [Path(args.out)],
            cwd=ROOT,
            extra={"kind": "real_hpo_smoke_cli", "objective": manifest["objective"]},
        )
        write_runtime_manifest(runtime, args.runtime_manifest)
        print("runtime_manifest", args.runtime_manifest)
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else Path(args.out).resolve().parent
    manifest["mandatory_artifacts"] = _write_hpo_artifacts(
        manifest=manifest,
        artifact_dir=artifact_dir,
        command=sys.argv,
        seed=args.seed,
        study_db=args.study_db,
    )
    Path(args.out).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("hpo_artifact_dir", artifact_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
