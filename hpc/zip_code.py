from __future__ import annotations

import argparse
import os
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parent.parent
ZIP_NAME = "tsgnn_sync.zip"
DEFAULT_EXCLUDES = {
    ".git",
    ".vscode",
    "checkpoints",
    "outputs",
    "logs",
    "results",
    "__pycache__",
    ".pytest_cache",
    "figures",
    "graphify-out",
}
DATA_CHOICES = ("raw", "processed", "external")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create TS-GNN sync zip for HPC.")
    parser.add_argument(
        "--include-data",
        nargs="*",
        choices=DATA_CHOICES,
        default=[],
        help="Optional data subtrees to include from data/: raw processed external",
    )
    return parser


def should_skip_dir(rel_parts: tuple[str, ...], include_data: set[str]) -> bool:
    if any(part in DEFAULT_EXCLUDES for part in rel_parts):
        return True
    if not rel_parts:
        return False
    if rel_parts[0] != "data":
        return False
    if len(rel_parts) == 1:
        return False
    data_kind = rel_parts[1]
    return data_kind not in include_data


def iter_files(include_data: set[str]):
    for current_root, dirs, files in os.walk(ROOT):
        rel_root = Path(current_root).relative_to(ROOT)
        rel_parts = rel_root.parts
        dirs[:] = [
            directory
            for directory in dirs
            if not should_skip_dir(rel_parts + (directory,), include_data)
        ]
        for filename in files:
            if filename.endswith(".zip"):
                continue
            rel_path = rel_root / filename if rel_parts else Path(filename)
            if should_skip_dir(rel_path.parts[:-1], include_data):
                continue
            yield rel_path


def zip_project(include_data: set[str]) -> None:
    zip_path = ROOT / ZIP_NAME
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for rel_path in iter_files(include_data):
            print(f"Adding: {rel_path.as_posix()}")
            archive.write(ROOT / rel_path, rel_path.as_posix())


def main() -> None:
    args = build_parser().parse_args()
    zip_project(set(args.include_data))


if __name__ == "__main__":
    main()
