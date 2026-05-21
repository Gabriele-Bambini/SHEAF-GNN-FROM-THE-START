from __future__ import annotations

import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON_TARGETS = [
    *ROOT.glob("hpc/*.py"),
    ROOT / "force_upload.py",
    ROOT / "sync_fts.py",
    ROOT / "scratch" / "download_results.py",
]
TEXT_TARGETS = [
    *ROOT.glob("hpc/*.ps1"),
    *ROOT.glob("hpc/*.sh"),
    *ROOT.glob("hpc/*.sbatch"),
]


def _is_secret_name(name: str) -> bool:
    lowered = name.lower()
    return lowered in {"pass", "pwd", "password", "passwd"} or lowered.endswith("_pass")


def test_hpc_python_scripts_do_not_hardcode_password_literals() -> None:
    failures: list[str] = []
    for path in PYTHON_TARGETS:
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
                if any(_is_secret_name(name) for name in targets) and node.value.value:
                    failures.append(f"{path.relative_to(ROOT)}:{node.lineno}")
            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if keyword.arg == "password" and isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
                        failures.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not failures, "hardcoded password literals found in active HPC scripts: " + ", ".join(failures)


def test_hpc_shell_scripts_do_not_embed_password_literals() -> None:
    assignment = re.compile(r"(?i)\b(pass|pwd|password|passwd)\b\s*=\s*['\"][^'\"]+['\"]")
    sshpass = re.compile(r"(?i)\bsshpass\s+-p\b")
    failures: list[str] = []
    for path in TEXT_TARGETS:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if assignment.search(text) or sshpass.search(text):
            failures.append(str(path.relative_to(ROOT)))
    assert not failures, "plaintext password usage found in active HPC scripts: " + ", ".join(failures)
