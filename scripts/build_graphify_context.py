#!/usr/bin/env python
"""
Build a Graphify-compatible project knowledge graph for TS-GNN.

The generated graph is optimized for targeted repo queries:
- directories/files are represented as nodes
- Python symbols (classes/functions) are added as first-class nodes
- project-local imports become graph edges
- concise text summaries are attached to aid Graphify retrieval

Output:
    graphify-out/graph.json
"""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "graphify-out"
OUT_GRAPH = OUT_DIR / "graph.json"

EXCLUDE_DIRS = {
    ".git",
    ".idea",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".vscode",
    "__pycache__",
}
TEXT_EXTENSIONS = {
    ".py",
    ".md",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".sh",
    ".ps1",
    ".ini",
    ".cfg",
    ".env",
    ".ipynb",
    ".tex",
    ".csv",
}
CODE_EXTENSIONS = {".py", ".sh", ".ps1"}


def rel_id(path: Path) -> str:
    return path.relative_to(ROOT).as_posix() if path != ROOT else "."


def should_skip(path: Path) -> bool:
    return any(part in EXCLUDE_DIRS for part in path.parts)


def summarize_python(path: Path, content: str) -> str:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return "Python source file"

    parts: list[str] = []
    module_doc = ast.get_docstring(tree)
    if module_doc:
        parts.append(module_doc.strip().splitlines()[0][:220])

    top_level = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    if top_level:
        snippet = ", ".join(top_level[:12])
        if len(top_level) > 12:
            snippet += f", ... (+{len(top_level) - 12})"
        parts.append(f"Top-level symbols: {snippet}")

    return " | ".join(parts) if parts else "Python source file"


def summarize_text(path: Path) -> str:
    ext = path.suffix.lower()
    size = path.stat().st_size
    if ext not in TEXT_EXTENSIONS:
        return f"Binary or unsupported content ({size} bytes)"
    if size > 2 * 1024 * 1024:
        return f"Large text file ({size} bytes)"

    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return f"Unreadable text ({size} bytes)"

    if ext == ".py":
        return summarize_python(path, content)

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if ext == ".md":
        heading = next((line.lstrip("# ").strip() for line in lines if line.startswith("#")), None)
        if heading:
            return f"Markdown doc: {heading[:220]}"

    return (" ".join(lines[:4])[:280]) if lines else "Empty file"


def module_from_src(path: Path) -> str | None:
    try:
        relative = path.relative_to(ROOT / "src")
    except ValueError:
        return None

    if relative.suffix != ".py":
        return None

    parts = list(relative.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    nodes: dict[str, dict] = {
        ".": {
            "id": ".",
            "label": "ts-gnn",
            "kind": "root",
            "path": ".",
            "text": "Project root for the TS-GNN repository.",
        }
    }
    links: set[tuple[str, str, str]] = set()
    python_files: list[Path] = []

    for path in ROOT.rglob("*"):
        if should_skip(path):
            continue

        node_id = rel_id(path)
        parent_id = rel_id(path.parent) if path != ROOT else "."

        if path.is_dir():
            nodes[node_id] = {
                "id": node_id,
                "label": path.name or "root",
                "kind": "directory",
                "path": node_id,
                "text": f"Directory {node_id}",
            }
            if node_id != ".":
                links.add((parent_id, node_id, "contains"))
            continue

        if not path.is_file():
            continue

        ext = path.suffix.lower()
        node = {
            "id": node_id,
            "label": path.name,
            "kind": "file",
            "path": node_id,
            "ext": ext,
            "size": path.stat().st_size,
            "text": summarize_text(path),
        }
        if ext in CODE_EXTENSIONS:
            try:
                with path.open("r", encoding="utf-8", errors="ignore") as handle:
                    node["loc"] = sum(1 for _ in handle)
            except OSError:
                pass

        nodes[node_id] = node
        links.add((parent_id, node_id, "contains"))

        if ext == ".py":
            python_files.append(path)

    module_to_file = {
        module: rel_id(path)
        for path in python_files
        for module in [module_from_src(path)]
        if module is not None
    }

    for path in python_files:
        file_id = rel_id(path)
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(content)
        except (OSError, SyntaxError):
            continue

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                symbol_id = f"{file_id}::{node.name}"
                symbol_kind = "class" if isinstance(node, ast.ClassDef) else "function"
                doc = ast.get_docstring(node) or ""
                nodes[symbol_id] = {
                    "id": symbol_id,
                    "label": node.name,
                    "kind": symbol_kind,
                    "path": file_id,
                    "line": getattr(node, "lineno", None),
                    "text": doc.strip().splitlines()[0][:220] if doc else f"{symbol_kind} {node.name}",
                }
                links.add((file_id, symbol_id, "defines"))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            else:
                continue

            for module_name in names:
                target = module_to_file.get(module_name)
                if target:
                    links.add((file_id, target, "imports"))

    graph = {
        "directed": True,
        "multigraph": False,
        "graph": {
            "name": "ts-gnn project knowledge graph",
            "root": ".",
        },
        "nodes": list(nodes.values()),
        "links": [
            {"source": source, "target": target, "relation": relation, "type": relation}
            for source, target, relation in sorted(links)
        ],
    }

    OUT_GRAPH.write_text(
        json.dumps(graph, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Graph written to {OUT_GRAPH}")
    print(f"Nodes: {len(graph['nodes'])}")
    print(f"Links: {len(graph['links'])}")


if __name__ == "__main__":
    main()
