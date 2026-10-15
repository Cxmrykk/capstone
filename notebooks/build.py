#!/usr/bin/env python3
"""Assemble notebooks/<name>.ipynb from the chunk files in notebooks/<name>/.

Layout of a notebook directory:
    notebook.json      {"nbformat": 4, "nbformat_minor": 4, "metadata": {...},
                        "cell_metadata": {"NN-name.py": {...}}}   (cell_metadata optional)
    NN-name.py         code cell
    NN-name.md         markdown cell
    NN-name.raw        raw cell
Chunks are ordered by the numeric prefix NN (duplicates are an error). Files named
notebook.json, README*, or starting with "_" or "." are ignored.

"cell_metadata" maps chunk file names to per-cell metadata (e.g. Colab cell ids).
It is not written to the notebook itself.

Shared snippets (paths relative to notebooks/, usually notebooks/_shared/):
    code:      # %include _shared/run_step.py
    markdown:  <!-- %include _shared/kaggle-setup.md -->
The directive line is replaced by the file's content, indented like the directive.

Usage:
    python notebooks/build.py                  # build every notebook directory
    python notebooks/build.py training-kaggle  # build one (name, dir or .ipynb path)
    python notebooks/build.py --check          # exit 1 if any .ipynb is stale (CI / pre-commit)
    python notebooks/build.py --list           # show the chunk order per notebook
    python notebooks/build.py --validate       # also validate with nbformat (if installed)

Exit codes: 0 ok, 1 stale notebooks (--check), 2 build errors.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

NOTEBOOKS_DIR = Path(__file__).resolve().parent
META_FILE = "notebook.json"
CELL_TYPES = {".py": "code", ".md": "markdown", ".raw": "raw"}
CHUNK_RE = re.compile(r"^(\d+)[-_][^/]+$")
INCLUDE_PATTERNS = {
    "code": re.compile(r"^(?P<indent>[ \t]*)#\s*%include\s+(?P<path>\S+)\s*$"),
    "markdown": re.compile(r"^(?P<indent>[ \t]*)<!--\s*%include\s+(?P<path>\S+)\s*-->\s*$"),
}
MAX_INCLUDE_DEPTH = 8

Chunk = Tuple[int, str, Path, str]  # (order, file name, path, cell type)


class BuildError(Exception):
    pass


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(NOTEBOOKS_DIR))
    except ValueError:
        return str(path)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def notebook_dirs() -> List[Path]:
    return sorted(
        p for p in NOTEBOOKS_DIR.iterdir()
        if p.is_dir() and not p.name.startswith(("_", ".")) and (p / META_FILE).is_file()
    )


def resolve_dir(arg: str) -> Path:
    name = Path(arg.rstrip("/")).name
    if name.endswith(".ipynb"):
        name = name[: -len(".ipynb")]
    nb_dir = NOTEBOOKS_DIR / name
    if not (nb_dir / META_FILE).is_file():
        known = ", ".join(p.name for p in notebook_dirs()) or "none"
        raise BuildError(f"No notebook directory '{name}' with {META_FILE} (known: {known}).")
    return nb_dir


def _ignored(path: Path) -> bool:
    name = path.name
    return name == META_FILE or name.startswith(("_", ".")) or name.upper().startswith("README")


def collect_chunks(nb_dir: Path) -> List[Chunk]:
    chunks: List[Chunk] = []
    for path in sorted(nb_dir.iterdir()):
        if _ignored(path):
            continue
        if path.is_dir():
            raise BuildError(f"{_rel(path)}: subdirectories are not supported; "
                             "put shared code in notebooks/_shared and use %include.")
        m = CHUNK_RE.match(path.name)
        if not m:
            raise BuildError(f"{_rel(path)}: chunk names must look like NN-name.py / NN-name.md "
                             "(prefix with '_' to exclude a file).")
        kind = CELL_TYPES.get(path.suffix)
        if kind is None:
            raise BuildError(f"{_rel(path)}: unknown extension (use {', '.join(CELL_TYPES)}).")
        chunks.append((int(m.group(1)), path.name, path, kind))

    if not chunks:
        raise BuildError(f"{_rel(nb_dir)}: no chunks found; refusing to write an empty notebook.")
    chunks.sort()

    seen: Dict[int, str] = {}
    for order, name, _path, _kind in chunks:
        if order in seen:
            raise BuildError(f"{_rel(nb_dir)}: {seen[order]} and {name} share the order "
                             f"prefix {order}; renumber one of them.")
        seen[order] = name
    return chunks


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def expand_includes(text: str, origin: Path, kind: str,
                    stack: Tuple[Path, ...] = ()) -> str:
    pattern = INCLUDE_PATTERNS.get(kind)
    if pattern is None:
        return text
    out: List[str] = []
    for line in text.split("\n"):
        m = pattern.match(line)
        if not m:
            out.append(line)
            continue
        target = (NOTEBOOKS_DIR / m.group("path")).resolve()
        if NOTEBOOKS_DIR not in target.parents:
            raise BuildError(f"{_rel(origin)}: include path escapes notebooks/: {m.group('path')}")
        if not target.is_file():
            raise BuildError(f"{_rel(origin)}: include not found: {m.group('path')}")
        if target in stack or target == origin.resolve():
            chain = " -> ".join(_rel(p) for p in (*stack, target))
            raise BuildError(f"{_rel(origin)}: include cycle: {chain}")
        if len(stack) >= MAX_INCLUDE_DEPTH:
            raise BuildError(f"{_rel(origin)}: includes nested deeper than {MAX_INCLUDE_DEPTH}.")
        body = expand_includes(_read(target), target, kind, (*stack, target)).strip("\n")
        indent = m.group("indent")
        out.extend((indent + ln) if ln.strip() else ln for ln in body.split("\n"))
    return "\n".join(out)


def to_source(text: str) -> List[str]:
    """nbformat line list: every line keeps its newline except the last."""
    lines = text.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def make_cell(kind: str, text: str, metadata: Optional[Dict] = None) -> Dict:
    cell: Dict = {"cell_type": kind}
    meta = dict(metadata or {})
    if kind == "code":
        cell.update(execution_count=None, metadata=meta, outputs=[], source=to_source(text))
    else:
        cell.update(metadata=meta, source=to_source(text))
    return cell


def build_notebook(nb_dir: Path) -> str:
    try:
        meta = json.loads(_read(nb_dir / META_FILE))
    except json.JSONDecodeError as exc:
        raise BuildError(f"{_rel(nb_dir / META_FILE)}: invalid JSON: {exc}") from exc

    cell_meta = meta.get("cell_metadata", {})
    if not isinstance(cell_meta, dict):
        raise BuildError(f"{_rel(nb_dir / META_FILE)}: cell_metadata must be an object.")

    chunks = collect_chunks(nb_dir)
    unknown = sorted(set(cell_meta) - {name for _o, name, _p, _k in chunks})
    if unknown:
        raise BuildError(f"{_rel(nb_dir / META_FILE)}: cell_metadata names missing chunk(s): "
                         + ", ".join(unknown))

    cells = []
    for _order, name, path, kind in chunks:
        text = expand_includes(_read(path), path, kind).strip("\n")
        if not text.strip():
            raise BuildError(f"{_rel(path)}: chunk is empty.")
        cells.append(make_cell(kind, text, cell_meta.get(name)))

    notebook = {
        "nbformat": meta.get("nbformat", 4),
        "nbformat_minor": meta.get("nbformat_minor", 4),
        "metadata": meta.get("metadata", {}),
        "cells": cells,
    }
    return json.dumps(notebook, indent=2, ensure_ascii=False) + "\n"


def validate(text: str, name: str) -> None:
    try:
        import nbformat
    except ImportError as exc:
        raise BuildError("--validate needs nbformat: pip install nbformat") from exc
    try:
        nbformat.validate(nbformat.reads(text, as_version=4))
    except Exception as exc:
        raise BuildError(f"{name}: nbformat validation failed: {exc}") from exc


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Assemble notebooks from chunk directories.")
    ap.add_argument("notebooks", nargs="*", help="Notebook names (default: all directories).")
    ap.add_argument("--check", action="store_true", help="Do not write; exit 1 if any notebook is stale.")
    ap.add_argument("--list", action="store_true", help="Print chunk order and exit.")
    ap.add_argument("--validate", action="store_true", help="Validate output with nbformat.")
    args = ap.parse_args(argv)

    try:
        dirs = [resolve_dir(a) for a in args.notebooks] if args.notebooks else notebook_dirs()
    except BuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not dirs:
        print(f"error: no notebook directories with {META_FILE} under {NOTEBOOKS_DIR}", file=sys.stderr)
        return 2

    if args.list:
        rc = 0
        for nb_dir in dirs:
            print(f"{nb_dir.name}.ipynb")
            try:
                for i, (_order, name, _path, kind) in enumerate(collect_chunks(nb_dir), 1):
                    print(f"  [{i:02d}] {kind:<8} {name}")
            except BuildError as exc:
                print(f"  error: {exc}")
                rc = 2
        return rc

    errors = stale = 0
    for nb_dir in dirs:
        out_path = NOTEBOOKS_DIR / f"{nb_dir.name}.ipynb"
        try:
            text = build_notebook(nb_dir)
            if args.validate:
                validate(text, out_path.name)
        except BuildError as exc:
            print(f"error: {exc}", file=sys.stderr)
            errors += 1
            continue

        current = _read(out_path) if out_path.exists() else None
        n_cells = text.count('"cell_type":')
        if args.check:
            if current != text:
                print(f"STALE     {out_path.name} (run: python notebooks/build.py {nb_dir.name})")
                stale += 1
            else:
                print(f"ok        {out_path.name}")
        elif current == text:
            print(f"unchanged {out_path.name} ({n_cells} cells)")
        else:
            out_path.write_text(text, encoding="utf-8")
            print(f"built     {out_path.name} ({n_cells} cells)")

    if errors:
        return 2
    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
