"""Path resolution that behaves identically on local CPU and on Colab."""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

_MARKERS = ("app.py", "download_data.sh", "AGENT.md")


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """Locate the repository root by walking up from this file."""
    env = os.environ.get("T2C_REPO_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    here = Path(__file__).resolve()
    for candidate in [here.parent.parent, *here.parents]:
        if any((candidate / m).exists() for m in _MARKERS):
            return candidate
    return here.parent.parent


@lru_cache(maxsize=1)
def is_colab() -> bool:
    if "google.colab" in sys.modules:
        return True
    return Path("/content").is_dir() and Path("/usr/local/lib/python3*/dist-packages").parent.exists()


@lru_cache(maxsize=1)
def data_dir() -> Path:
    env = os.environ.get("T2C_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return repo_root() / "data"


@lru_cache(maxsize=1)
def artifacts_dir() -> Path:
    """
    Scratch space for runs, predictions, reports and exports.

    On Colab this defaults to local (fast, ephemeral) disk; durability comes
    from the Hub checkpoint sync rather than from the filesystem.
    """
    env = os.environ.get("T2C_ARTIFACTS_DIR")
    if env:
        path = Path(env).expanduser().resolve()
    elif is_colab():
        path = Path("/content/artifacts")
    else:
        path = repo_root() / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def runs_dir() -> Path:
    p = artifacts_dir() / "runs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def predictions_dir() -> Path:
    p = artifacts_dir() / "predictions"
    p.mkdir(parents=True, exist_ok=True)
    return p


def reports_dir() -> Path:
    p = artifacts_dir() / "reports"
    p.mkdir(parents=True, exist_ok=True)
    return p


def merged_dir() -> Path:
    p = artifacts_dir() / "merged"
    p.mkdir(parents=True, exist_ok=True)
    return p


def gguf_dir() -> Path:
    p = artifacts_dir() / "gguf"
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_dir() -> Path:
    p = artifacts_dir() / "cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def dataset_dir(override: str | None = None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    env = os.environ.get("T2C_DATASET_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return data_dir() / "text2cypher-2024v1"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p
