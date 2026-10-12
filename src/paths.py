"""Path resolution that behaves identically on a local machine and on Colab."""
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
    """True inside a Colab runtime, including `!python app.py ...` subprocesses.

    Subprocesses never import google.colab, so the runtime's environment
    variables are checked as well.
    """
    if "google.colab" in sys.modules:
        return True
    return bool(os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU"))


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

    Always <repo>/artifacts unless T2C_ARTIFACTS_DIR is set. On Colab the repo
    lives on the VM's local disk (/content/capstone), so this is already fast,
    and it keeps the relative `artifacts/...` paths used in the notebook valid.
    Durability across Colab sessions comes from the Hub checkpoint sync.
    """
    env = os.environ.get("T2C_ARTIFACTS_DIR")
    if env:
        path = Path(env).expanduser().resolve()
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
