"""Environment diagnostics and system compatibility checks for `app.py doctor`.

Verifies GPU accelerator availability, VRAM allocation, precision capabilities
(fp16 on NVIDIA Turing T4 vs bf16 on Ampere+), visible GPU count, Python
dependencies, and Neo4j connectivity. Runs on Colab, Kaggle and local machines.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.paths import (
    artifacts_dir,
    data_dir,
    dataset_dir,
    is_colab,
    is_kaggle,
    platform_name,
    repo_root,
)

# (display name, distribution name). Versions are read from package metadata,
# never by importing: importing unsloth after transformers/peft prints an
# import-order warning and patches both libraries inside this process.
_PACKAGES: List[Tuple[str, str]] = [
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("peft", "peft"),
    ("accelerate", "accelerate"),
    ("bitsandbytes", "bitsandbytes"),
    ("datasets", "datasets"),
    ("huggingface_hub", "huggingface_hub"),
    ("unsloth", "unsloth"),
    ("neo4j", "neo4j"),
    ("yaml", "PyYAML"),
    ("pandas", "pandas"),
    ("pyarrow", "pyarrow"),
    ("sentence_transformers", "sentence-transformers"),
    ("requests", "requests"),
]

# Models supported on the 16 GB T4: the two 2B models under study plus the
# Granite 4.2 3B dense-transformer control.
_MODEL_DIRS = ["gemma-4-E2B", "Qwen3.5-2B", "granite-4.2-3b"]

_PLATFORM_LABELS = {
    "colab": "Google Colab",
    "kaggle": "Kaggle",
    "local": "Standalone / Local",
}

_GPU_HINTS = {
    "colab": "Colab: Runtime > Change runtime type > T4 GPU",
    "kaggle": "Kaggle: Settings > Accelerator > GPU T4 x2 (phone-verified account)",
    "local": "attach an NVIDIA GPU with CUDA drivers",
}


def _package_versions() -> Dict[str, str]:
    """Reads installed package versions from metadata without importing them."""
    out: Dict[str, str] = {}
    for label, dist in _PACKAGES:
        try:
            out[label] = importlib_metadata.version(dist)
        except importlib_metadata.PackageNotFoundError:
            out[label] = "MISSING"
        except Exception:
            out[label] = "unknown"
    return out


def _torch_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "available": False,
        "cuda": False,
        "device_name": None,
        "capability": None,
        "total_vram_gb": None,
        "bf16_supported": False,
        "recommended_dtype": "float32",
        "device_count": 0,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch
    except ImportError:
        return info

    info["available"] = True
    info["version"] = torch.__version__
    if not torch.cuda.is_available():
        info["recommended_dtype"] = "float32"
        return info

    info["cuda"] = True
    info["cuda_version"] = torch.version.cuda
    info["device_count"] = torch.cuda.device_count()
    props = torch.cuda.get_device_properties(0)
    info["device_name"] = props.name
    info["capability"] = f"{props.major}.{props.minor}"
    info["capability_major"] = props.major
    info["total_vram_gb"] = round(props.total_memory / (1024 ** 3), 2)

    # Native bf16 needs Ampere (sm_80) or newer. torch.cuda.is_bf16_supported()
    # also reports emulated bf16, which recent PyTorch claims for the T4.
    bf16 = props.major >= 8
    info["bf16_supported"] = bf16
    try:
        info["bf16_emulated"] = (not bf16) and bool(torch.cuda.is_bf16_supported())
    except Exception:
        info["bf16_emulated"] = False
    info["recommended_dtype"] = "bfloat16" if bf16 else "float16"
    return info


def _which_all(names: List[str]) -> Dict[str, str | None]:
    return {n: shutil.which(n) for n in names}


def _neo4j_service_status() -> str:
    if platform.system() != "Linux":
        return "n/a (non-Linux)"
    if not shutil.which("systemctl"):
        return "unknown (no systemctl)"
    try:
        res = subprocess.run(
            ["systemctl", "is-active", "neo4j"],
            capture_output=True, text=True, timeout=5,
        )
        return res.stdout.strip() or "unknown"
    except Exception as exc:
        return f"unknown ({exc})"


def _dataset_status() -> Dict[str, Any]:
    root = dataset_dir()
    train = root / "data" / "train-00000-of-00001.parquet"
    test = root / "data" / "test-00000-of-00001.parquet"
    return {
        "dir": str(root),
        "exists": root.exists(),
        "train_parquet": train.exists(),
        "test_parquet": test.exists(),
        "train_size_mb": round(train.stat().st_size / 1e6, 1) if train.exists() else None,
        "test_size_mb": round(test.stat().st_size / 1e6, 1) if test.exists() else None,
    }


def _model_status() -> Dict[str, Any]:
    out = {}
    for name in _MODEL_DIRS:
        path = data_dir() / name
        has_weights = False
        if path.exists():
            has_weights = any(path.glob("*.safetensors")) or any(path.glob("*.bin"))
        out[name] = {
            "dir_exists": path.exists(),
            "has_weights": has_weights,
        }
    return out


def _probe_neo4j() -> Dict[str, Any]:
    from src.config import RunConfig
    from src.evaluation.neo4j_client import Neo4jExecutor, resolve_target

    cfg = RunConfig()
    results = {}
    for alias in ("neo4jlabs_demo_db_movies", "neo4jlabs_demo_db_fincen"):
        target = resolve_target(alias, cfg.neo4j)
        if target is None:
            results[alias] = "unresolvable"
            continue
        ex = Neo4jExecutor(cfg.neo4j, use_cache=False)
        ok, detail = ex.ping(target)
        ex.close()
        results[alias] = "ok" if ok else f"failed: {detail}"
    return results


def environment_report(check_neo4j: bool = False) -> Dict[str, Any]:
    """Collects comprehensive hardware, software, and dependency diagnostics."""
    report: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "platform_name": platform_name(),
        "is_colab": is_colab(),
        "is_kaggle": is_kaggle(),
        "kaggle_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
        "repo_root": str(repo_root()),
        "data_dir": str(data_dir()),
        "artifacts_dir": str(artifacts_dir()),
        "packages": _package_versions(),
        "torch": _torch_info(),
        "dataset": _dataset_status(),
        "models": _model_status(),
        "binaries": _which_all([
            "git", "cmake", "make", "nvidia-smi", "hf", "neo4j", "cypher-shell", "java",
            "llama-cli", "llama-server", "llama-quantize",
        ]),
        "neo4j_service": _neo4j_service_status(),
        "hf_token_present": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")),
        "hub_repo": os.environ.get("T2C_HUB_REPO"),
    }

    llama_cpp = os.environ.get("LLAMA_CPP_DIR")
    for candidate in filter(None, [llama_cpp, str(Path.home() / "llama.cpp"),
                                   str(repo_root() / "vendor" / "llama.cpp")]):
        if Path(candidate).exists():
            report["llama_cpp_dir"] = candidate
            break
    else:
        report["llama_cpp_dir"] = None

    if check_neo4j:
        report["neo4j_probe"] = _probe_neo4j()

    report["warnings"] = _warnings(report)
    return report


def _warnings(r: Dict[str, Any]) -> List[str]:
    warns: List[str] = []
    t = r["torch"]
    plat = r.get("platform_name", "local")

    if not t["available"]:
        warns.append("PyTorch is not installed -- training and PyTorch inference are unavailable.")
    elif not t["cuda"]:
        warns.append("No CUDA GPU detected. Model fine-tuning requires GPU acceleration "
                     f"({_GPU_HINTS.get(plat, _GPU_HINTS['local'])}).")
    else:
        major = t.get("capability_major") or 0
        if major and major < 7:
            warns.append(
                f"{t['device_name']} (sm_{t['capability']}) is older than Turing. Recent PyTorch "
                "CUDA builds are dropping Pascal support and the configs are tuned for a T4. "
                "On Kaggle, choose the GPU T4 x2 accelerator instead of P100."
            )
        if t.get("device_count", 1) > 1:
            warns.append(
                f"{t['device_count']} CUDA devices are visible. Without torchrun, HF Trainer wraps "
                "the model in nn.DataParallel, which is fragile with 4-bit bitsandbytes models and "
                "doubles the effective batch size versus single-GPU runs. Set CUDA_VISIBLE_DEVICES=0 "
                "(kaggle.ipynb does this automatically)."
            )
        if not t["bf16_supported"]:
            emulated = " (PyTorch reports emulated bf16 only)" if t.get("bf16_emulated") else ""
            warns.append(
                f"{t['device_name']} (sm_{t['capability']}) has no native bfloat16{emulated}. "
                "Training uses fp16 automatically; the Gemma and Granite configs disable Unsloth "
                "(it upcasts Gemma 4 to fp32 on this hardware). Expected on a T4."
            )
        if t["total_vram_gb"] and t["total_vram_gb"] < 20:
            warns.append(
                f"Detected {t['total_vram_gb']} GB VRAM. Only the 2B configs and the Granite 3B "
                "control config are supported; keep 4-bit loading, gradient checkpointing and "
                "target_only_loss enabled."
            )

    if r["packages"].get("bitsandbytes") == "MISSING" and t.get("cuda"):
        warns.append("bitsandbytes is not installed -- 4-bit QLoRA training is unavailable.")
    if r["packages"].get("peft") == "MISSING":
        warns.append("peft is not installed -- LoRA training is unavailable.")
    if r["packages"].get("neo4j") == "MISSING":
        warns.append("The neo4j driver is not installed -- execution-based evaluation is unavailable.")

    ds = r["dataset"]
    if not (ds["train_parquet"] and ds["test_parquet"]):
        warns.append(f"Dataset Parquet files missing from {ds['dir']}. Run: bash download_data.sh")

    missing_models = [n for n, st in r["models"].items() if not st["has_weights"]]
    if missing_models:
        warns.append("Base model weights not found locally for: " + ", ".join(missing_models)
                     + " (download_data.sh, or they will be fetched from the Hub on first load).")

    if not r["hf_token_present"]:
        warns.append("HF_TOKEN is not set -- gated model access and remote checkpoint sync may fail.")
    if not r.get("hub_repo"):
        lost = {
            "colab": "lost when the Colab runtime resets",
            "kaggle": "lost when the Kaggle session ends",
        }.get(plat, "not backed up remotely")
        warns.append("T2C_HUB_REPO is not set and configs use hub.repo_id: null -- "
                     f"checkpoints will only be kept locally and are {lost}.")

    if not r["llama_cpp_dir"] and not r["binaries"].get("llama-cli"):
        warns.append("llama.cpp not found -- run scripts/build_llama_cpp.sh before GGUF export.")

    return warns


def render_report(r: Dict[str, Any]) -> str:
    lines: List[str] = []
    add = lines.append

    plat = r.get("platform_name", "local")
    env_label = _PLATFORM_LABELS.get(plat, plat)
    if plat == "kaggle" and r.get("kaggle_run_type"):
        env_label = f"{env_label} ({r['kaggle_run_type']} session)"

    add("=" * 68)
    add("  Text2Cypher System Diagnostic Report")
    add("=" * 68)
    add(f"Python          : {r['python']}")
    add(f"Platform        : {r['platform']}")
    add(f"Environment     : {env_label}")
    add(f"Repository Root : {r['repo_root']}")
    add(f"Artifacts Dir   : {r['artifacts_dir']}")
    add(f"Hub Repo        : {r.get('hub_repo') or '-'}")
    add("")

    t = r["torch"]
    add("-- Accelerator ----------------------------------------------------")
    if t["cuda"]:
        if t["bf16_supported"]:
            bf16_text = "Supported (native)"
        elif t.get("bf16_emulated"):
            bf16_text = "Emulated only (not used)"
        else:
            bf16_text = "Not Supported"
        visible = t.get("cuda_visible_devices")
        add(f"Device          : {t['device_name']} (sm_{t['capability']})")
        add(f"Visible GPUs    : {t.get('device_count', 1)}"
            + (f" (CUDA_VISIBLE_DEVICES={visible})" if visible is not None else ""))
        add(f"VRAM            : {t['total_vram_gb']} GB")
        add(f"CUDA Version    : {t.get('cuda_version')}")
        add(f"bfloat16        : {bf16_text}")
        add(f"Training Dtype  : {t['recommended_dtype']}")
    else:
        add("Device          : CPU only")
    add("")

    add("-- Python Packages ------------------------------------------------")
    for name, ver in r["packages"].items():
        add(f"  {name:<22} {ver}")
    add("")

    add("-- Dataset Status -------------------------------------------------")
    ds = r["dataset"]
    add(f"  Location      : {ds['dir']}")
    add(f"  Train Split   : Present ({ds['train_size_mb']} MB)" if ds["train_parquet"]
        else "  Train Split   : MISSING")
    add(f"  Test Split    : Present ({ds['test_size_mb']} MB)" if ds["test_parquet"]
        else "  Test Split    : MISSING")
    add("")

    add("-- Local Base Models ----------------------------------------------")
    for name, st in r["models"].items():
        state = "Weights Present" if st["has_weights"] else (
            "Directory Exists (No Weights)" if st["dir_exists"] else "Not Found")
        add(f"  {name:<16} {state}")
    add("")

    add("-- Tools & Services -----------------------------------------------")
    for name, path in r["binaries"].items():
        add(f"  {name:<16} {path or '-'}")
    add(f"  llama.cpp      : {r['llama_cpp_dir'] or '-'}")
    add(f"  Neo4j Service  : {r['neo4j_service']}")
    add(f"  HF_TOKEN       : {'Configured' if r['hf_token_present'] else 'NOT CONFIGURED'}")
    add("")

    if "neo4j_probe" in r:
        add("-- Neo4j Connectivity ---------------------------------------------")
        for alias, status in r["neo4j_probe"].items():
            add(f"  {alias:<32} {status}")
        add("")

    if r["warnings"]:
        add("-- Diagnostics Warnings -------------------------------------------")
        for w in r["warnings"]:
            add(f"  ! {w}")
    else:
        add("Diagnostics passed without warnings. Environment is ready.")
    add("=" * 68)
    return "\n".join(lines)
