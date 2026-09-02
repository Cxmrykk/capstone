"""Environment introspection -- powers `python app.py doctor`.

The point is to catch common failure modes in resource-constrained or Colab
environments: no GPU, bf16 assumed on a T4, bitsandbytes missing, missing HF
token, or unpulled dataset files.
"""
from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from src.paths import artifacts_dir, data_dir, dataset_dir, is_colab, repo_root

_PACKAGES = [
    "torch", "transformers", "peft", "accelerate", "bitsandbytes", "datasets",
    "huggingface_hub", "trl", "unsloth", "neo4j", "yaml", "pandas", "pyarrow",
    "sentence_transformers", "requests",
]

_MODEL_DIRS = ["gemma-4-E2B", "gemma-4-E4B", "Qwen3.5-2B", "Qwen3.5-4B"]


def _package_versions() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name in _PACKAGES:
        try:
            mod = importlib.import_module(name)
            out[name] = str(getattr(mod, "__version__", "installed"))
        except Exception:
            out[name] = "MISSING"
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
    props = torch.cuda.get_device_properties(0)
    info["device_name"] = props.name
    info["capability"] = f"{props.major}.{props.minor}"
    info["total_vram_gb"] = round(props.total_memory / (1024 ** 3), 2)

    try:
        bf16 = bool(torch.cuda.is_bf16_supported())
    except Exception:
        bf16 = props.major >= 8
    info["bf16_supported"] = bf16
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
    report: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "is_colab": is_colab(),
        "repo_root": str(repo_root()),
        "data_dir": str(data_dir()),
        "artifacts_dir": str(artifacts_dir()),
        "packages": _package_versions(),
        "torch": _torch_info(),
        "dataset": _dataset_status(),
        "models": _model_status(),
        "binaries": _which_all([
            "git", "cmake", "make", "nvidia-smi", "neo4j", "cypher-shell", "java",
            "llama-cli", "llama-server", "llama-quantize",
        ]),
        "neo4j_service": _neo4j_service_status(),
        "hf_token_present": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")),
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

    if not t["available"]:
        warns.append("PyTorch is not installed -- training and HF inference are unavailable.")
    elif not t["cuda"]:
        warns.append("No CUDA device detected. Fine-tuning is not practical on CPU; consider using a GPU instance such as Colab.")
    else:
        if not t["bf16_supported"]:
            warns.append(
                f"{t['device_name']} (sm_{t['capability']}) does not support bfloat16. "
                "Training will run in fp16 -- this is expected on a T4 and handled automatically."
            )
        if t["total_vram_gb"] and t["total_vram_gb"] < 20:
            warns.append(
                f"Detected {t['total_vram_gb']} GB VRAM. Maintain 4-bit quantization, "
                "gradient checkpointing, and small per-device batch sizes."
            )

    if r["packages"].get("bitsandbytes") == "MISSING" and t.get("cuda"):
        warns.append("bitsandbytes is missing -- 4-bit QLoRA will not work.")
    if r["packages"].get("peft") == "MISSING":
        warns.append("peft is missing -- LoRA training will not work.")
    if r["packages"].get("neo4j") == "MISSING":
        warns.append("The neo4j driver is missing -- execution-based evaluation is unavailable.")

    ds = r["dataset"]
    if not (ds["train_parquet"] and ds["test_parquet"]):
        warns.append(f"Dataset parquet files not found under {ds['dir']}. Run ./download_data.sh")

    if not r["hf_token_present"]:
        warns.append("No HF_TOKEN in the environment -- gated models and checkpoint sync will fail.")

    if not r["llama_cpp_dir"] and not r["binaries"].get("llama-cli"):
        warns.append("llama.cpp not found -- run scripts/build_llama_cpp.sh before GGUF work.")

    return warns


def render_report(r: Dict[str, Any]) -> str:
    lines: List[str] = []
    add = lines.append

    add("=" * 68)
    add("  Text2Cypher -- environment report")
    add("=" * 68)
    add(f"Python          : {r['python']}")
    add(f"Platform        : {r['platform']}")
    add(f"Google Colab    : {'yes' if r['is_colab'] else 'no'}")
    add(f"Repo root       : {r['repo_root']}")
    add(f"Artifacts       : {r['artifacts_dir']}")
    add("")

    t = r["torch"]
    add("-- Accelerator ----------------------------------------------------")
    if t["cuda"]:
        add(f"Device          : {t['device_name']} (sm_{t['capability']})")
        add(f"VRAM            : {t['total_vram_gb']} GB")
        add(f"CUDA            : {t.get('cuda_version')}")
        add(f"bfloat16        : {'supported' if t['bf16_supported'] else 'NOT supported'}")
        add(f"Training dtype  : {t['recommended_dtype']}")
    else:
        add("Device          : CPU only")
    add("")

    add("-- Packages -------------------------------------------------------")
    for name, ver in r["packages"].items():
        add(f"  {name:<22} {ver}")
    add("")

    add("-- Dataset --------------------------------------------------------")
    ds = r["dataset"]
    add(f"  dir           : {ds['dir']}")
    add(f"  train parquet : {'yes' if ds['train_parquet'] else 'NO'} "
        f"({ds['train_size_mb']} MB)" if ds["train_parquet"] else "  train parquet : NO")
    add(f"  test parquet  : {'yes' if ds['test_parquet'] else 'NO'} "
        f"({ds['test_size_mb']} MB)" if ds["test_parquet"] else "  test parquet  : NO")
    add("")

    add("-- Base models ----------------------------------------------------")
    for name, st in r["models"].items():
        state = "weights present" if st["has_weights"] else (
            "directory only (submodule not pulled)" if st["dir_exists"] else "absent")
        add(f"  {name:<16} {state}")
    add("")

    add("-- Tooling --------------------------------------------------------")
    for name, path in r["binaries"].items():
        add(f"  {name:<16} {path or '-'}")
    add(f"  llama.cpp dir  : {r['llama_cpp_dir'] or '-'}")
    add(f"  neo4j service  : {r['neo4j_service']}")
    add(f"  HF_TOKEN       : {'set' if r['hf_token_present'] else 'NOT SET'}")
    add("")

    if "neo4j_probe" in r:
        add("-- Neo4j connectivity ---------------------------------------------")
        for alias, status in r["neo4j_probe"].items():
            add(f"  {alias:<32} {status}")
        add("")

    if r["warnings"]:
        add("-- Warnings -------------------------------------------------------")
        for w in r["warnings"]:
            add(f"  ! {w}")
    else:
        add("No warnings. Environment looks ready.")
    add("=" * 68)
    return "\n".join(lines)
