"""GGUF export and quantization via llama.cpp tooling.

Converts merged Hugging Face model weights to GGUF format and applies k-quantization
(e.g., Q4_K_M, Q5_K_M) for efficient CPU and edge device deployment.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from src.logging_utils import get_logger
from src.paths import gguf_dir, repo_root

log = get_logger(__name__)

QUANT_TYPES = [
    "Q2_K", "Q3_K_S", "Q3_K_M", "Q4_0", "Q4_K_S", "Q4_K_M",
    "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0",
]


def find_llama_cpp_dir(override: Optional[str] = None) -> Optional[Path]:
    """Finds the root directory of a compiled llama.cpp repository."""
    candidates: List[Path] = []
    if override:
        candidates.append(Path(override))
    env = os.environ.get("LLAMA_CPP_DIR")
    if env:
        candidates.append(Path(env))
    candidates += [
        repo_root() / "vendor" / "llama.cpp",
        Path.home() / "llama.cpp",
        Path("/opt/llama.cpp"),
        Path("/content/llama.cpp"),
    ]
    for candidate in candidates:
        if (candidate / "convert_hf_to_gguf.py").exists():
            return candidate
    return None


def _find_quantize_binary(llama_dir: Path) -> Optional[str]:
    on_path = shutil.which("llama-quantize")
    if on_path:
        return on_path
    for rel in ("build/bin/llama-quantize", "build/llama-quantize",
                "llama-quantize", "quantize", "build/bin/quantize"):
        candidate = llama_dir / rel
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _run(cmd: List[str], cwd: Optional[Path] = None) -> None:
    log.info("$ %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd, cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    assert proc.stdout is not None
    tail: List[str] = []
    for line in proc.stdout:
        line = line.rstrip()
        tail.append(line)
        if len(tail) > 40:
            tail.pop(0)
        print(line)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {proc.returncode}:\n  {' '.join(cmd)}\n"
            f"Last output:\n" + "\n".join(tail)
        )


def convert_to_gguf(
    merged_dir: str,
    out_dir: Optional[str] = None,
    quant: str = "Q4_K_M",
    llama_cpp_dir: Optional[str] = None,
    keep_f16: bool = False,
) -> Path:
    """Converts a merged Hugging Face model directory to a quantized GGUF file."""
    merged = Path(merged_dir)
    if not merged.exists():
        raise FileNotFoundError(f"Merged model directory not found: {merged}")

    if quant.upper() not in QUANT_TYPES and quant.upper() not in {"F16", "F32"}:
        log.warning("Quantization type %r is not standard; passing to llama-quantize directly.", quant)

    llama_dir = find_llama_cpp_dir(llama_cpp_dir)
    if llama_dir is None:
        raise RuntimeError(
            "Could not locate llama.cpp repository containing convert_hf_to_gguf.py.\n"
            "Build with: bash scripts/build_llama_cpp.sh\n"
            "or set LLAMA_CPP_DIR to an existing checkout."
        )
    log.info("Using llama.cpp at %s", llama_dir)

    out = Path(out_dir) if out_dir else gguf_dir()
    out.mkdir(parents=True, exist_ok=True)

    name = merged.name.replace("-merged", "")
    f16_path = out / f"{name}-f16.gguf"

    log.info("Step 1/2: Converting Hugging Face weights to f16 GGUF format...")
    try:
        _run([
            sys.executable,
            str(llama_dir / "convert_hf_to_gguf.py"),
            str(merged),
            "--outfile", str(f16_path),
            "--outtype", "f16",
        ])
    except RuntimeError as exc:
        raise RuntimeError(
            f"{exc}\n\n"
            "The llama.cpp converter may require an update for this model architecture.\n"
            "Options:\n"
            f"  1. Update repository: git -C {llama_dir} pull && bash scripts/build_llama_cpp.sh\n"
            "  2. Evaluate the merged fp16 model using the 'hf' backend on CPU:\n"
            f"     python app.py predict --config <cfg> --backend hf --model-path {merged} --limit 50\n"
        ) from exc

    if quant.upper() in {"F16", "F32"}:
        log.info("Requested %s precision; skipping quantization step.", quant)
        _write_sidecar(f16_path, merged, quant)
        return f16_path

    log.info("Step 2/2: Quantizing weights to %s ...", quant)
    quantize_bin = _find_quantize_binary(llama_dir)
    if not quantize_bin:
        raise RuntimeError(
            f"llama-quantize binary not found under {llama_dir}. "
            "Build with: bash scripts/build_llama_cpp.sh"
        )

    quant_path = out / f"{name}-{quant.lower()}.gguf"
    _run([quantize_bin, str(f16_path), str(quant_path), quant])

    if not keep_f16:
        try:
            f16_path.unlink()
            log.info("Removed intermediate f16 GGUF file (pass --keep-f16 to retain it).")
        except Exception as exc:
            log.warning("Could not delete %s: %s", f16_path, exc)

    _write_sidecar(quant_path, merged, quant)

    size_gb = quant_path.stat().st_size / 1e9
    log.info("GGUF artifact ready: %s (%.2f GB)", quant_path, size_gb)
    log.info("Test with:\n  python app.py predict --config <cfg> --backend llamacpp --gguf %s --limit 50", quant_path)
    return quant_path


def _write_sidecar(gguf_path: Path, merged: Path, quant: str) -> None:
    meta = {"gguf": str(gguf_path), "merged_from": str(merged), "quantisation": quant}
    merge_meta = merged / "merge_metadata.json"
    if merge_meta.exists():
        try:
            meta["merge_metadata"] = json.loads(merge_meta.read_text(encoding="utf-8"))
        except Exception:
            pass
    gguf_path.with_suffix(".gguf.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
