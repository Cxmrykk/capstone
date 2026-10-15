#!/usr/bin/env bash
#
# Google Colab / Kaggle environment bootstrap (GPU or CPU).
#
#   From the notebook:  !bash scripts/colab_bootstrap.sh
#   Skip Unsloth:       !INSTALL_UNSLOTH=0 bash scripts/colab_bootstrap.sh
#
# Colab already ships a CUDA-enabled torch; requirements.txt only sets a floor,
# so pip keeps the preinstalled build.
set -euo pipefail

BOLD=$'\033[1m'; RESET=$'\033[0m'; YELLOW=$'\033[33m'
say()  { echo "${BOLD}==>${RESET} $*"; }
warn() { echo "${YELLOW}!!${RESET} $*"; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

INSTALL_UNSLOTH="${INSTALL_UNSLOTH:-1}"
TRANSFORMERS_FLOOR="transformers>=5.5.0"

say "Checking GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv \
    || warn "No GPU visible. In Colab: Runtime > Change runtime type > T4 GPU (expected on CPU-only sessions)."

say "Installing Python requirements"
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet -r requirements.txt

if [[ "${INSTALL_UNSLOTH}" == "1" ]]; then
    say "Installing Unsloth (optional; only the Qwen configs use it)"
    if python3 -m pip install --quiet unsloth; then
        # Unsloth may pin an older transformers. Gemma 4 / Qwen3.5 need a recent
        # release, so restore the floor. If this breaks Unsloth, the Qwen configs
        # fall back to Transformers + PEFT automatically (use_unsloth: auto).
        python3 -m pip install --quiet "${TRANSFORMERS_FLOOR}" \
            || warn "Could not restore ${TRANSFORMERS_FLOOR}; check 'python app.py doctor'."
    else
        warn "Unsloth install failed; training will use Transformers + PEFT."
    fi
fi

say "Checking torchao compatibility with PEFT"
# Some images (e.g. Kaggle CPU) preinstall an old torchao. Recent PEFT raises
# ImportError for torchao <= 0.16.0 while injecting LoRA layers, even though no
# torchao layer is used (history/033.md). Nothing in this project needs torchao,
# so an incompatible version is removed.
python3 - <<'EOF'
import subprocess
import sys
from importlib import metadata

try:
    version = metadata.version("torchao")
except metadata.PackageNotFoundError:
    print("  torchao not installed (ok)")
    sys.exit(0)

from packaging.version import Version

if Version(version) <= Version("0.16.0"):
    print(f"  torchao {version} is incompatible with PEFT; uninstalling")
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "--quiet", "torchao"])
else:
    print(f"  torchao {version} (ok)")
EOF

say "Checking the Hugging Face CLI"
if ! command -v hf > /dev/null 2>&1; then
    python3 -m pip install --quiet --upgrade "huggingface_hub[cli]"
fi
hf version 2>/dev/null || hf --version 2>/dev/null || warn "hf CLI not on PATH."

chmod +x download_data.sh scripts/*.sh 2>/dev/null || true

say "Installed versions"
python3 - <<'EOF'
import importlib
for name in ("torch", "transformers", "peft", "bitsandbytes", "accelerate",
             "huggingface_hub", "unsloth", "neo4j", "torchao"):
    try:
        mod = importlib.import_module(name)
        print(f"  {name:<16} {getattr(mod, '__version__', 'installed')}")
    except Exception as exc:
        print(f"  {name:<16} MISSING ({type(exc).__name__})")
try:
    import torch
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"  GPU              {p.name} sm_{p.major}{p.minor} "
              f"{p.total_memory / 1024**3:.1f} GB, bf16={torch.cuda.is_bf16_supported()}")
    else:
        print("  GPU              none (CPU only)")
except Exception:
    pass
EOF

say "Bootstrap complete. Next: bash download_data.sh"
