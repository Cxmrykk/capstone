#!/usr/bin/env bash
#
# One-shot Colab setup. Run from a cell:
#
#   !git clone https://github.com/<you>/<repo>.git /content/capstone
#   %cd /content/capstone
#   !bash scripts/colab_bootstrap.sh
#
# Then set your token and start training:
#
#   import os; os.environ["HF_TOKEN"] = "hf_..."
#   !python app.py train --config configs/qwen3.5-2b.yaml --resume auto
set -euo pipefail

BOLD=$'\033[1m'; RESET=$'\033[0m'
say() { echo "${BOLD}==>${RESET} $*"; }

say "Installing training dependencies"
pip install -q --upgrade pip
pip install -q -r requirements.txt
pip install -q -r requirements-train.txt

say "Attempting Unsloth (optional -- the toolkit falls back to transformers+peft)"
pip install -q unsloth || echo "Unsloth unavailable; continuing without it."

say "Installing evaluation dependencies"
pip install -q -r requirements-eval.txt

say "Enabling fast Hub transfers"
export HF_HUB_ENABLE_HF_TRANSFER=1
echo "export HF_HUB_ENABLE_HF_TRANSFER=1" >> ~/.bashrc

say "Environment report"
python app.py doctor

cat <<EOF

$(say "Bootstrap complete")

Before training, set a WRITE token so checkpoints survive a session loss:

  import os
  os.environ["HF_TOKEN"] = "hf_..."

And set hub.repo_id in your config (or export T2C_HUB_REPO), e.g.

  hub:
    repo_id: your-username/t2c-capstone-checkpoints

Datasets/models: the base weights download automatically from the Hub on first
use. To pre-fetch them into data/ instead, run ./download_data.sh.
EOF
