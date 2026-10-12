#!/bin/bash
#
# Download the benchmark dataset and the 2B-parameter base models.
#
#   bash download_data.sh          # dataset + Gemma 4 E2B + Qwen3.5-2B
#   bash download_data.sh gemma    # dataset + Gemma 4 E2B only
#   bash download_data.sh qwen     # dataset + Qwen3.5-2B only
#   bash download_data.sh dataset  # dataset only
#
# Only 2B models are supported: 4B models do not fit the 16 GB T4 used on Colab.
set -euo pipefail

cd "$(dirname "$0")"

if [ -f .env ]; then
    echo "Loading environment variables from .env..."
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

export HF_XET_HIGH_PERFORMANCE=1

if ! command -v hf > /dev/null 2>&1; then
    echo "ERROR: 'hf' CLI not found. Install with: pip install -U 'huggingface_hub[cli]'"
    exit 1
fi

TARGET="${1:-all}"

DATASET="dataset neo4j/text2cypher-2024v1 data/text2cypher-2024v1"
GEMMA="model google/gemma-4-E2B data/gemma-4-E2B"
QWEN="model Qwen/Qwen3.5-2B data/Qwen3.5-2B"

case "$TARGET" in
    all)     REPOS=("$DATASET" "$GEMMA" "$QWEN") ;;
    gemma)   REPOS=("$DATASET" "$GEMMA") ;;
    qwen)    REPOS=("$DATASET" "$QWEN") ;;
    dataset) REPOS=("$DATASET") ;;
    *)
        echo "Usage: bash download_data.sh [all|gemma|qwen|dataset]"
        exit 1
        ;;
esac

# Sequential downloads: concurrent multi-GB pulls exhausted Colab disk/RAM (history/006.md).
for ENTRY in "${REPOS[@]}"; do
    TYPE=$(echo "$ENTRY" | cut -d' ' -f1)
    REPO_ID=$(echo "$ENTRY" | cut -d' ' -f2)
    DEST=$(echo "$ENTRY" | cut -d' ' -f3)

    echo "=========================================="
    echo "Downloading $TYPE: $REPO_ID -> $DEST"
    echo "=========================================="

    if [ "$TYPE" == "dataset" ]; then
        hf download "$REPO_ID" --repo-type dataset --local-dir "$DEST"
    else
        hf download "$REPO_ID" --local-dir "$DEST"
    fi
done

echo "=========================================="
echo "Downloads complete ($TARGET)."
df -h . | tail -n 1 | awk '{print "Disk free: " $4 " of " $2}'
echo "=========================================="
