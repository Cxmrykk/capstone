#!/bin/bash
#
# Download the benchmark dataset and the base models.
#
#   bash download_data.sh          # dataset + Gemma 4 E2B + Qwen3.5-2B + Granite 4.2 3B
#   bash download_data.sh gemma    # dataset + Gemma 4 E2B only
#   bash download_data.sh qwen     # dataset + Qwen3.5-2B only
#   bash download_data.sh granite  # dataset + Granite 4.2 3B only
#   bash download_data.sh dataset  # dataset only
#
# Supported models: the two 2B models under study plus Granite 4.2 3B as a
# dense-transformer control. 4B models do not fit the 16 GB T4 used on Colab.
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
GRANITE="model ibm-granite/granite-4.2-3b data/granite-4.2-3b"

case "$TARGET" in
    all)     REPOS=("$DATASET" "$GEMMA" "$QWEN" "$GRANITE") ;;
    gemma)   REPOS=("$DATASET" "$GEMMA") ;;
    qwen)    REPOS=("$DATASET" "$QWEN") ;;
    granite) REPOS=("$DATASET" "$GRANITE") ;;
    dataset) REPOS=("$DATASET") ;;
    *)
        echo "Usage: bash download_data.sh [all|gemma|qwen|granite|dataset]"
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
