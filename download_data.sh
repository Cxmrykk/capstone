#!/bin/bash
set -e

# Load .env if present
if [ -f .env ]; then
    echo "Loading environment variables from .env..."
    export $(grep -v '^#' .env | xargs)
fi

# Enable high-performance Xet transfer engine
export HF_XET_HIGH_PERFORMANCE=1

# List of targets: "repo_type repo_id local_dir"
REPOS=(
    "dataset neo4j/text2cypher-2024v1 data/text2cypher-2024v1"
    "model google/gemma-4-E2B data/gemma-4-E2B"
    "model google/gemma-4-E4B data/gemma-4-E4B"
    "model Qwen/Qwen3.5-2B data/Qwen3.5-2B"
    "model Qwen/Qwen3.5-4B data/Qwen3.5-4B"
)

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
echo "All datasets and models downloaded successfully!"
echo "=========================================="