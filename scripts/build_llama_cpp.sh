#!/usr/bin/env bash
#
# Build llama.cpp for GGUF conversion and CPU inference.
#
#   Laptop (Debian 13):  bash scripts/build_llama_cpp.sh
#   Colab (CUDA build):  BUILD_CUDA=1 bash scripts/build_llama_cpp.sh
#
# Installs into vendor/llama.cpp inside the repo by default.
set -euo pipefail

BOLD=$'\033[1m'; RESET=$'\033[0m'
say() { echo "${BOLD}==>${RESET} $*"; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${LLAMA_CPP_DIR:-${REPO_ROOT}/vendor/llama.cpp}"
BUILD_CUDA="${BUILD_CUDA:-0}"
JOBS="${JOBS:-$(nproc)}"

if ! command -v cmake > /dev/null 2>&1 || ! command -v git > /dev/null 2>&1; then
    say "Installing build dependencies"
    if [[ $EUID -eq 0 ]]; then SUDO=""; else SUDO="sudo"; fi
    $SUDO apt-get update
    $SUDO apt-get install -y build-essential cmake git libcurl4-openssl-dev
fi

if [[ -d "${DEST}/.git" ]]; then
    say "Updating existing checkout at ${DEST}"
    git -C "${DEST}" pull --ff-only
else
    say "Cloning llama.cpp into ${DEST}"
    mkdir -p "$(dirname "${DEST}")"
    git clone --depth 1 https://github.com/ggml-org/llama.cpp "${DEST}"
fi

CMAKE_FLAGS=(-B "${DEST}/build" -S "${DEST}" -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=ON)

if [[ "${BUILD_CUDA}" == "1" ]]; then
    say "Configuring with CUDA support"
    CMAKE_FLAGS+=(-DGGML_CUDA=ON)
else
    say "Configuring CPU build with native optimisations"
    CMAKE_FLAGS+=(-DGGML_NATIVE=ON)
fi

cmake "${CMAKE_FLAGS[@]}"
cmake --build "${DEST}/build" --config Release -j "${JOBS}"

say "Installing Python requirements for the GGUF converter"
python3 -m pip install --quiet --upgrade -r "${DEST}/requirements/requirements-convert_hf_to_gguf.txt" \
    || python3 -m pip install --quiet --upgrade "numpy" "sentencepiece" "gguf" "protobuf" "torch" "transformers"

cat <<EOF

$(say "Build complete")

Binaries:   ${DEST}/build/bin
Converter:  ${DEST}/convert_hf_to_gguf.py

Add this to your shell profile so the toolkit finds it automatically:

  ${BOLD}export LLAMA_CPP_DIR="${DEST}"${RESET}

Quick check:

  ${BOLD}${DEST}/build/bin/llama-cli --version${RESET}
EOF
