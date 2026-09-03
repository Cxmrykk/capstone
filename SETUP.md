# GCP Deep Learning VM Setup & Execution Guide

This document outlines the end-to-end workflow for deploying, training, and evaluating the Text2Cypher toolkit on a Google Cloud Platform (GCP) Virtual Machine. 

By migrating to an NVIDIA L4 or A100 GPU, the pipeline leverages Ampere/Ada architecture, natively supporting `bfloat16` precision. This eliminates previous OOM bottlenecks and numerical instabilities experienced on older Turing (T4) hardware.

## 1. Provisioning the GCP Instance

1. Navigate to **Compute Engine -> VM Instances** in the GCP Console.
2. Click **Create Instance**.
3. **Machine Configuration:**
   * **GPU:** Select 1x **NVIDIA L4** (24GB VRAM) or **NVIDIA A100** (40GB VRAM).
   * **Machine Type:** `g2-standard-8` (for L4) or `a2-highgpu-1g` (for A100).
4. **Boot Disk (Crucial):**
   * Click **Change** under Boot Disk.
   * **Operating System:** Deep Learning on Linux.
   * **Version:** Deep Learning VM with PyTorch 2.x and CUDA 12.x (Ubuntu 22.04).
   * **Size:** At least **100 GB** Standard Persistent Disk.
5. Check **"Allow HTTP/HTTPS traffic"**.
6. Create the instance.

## 2. Environment Setup

SSH into your new VM and execute the following commands to initialize the repository:

```bash
# 1. Clone the repository
git clone https://github.com/Cxmrykk/capstone.git
cd capstone

# 2. Setup Environment Variables
cp .env.example .env
nano .env  # PASTE YOUR HF_TOKEN HERE, save (Ctrl+O, Enter), and exit (Ctrl+X)

# 3. Source the environment variables
export $(grep -v '^#' .env | xargs)

# 4. Install Dependencies
pip install -r requirements.txt
pip install unsloth
```

## 3. Downloading Data & Models

The toolkit includes a high-performance download script that leverages `hf_transfer` to pull the `text2cypher-2024v1` dataset and base models directly to the local disk.

```bash
bash download_data.sh
```

## 4. Diagnostics & Hardware Verification

Verify that the system detects the L4/A100 GPU and has successfully defaulted to `bfloat16` precision.

```bash
python app.py doctor
```
*Expected Output: `bfloat16: Supported` and `Training Dtype: bfloat16`.*

## 5. Model Fine-Tuning (Background Execution)

Because SSH connections can drop, you should run the training loop inside a `tmux` session. This ensures training continues even if you close your laptop.

```bash
# Start a new tmux session
tmux new -s training

# Execute the training pipeline
python app.py train --config configs/gemma-4-e2b.yaml
```
*(To detach from the tmux session while it runs, press `Ctrl+B`, then release and press `D`. To reattach later, type `tmux attach -t training`).*

## 6. Inference (Cypher Generation)

Generate Cypher query predictions on the holdout test split using the newly trained adapter.

```bash
python app.py predict \
    --config configs/gemma-4-e2b.yaml \
    --adapter artifacts/runs/gemma-4-e2b-enhanced/final \
    --split test \
    --limit 100 \
    --out artifacts/predictions/gemma-4-e2b-test.jsonl
```

## 7. Evaluation

Score the model outputs across syntactic validity, translation fidelity (Google-BLEU), and logical execution against the remote Neo4j Labs demo databases.

```bash
python app.py evaluate \
    --predictions artifacts/predictions/gemma-4-e2b-test.jsonl \
    --execute \
    --validate-syntax
```

## 8. Export to Edge Format (GGUF)

Merge the LoRA weights back into the base model and quantize it to a 4-bit `GGUF` file for local edge deployment.

```bash
# 1. Merge LoRA adapter into full-precision base weights
python app.py export merge \
    --config configs/gemma-4-e2b.yaml \
    --adapter artifacts/runs/gemma-4-e2b-enhanced/final \
    --out artifacts/merged/gemma-4-e2b-enhanced

# 2. Build llama.cpp with CUDA support
BUILD_CUDA=1 bash scripts/build_llama_cpp.sh

# 3. Quantize the merged weights to Q4_K_M
python app.py export gguf \
    --merged artifacts/merged/gemma-4-e2b-enhanced \
    --quant Q4_K_M
```
