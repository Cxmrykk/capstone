# Text2Cypher Toolkit: Edge (CPU) Evaluation on Kaggle

Companion to `final-eval-kaggle.ipynb`. It measures edge (CPU) deployment and adds the analyses the GPU evaluation does not produce. Every result is written as structured tables for the thesis.

### What it measures
- **Export:** each final adapter is merged into the full-precision base model, converted to an f16 GGUF and quantised with a pinned llama.cpp build. Adapter verification, file sizes and SHA-256 hashes are recorded.
- **Throughput:** `llama-bench` prompt-processing (ppN) and generation (tgN) speed per model, quantisation and thread count.
- **Footprint:** `llama-server` load time, resident memory after load, peak resident memory and the first (cold) request.
- **Accuracy on the CPU:** GGUF predictions for every run (Q4_K_M), the cross-schema condition and extra quantisations if time allows, on a seeded subset of the 500 rows used by `final-eval`. Prompts are built by the same code as `app.py predict`; scoring uses `app.py evaluate --execute --validate-syntax`.
- **Fidelity:** paired comparison of every CPU job with the matching GPU (4-bit HF) job on identical rows, including the share of identical predictions.
- **Per-query latency:** prefill and decode time from the server's own timings.
- **Tokenizer parity:** prompt tokens counted by the GGUF tokenizer vs the training tokenizer (catches a doubled or missing BOS).
- **Supplementary analyses of the GPU reference jobs:**
  - metrics without the near-saturated `neo4jLabs_functional_cypher` source;
  - whether the exact-match filter keeps every element the gold query needs;
  - schema hallucination;
  - execution matches where the gold query returns no rows;
  - cross-model agreement;
  - sample composition.

### Setup (per Kaggle account)
1. **Settings > Accelerator > None** (CPU session; no GPU quota is used). **Settings > Internet > On.**
2. **Add-ons > Secrets:** `HF_TOKEN` with read access to the checkpoint repo and write access for uploads.
3. Pin `GIT_REF` (and optionally `LLAMA_CPP_REF`) in the configuration cell.

### Running and resuming
- **Save Version > Save & Run All (Commit).**
- Each finished job, benchmark and export record is uploaded to `edge-eval/<edge_tag>/` on the Hub in one commit.
- If the session ends early, commit a new version. Finished work is reused when the code commit, llama.cpp commit, CPU model and row set all match. Only runs with unfinished work are exported again.
- The first session records its llama.cpp commit on the Hub. Later sessions with a blank `LLAMA_CPP_REF` build that same commit.

### Caveats
- **Hardware:** timings come from the Kaggle CPU recorded in the manifest (`lscpu`), not from a specific edge device. The thread sweep shows how speed scales with fewer cores.
- **Quantisation:** adapters were trained against an NF4 base and are merged into the full-precision base before GGUF quantisation. The GPU vs CPU comparison measures both changes together.
