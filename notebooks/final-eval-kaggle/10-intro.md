# Text2Cypher Toolkit: Final Evaluation on Kaggle (T4 x2)

Kaggle counterpart of `final-evaluation.ipynb`. Same protocol and seeded rows. Jobs are reused from the Hub only when they ran on the same commit and platform (see the re-run plan below), so the bundle is a single-protocol result.

### How the two GPUs are used
- **GPU workers (one per T4):** pull jobs from a shared queue and run `app.py predict` with `CUDA_VISIBLE_DEVICES` set to their own GPU. Each process sees exactly one device.
- **Evaluation workers (CPU / network):** run `app.py evaluate --execute --validate-syntax` as soon as a prediction file is ready, so a GPU never waits on the Neo4j demo server.
- Fine-tuned jobs are queued first. The second GPU worker starts `WORKER_STAGGER_S` seconds after the first, so model loading does not compete for disk and the 4 vCPUs.

### Protocol
- `EVAL_LIMIT` rows from the test split, seed 3407, identical for every job.
- Fine-tuned (and cross-schema) jobs must log `Adapter verified` (history/027.md). Zero-shot jobs must not load an adapter.
- Greedy decoding; batch size and `max_new_tokens` from each config. Prompts are fitted to the training sequence budget (history/030.md).

### One-time setup per Kaggle account
1. Verify the account by phone (GPU and internet).
2. **Settings > Accelerator > GPU T4 x2** (not P100). **Settings > Internet > On.**
3. **Add-ons > Secrets:** add `HF_TOKEN` (read access to the checkpoint repo, write access to upload results) and attach it.

### Running
- Use **Save Version > Save & Run All (Commit)**.
- Lane output is not streamed (interleaved streams are unreadable). A heartbeat prints every `HEARTBEAT_S` seconds with each worker's job, step, latest log line, GPU memory and queue sizes.
- Each finished job uploads to `final-eval/<eval_tag>/jobs/<job_id>/` in one commit. If the session ends early, commit a new version: finished jobs on the same commit and platform are reused.
- Logs and the results bundle are written to `/kaggle/working` and saved with the version; the bundle is also uploaded to `final-eval/<eval_tag>/thesis/<session>/`.

### Caveat
- Latency (`s/item`) for jobs run here is measured with two predictions sharing 4 vCPUs. Each job records `platform` and `concurrent_gpu_workers`.
