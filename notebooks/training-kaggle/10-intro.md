# Text2Cypher Toolkit: Kaggle Pipeline (parallel lanes)

Runs up to **two independent experiments in parallel**, one per T4 (a *lane*). Each lane is a separate process with its own GPU, config and Hub run: fine-tuning, inference, evaluation and GGUF export. Lanes share only the repo checkout, the dependency install and the base-model download.

Each lane keeps the single-GPU maths of the Colab runs (batch 1 x accumulation 16), so results stay comparable and Hub checkpoints resume across platforms.

### One-time setup per Kaggle account
1. Verify the account by phone (required for GPU and internet).
2. **Settings > Accelerator > GPU T4 x2.** Not P100.
3. **Settings > Internet > On.**
4. **Add-ons > Secrets:** add `HF_TOKEN` (write access) and attach it to this notebook.

### Running
- Set `INSTANCE_LABEL` and `LANES` in the configuration cell (the three standard instances are listed there).
- Use **Save Version > Save & Run All (Commit)** for unattended runs.
- Lane output is not streamed to the cell (two interleaved streams are unreadable). A heartbeat prints every `HEARTBEAT_S` seconds with each lane's step, latest log line, GPU memory and free RAM.
- Lane starts are staggered by `LANE_STAGGER_S` so model loading and tokenization do not compete for the 4 vCPUs at the same moment.
- Merge and GGUF export run one lane at a time (disk space, shared llama.cpp build). Merged fp16 weights are deleted after conversion unless `KEEP_MERGED` is set.

### Time budget
- Kaggle GPU sessions are capped at about 12 h. Each lane's training limit is the smaller of `TIME_LIMIT_MINUTES` and the session time left after `RESERVE_MINUTES`.
- A lane that pauses at its limit has its checkpoint on the Hub; it skips its remaining steps while the other lane continues. Commit a new version to resume. Lanes whose final adapter is on the Hub skip training.

### Logs, results and Hub commits
- Logs are **not** uploaded to the Hub (commit rate limits with several instances running). They are written to `OUTPUT_DIR/text2cypher-logs/<instance>/<session>/<lane>/` and saved with the notebook version.
- Each lane uploads its results folder to `results/<lane>/<session>/` in **one** commit when it ends (`UPLOAD_RESULTS_TO_HUB`). GGUF files are one commit each. Checkpoint sync is unchanged; training also mirrors `metrics.jsonl` to `runs/<run>/telemetry/<session>/` at most once an hour.

### Notifications
- With `NTFY_TOPIC` set, alerts go to `<NTFY_SERVER>/<NTFY_TOPIC>`: pipeline start, per-lane training start, step failures, time-limit pauses, per-lane completion with metrics, pipeline end, and kernel shutdown before the end.
- A hard kill at the session cap may not deliver an alert. ntfy.sh topics are public; only status text and log tails are sent.
