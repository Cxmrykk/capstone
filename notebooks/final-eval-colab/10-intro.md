# Text2Cypher Toolkit: Final Evaluation (thesis results)

Runs the final, reproducible test of every model trained on Kaggle and writes all results as structured data for the thesis.

### Protocol
- **Same rows for every job:** `EVAL_LIMIT` rows are drawn from the test split with the shared config seed (3407), so all jobs are scored on identical instances and can be compared pairwise.
- **Fine-tuned jobs:** each final adapter is pulled from `runs/<run_name>/final` on the Hub. A job is only scored if its predict log contains the `Adapter verified` line from the verified loader (history/027.md).
- **Zero-shot jobs:** the same base model, prompts, schema mode and rows, with no adapter.
- **Decoding:** greedy, with `max_new_tokens` and batch size taken from each config.
- **Metrics:** Google-BLEU, exact match, normalised exact match, executable rate, EXPLAIN validity, and execution accuracy against the Neo4j Labs demo server.
- **Optional GGUF check:** Q4_K_M GGUF inference on the Colab CPU (latency and quantisation effect).

### Outputs (`artifacts/thesis/<eval_tag>/`)
- **`tables/*.csv`:** headline table (also `.tex`), overall metrics with 95% confidence intervals, metrics by complexity, data source and database, error categories, paired significance tests (McNemar and bootstrap, Holm-corrected), schema-filtering token efficiency, training summary and curves, integrity checks, job registry and published baselines.
- **Per-row data:** `predictions_long.parquet` and `predictions_long.csv.gz`, one row per job and instance.
- **JSON:** `results.json` (all tables plus manifest) and `manifest.json` (commit, GPU, package versions, settings).
- **Figures:** `figures/*.png` (300 dpi) and `*.pdf`.
- **Delivery:** the bundle is zipped, uploaded to `final-eval/<eval_tag>/thesis/<session>/` on the Hub, and optionally copied to Drive and downloaded.

### How to use
1. Runtime > Change runtime type > **T4 GPU**.
2. Add a Colab secret `HF_TOKEN` with access to the checkpoint repo, and allow this notebook to use it.
3. Check the configuration cell, then **Runtime > Run all**.

### Resuming
- Each job uploads its outputs to `final-eval/<eval_tag>/jobs/<job_id>/` as soon as it finishes.
- If the runtime dies, open a fresh runtime and **Run all**: finished jobs are downloaded instead of recomputed.
- If an import error appears right after the dependency install, use Runtime > Restart session, then Run all. Downloads and finished jobs are reused.
