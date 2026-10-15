## Re-run plan (history/030.md)

The thesis bundle must come from **one commit and one platform**. Set `GIT_REF` to a tag, keep `EVAL_TAG_SUFFIX = 'r030'` (a fresh Hub namespace) and leave `REUSE_SAME_COMMIT_ONLY` / `REUSE_SAME_PLATFORM_ONLY` on.

Jobs that must run under that commit on Kaggle (12 GPU jobs):

| Run | fine-tuned | zero-shot | cross-schema |
|---|---|---|---|
| gemma-4-e2b-enhanced | yes (previously run on Colab) | yes | yes (exact-match prompts) |
| gemma-4-e2b-exact-match | yes (previously run on Colab) | yes | - |
| qwen3.5-2b-enhanced | yes | yes | yes (exact-match prompts) |
| qwen3.5-2b-exact-match | yes | yes | - |
| granite-4.2-3b-enhanced | yes | yes | - |

Every earlier job is invalidated for three reasons: inference prompts are now trimmed to the training budget, Neo4j outages were previously cached as query failures, and earlier jobs mixed commits and platforms. No retraining is needed.
