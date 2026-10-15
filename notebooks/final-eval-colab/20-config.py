# 1. Final evaluation configuration (edit, then Runtime > Run all)

HUB_REPO = '2l3c34o1/text2cypher-checkpoints'
GITHUB_REPO = 'https://github.com/Cxmrykk/capstone.git'
GIT_REF = ''                  # Optional tag/commit to pin the code version (blank = latest default branch)

# Experiments trained on Kaggle: (model, schema variant, config).
# Each gets one fine-tuned job and, if RUN_ZERO_SHOT, one zero-shot job.
EXPERIMENTS = [
    ('gemma-4-e2b', 'enhanced', 'configs/gemma-4-e2b.yaml'),
    ('gemma-4-e2b', 'pruned', 'configs/gemma-4-e2b-pruned.yaml'),
    ('qwen3.5-2b', 'enhanced', 'configs/qwen3.5-2b.yaml'),
    ('qwen3.5-2b', 'pruned', 'configs/qwen3.5-2b-pruned.yaml'),
    ('granite-4.2-3b', 'enhanced', 'configs/granite-4.2-3b.yaml'),
]

SPLIT = 'test'
EVAL_LIMIT = 500              # Seeded sample shared by every job; None = full split (4,833 rows, many hours)
RUN_ZERO_SHOT = True          # Base model without adapter, same prompts and rows
RUN_EXECUTION = True          # Execution accuracy against the Neo4j Labs demo server
RUN_SYNTAX_CHECK = True       # EXPLAIN-based syntax/schema validity
SKIP_COMPLETED_JOBS = True    # Reuse job outputs already on the Hub (resume after a dead runtime)
MAX_SESSION_MINUTES = 660     # Stop starting new jobs after this; Run all again to continue

BOOTSTRAP_SAMPLES = 2000      # Resamples for bootstrap confidence intervals
CI_LEVEL = 0.95
STATS_SEED = 20261014

RUN_GGUF_CPU_BENCH = False    # Optional: Q4_K_M GGUF on the Colab CPU via llama.cpp (slow on 2 vCPUs)
GGUF_BENCH_LIMIT = 50         # A subset of the EVAL_LIMIT rows (same seed)

UPLOAD_TO_HUB = True          # Per-job outputs and the final results bundle
SAVE_TO_DRIVE = False         # Also copy the results bundle to Google Drive
DOWNLOAD_ZIP = True           # Offer the results bundle as a browser download

# Optional reference numbers copied by hand from published papers (e.g. Ozsoy et al., 2025).
# They are kept in a separate table and never mixed with measured results. Example entry:
# {'source': 'Ozsoy et al. (2025)', 'model': 'GPT-4o (zero-shot)', 'metric': 'google_bleu',
#  'value': <number from the paper>, 'notes': 'figure/table reference, different sample and protocol'}
PUBLISHED_BASELINES = []

NTFY_TOPIC = 'cxmrykk'        # Blank disables push notifications
NTFY_SERVER = 'https://ntfy.sh'
