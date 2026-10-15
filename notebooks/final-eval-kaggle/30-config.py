# 1. Final evaluation configuration (edit, then Save Version > Save & Run All)
import time

PIPELINE_T0 = time.time()  # Session time budget is measured from here

HUB_REPO = '2l3c34o1/text2cypher-checkpoints'
GITHUB_REPO = 'https://github.com/Cxmrykk/capstone.git'
GIT_REF = ''                  # Pin for thesis runs (tag or commit, e.g. 'thesis-r030'); blank is warned and flagged
EVAL_TAG_SUFFIX = 'r030'      # Fresh Hub namespace: outputs from before history/030.md are never reused
REUSE_SAME_COMMIT_ONLY = True     # Reuse a Hub job only if it ran on this exact commit
REUSE_SAME_PLATFORM_ONLY = True   # ...and on this platform
RUN_CROSS_SCHEMA = True       # Enhanced-trained adapter scored with exact-match prompts
DEAD_DB_POLICY = 'flag'       # 'flag' or 'exclude' databases whose gold queries all fail
EQUIV_MARGIN = 0.05           # TOST margin on execution-accuracy differences (absolute)
PRUNED_RATIO_THRESHOLD = 0.95 # Rows where the filter kept < 95% of the schema characters

# Experiments trained on Kaggle: (model, schema variant, config).
# Each gets one fine-tuned job and, if RUN_ZERO_SHOT, one zero-shot job.
EXPERIMENTS = [
    ('gemma-4-e2b', 'enhanced', 'configs/gemma-4-e2b.yaml'),
    ('gemma-4-e2b', 'pruned', 'configs/gemma-4-e2b-pruned.yaml'),
    ('qwen3.5-2b', 'enhanced', 'configs/qwen3.5-2b.yaml'),
    ('qwen3.5-2b', 'pruned', 'configs/qwen3.5-2b-pruned.yaml'),
    ('granite-4.2-3b', 'enhanced', 'configs/granite-4.2-3b.yaml'),
]

GPU_IDS = ['0', '1']          # One prediction worker per GPU (T4 x2)
N_EVAL_WORKERS = 2            # Concurrent `app.py evaluate` processes (CPU + Neo4j network)
WORKER_STAGGER_S = 60         # Delay before the second GPU worker starts

SPLIT = 'test'
EVAL_LIMIT = 500              # None = full split
RUN_ZERO_SHOT = True          # Base model without adapter, same prompts and rows
RUN_EXECUTION = True          # Execution accuracy against the Neo4j Labs demo server
RUN_SYNTAX_CHECK = True       # EXPLAIN-based syntax/schema validity
SKIP_COMPLETED_JOBS = True    # Reuse job outputs already on the Hub (same commit and platform only)

SESSION_BUDGET_MINUTES = 720  # Kaggle GPU session cap (12 h)
RESERVE_MINUTES = 60          # Kept for aggregation, figures and uploads at the end
MAX_JOB_MINUTES = 60          # A GPU worker does not start a job unless this much time remains before the reserve

BOOTSTRAP_SAMPLES = 2000      # Resamples for bootstrap confidence intervals
CI_LEVEL = 0.95
STATS_SEED = 20261014

RUN_GGUF_CPU_BENCH = False    # Optional: Q4_K_M GGUF on the Kaggle CPU via llama.cpp (runs after the GPU jobs)
GGUF_BENCH_LIMIT = 50         # A subset of the EVAL_LIMIT rows (same seed)

UPLOAD_TO_HUB = True          # Per-job outputs and the final results bundle
HEARTBEAT_S = 300

# Optional reference numbers copied by hand from published papers (e.g. Ozsoy et al., 2025).
# They are kept in a separate table and never mixed with measured results. Example entry:
# {'source': 'Ozsoy et al. (2025)', 'model': 'GPT-4o (zero-shot)', 'metric': 'google_bleu',
#  'value': <number from the paper>, 'notes': 'figure/table reference, different sample and protocol'}
PUBLISHED_BASELINES = []

NTFY_TOPIC = 'cxmrykk'        # Blank disables push notifications
NTFY_SERVER = 'https://ntfy.sh'

SCRATCH_DIR = '/kaggle/temp'  # Repo, models, artifacts (not saved; falls back to /tmp)
OUTPUT_DIR = '/kaggle/working'  # Logs and the results bundle (saved with the notebook version)

if len(GPU_IDS) != len(set(GPU_IDS)):
    raise ValueError('GPU_IDS must not repeat a GPU.')
print('GPU workers  : ' + ', '.join('GPU ' + g for g in GPU_IDS))
print('Eval workers : ' + str(N_EVAL_WORKERS))
print('Eval limit   : ' + str(EVAL_LIMIT))
print('Budget       : ' + str(SESSION_BUDGET_MINUTES) + ' min (reserve ' + str(RESERVE_MINUTES)
      + ' min, no new job within ' + str(MAX_JOB_MINUTES) + ' min of the reserve)')
