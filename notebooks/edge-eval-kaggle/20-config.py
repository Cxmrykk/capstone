# 1. Edge evaluation configuration (edit, then Save Version > Save & Run All)
import time

PIPELINE_T0 = time.time()  # Session time budget is measured from here

HUB_REPO = '2l3c34o1/text2cypher-checkpoints'
GITHUB_REPO = 'https://github.com/Cxmrykk/capstone.git'
GIT_REF = ''                  # Pin for thesis runs (tag or commit); blank is warned and flagged
LLAMA_CPP_REPO = 'https://github.com/ggml-org/llama.cpp'
LLAMA_CPP_REF = ''            # Tag or commit. Blank: the commit recorded by the first session, else latest

REFERENCE_EVAL_TAG = 'test-n500-seed3407-r030'   # GPU jobs written by final-eval-kaggle (history/030.md)
EDGE_TAG_SUFFIX = 'e032'                         # Hub namespace: edge-eval/<edge_tag>/
REUSE_SAME_COMMIT_ONLY = True # Reuse Hub work only from this code commit (llama.cpp commit, CPU and rows always must match)
SKIP_COMPLETED_JOBS = True

# Experiments trained on Kaggle: (model, schema variant, config).
EXPERIMENTS = [
    ('gemma-4-e2b', 'enhanced', 'configs/gemma-4-e2b.yaml'),
    ('gemma-4-e2b', 'pruned', 'configs/gemma-4-e2b-pruned.yaml'),
    ('qwen3.5-2b', 'enhanced', 'configs/qwen3.5-2b.yaml'),
    ('qwen3.5-2b', 'pruned', 'configs/qwen3.5-2b-pruned.yaml'),
    ('granite-4.2-3b', 'enhanced', 'configs/granite-4.2-3b.yaml'),
]

SPLIT = 'test'
REFERENCE_LIMIT = 500         # Reference EVAL_LIMIT; the rows are re-derived and checked against the Hub jobs
EDGE_LIMIT = 100              # Rows scored on the CPU: a seeded subset of the reference rows
EDGE_SUBSET = 'random'        # 'random' or 'stratified' (keeps the reference share of rows with a database)
EDGE_SUBSET_SEED = 20261015

# Export: merge -> f16 GGUF -> quantise (only for runs with unfinished work in this session)
MERGE_DTYPE = 'float32'       # Exact LoRA merge before conversion; the GGUF intermediate is f16
KEEP_MERGED = False           # Delete merged weights after conversion (disk)
KEEP_F16 = False              # Delete the f16 GGUF after quantising unless F16 is benchmarked
UPLOAD_GGUF = False           # Upload GGUFs (UPLOAD_GGUF_QUANTS) to edge-eval/<edge_tag>/gguf/
UPLOAD_GGUF_QUANTS = ['Q4_K_M']
EXPORT_MINUTES_ESTIMATE = 30  # Per run (merge + convert + quantise); no export starts with less time left

# Accuracy jobs on the CPU (queued by priority; later ones run only if time remains)
ACCURACY_QUANT = 'Q4_K_M'     # Priority 0: every run. Priority 1: cross-schema (enhanced-trained, exact-match prompts)
RUN_CROSS_SCHEMA = True
EXTRA_ACCURACY_QUANTS = ['Q8_0']      # Priority 2: quantisation fidelity
EXTRA_QUANT_VARIANTS = ('enhanced',)  # ...for these schema variants only
EDGE_STOP_MODE = 'hf'         # 'hf': stop on EOS only, as the GPU jobs; 'config': also generation.stop_strings
DEFAULT_S_PER_ITEM = 30       # Planning estimate before anything has been measured

# Throughput and footprint (one run per model: speed does not depend on the adapter)
LATENCY_QUANTS = ['Q4_K_M', 'Q5_K_M', 'Q8_0', 'F16']
THREAD_SWEEP = [1, 2, 4]      # Capped at the CPUs available; the full count is always included
THREAD_SWEEP_QUANTS = ['Q4_K_M']      # Other quantisations are measured at the full thread count only
BENCH_PROMPT_TOKENS = 512     # llama-bench ppN
BENCH_GEN_TOKENS = 64         # llama-bench tgN
BENCH_REPS = 3
BENCH_TIMEOUT_S = 3600
TYPICAL_GEN_TOKENS = 48       # With the median edge prompt length, picks the fastest thread count for 'auto'
SERVER_THREADS = 'auto'       # 'auto' or an integer
CACHE_PROMPT = False          # True lets llama-server reuse the shared instruction prefix between queries
REQUEST_TIMEOUT_S = 900

# Scoring and statistics
RUN_EXECUTION = True
RUN_SYNTAX_CHECK = True
DEAD_DB_POLICY = 'flag'
BOOTSTRAP_SAMPLES = 2000
CI_LEVEL = 0.95
STATS_SEED = 20261015
EQUIV_MARGIN = 0.05           # TOST margin on CPU vs GPU execution-accuracy differences (absolute)

RUN_SUPPLEMENTARY = True      # Analyses of the GPU reference jobs that final-eval does not produce

SESSION_BUDGET_MINUTES = 720  # Kaggle session cap (12 h)
RESERVE_MINUTES = 45          # Kept for supplementary analysis, aggregation, figures and uploads

UPLOAD_TO_HUB = True
NTFY_TOPIC = 'cxmrykk'        # Blank disables push notifications
NTFY_SERVER = 'https://ntfy.sh'

SCRATCH_DIR = '/kaggle/temp'    # Repo, models, GGUFs (not saved; falls back to /tmp)
OUTPUT_DIR = '/kaggle/working'  # Logs and the results bundle (saved with the notebook version)

if EDGE_SUBSET not in ('random', 'stratified'):
    raise ValueError("EDGE_SUBSET must be 'random' or 'stratified'.")
if EDGE_STOP_MODE not in ('hf', 'config'):
    raise ValueError("EDGE_STOP_MODE must be 'hf' or 'config'.")
if EDGE_LIMIT > REFERENCE_LIMIT:
    raise ValueError('EDGE_LIMIT cannot exceed REFERENCE_LIMIT.')
print('Reference tag : ' + REFERENCE_EVAL_TAG)
print('Edge rows     : ' + str(EDGE_LIMIT) + ' of ' + str(REFERENCE_LIMIT) + ' (' + EDGE_SUBSET + ')')
print('Accuracy      : ' + ACCURACY_QUANT + ' (all runs)' + (', cross-schema' if RUN_CROSS_SCHEMA else '')
      + ', extra ' + str(EXTRA_ACCURACY_QUANTS) + ' for ' + str(EXTRA_QUANT_VARIANTS))
print('Latency       : ' + str(LATENCY_QUANTS) + ', threads ' + str(THREAD_SWEEP) + ' for ' + str(THREAD_SWEEP_QUANTS))
print('Budget        : ' + str(SESSION_BUDGET_MINUTES) + ' min (reserve ' + str(RESERVE_MINUTES) + ' min)')
