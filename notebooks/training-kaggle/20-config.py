# 1. Pipeline configuration (edit these values, then Save Version > Save & Run All)
import time

PIPELINE_T0 = time.time()  # Session time budget is measured from here

# One lane per GPU: (model, schema variant, GPU index). Standard instances:
#   kaggle-gemma.ipynb   : INSTANCE_LABEL = "gemma-pair"
#                          LANES = [("gemma-4-e2b", "enhanced", "0"), ("gemma-4-e2b", "pruned", "1")]
#   kaggle-qwen.ipynb    : INSTANCE_LABEL = "qwen-pair"
#                          LANES = [("qwen3.5-2b", "enhanced", "0"), ("qwen3.5-2b", "pruned", "1")]
#   kaggle-granite.ipynb : INSTANCE_LABEL = "granite"
#                          LANES = [("granite-4.2-3b", "enhanced", "0")]
INSTANCE_LABEL = "gemma-pair"
LANES = [
    ("gemma-4-e2b", "enhanced", "0"),
    ("gemma-4-e2b", "pruned", "1"),
]

HUB_REPO = "2l3c34o1/text2cypher-checkpoints"
GITHUB_REPO = "https://github.com/Cxmrykk/capstone.git"

SESSION_BUDGET_MINUTES = 720     # Kaggle GPU session cap (12 h)
RESERVE_MINUTES = 90             # Setup before the training clock + upload + predict/eval/export
TIME_LIMIT_MINUTES = 600         # Upper bound on training wall-clock time per session
SKIP_TRAINING_IF_FINAL_EXISTS = True
LANE_STAGGER_S = 90              # Delay between lane starts (CPU-heavy loading/tokenization)

PREDICT_LIMIT = 100
RUN_EXECUTION_EVAL = True
RUN_EXPORT = True
UPLOAD_GGUF = True
KEEP_MERGED = False              # Delete merged fp16 weights after GGUF conversion (disk)
UPLOAD_RESULTS_TO_HUB = True     # One commit per lane at the end; logs are never uploaded

HEARTBEAT_S = 300
NTFY_TOPIC = "cxmrykk"           # Blank disables notifications
NTFY_SERVER = "https://ntfy.sh"

SCRATCH_DIR = "/kaggle/temp"     # Repo, models, artifacts (not saved; falls back to /tmp)
OUTPUT_DIR = "/kaggle/working"   # Logs and small results (saved with the notebook version)

CONFIGS = {
    ("gemma-4-e2b", "enhanced"): "configs/gemma-4-e2b.yaml",
    ("gemma-4-e2b", "pruned"): "configs/gemma-4-e2b-pruned.yaml",
    ("qwen3.5-2b", "enhanced"): "configs/qwen3.5-2b.yaml",
    ("qwen3.5-2b", "pruned"): "configs/qwen3.5-2b-pruned.yaml",
    ("granite-4.2-3b", "enhanced"): "configs/granite-4.2-3b.yaml",
}
DOWNLOAD_TARGETS = {"gemma-4-e2b": "gemma", "qwen3.5-2b": "qwen", "granite-4.2-3b": "granite"}

LANE_SPECS = []
for _model, _variant, _gpu in LANES:
    if (_model, _variant) not in CONFIGS:
        raise ValueError(f"No config for {_model} / {_variant} (Granite is enhanced-only).")
    LANE_SPECS.append({
        "key": f"{_model}-{_variant}",
        "model": _model,
        "variant": _variant,
        "gpu": str(_gpu),
        "config": CONFIGS[(_model, _variant)],
        "download_target": DOWNLOAD_TARGETS[_model],
    })
if not LANE_SPECS:
    raise ValueError("LANES is empty.")
if len(LANE_SPECS) > 2:
    raise ValueError("Kaggle T4 x2 has two GPUs; use at most two lanes.")
if len({lane["key"] for lane in LANE_SPECS}) != len(LANE_SPECS):
    raise ValueError("Each lane must be a different model/variant.")
if len({lane["gpu"] for lane in LANE_SPECS}) != len(LANE_SPECS):
    raise ValueError("Each lane needs its own GPU index.")

# Sequential downloads (history/006.md); a shared base model is downloaded once.
DOWNLOAD_ORDER = list(dict.fromkeys(lane["download_target"] for lane in LANE_SPECS))
# Only the Qwen configs use Unsloth.
INSTALL_UNSLOTH = any(lane["model"] == "qwen3.5-2b" for lane in LANE_SPECS)

print(f"Instance         : {INSTANCE_LABEL}")
for lane in LANE_SPECS:
    print(f"  Lane GPU {lane['gpu']}     : {lane['key']} ({lane['config']})")
print(f"Downloads        : {', '.join(DOWNLOAD_ORDER)}")
print(f"Install Unsloth  : {INSTALL_UNSLOTH}")
print(f"Hub repo         : {HUB_REPO} (checkpoints; results={UPLOAD_RESULTS_TO_HUB}; logs never)")
print(f"Session budget   : {SESSION_BUDGET_MINUTES} min (reserve {RESERVE_MINUTES} min, "
      f"training cap {TIME_LIMIT_MINUTES} min)")
print(f"Notifications    : {NTFY_SERVER.rstrip('/')}/{NTFY_TOPIC.strip()}" if NTFY_TOPIC.strip()
      else "Notifications    : disabled")
