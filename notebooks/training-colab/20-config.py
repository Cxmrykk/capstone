# @title 1. Pipeline configuration { display-mode: "form" }
MODEL = "gemma-4-e2b"  # @param ["gemma-4-e2b", "qwen3.5-2b", "granite-4.2-3b"]
SCHEMA_VARIANT = "enhanced"  # @param ["enhanced", "pruned"]
HUB_REPO = "2l3c34o1/text2cypher-checkpoints"  # @param {type:"string"}
GITHUB_REPO = "https://github.com/Cxmrykk/capstone.git"  # @param {type:"string"}
TIME_LIMIT_MINUTES = 270  # @param {type:"integer"}
SKIP_TRAINING_IF_FINAL_EXISTS = True  # @param {type:"boolean"}
PREDICT_LIMIT = 100  # @param {type:"integer"}
RUN_EXECUTION_EVAL = True  # @param {type:"boolean"}
RUN_EXPORT = True  # @param {type:"boolean"}
UPLOAD_GGUF = True  # @param {type:"boolean"}
SAVE_LOGS_TO_DRIVE = False  # @param {type:"boolean"}
LOG_UPLOAD_INTERVAL_S = 300  # @param {type:"integer"}
NTFY_TOPIC = "cxmrykk"  # @param {type:"string"}
NTFY_SERVER = "https://ntfy.sh"  # @param {type:"string"}

CONFIGS = {
    ("gemma-4-e2b", "enhanced"): "configs/gemma-4-e2b.yaml",
    ("gemma-4-e2b", "pruned"): "configs/gemma-4-e2b-pruned.yaml",
    ("qwen3.5-2b", "enhanced"): "configs/qwen3.5-2b.yaml",
    ("qwen3.5-2b", "pruned"): "configs/qwen3.5-2b-pruned.yaml",
    ("granite-4.2-3b", "enhanced"): "configs/granite-4.2-3b.yaml",
}
DOWNLOAD_TARGETS = {"gemma-4-e2b": "gemma", "qwen3.5-2b": "qwen", "granite-4.2-3b": "granite"}

if (MODEL, SCHEMA_VARIANT) not in CONFIGS:
    print(f"Schema variant '{SCHEMA_VARIANT}' is not defined for {MODEL} "
          "(Granite is the enhanced-schema control); using 'enhanced'.")
    SCHEMA_VARIANT = "enhanced"

CONFIG = CONFIGS[(MODEL, SCHEMA_VARIANT)]
DOWNLOAD_TARGET = DOWNLOAD_TARGETS[MODEL]
# Only the Qwen configs use Unsloth. Skipping it for the other models saves
# install time and stops Unsloth patching transformers for no benefit.
INSTALL_UNSLOTH = MODEL == "qwen3.5-2b"

print(f"Model            : {MODEL}")
print(f"Schema variant   : {SCHEMA_VARIANT}")
print(f"Config           : {CONFIG}")
print(f"Download target  : {DOWNLOAD_TARGET}")
print(f"Install Unsloth  : {INSTALL_UNSLOTH}")
print(f"Hub repo         : {HUB_REPO}")
print(f"Notifications    : {NTFY_SERVER.rstrip('/')}/{NTFY_TOPIC.strip()}" if NTFY_TOPIC.strip()
      else "Notifications    : disabled")
