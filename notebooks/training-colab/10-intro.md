# Text2Cypher Toolkit: Set-and-Go Pipeline

Runs the complete pipeline for **one** model: bootstrap, download, diagnostics, LoRA fine-tuning, inference, evaluation and GGUF export.

### How to use
1. Add a Colab Secret named `HF_TOKEN` (with **Write** permissions) and allow this notebook to access it.
2. Choose the model and options in the **Pipeline configuration** form below.
3. `Runtime > Run all`.

### Logs and recovery
- Every step streams to its cell **and** to `LOG_DIR/NN-<step>.log`. Logs are mirrored to the Hub repo under `logs/<model>-<variant>/<session>/` every `LOG_UPLOAD_INTERVAL_S` seconds and when each step ends, so they survive a dead runtime. Enable `SAVE_LOGS_TO_DRIVE` to also write them live to Google Drive (needs one authorisation click).
- Training checkpoints sync to the Hub. If training reaches `TIME_LIMIT_MINUTES`, the notebook stops with a message: open a fresh runtime and **Run all** again to resume. Once a final adapter exists on the Hub, later runs skip training.
- Predictions, evaluation reports and training metrics are uploaded under `results/<model>-<variant>/<session>/`.
- A failing step raises an error, which stops **Run all** so later steps never run on missing inputs. GGUF export is best-effort and never stops the run.

### Notifications
- With `NTFY_TOPIC` set, push alerts go to `<NTFY_SERVER>/<NTFY_TOPIC>` (subscribe in the ntfy app or browser) when the pipeline starts, training starts, a step fails, training pauses at `TIME_LIMIT_MINUTES`, the pipeline finishes, or the kernel shuts down before finishing.
- A hard VM reclaim by Colab (usage limit, idle timeout) kills the machine without warning, so no alert can be sent. If the expected time in the *training started* alert passes in silence, assume the runtime died.
- ntfy.sh topics are public. Only status text and log tails are sent, never tokens.
