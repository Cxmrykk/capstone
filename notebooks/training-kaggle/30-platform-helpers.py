# 2. Platform checks, helpers, notifications and secrets
import atexit
import codecs
import datetime
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path

CR, LF = chr(13), chr(10)

# --------------------------------------------------------------------------- #
# Platform checks (fail fast, before anything slow runs)
# --------------------------------------------------------------------------- #
ON_KAGGLE = Path("/kaggle").is_dir() or bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))
if not ON_KAGGLE:
    raise RuntimeError("This notebook targets Kaggle. On Google Colab use google-colab.ipynb.")
RUN_TYPE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "unknown")  # Interactive | Batch (commit)

# Shared steps (doctor, previews) see only the first lane's GPU. Every lane step
# overrides CUDA_VISIBLE_DEVICES with its own GPU, so each training process sees
# exactly one device and HF Trainer never wraps the model in nn.DataParallel.
os.environ["CUDA_VISIBLE_DEVICES"] = LANE_SPECS[0]["gpu"]


def _query_gpus():
    """Returns {index: name} for physical GPUs (nvidia-smi ignores CUDA_VISIBLE_DEVICES)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        ).stdout
    except Exception:
        return {}
    gpus = {}
    for line in out.splitlines():
        if "," in line:
            idx, name = line.split(",", 1)
            gpus[idx.strip()] = name.strip()
    return gpus


GPUS = _query_gpus()
if not GPUS:
    raise RuntimeError("No GPU visible. Settings > Accelerator > GPU T4 x2 "
                       "(requires a phone-verified Kaggle account).")
for lane in LANE_SPECS:
    gpu_name = GPUS.get(lane["gpu"])
    if gpu_name is None:
        raise RuntimeError(f"Lane {lane['key']} wants GPU {lane['gpu']}, but only GPUs "
                           f"{sorted(GPUS)} exist. Two lanes need the GPU T4 x2 accelerator.")
    if "P100" in gpu_name:
        raise RuntimeError("Tesla P100 (Pascal, sm_60) selected. Recent PyTorch CUDA builds are "
                           "dropping Pascal support and the configs are tuned for a T4. "
                           "Switch the accelerator to GPU T4 x2.")
    if "T4" not in gpu_name:
        print(f"WARNING: lane {lane['key']} expected a Tesla T4 on GPU {lane['gpu']}, "
              f"found {gpu_name}. Configs are sized for 16 GB fp16.")
    lane["gpu_name"] = gpu_name


def _internet_ok(url="https://huggingface.co", timeout=15):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status < 500
    except Exception:
        return False


if not _internet_ok():
    raise RuntimeError("No internet access. Settings > Internet > On "
                       "(requires a phone-verified Kaggle account).")


def _writable_dir(path):
    try:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        probe = p / ".t2c-write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except Exception:
        return False


def _free_gb(path):
    try:
        return shutil.disk_usage(path).free / 1e9
    except Exception:
        return float("nan")


WORK_ROOT = Path(SCRATCH_DIR) if _writable_dir(SCRATCH_DIR) else Path("/tmp")
REPO_DIR = WORK_ROOT / "capstone"
# Each lane runs from its own working directory, so per-process caches written
# to the cwd (e.g. Unsloth's compiled cache) never collide between lanes.
LANES_DIR = WORK_ROOT / "lanes"

# --------------------------------------------------------------------------- #
# Session, log locations and secrets
# --------------------------------------------------------------------------- #
SESSION = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S") + "-kaggle"
LOG_DIR = Path(OUTPUT_DIR) / "text2cypher-logs" / INSTANCE_LABEL / SESSION
LOG_DIR.mkdir(parents=True, exist_ok=True)

try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN loaded from Kaggle Secrets.")
except Exception as exc:
    # Scratch storage is wiped when the session ends, so training without Hub
    # sync would lose all progress. Refuse to start instead.
    raise RuntimeError(
        f"Could not read the 'HF_TOKEN' Kaggle secret ({type(exc).__name__}). "
        "Add-ons > Secrets: add HF_TOKEN (write access) and attach it to this notebook."
    ) from exc

os.environ["T2C_HUB_REPO"] = HUB_REPO
# Reduce PyTorch memory fragmentation on the 16 GB T4.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Child Python processes flush output immediately, so logs stay current.
os.environ["PYTHONUNBUFFERED"] = "1"

# --------------------------------------------------------------------------- #
# Pipeline state
# --------------------------------------------------------------------------- #
STATUS = {
    "platform": "kaggle",
    "run_type": RUN_TYPE,
    "instance": INSTANCE_LABEL,
    "session": SESSION,
    "gpus": GPUS,
    # None while running; then completed | failed | interrupted | runtime-ended.
    "outcome": None,
    "shared": {"current_step": None, "step_started": None, "last_line": "", "steps": []},
    "lanes": {
        lane["key"]: {
            "model": lane["model"], "variant": lane["variant"], "gpu": lane["gpu"],
            "gpu_name": lane["gpu_name"], "config": lane["config"], "run_name": None,
            "current_step": None, "step_started": None, "last_line": "", "steps": [],
            "train_limit": None, "export": None,
            # None while running; then completed | paused | no-time | failed | interrupted.
            "outcome": None,
        }
        for lane in LANE_SPECS
    },
}
_status_lock = threading.RLock()
_hub_lock = threading.Lock()
_hub_state = {"repo_ready": False}
STOP_EVENT = threading.Event()
ACTIVE_PROCS = {}
_procs_lock = threading.Lock()


def _scope(lane_key):
    return STATUS["lanes"][lane_key] if lane_key else STATUS["shared"]


def save_status():
    """Writes pipeline_status.json locally (saved with the version; never uploaded)."""
    with _status_lock:
        try:
            text = json.dumps(STATUS, indent=2, default=str)
        except RuntimeError:
            return  # A lane thread mutated STATUS mid-dump; the next save catches up.
        (LOG_DIR / "pipeline_status.json").write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------- #
# ntfy push notifications
# --------------------------------------------------------------------------- #
NTFY_URL = f"{NTFY_SERVER.rstrip('/')}/{NTFY_TOPIC.strip()}" if NTFY_TOPIC.strip() else None


def _title(label, text):
    return f"T2C Kaggle {label}: {text}"


def notify(title, message, priority="default", tags=(), click=None):
    """Best-effort push notification via ntfy. Never raises.

    The topic is public: anyone who knows its name can read it, so only
    status text and log tails are sent, never tokens.
    """
    if not NTFY_URL:
        return False
    headers = {
        # HTTP header values must be latin-1; keep the title plain ASCII.
        "Title": title.encode("ascii", "replace").decode("ascii"),
        "Priority": str(priority),
    }
    if tags:
        headers["Tags"] = ",".join(tags)
    if click:
        headers["Click"] = click
    # ntfy turns bodies over 4096 bytes into attachments; keep the end of the text.
    body = message if len(message) <= 3000 else "..." + message[-3000:]
    req = urllib.request.Request(NTFY_URL, data=body.encode("utf-8"),
                                 headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
        return True
    except Exception as exc:
        print(f"[ntfy] Notification failed: {exc}", file=sys.stderr)
        return False


def _log_tail(path, n=15):
    """Returns the last n lines of a step log (empty string if unreadable)."""
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return ""
    return LF.join(lines[-n:])


# --------------------------------------------------------------------------- #
# Hub uploads (results and GGUF only: one commit per call, never logs)
# --------------------------------------------------------------------------- #
def hub_upload(local_path, path_in_repo, message):
    """Uploads a file or folder to the Hub repo in a single commit. Never raises."""
    if not HUB_REPO or not os.environ.get("HF_TOKEN"):
        return False
    local_path = Path(local_path)
    if not local_path.exists():
        return False
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ["HF_TOKEN"])
        with _hub_lock:
            if not _hub_state["repo_ready"]:
                api.create_repo(HUB_REPO, repo_type="model", private=True, exist_ok=True)
                _hub_state["repo_ready"] = True
            if local_path.is_dir():
                api.upload_folder(folder_path=str(local_path), path_in_repo=path_in_repo,
                                  repo_id=HUB_REPO, repo_type="model", commit_message=message)
            else:
                api.upload_file(path_or_fileobj=str(local_path), path_in_repo=path_in_repo,
                                repo_id=HUB_REPO, repo_type="model", commit_message=message)
        return True
    except Exception as exc:
        print(f"[hub] Upload of {local_path.name} failed: {exc}", file=sys.stderr)
        return False


def archive_results(lane_key, paths):
    """Copies result files into the lane's results folder (saved output, no upload)."""
    dest = LOG_DIR / lane_key / "results"
    dest.mkdir(parents=True, exist_ok=True)
    for p in map(Path, paths):
        if not p.exists():
            continue
        if p.is_dir():
            shutil.copytree(p, dest / p.name, dirs_exist_ok=True)
        else:
            shutil.copy2(p, dest / p.name)


def publish_results(lane_key):
    """Uploads the lane's results folder plus its status to the Hub in ONE commit."""
    dest = LOG_DIR / lane_key / "results"
    dest.mkdir(parents=True, exist_ok=True)
    with _status_lock:
        lane_status = json.dumps(STATUS["lanes"][lane_key], indent=2, default=str)
    (dest / "lane_status.json").write_text(lane_status, encoding="utf-8")
    if not UPLOAD_RESULTS_TO_HUB:
        return False
    ok = hub_upload(dest, f"results/{lane_key}/{SESSION}", f"[{lane_key}] results {SESSION}")
    print(f"[{lane_key}] Results upload: {'OK' if ok else 'FAILED (kept in saved output)'}", flush=True)
    return ok


# --------------------------------------------------------------------------- #
# Step runner
# --------------------------------------------------------------------------- #
def _clean_lines(buf, text):
    """Splits streamed output into finished lines. Only the last carriage-return
    segment of each line is kept, so tqdm redraws do not bloat the log file."""
    text = buf + text.replace(CR + LF, LF)
    parts = text.split(LF)
    lines = [p.split(CR)[-1] for p in parts[:-1]]
    return lines, parts[-1].split(CR)[-1]


TRAIN_EXIT_HINTS = {
    2: "The config fingerprint differs from the remote checkpoint. Change run_name, or delete the old run on the Hub.",
    3: "Training loss became non-finite (fp16 overflow); no final adapter was saved.",
    4: "Training reached its time budget and checkpointed to the Hub. Commit a new version "
       "(Save Version > Save & Run All): training resumes automatically.",
    5: "Training finished but the final adapter did NOT reach the Hub (it exists only in this session). "
       "The lane retries once with `checkpoint push-final`; if that fails, the next session resumes from "
       "the end-of-training checkpoint and writes the final adapter again.",
}


def _interrupt_proc(proc):
    """Sends SIGINT to a step's process group (training writes an emergency checkpoint)."""
    try:
        os.killpg(proc.pid, signal.SIGINT)
        return proc.wait(timeout=180)
    except Exception:
        proc.kill()
        return proc.wait()


def _stop_all_procs():
    """Signals every running step; the owning threads collect the exit codes."""
    with _procs_lock:
        procs = list(ACTIVE_PROCS.values())
    for proc in procs:
        try:
            os.killpg(proc.pid, signal.SIGINT)
        except Exception:
            pass


def run_step(name, cmd, lane=None, allow_fail=False, env=None, cwd=None, echo=True):
    """Runs a shell command and writes its output to a log file.

    Shared steps (lane=None) also stream to the cell. Lane steps run in worker
    threads with echo=False; the heartbeat in cell 7 shows their progress.
    """
    if STOP_EVENT.is_set():
        raise KeyboardInterrupt(f"Stop requested before step '{name}'.")
    scope = _scope(lane)
    label = lane or "shared"
    with _status_lock:
        index = len(scope["steps"]) + 1
        scope["current_step"] = name
        scope["step_started"] = time.time()
        scope["last_line"] = ""
    log_dir = LOG_DIR / label
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{index:02d}-{name}.log"
    workdir = cwd or (str(REPO_DIR) if REPO_DIR.exists() else str(WORK_ROOT))
    started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

    banner = f"[{label}] [{index:02d}] {name}: {cmd}"
    if echo:
        print("=" * 72)
        print(banner)
        print(f"     log: {log_path}")
        print("=" * 72, flush=True)
    else:
        print(f"{banner}" + LF + f"     log: {log_path}", flush=True)

    t0 = time.time()
    buf = ""
    interrupted = False
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    proc_key = (label, index)
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write(f"$ {cmd}" + LF + f"# cwd: {workdir}" + LF + f"# started: {started}" + LF + LF)
        fh.flush()
        proc = subprocess.Popen(
            cmd, shell=True, cwd=workdir, env={**os.environ, **(env or {})},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True,
        )
        with _procs_lock:
            ACTIVE_PROCS[proc_key] = proc
        try:
            while True:
                chunk = proc.stdout.read1(8192)
                if not chunk:
                    break
                text = decoder.decode(chunk)
                if echo:
                    sys.stdout.write(text)
                    sys.stdout.flush()
                lines, buf = _clean_lines(buf, text)
                for line in lines:
                    fh.write(line + LF)
                fh.flush()
                current = buf.strip() or (lines[-1].strip() if lines else "")
                if current:
                    scope["last_line"] = current
            rc = proc.wait()
        except KeyboardInterrupt:
            # Main-thread (shared) step: forward the interrupt to the child.
            interrupted = True
            rc = _interrupt_proc(proc)
        finally:
            with _procs_lock:
                ACTIVE_PROCS.pop(proc_key, None)
        if buf:
            fh.write(buf + LF)
        elapsed = time.time() - t0
        fh.write(LF + f"# exit code: {rc}" + LF + f"# elapsed: {elapsed:.1f}s" + LF)

    # Lane steps are stopped by the main thread via STOP_EVENT + SIGINT.
    interrupted = interrupted or STOP_EVENT.is_set()
    with _status_lock:
        scope["current_step"] = None
        scope["step_started"] = None
        scope["steps"].append({
            "index": index, "name": name, "cmd": cmd, "exit_code": rc,
            "elapsed_s": round(elapsed, 1), "started": started, "log": str(log_path),
            "interrupted": interrupted,
        })
    save_status()

    outcome = "OK" if rc == 0 else f"FAILED (exit {rc})"
    print(f"[{label}] [{index:02d}] {name}: {outcome} in {elapsed / 60:.1f} min", flush=True)

    if interrupted:
        if lane:
            scope["outcome"] = "interrupted"
        save_status()
        raise KeyboardInterrupt(f"Step '{name}' ({label}) interrupted.")

    if rc != 0 and not allow_fail:
        hint = TRAIN_EXIT_HINTS.get(rc) if name == "train" else None
        tail = _log_tail(log_path)
        paused = name == "train" and rc == 4
        if paused:
            scope["outcome"] = "paused"
            notify(_title(label, "paused at time limit"), LF.join([
                hint,
                f"Session {SESSION}, trained for {elapsed / 60:.0f} min.",
                "",
                tail,
            ]), priority="high", tags=["hourglass"])
        else:
            if lane:
                scope["outcome"] = "failed"
            notify(_title(label, f"step '{name}' failed"), LF.join([
                f"Exit code {rc} after {elapsed / 60:.1f} min. {hint or ''}".strip(),
                f"Session {SESSION}. Last output:",
                tail,
            ]), priority="urgent", tags=["rotating_light"])
        save_status()
        error = RuntimeError(f"Step '{name}' ({label}) failed with exit code {rc}. "
                             f"{hint or f'See {log_path}'}")
        # Already reported; stops other handlers from sending a duplicate alert.
        error.ntfy_sent = True
        # Lets a lane treat a time-limit pause as a clean halt.
        error.halt = "paused" if paused else None
        error.exit_code = rc
        raise error
    return rc


# --------------------------------------------------------------------------- #
# Crash / shutdown hooks
# --------------------------------------------------------------------------- #
def _notify_unfinished(reason):
    """Sends one alert if the kernel goes away before the pipeline reached an outcome."""
    if STATUS.get("outcome") is not None:
        return
    STATUS["outcome"] = "runtime-ended"
    running = [f"{key}: {st['current_step']}" for key, st in STATUS["lanes"].items()
               if st.get("current_step")]
    if STATUS["shared"].get("current_step"):
        running.insert(0, f"shared: {STATUS['shared']['current_step']}")
    where = ("Running: " + "; ".join(running)) if running else "No step was running."
    notify(_title(INSTANCE_LABEL, "runtime ended"), LF.join([
        f"{reason} before the pipeline finished.",
        where,
        f"Session {SESSION}. Hub checkpoints are kept; commit a new version to resume.",
    ]), priority="urgent", tags=["skull"])


def _on_sigterm(signum, frame):
    _notify_unfinished("The Kaggle session was shut down (SIGTERM)")
    # Give running trainers the chance to write an emergency checkpoint.
    STOP_EVENT.set()
    _stop_all_procs()
    prev = globals().get("_NTFY_PREV_SIGTERM")
    if callable(prev):
        prev(signum, frame)
    else:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGTERM)


def _on_cell_exception(shell, etype, evalue, tb, tb_offset=None):
    """Alerts on any uncaught cell error that run_step has not already reported."""
    if not getattr(evalue, "ntfy_sent", False):
        STATUS["outcome"] = "failed"
        try:
            save_status()
        except Exception:
            pass
        notify(_title(INSTANCE_LABEL, "notebook error"),
               f"{etype.__name__}: {evalue}" + LF + f"Session {SESSION}.",
               priority="urgent", tags=["rotating_light"])
    elif STATUS.get("outcome") is None:
        STATUS["outcome"] = "failed"
        save_status()
    shell.showtraceback((etype, evalue, tb), tb_offset=tb_offset)


# Install the exit hooks once per kernel, even if this cell is re-run.
if not globals().get("_NTFY_HOOKS_INSTALLED"):
    _NTFY_PREV_SIGTERM = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _on_sigterm)
    atexit.register(lambda: _notify_unfinished("The notebook kernel exited"))
    _NTFY_HOOKS_INSTALLED = True
get_ipython().set_custom_exc((Exception,), _on_cell_exception)


save_status()
min_free = 40 if len(LANE_SPECS) == 1 else 60
print(f"Platform   : Kaggle ({RUN_TYPE} session)")
print(f"GPUs       : {GPUS}")
for lane in LANE_SPECS:
    print(f"Lane       : {lane['key']} -> GPU {lane['gpu']} ({lane['gpu_name']})")
print(f"Work root  : {WORK_ROOT} ({_free_gb(WORK_ROOT):.0f} GB free, not saved)")
print(f"Output dir : {OUTPUT_DIR} ({_free_gb(OUTPUT_DIR):.0f} GB free, saved with the version)")
if _free_gb(WORK_ROOT) < min_free:
    print(f"WARNING: under {min_free} GB free in the work root. Base models, the merged model and "
          "the f16 GGUF intermediate may not fit; consider RUN_EXPORT = False.")
print(f"Session    : {SESSION}")
print(f"Local logs : {LOG_DIR}")
if NTFY_URL:
    print(f"Notify     : {NTFY_URL}")
    notify(_title(INSTANCE_LABEL, "pipeline started"),
           f"Session {SESSION} ({RUN_TYPE}). Lanes: "
           + ", ".join(f"{lane['key']} on GPU {lane['gpu']}" for lane in LANE_SPECS) + ".",
           priority="low", tags=["rocket"])
