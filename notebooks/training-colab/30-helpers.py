# @title 2. Logging helpers, notifications, secrets and environment
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
import time
import urllib.request
from pathlib import Path

CR, LF = chr(13), chr(10)
REPO_DIR = Path("/content/capstone")
SESSION = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
RUN_KEY = f"{MODEL}-{SCHEMA_VARIANT}"

if SAVE_LOGS_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    LOG_ROOT = Path("/content/drive/MyDrive/text2cypher-logs")
else:
    LOG_ROOT = Path("/content/text2cypher-logs")
LOG_DIR = LOG_ROOT / RUN_KEY / SESSION
LOG_DIR.mkdir(parents=True, exist_ok=True)
HUB_LOG_PREFIX = f"logs/{RUN_KEY}/{SESSION}"
HUB_RESULTS_PREFIX = f"results/{RUN_KEY}/{SESSION}"
HUB_LOGS_URL = f"https://huggingface.co/{HUB_REPO}/tree/main/{HUB_LOG_PREFIX}" if HUB_REPO else None

from google.colab import userdata
try:
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    print("HF_TOKEN loaded from Colab Secrets.")
except Exception:
    print("WARNING: set the 'HF_TOKEN' Colab secret (write access). "
          "Without it, Hub checkpoints and log mirroring are disabled.")

os.environ["T2C_HUB_REPO"] = HUB_REPO
# Reduce PyTorch memory fragmentation on the 16 GB T4.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Child Python processes flush output immediately, so logs stay current.
os.environ["PYTHONUNBUFFERED"] = "1"

STATUS = {
    "model": MODEL,
    "schema_variant": SCHEMA_VARIANT,
    "config": CONFIG,
    "session": SESSION,
    "run_name": None,
    "current_step": None,
    # None while running; then completed | failed | paused | interrupted | runtime-ended.
    "outcome": None,
    "steps": [],
}
_hub_state = {"repo_ready": False}
_hub_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# ntfy push notifications
# --------------------------------------------------------------------------- #
NTFY_URL = f"{NTFY_SERVER.rstrip('/')}/{NTFY_TOPIC.strip()}" if NTFY_TOPIC.strip() else None
NTFY_PREFIX = f"T2C {RUN_KEY}"


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
    if click or HUB_LOGS_URL:
        headers["Click"] = click or HUB_LOGS_URL
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


def hub_upload(local_path, path_in_repo, blocking=False):
    """Best-effort upload of a file or folder to the Hub repo. Never raises."""
    if not HUB_REPO or not os.environ.get("HF_TOKEN"):
        return
    local_path = Path(local_path)
    if not local_path.exists():
        return

    def _do():
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=os.environ["HF_TOKEN"])
            with _hub_lock:
                if not _hub_state["repo_ready"]:
                    api.create_repo(HUB_REPO, repo_type="model", private=True, exist_ok=True)
                    _hub_state["repo_ready"] = True
                if local_path.is_dir():
                    api.upload_folder(folder_path=str(local_path), path_in_repo=path_in_repo,
                                      repo_id=HUB_REPO, repo_type="model",
                                      commit_message=f"[{RUN_KEY}] {path_in_repo}")
                else:
                    api.upload_file(path_or_fileobj=str(local_path), path_in_repo=path_in_repo,
                                    repo_id=HUB_REPO, repo_type="model",
                                    commit_message=f"[{RUN_KEY}] {path_in_repo}")
        except Exception as exc:
            print(f"[log-sync] Upload of {local_path.name} failed: {exc}", file=sys.stderr)

    if blocking:
        _do()
    else:
        threading.Thread(target=_do, daemon=True).start()


def save_status():
    path = LOG_DIR / "pipeline_status.json"
    path.write_text(json.dumps(STATUS, indent=2), encoding="utf-8")
    hub_upload(path, f"{HUB_LOG_PREFIX}/pipeline_status.json")


def archive_results(paths):
    """Copies result files next to the logs and uploads them to the Hub."""
    dest = LOG_DIR / "results"
    dest.mkdir(parents=True, exist_ok=True)
    for p in map(Path, paths):
        if not p.exists():
            continue
        if p.is_dir():
            shutil.copytree(p, dest / p.name, dirs_exist_ok=True)
        else:
            shutil.copy2(p, dest / p.name)
        hub_upload(p, f"{HUB_RESULTS_PREFIX}/{p.name}", blocking=True)


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
    4: "Training reached TIME_LIMIT_MINUTES and checkpointed to the Hub. Open a fresh runtime "
       "and Run all again: training resumes automatically.",
}


def run_step(name, cmd, allow_fail=False, env=None, cwd=None):
    """Runs a shell command, streaming its output to the cell and to a log file."""
    index = len(STATUS["steps"]) + 1
    log_path = LOG_DIR / f"{index:02d}-{name}.log"
    hub_path = f"{HUB_LOG_PREFIX}/{log_path.name}"
    workdir = cwd or (str(REPO_DIR) if REPO_DIR.exists() else "/content")
    started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    STATUS["current_step"] = name
    print("=" * 72)
    print(f"[{index:02d}] {name}: {cmd}")
    print(f"     log: {log_path}")
    print("=" * 72, flush=True)

    t0 = time.time()
    last_sync = t0
    buf = ""
    interrupted = False
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write(f"$ {cmd}" + LF + f"# cwd: {workdir}" + LF + f"# started: {started}" + LF + LF)
        fh.flush()
        proc = subprocess.Popen(
            cmd, shell=True, cwd=workdir, env={**os.environ, **(env or {})},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            while True:
                chunk = proc.stdout.read1(8192)
                if not chunk:
                    break
                text = decoder.decode(chunk)
                sys.stdout.write(text)
                sys.stdout.flush()
                lines, buf = _clean_lines(buf, text)
                for line in lines:
                    fh.write(line + LF)
                fh.flush()
                if time.time() - last_sync >= LOG_UPLOAD_INTERVAL_S:
                    hub_upload(log_path, hub_path)
                    last_sync = time.time()
            rc = proc.wait()
        except KeyboardInterrupt:
            # Forward the interrupt so training can write its emergency checkpoint.
            interrupted = True
            try:
                os.killpg(proc.pid, signal.SIGINT)
                rc = proc.wait(timeout=120)
            except Exception:
                proc.kill()
                rc = proc.wait()
        if buf:
            fh.write(buf + LF)
        elapsed = time.time() - t0
        fh.write(LF + f"# exit code: {rc}" + LF + f"# elapsed: {elapsed:.1f}s" + LF)

    hub_upload(log_path, hub_path, blocking=True)
    STATUS["current_step"] = None
    STATUS["steps"].append({
        "index": index, "name": name, "cmd": cmd, "exit_code": rc,
        "elapsed_s": round(elapsed, 1), "started": started, "log": str(log_path),
        "interrupted": interrupted,
    })
    save_status()

    outcome = "OK" if rc == 0 else f"FAILED (exit {rc})"
    print(f"[{index:02d}] {name}: {outcome} in {elapsed / 60:.1f} min", flush=True)

    if interrupted:
        STATUS["outcome"] = "interrupted"
        save_status()
        notify(f"{NTFY_PREFIX}: interrupted",
               f"Step '{name}' was stopped by the user after {elapsed / 60:.1f} min. Session {SESSION}.",
               priority="high", tags=["stop_sign"])
        raise KeyboardInterrupt(f"Step '{name}' interrupted by user.")

    if rc != 0 and not allow_fail:
        hint = TRAIN_EXIT_HINTS.get(rc) if name == "train" else None
        tail = _log_tail(log_path)
        if name == "train" and rc == 4:
            STATUS["outcome"] = "paused"
            notify(f"{NTFY_PREFIX}: paused at time limit", LF.join([
                hint,
                f"Session {SESSION}, trained for {elapsed / 60:.0f} min.",
                "",
                tail,
            ]), priority="high", tags=["hourglass"])
        else:
            STATUS["outcome"] = "failed"
            notify(f"{NTFY_PREFIX}: step '{name}' failed", LF.join([
                f"Exit code {rc} after {elapsed / 60:.1f} min. {hint or ''}".strip(),
                f"Session {SESSION}. Last output:",
                tail,
            ]), priority="urgent", tags=["rotating_light"])
        save_status()
        error = RuntimeError(f"Step '{name}' failed with exit code {rc}. {hint or f'See {log_path}'}")
        # Already reported; stops the cell-error hook from sending a duplicate.
        error.ntfy_sent = True
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
    step = STATUS.get("current_step")
    where = f"while step '{step}' was running" if step else "between steps"
    done = ", ".join(s["name"] for s in STATUS["steps"]) or "none"
    notify(f"{NTFY_PREFIX}: runtime ended", LF.join([
        f"{reason} {where}, before the pipeline finished.",
        f"Session {SESSION}. Completed steps: {done}.",
        "Hub checkpoints are kept; open a fresh runtime and Run all to resume.",
    ]), priority="urgent", tags=["skull"])


def _on_sigterm(signum, frame):
    _notify_unfinished("The Colab runtime was shut down (SIGTERM)")
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
        notify(f"{NTFY_PREFIX}: notebook error",
               f"{etype.__name__}: {evalue}" + LF + f"Session {SESSION}.",
               priority="urgent", tags=["rotating_light"])
    shell.showtraceback((etype, evalue, tb), tb_offset=tb_offset)


# Install the exit hooks once per kernel, even if this cell is re-run.
if not globals().get("_NTFY_HOOKS_INSTALLED"):
    _NTFY_PREV_SIGTERM = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _on_sigterm)
    atexit.register(lambda: _notify_unfinished("The notebook kernel exited"))
    _NTFY_HOOKS_INSTALLED = True
get_ipython().set_custom_exc((Exception,), _on_cell_exception)


save_status()
print(f"Session   : {SESSION}")
print(f"Local logs: {LOG_DIR}")
if HUB_LOGS_URL:
    print(f"Hub logs  : {HUB_LOGS_URL}")
if NTFY_URL:
    print(f"Notify    : {NTFY_URL}")
    notify(f"{NTFY_PREFIX}: pipeline started",
           f"Session {SESSION} started on Colab ({CONFIG}).",
           priority="low", tags=["rocket"])
