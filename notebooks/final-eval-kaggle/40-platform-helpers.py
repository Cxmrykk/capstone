# 2. Platform checks, helpers, notifications and Hub access
import atexit
import codecs
import datetime
import json
import os
import platform
import queue
import shutil
import signal
import subprocess
import sys
import threading
import traceback
import urllib.request
from pathlib import Path

CR, LF = chr(13), chr(10)
PLATFORM = 'kaggle'  # The Colab notebook uses 'colab'

# --------------------------------------------------------------------------- #
# Platform checks (fail fast, before anything slow runs)
# --------------------------------------------------------------------------- #
ON_KAGGLE = Path('/kaggle').is_dir() or bool(os.environ.get('KAGGLE_KERNEL_RUN_TYPE'))
if not ON_KAGGLE:
    raise RuntimeError('This notebook targets Kaggle. On Google Colab use final-evaluation.ipynb.')
RUN_TYPE = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', 'unknown')  # Interactive | Batch (commit)


def _query_gpus():
    """Returns {index: name} for physical GPUs (nvidia-smi ignores CUDA_VISIBLE_DEVICES)."""
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=index,name', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return {}
    gpus = {}
    for line in out.splitlines():
        if ',' in line:
            idx, name = line.split(',', 1)
            gpus[idx.strip()] = name.strip()
    return gpus


GPUS = _query_gpus()
if not GPUS:
    raise RuntimeError('No GPU visible. Settings > Accelerator > GPU T4 x2 (phone-verified account).')
for gid in GPU_IDS:
    name = GPUS.get(gid)
    if name is None:
        raise RuntimeError('GPU ' + gid + ' does not exist (found ' + str(sorted(GPUS)) + '). '
                           'Two workers need the GPU T4 x2 accelerator, or set GPU_IDS = ["0"].')
    if 'P100' in name:
        raise RuntimeError('Tesla P100 (Pascal) selected. Switch the accelerator to GPU T4 x2.')
    if 'T4' not in name:
        print('WARNING: expected a Tesla T4 on GPU ' + gid + ', found ' + name + '. Accuracy is unaffected, '
              'but latency figures will not match the T4 used for training.')
GPU_NAME = GPUS[GPU_IDS[0]]
GPU_LABEL = str(len(GPU_IDS)) + 'x ' + GPU_NAME if len(set(GPUS[g] for g in GPU_IDS)) == 1 \
    else ' + '.join(GPUS[g] for g in GPU_IDS)

# Shared steps (doctor, neo4j checks) see one GPU. Every predict step overrides
# CUDA_VISIBLE_DEVICES with its worker's GPU, so no process ever sees two devices.
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_IDS[0]


def _internet_ok(url='https://huggingface.co', timeout=15):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status < 500
    except Exception:
        return False


if not _internet_ok():
    raise RuntimeError('No internet access. Settings > Internet > On (phone-verified account).')


def _writable_dir(path):
    try:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        probe = p / '.t2c-write-test'
        probe.write_text('ok', encoding='utf-8')
        probe.unlink()
        return True
    except Exception:
        return False


def _free_gb(path):
    try:
        return shutil.disk_usage(path).free / 1e9
    except Exception:
        return float('nan')


WORK_ROOT = Path(SCRATCH_DIR) if _writable_dir(SCRATCH_DIR) else Path('/tmp')
REPO_DIR = WORK_ROOT / 'capstone'
SESSION = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d-%H%M%S') + '-kaggle'
LOG_DIR = Path(OUTPUT_DIR) / 'final-eval-logs' / SESSION
LOG_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Secrets and environment
# --------------------------------------------------------------------------- #
try:
    from kaggle_secrets import UserSecretsClient
    os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')
except Exception as exc:
    raise RuntimeError('Could not read the HF_TOKEN Kaggle secret (' + type(exc).__name__ + '). '
                       'Add-ons > Secrets: add HF_TOKEN and attach it. The adapters live in a private repo.') from exc

os.environ['T2C_HUB_REPO'] = HUB_REPO
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# --------------------------------------------------------------------------- #
# Shared state
# --------------------------------------------------------------------------- #
STATUS = {'session': SESSION, 'platform': PLATFORM, 'run_type': RUN_TYPE, 'gpus': GPUS,
          'jobs': {}, 'outcome': None}
WORKERS = {}                   # scope -> {'job', 'step', 'started', 'last_line'}
_state_lock = threading.RLock()
STOP_EVENT = threading.Event()
ACTIVE_PROCS = {}
_procs_lock = threading.Lock()

# --------------------------------------------------------------------------- #
# ntfy push notifications (best-effort; the topic is public, so no secrets)
# --------------------------------------------------------------------------- #
NTFY_URL = (NTFY_SERVER.rstrip('/') + '/' + NTFY_TOPIC.strip()) if NTFY_TOPIC.strip() else None


def notify(title, message, priority='default', tags=()):
    if not NTFY_URL:
        return False
    headers = {'Title': ('T2C Kaggle final eval: ' + title).encode('ascii', 'replace').decode('ascii'),
               'Priority': str(priority)}
    if tags:
        headers['Tags'] = ','.join(tags)
    body = message if len(message) <= 3000 else '...' + message[-3000:]
    req = urllib.request.Request(NTFY_URL, data=body.encode('utf-8'), headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
        return True
    except Exception as exc:
        print('[ntfy] Notification failed: ' + str(exc), file=sys.stderr)
        return False


# --------------------------------------------------------------------------- #
# Step runner: thread-safe, writes a log file (tqdm redraws collapsed)
# --------------------------------------------------------------------------- #
def _clean_lines(buf, text):
    text = buf + text.replace(CR + LF, LF)
    parts = text.split(LF)
    return [p.split(CR)[-1] for p in parts[:-1]], parts[-1].split(CR)[-1]


def _interrupt_proc(proc):
    try:
        os.killpg(proc.pid, signal.SIGINT)
        return proc.wait(timeout=60)
    except Exception:
        proc.kill()
        return proc.wait()


def _stop_all_procs():
    with _procs_lock:
        procs = list(ACTIVE_PROCS.values())
    for proc in procs:
        try:
            os.killpg(proc.pid, signal.SIGINT)
        except Exception:
            pass


def run_step(name, cmd, log_path=None, allow_fail=False, env=None, cwd=None, echo=True, scope='main'):
    """Runs a shell command, writing its output to a log file.

    Main-thread steps stream to the cell (echo=True). Worker steps use echo=False;
    the heartbeat shows their latest log line instead.
    """
    if STOP_EVENT.is_set():
        raise KeyboardInterrupt('Stop requested before step ' + name + '.')
    log_path = Path(log_path) if log_path else LOG_DIR / (scope + '-' + name + '.log')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    workdir = cwd or (str(REPO_DIR) if REPO_DIR.exists() else str(WORK_ROOT))
    with _state_lock:
        WORKERS.setdefault(scope, {}).update(step=name, started=time.time(), last_line='')
    if echo:
        print('=' * 72)
        print('[' + scope + '] ' + name + ': ' + cmd)
        print('    log: ' + str(log_path))
        print('=' * 72, flush=True)
    else:
        print('[' + scope + '] ' + name + ' started (log: ' + str(log_path) + ')', flush=True)

    t0 = time.time()
    buf = ''
    interrupted = False
    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
    proc_key = (scope, name, t0)
    with open(log_path, 'w', encoding='utf-8') as fh:
        fh.write('$ ' + cmd + LF + '# cwd: ' + workdir + LF + LF)
        fh.flush()
        proc = subprocess.Popen(cmd, shell=True, cwd=workdir, env={**os.environ, **(env or {})},
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
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
                current = buf.strip() or (lines[-1].strip() if lines else '')
                if current:
                    with _state_lock:
                        WORKERS[scope]['last_line'] = current
            rc = proc.wait()
        except KeyboardInterrupt:
            interrupted = True
            rc = _interrupt_proc(proc)
        finally:
            with _procs_lock:
                ACTIVE_PROCS.pop(proc_key, None)
        if buf:
            fh.write(buf + LF)
        elapsed = time.time() - t0
        fh.write(LF + '# exit code: ' + str(rc) + LF + '# elapsed: ' + format(elapsed, '.1f') + 's' + LF)

    with _state_lock:
        WORKERS[scope].update(step=None, started=None)
    print('[' + scope + '] ' + name + ': exit ' + str(rc) + ' after ' + format(elapsed / 60, '.1f') + ' min',
          flush=True)
    if interrupted or (STOP_EVENT.is_set() and rc != 0):
        raise KeyboardInterrupt('Step ' + name + ' (' + scope + ') interrupted.')
    if rc != 0 and not allow_fail:
        raise RuntimeError('Step ' + name + ' failed with exit code ' + str(rc) + '. See ' + str(log_path))
    return rc


# --------------------------------------------------------------------------- #
# Hub access (uploads are serialised: concurrent commits to one repo conflict)
# --------------------------------------------------------------------------- #
_HUB = {'files': None}
_hub_lock = threading.Lock()


def hub_api():
    from huggingface_hub import HfApi
    return HfApi(token=os.environ['HF_TOKEN'])


def hub_files(refresh=False):
    if _HUB['files'] is None or refresh:
        _HUB['files'] = set(hub_api().list_repo_files(HUB_REPO, repo_type='model'))
    return _HUB['files']


def hub_download_prefix(prefix, dest):
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=HUB_REPO, repo_type='model', token=os.environ['HF_TOKEN'],
                      allow_patterns=[prefix + '/*'], local_dir=str(dest))
    return Path(dest) / prefix


def hub_download_file(path_in_repo):
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=HUB_REPO, filename=path_in_repo, repo_type='model',
                                token=os.environ['HF_TOKEN']))


def hub_upload_folder(local_dir, path_in_repo, message):
    """Uploads a folder in ONE commit, with retries. Never raises."""
    if not UPLOAD_TO_HUB:
        return False
    for attempt in range(3):
        try:
            with _hub_lock:
                hub_api().upload_folder(folder_path=str(local_dir), path_in_repo=path_in_repo,
                                        repo_id=HUB_REPO, repo_type='model', commit_message=message)
            return True
        except Exception as exc:
            print('[hub] Upload of ' + path_in_repo + ' failed (attempt ' + str(attempt + 1) + '/3): ' + str(exc),
                  flush=True)
            time.sleep(30 * (attempt + 1))
    return False


# --------------------------------------------------------------------------- #
# Crash hooks
# --------------------------------------------------------------------------- #
def _notify_unfinished(reason):
    if STATUS.get('outcome') is not None:
        return
    STATUS['outcome'] = 'runtime-ended'
    done = [k for k, v in STATUS['jobs'].items() if v == 'completed']
    notify('runtime ended', reason + ' before the final cell. Session ' + SESSION + '. Completed jobs: '
           + (', '.join(done) or 'none') + '. Commit a new version: finished jobs are reused from the Hub.',
           priority='urgent', tags=['skull'])


def _on_sigterm(signum, frame):
    _notify_unfinished('The Kaggle session was shut down (SIGTERM)')
    STOP_EVENT.set()
    _stop_all_procs()
    prev = globals().get('_PREV_SIGTERM')
    if callable(prev):
        prev(signum, frame)
    else:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGTERM)


def _on_cell_exception(shell, etype, evalue, tb, tb_offset=None):
    notify('notebook error', etype.__name__ + ': ' + str(evalue) + LF + 'Session ' + SESSION,
           priority='urgent', tags=['rotating_light'])
    shell.showtraceback((etype, evalue, tb), tb_offset=tb_offset)


if not globals().get('_HOOKS_INSTALLED'):
    _PREV_SIGTERM = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _on_sigterm)
    atexit.register(lambda: _notify_unfinished('The notebook kernel exited'))
    _HOOKS_INSTALLED = True
get_ipython().set_custom_exc((Exception,), _on_cell_exception)

print('Platform   : Kaggle (' + RUN_TYPE + ' session)')
print('GPUs       : ' + str(GPUS))
print('Work root  : ' + str(WORK_ROOT) + ' (' + format(_free_gb(WORK_ROOT), '.0f') + ' GB free, not saved)')
print('Output dir : ' + OUTPUT_DIR + ' (' + format(_free_gb(OUTPUT_DIR), '.0f') + ' GB free, saved)')
print('Session    : ' + SESSION)
print('Logs       : ' + str(LOG_DIR))
notify('started', 'Session ' + SESSION + ' (' + RUN_TYPE + ') on ' + GPU_LABEL + '.', priority='low', tags=['rocket'])
