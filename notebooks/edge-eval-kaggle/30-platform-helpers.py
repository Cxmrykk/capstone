# 2. Platform checks (Kaggle CPU session), helpers, notifications and Hub access
import atexit
import codecs
import datetime
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import urllib.request
from pathlib import Path

CR, LF = chr(13), chr(10)
PLATFORM = 'kaggle'

ON_KAGGLE = Path('/kaggle').is_dir() or bool(os.environ.get('KAGGLE_KERNEL_RUN_TYPE'))
if not ON_KAGGLE:
    raise RuntimeError('This notebook targets a Kaggle CPU session.')
RUN_TYPE = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', 'unknown')  # Interactive | Batch (commit)


def _gpu_names():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return []
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


GPUS_PRESENT = _gpu_names()
if GPUS_PRESENT:
    print('WARNING: a GPU accelerator is attached (' + ', '.join(GPUS_PRESENT) + '). This notebook never uses it; '
          'select Settings > Accelerator > None to keep GPU quota.')
# Every child process (merge, conversion, tokenisation) runs on the CPU only.
os.environ['CUDA_VISIBLE_DEVICES'] = ''


def _lscpu():
    try:
        out = subprocess.run(['lscpu'], capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return {}
    info = {}
    for line in out.splitlines():
        if ':' in line:
            key, value = line.split(':', 1)
            info[key.strip()] = value.strip()
    return info


def _cgroup_cpu_limit():
    try:
        quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()[:2]
        return None if quota == 'max' else round(int(quota) / int(period), 2)
    except Exception:
        pass
    try:
        quota = int(Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us').read_text())
        period = int(Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us').read_text())
        return None if quota <= 0 else round(quota / period, 2)
    except Exception:
        return None


def _meminfo_gb(key):
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith(key + ':'):
                return int(line.split()[1]) / 1024 ** 2
    except Exception:
        pass
    return float('nan')


LSCPU = _lscpu()
CPU_MODEL = LSCPU.get('Model name') or platform.processor() or 'unknown'
try:
    LOGICAL_CPUS = len(os.sched_getaffinity(0))
except Exception:
    LOGICAL_CPUS = os.cpu_count() or 1
try:
    PHYSICAL_CORES = int(LSCPU.get('Core(s) per socket', '0')) * int(LSCPU.get('Socket(s)', '1')) or None
except ValueError:
    PHYSICAL_CORES = None
CPU_FLAGS = set(LSCPU.get('Flags', '').split())
SIMD_FLAGS = [f for f in ('sse4_2', 'avx', 'avx2', 'fma', 'f16c', 'avx512f', 'avx512_vnni', 'avx512_bf16',
                          'avx_vnni', 'amx_tile') if f in CPU_FLAGS]
CGROUP_CPU_LIMIT = _cgroup_cpu_limit()
RAM_TOTAL_GB = _meminfo_gb('MemTotal')


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
SESSION = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d-%H%M%S') + '-kaggle-cpu'
LOG_DIR = Path(OUTPUT_DIR) / 'edge-eval-logs' / SESSION
LOG_DIR.mkdir(parents=True, exist_ok=True)

try:
    from kaggle_secrets import UserSecretsClient
    os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')
except Exception as exc:
    raise RuntimeError('Could not read the HF_TOKEN Kaggle secret (' + type(exc).__name__ + '). '
                       'Add-ons > Secrets: add HF_TOKEN and attach it.') from exc

os.environ['T2C_HUB_REPO'] = HUB_REPO
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

STATUS = {'session': SESSION, 'platform': PLATFORM, 'run_type': RUN_TYPE, 'cpu': CPU_MODEL,
          'current_step': None, 'jobs': {}, 'outcome': None}

# --------------------------------------------------------------------------- #
# ntfy push notifications (best-effort; the topic is public, so no secrets)
# --------------------------------------------------------------------------- #
NTFY_URL = (NTFY_SERVER.rstrip('/') + '/' + NTFY_TOPIC.strip()) if NTFY_TOPIC.strip() else None


def notify(title, message, priority='default', tags=()):
    if not NTFY_URL:
        return False
    headers = {'Title': ('T2C edge eval: ' + title).encode('ascii', 'replace').decode('ascii'),
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
# Step runner: streams to the cell and to a log file (tqdm redraws collapsed)
# --------------------------------------------------------------------------- #
def _clean_lines(buf, text):
    text = buf + text.replace(CR + LF, LF)
    parts = text.split(LF)
    return [p.split(CR)[-1] for p in parts[:-1]], parts[-1].split(CR)[-1]


def _log_tail(path, n=25):
    """Last n lines of a step log, without the runner's own trailer lines."""
    try:
        lines = Path(path).read_text(encoding='utf-8', errors='replace').splitlines()
    except Exception:
        return ''
    lines = [ln for ln in lines if not ln.startswith(('# exit code:', '# elapsed:'))]
    while lines and not lines[-1].strip():
        lines.pop()
    return LF.join(lines[-n:])


def run_step(name, cmd, log_path=None, allow_fail=False, env=None, cwd=None, echo=True):
    log_path = Path(log_path) if log_path else LOG_DIR / (name + '.log')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    workdir = cwd or (str(REPO_DIR) if REPO_DIR.exists() else str(WORK_ROOT))
    STATUS['current_step'] = name
    print('=' * 72)
    print('[' + name + '] ' + cmd)
    print('    log: ' + str(log_path))
    print('=' * 72, flush=True)
    t0 = time.time()
    buf = ''
    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
    with open(log_path, 'w', encoding='utf-8') as fh:
        fh.write('$ ' + cmd + LF + '# cwd: ' + workdir + LF + LF)
        fh.flush()
        proc = subprocess.Popen(cmd, shell=True, cwd=workdir, env={**os.environ, **(env or {})},
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
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
            rc = proc.wait()
        except KeyboardInterrupt:
            try:
                os.killpg(proc.pid, signal.SIGINT)
                proc.wait(timeout=60)
            except Exception:
                proc.kill()
            raise
        if buf:
            fh.write(buf + LF)
        elapsed = time.time() - t0
        fh.write(LF + '# exit code: ' + str(rc) + LF + '# elapsed: ' + format(elapsed, '.1f') + 's' + LF)
    STATUS['current_step'] = None
    print('[' + name + '] exit ' + str(rc) + ' after ' + format(elapsed / 60, '.1f') + ' min', flush=True)
    if rc != 0 and not allow_fail:
        # Include the tail so the cause is visible in the traceback and the ntfy alert,
        # not only in the log file (history/035.md).
        tail = _log_tail(log_path)
        raise RuntimeError('Step ' + name + ' failed with exit code ' + str(rc) + '. See ' + str(log_path)
                           + ((LF + 'Last output:' + LF + tail) if tail else ''))
    return rc


def minutes_left():
    """Minutes left before the reserve kept for analysis and uploads."""
    return SESSION_BUDGET_MINUTES - RESERVE_MINUTES - (time.time() - PIPELINE_T0) / 60


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Hub access
# --------------------------------------------------------------------------- #
_HUB = {'files': None}


def hub_api():
    from huggingface_hub import HfApi
    return HfApi(token=os.environ['HF_TOKEN'])


def hub_files(refresh=False):
    if _HUB['files'] is None or refresh:
        _HUB['files'] = set(hub_api().list_repo_files(HUB_REPO, repo_type='model'))
    return _HUB['files']


def hub_download_file(path_in_repo):
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=HUB_REPO, filename=path_in_repo, repo_type='model',
                                token=os.environ['HF_TOKEN']))


def hub_download_patterns(patterns, dest):
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=HUB_REPO, repo_type='model', token=os.environ['HF_TOKEN'],
                      allow_patterns=list(patterns), local_dir=str(dest))
    return Path(dest)


def hub_upload_folder(local_dir, path_in_repo, message):
    """Uploads a folder in ONE commit, with retries. Never raises."""
    if not UPLOAD_TO_HUB:
        return False
    for attempt in range(3):
        try:
            hub_api().upload_folder(folder_path=str(local_dir), path_in_repo=path_in_repo,
                                    repo_id=HUB_REPO, repo_type='model', commit_message=message)
            return True
        except Exception as exc:
            print('[hub] Upload of ' + path_in_repo + ' failed (attempt ' + str(attempt + 1) + '/3): ' + str(exc),
                  flush=True)
            time.sleep(30 * (attempt + 1))
    return False


def hub_upload_file(local_path, path_in_repo, message):
    """Uploads one file in one commit, with retries. Never raises."""
    if not UPLOAD_TO_HUB:
        return False
    for attempt in range(3):
        try:
            hub_api().upload_file(path_or_fileobj=str(local_path), path_in_repo=path_in_repo,
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
    notify('runtime ended', reason + ' before the final cell (step: ' + str(STATUS.get('current_step')) + '). Session '
           + SESSION + '. Completed jobs: ' + (', '.join(done) or 'none')
           + '. Commit a new version: finished work is reused from the Hub.', priority='urgent', tags=['skull'])


def _on_sigterm(signum, frame):
    _notify_unfinished('The Kaggle session was shut down (SIGTERM)')
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

print('Platform   : Kaggle (' + RUN_TYPE + ' session, CPU only)')
print('CPU        : ' + CPU_MODEL)
print('CPUs       : ' + str(LOGICAL_CPUS) + ' logical, ' + str(PHYSICAL_CORES) + ' physical, cgroup limit '
      + str(CGROUP_CPU_LIMIT))
print('SIMD       : ' + ', '.join(SIMD_FLAGS))
print('RAM        : ' + format(RAM_TOTAL_GB, '.1f') + ' GB')
print('Work root  : ' + str(WORK_ROOT) + ' (' + format(_free_gb(WORK_ROOT), '.0f') + ' GB free, not saved)')
print('Output dir : ' + OUTPUT_DIR + ' (' + format(_free_gb(OUTPUT_DIR), '.0f') + ' GB free, saved)')
print('Session    : ' + SESSION)
notify('started', 'Session ' + SESSION + ' (' + RUN_TYPE + ') on ' + CPU_MODEL + ', ' + str(LOGICAL_CPUS) + ' CPUs.',
       priority='low', tags=['rocket'])
