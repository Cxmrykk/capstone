# 2. Helpers: environment, secrets, logging, notifications and Hub access
import atexit
import codecs
import datetime
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

T0 = time.time()
CR, LF = chr(13), chr(10)
REPO_DIR = Path('/content/capstone')
SESSION = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d-%H%M%S') + '-colab'
LOG_DIR = Path('/content/final-eval-logs') / SESSION
LOG_DIR.mkdir(parents=True, exist_ok=True)

try:
    from google.colab import userdata
    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
except Exception as exc:
    raise RuntimeError('Add a Colab secret named HF_TOKEN with access to ' + HUB_REPO + ' and allow '
                       'this notebook to use it. The adapters live in a private repo.') from exc

os.environ['T2C_HUB_REPO'] = HUB_REPO
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def gpu_name():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        return out.splitlines()[0].strip() if out else None
    except Exception:
        return None


GPU_NAME = gpu_name()
if not GPU_NAME:
    raise RuntimeError('No GPU visible. Runtime > Change runtime type > T4 GPU.')
if 'T4' not in GPU_NAME:
    print('WARNING: expected a Tesla T4, found ' + GPU_NAME + '. Accuracy is unaffected, but '
          'latency figures will not match the T4 used for training.')

STATUS = {'session': SESSION, 'gpu': GPU_NAME, 'jobs': {}, 'outcome': None}

# --------------------------------------------------------------------------- #
# ntfy push notifications (best-effort; the topic is public, so no secrets)
# --------------------------------------------------------------------------- #
NTFY_URL = (NTFY_SERVER.rstrip('/') + '/' + NTFY_TOPIC.strip()) if NTFY_TOPIC.strip() else None


def notify(title, message, priority='default', tags=()):
    if not NTFY_URL:
        return False
    headers = {'Title': ('T2C final eval: ' + title).encode('ascii', 'replace').decode('ascii'),
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


def run_step(name, cmd, log_path=None, allow_fail=False, env=None, cwd=None, echo=True):
    log_path = Path(log_path) if log_path else LOG_DIR / (name + '.log')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    workdir = cwd or (str(REPO_DIR) if REPO_DIR.exists() else '/content')
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
        fh.write(LF + '# exit code: ' + str(rc) + LF + f'# elapsed: {elapsed:.1f}s' + LF)
    print(f'[{name}] exit {rc} after {elapsed / 60:.1f} min', flush=True)
    if rc != 0 and not allow_fail:
        raise RuntimeError(f'Step {name} failed with exit code {rc}. See {log_path}')
    return rc


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
            hub_api().upload_folder(folder_path=str(local_dir), path_in_repo=path_in_repo,
                                    repo_id=HUB_REPO, repo_type='model', commit_message=message)
            return True
        except Exception as exc:
            print(f'[hub] Upload of {path_in_repo} failed (attempt {attempt + 1}/3): {exc}', flush=True)
            time.sleep(30 * (attempt + 1))
    return False


# --------------------------------------------------------------------------- #
# Crash hooks
# --------------------------------------------------------------------------- #
def _notify_unfinished():
    if STATUS.get('outcome') is None:
        done = [k for k, v in STATUS['jobs'].items() if v == 'completed']
        notify('runtime ended', 'Session ' + SESSION + ' ended before the final cell. Completed jobs: '
               + (', '.join(done) or 'none') + '. Run all again: finished jobs are reused from the Hub.',
               priority='urgent', tags=['skull'])


def _on_cell_exception(shell, etype, evalue, tb, tb_offset=None):
    notify('notebook error', etype.__name__ + ': ' + str(evalue) + LF + 'Session ' + SESSION,
           priority='urgent', tags=['rotating_light'])
    shell.showtraceback((etype, evalue, tb), tb_offset=tb_offset)


if not globals().get('_HOOKS_INSTALLED'):
    atexit.register(_notify_unfinished)
    _HOOKS_INSTALLED = True
get_ipython().set_custom_exc((Exception,), _on_cell_exception)

print('Session : ' + SESSION)
print('GPU     : ' + GPU_NAME)
print('Logs    : ' + str(LOG_DIR))
notify('started', 'Session ' + SESSION + ' on ' + GPU_NAME + '.', priority='low', tags=['rocket'])
