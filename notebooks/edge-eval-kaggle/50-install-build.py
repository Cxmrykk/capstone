# 4. Dependencies, dataset and base models, and a CPU build of llama.cpp
from importlib import metadata as _metadata

# --------------------------------------------------------------------------- #
# Freeze packages the kernel may already have imported (history/034.md).
# Kaggle's kernel imports numpy at startup (matplotlib inline backend). If pip
# replaces numpy on disk afterwards, lazily loaded numpy submodules come from the
# new files and fail against the old ones in memory (ImportError: _slice).
# PIP_CONSTRAINT applies to every pip call in this session, including the ones
# inside scripts/colab_bootstrap.sh. An unsatisfiable requirement now fails
# loudly at install time instead of breaking the kernel later.
# --------------------------------------------------------------------------- #
FROZEN_DISTS = {'numpy': 'numpy', 'scipy': 'scipy', 'pandas': 'pandas', 'scikit-learn': 'sklearn',
                'pyarrow': 'pyarrow'}


def _dist_version(dist):
    try:
        return _metadata.version(dist)
    except _metadata.PackageNotFoundError:
        return None


FROZEN_BEFORE = {d: _dist_version(d) for d in FROZEN_DISTS}
IMPORTED_AT_START = sorted(mod for mod in FROZEN_DISTS.values() if mod in sys.modules)
CONSTRAINTS_FILE = LOG_DIR / 'pip-constraints.txt'
CONSTRAINTS_FILE.write_text(''.join(d + '==' + v + LF for d, v in FROZEN_BEFORE.items() if v), encoding='utf-8')
os.environ['PIP_CONSTRAINT'] = str(CONSTRAINTS_FILE)
print('Already imported by the kernel: ' + (', '.join(IMPORTED_AT_START) or 'none'))
print('pip constraints (' + str(CONSTRAINTS_FILE) + '): '
      + ', '.join(d + '==' + v for d, v in FROZEN_BEFORE.items() if v))

# Unsloth is not needed: merging loads adapters through Transformers + PEFT.
run_step('bootstrap', 'bash scripts/colab_bootstrap.sh', env={'INSTALL_UNSLOTH': '0'})

# PEFT must be able to inject LoRA layers on this image (history/033.md). An old
# preinstalled torchao makes PEFT raise ImportError during adapter loading. The
# bootstrap removes it; this check fails in seconds instead of after loading
# every base model in the export cell.
run_step('peft-check', 'python -c "from peft.import_utils import is_torchao_available; '
                       'print(\'torchao usable by PEFT:\', is_torchao_available())"')

DOWNLOAD_TARGETS = {'gemma-4-e2b': 'gemma', 'qwen3.5-2b': 'qwen', 'granite-4.2-3b': 'granite'}
for target in dict.fromkeys(DOWNLOAD_TARGETS[m] for m, _v, _c in EXPERIMENTS):
    run_step('download-' + target, 'bash download_data.sh ' + target)

# --------------------------------------------------------------------------- #
# llama.cpp: pinned commit (LLAMA_CPP_REF, else the commit the first session recorded)
# --------------------------------------------------------------------------- #
LLAMA_DIR = WORK_ROOT / 'llama.cpp'
LLAMA_BIN = LLAMA_DIR / 'build' / 'bin'
LLAMA_PIN_PATH = 'edge-eval/' + EDGE_TAG_SUFFIX + '/llama_cpp_commit.txt'

pinned = None
if not LLAMA_CPP_REF:
    try:
        if LLAMA_PIN_PATH in hub_files(refresh=True):
            pinned = hub_download_file(LLAMA_PIN_PATH).read_text(encoding='utf-8').strip() or None
            print('Using the llama.cpp commit pinned by an earlier session: ' + str(pinned))
    except Exception as exc:
        print('Could not read the llama.cpp pin (' + str(exc) + '); building the latest commit.')
LLAMA_REF_USED = LLAMA_CPP_REF or pinned

run_step('build-tools', 'command -v cmake >/dev/null && command -v g++ >/dev/null || '
                        '(apt-get update -qq && apt-get install -y -qq cmake build-essential)', allow_fail=True)
if not (LLAMA_DIR / '.git').exists():
    if LLAMA_REF_USED:
        run_step('llama-clone', 'git clone --filter=blob:none ' + LLAMA_CPP_REPO + ' ' + str(LLAMA_DIR),
                 cwd=str(WORK_ROOT))
    else:
        run_step('llama-clone', 'git clone --depth 1 ' + LLAMA_CPP_REPO + ' ' + str(LLAMA_DIR), cwd=str(WORK_ROOT))
if LLAMA_REF_USED:
    run_step('llama-checkout', 'git -C ' + str(LLAMA_DIR) + ' checkout --detach ' + LLAMA_REF_USED, cwd=str(WORK_ROOT))

# GGML_NATIVE tunes kernels to this CPU, as an edge build would be tuned to its device.
run_step('llama-cmake', 'cmake -B ' + str(LLAMA_DIR / 'build') + ' -S ' + str(LLAMA_DIR)
         + ' -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_CUDA=OFF -DLLAMA_CURL=OFF', cwd=str(WORK_ROOT))
run_step('llama-build', 'cmake --build ' + str(LLAMA_DIR / 'build') + ' --config Release -j ' + str(LOGICAL_CPUS)
         + ' --target llama-server llama-quantize llama-bench', cwd=str(WORK_ROOT))
for binary in ('llama-server', 'llama-quantize', 'llama-bench'):
    if not (LLAMA_BIN / binary).exists():
        raise RuntimeError(binary + ' was not built; see the llama-build log.')

# --------------------------------------------------------------------------- #
# GGUF converter dependencies (history/035.md)
# convert_hf_to_gguf.py inserts the checkout's own gguf-py at the front of sys.path
# unless NO_LOCAL_GGUF is set, so gguf-py is deliberately NOT pip-installed: building
# it from source needs an isolated poetry-core build environment, which failed under
# PIP_CONSTRAINT. Only gguf-py's runtime dependencies and the converter's own imports
# are installed (all prebuilt wheels). The llama.cpp requirements file is avoided
# because it pins transformers < 5. PIP_CONSTRAINT still keeps numpy and friends at
# the versions the kernel has loaded.
# --------------------------------------------------------------------------- #
os.environ.pop('NO_LOCAL_GGUF', None)
CONVERT_DEPS = ['sentencepiece', 'protobuf', 'pyyaml', 'tqdm', 'requests', 'safetensors', 'transformers>=5.5.0']
run_step('llama-convert-deps', 'python -m pip install --disable-pip-version-check '
         + ' '.join('"' + d + '"' for d in CONVERT_DEPS))
os.environ['LLAMA_CPP_DIR'] = str(LLAMA_DIR)

# Fails in seconds if the converter cannot import (missing dependency, gguf-py/transformers
# mismatch) instead of after a 10-30 min merge in the export cell.
run_step('llama-convert-check',
         'python -c "import sys; sys.path.insert(1, \'gguf-py\'); import gguf, yaml, sentencepiece, '
         'google.protobuf, torch, transformers, safetensors; print(\'gguf-py from\', gguf.__file__)" '
         '&& python convert_hf_to_gguf.py --help > /dev/null && echo "convert_hf_to_gguf.py imports OK"',
         cwd=str(LLAMA_DIR))

# --------------------------------------------------------------------------- #
# Guard: nothing the kernel depends on may have changed on disk
# --------------------------------------------------------------------------- #
FROZEN_AFTER = {d: _dist_version(d) for d in FROZEN_DISTS}
changed = {d: (FROZEN_BEFORE[d], FROZEN_AFTER[d]) for d in FROZEN_DISTS if FROZEN_BEFORE[d] != FROZEN_AFTER[d]}
loaded_mismatch = []
for dist, mod in FROZEN_DISTS.items():
    loaded = sys.modules.get(mod)
    loaded_version = getattr(loaded, '__version__', None) if loaded is not None else None
    if loaded_version and FROZEN_AFTER[dist] and loaded_version != FROZEN_AFTER[dist]:
        loaded_mismatch.append(mod + ' ' + loaded_version + ' in memory vs ' + FROZEN_AFTER[dist] + ' on disk')
if changed or loaded_mismatch:
    raise RuntimeError('Installs changed packages the kernel depends on: '
                       + '; '.join(d + ' ' + str(a) + ' -> ' + str(b) for d, (a, b) in changed.items())
                       + ('; ' if changed and loaded_mismatch else '') + '; '.join(loaded_mismatch)
                       + '. Later in-kernel imports would fail (history/034.md). Check which install ignored '
                         'PIP_CONSTRAINT in the logs above.')
run_step('import-check', 'python -c "import numpy, numpy.char, scipy.sparse, sklearn, pandas, transformers; '
                         'from transformers import AutoTokenizer; print(\'numpy\', numpy.__version__, '
                         '\'transformers\', transformers.__version__)"')
print('Frozen packages unchanged: ' + ', '.join(d + ' ' + str(v) for d, v in FROZEN_AFTER.items() if v))

LLAMA_COMMIT = subprocess.run(['git', '-C', str(LLAMA_DIR), 'rev-parse', 'HEAD'],
                              capture_output=True, text=True).stdout.strip()
_ver = subprocess.run([str(LLAMA_BIN / 'llama-server'), '--version'], capture_output=True, text=True, timeout=60)
LLAMA_VERSION = ' | '.join(ln.strip() for ln in (_ver.stdout + _ver.stderr).splitlines() if ln.strip())[:300]
print('llama.cpp commit : ' + LLAMA_COMMIT)
print('llama.cpp version: ' + LLAMA_VERSION)

if not LLAMA_CPP_REF and not pinned:
    pin_file = LOG_DIR / 'llama_cpp_commit.txt'
    pin_file.write_text(LLAMA_COMMIT + LF, encoding='utf-8')
    if hub_upload_file(pin_file, LLAMA_PIN_PATH, '[edge-eval] pin llama.cpp ' + LLAMA_COMMIT[:10]):
        print('Recorded this llama.cpp commit on the Hub; later sessions will build the same one.')

run_step('doctor', 'python app.py doctor', allow_fail=True)
print('Free disk in work root: ' + format(_free_gb(WORK_ROOT), '.0f') + ' GB')
