# 4. Dependencies, dataset and base models, and a CPU build of llama.cpp
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

# The converter uses the checkout's own gguf-py. Its full requirements file pins older
# transformers releases, so only the light dependencies are installed and the floor re-asserted.
run_step('llama-convert-deps', 'python -m pip install --quiet ' + str(LLAMA_DIR / 'gguf-py')
         + ' sentencepiece protobuf && python -m pip install --quiet "transformers>=5.5.0"')
os.environ['LLAMA_CPP_DIR'] = str(LLAMA_DIR)

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
