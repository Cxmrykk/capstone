# 4. Install dependencies and download the dataset and base models (sequential downloads)
# Unsloth is not needed: inference loads adapters through Transformers + PEFT.
run_step('bootstrap', 'bash scripts/colab_bootstrap.sh', env={'INSTALL_UNSLOTH': '0'})

DOWNLOAD_TARGETS = {'gemma-4-e2b': 'gemma', 'qwen3.5-2b': 'qwen', 'granite-4.2-3b': 'granite'}
for target in dict.fromkeys(DOWNLOAD_TARGETS[m] for m, _v, _c in EXPERIMENTS):
    run_step('download-' + target, 'bash download_data.sh ' + target)

run_step('doctor', 'python app.py doctor', allow_fail=True)
print('Free disk in work root: ' + format(_free_gb(WORK_ROOT), '.0f') + ' GB')
