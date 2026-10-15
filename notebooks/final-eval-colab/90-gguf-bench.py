# 8. Optional: Q4_K_M GGUF on the Colab CPU (edge-deployment latency and quantisation check)
if not RUN_GGUF_CPU_BENCH:
    print('RUN_GGUF_CPU_BENCH is False; skipping the GGUF benchmark.')
else:
    run_step('build-llama-cpp', 'bash scripts/build_llama_cpp.sh', env={'BUILD_CUDA': '0'})
    os.environ['LLAMA_CPP_DIR'] = str(REPO_DIR / 'vendor' / 'llama.cpp')
    files = hub_files(refresh=True)
    for e in EXP:
        rn = e['run_name']
        jid = rn + '__fine_tuned__gguf_q4_k_m_cpu'
        job = {**e, 'condition': 'fine_tuned', 'backend': 'llamacpp', 'job_id': jid}
        done_marker = HUB_EVAL_PREFIX + '/jobs/' + jid + '/job.json'
        if not (SKIP_COMPLETED_JOBS and done_marker in files):
            ggufs = sorted(f for f in files if f.startswith('gguf/' + rn + '/') and f.endswith('-q4_k_m.gguf'))
            if not ggufs:
                print('[' + rn + '] No Q4_K_M GGUF on the Hub; skipping.')
                continue
            base = ggufs[-1].split('/')[-1]
            sidecars = sorted(f for f in files if f.endswith('/' + base + '.json'))
            merge_meta = {}
            if sidecars:
                try:
                    sidecar = json.loads(hub_download_file(sidecars[-1]).read_text(encoding='utf-8'))
                    merge_meta = sidecar.get('merge_metadata') or {}
                except Exception as exc:
                    print('[' + rn + '] Could not read the GGUF sidecar: ' + str(exc))
            # Only Qwen was affected by the wrapper-class bug (history/027.md). Its GGUF must
            # come from the fixed merge, which records model_class in merge_metadata.json.
            if e['model'] == 'qwen3.5-2b' and 'model_class' not in merge_meta:
                print('[' + rn + '] Skipping: no post-fix merge metadata, so this GGUF may not contain the adapter.')
                continue
            print('[' + rn + '] Downloading ' + ggufs[-1] + ' ...', flush=True)
            job['gguf_local'] = str(hub_download_file(ggufs[-1]))
            job['gguf_repo_path'] = ggufs[-1]
            job['gguf_merge_metadata'] = merge_meta
        try:
            JOB_RESULTS[jid] = run_job(job)
            STATUS['jobs'][jid] = 'completed'
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            STATUS['jobs'][jid] = 'failed'
            JOB_RESULTS[jid] = _job_record(job, status='failed', error=str(exc))
            print('[' + jid + '] FAILED (best-effort): ' + str(exc), flush=True)
