# 7. Run the evaluation jobs on both GPUs:
#    GPU workers: predict -> verify adapter -> hand off
#    Eval workers: evaluate -> record -> upload (one commit per job)
ADAPTER_CONDITIONS = ('fine_tuned', 'cross_schema')
CONDITION_ORDER = {'fine_tuned': 0, 'cross_schema': 1, 'zero_shot': 2}

# The expected job list comes from EXPERIMENTS alone. A job whose adapter is missing
# stays expected: it is recorded as not run and fails an integrity check (history/030.md).
JOBS = []
for e in EXP:
    for condition in ['fine_tuned'] + (['zero_shot'] if RUN_ZERO_SHOT else []):
        JOBS.append({**e, 'condition': condition, 'backend': 'hf',
                     'prompt_schema_mode': e['schema_mode'],
                     'train_schema_mode': e['schema_mode'] if condition == 'fine_tuned' else None,
                     'job_id': e['run_name'] + '__' + condition})
if RUN_CROSS_SCHEMA:
    # Enhanced-trained adapter scored with exact-match prompts: separates the train-time
    # effect of schema filtering from its inference-time effect.
    by_variant = {(e['model'], e['variant']): e for e in EXP}
    for (m, v), e in by_variant.items():
        pruned = by_variant.get((m, 'pruned'))
        if v != 'enhanced' or pruned is None:
            continue
        JOBS.append({**e, 'condition': 'cross_schema', 'backend': 'hf',
                     'schema_mode': pruned['schema_mode'],
                     'prompt_schema_mode': pruned['schema_mode'],
                     'train_schema_mode': e['schema_mode'],
                     'job_id': e['run_name'] + '__fine_tuned__' + pruned['schema_mode'] + '_prompts'})
JOBS.sort(key=lambda j: CONDITION_ORDER.get(j['condition'], 9))

JOB_RESULTS = {}
_results_lock = threading.Lock()
SKIPPED_FOR_TIME = []


def _set_result(jid, rec, status):
    with _results_lock:
        JOB_RESULTS[jid] = rec
        STATUS['jobs'][jid] = status


def _job_record(job, **extra):
    keep = ('job_id', 'model', 'variant', 'config', 'run_name', 'schema_mode', 'condition', 'backend',
            'prompt_schema_mode', 'train_schema_mode', 'plain_text_prompt',
            'fingerprint_now', 'fingerprint_trained', 'fingerprint_match')
    rec = {k: job.get(k) for k in keep}
    rec.update({'eval_tag': EVAL_TAG, 'split': SPLIT, 'limit': EVAL_LIMIT, 'sample_seed': SAMPLE_SEED,
                'git_ref': GIT_REF, 'git_commit': GIT_COMMIT, 'git_dirty': GIT_DIRTY, 'gpu': GPU_LABEL,
                'platform': PLATFORM, 'session': SESSION, 'execution': RUN_EXECUTION,
                'syntax_check': RUN_SYNTAX_CHECK, 'dead_db_policy': DEAD_DB_POLICY,
                'concurrent_gpu_workers': len(GPU_IDS)})
    if job['condition'] in ADAPTER_CONDITIONS:
        rec['adapter'] = str(job['adapter_dir'])
        rec['train_global_step'] = (job.get('train_meta') or {}).get('global_step')
    if job['backend'] == 'llamacpp':
        rec['gguf'] = job.get('gguf_repo_path')
        rec['limit'] = GGUF_BENCH_LIMIT
    rec.update(extra)
    return rec


def _reuse_from_hub(job, jdir):
    prefix = HUB_EVAL_PREFIX + '/jobs/' + job['job_id']
    if not SKIP_COMPLETED_JOBS or (prefix + '/job.json') not in hub_files():
        return None
    rec = json.loads(hub_download_file(prefix + '/job.json').read_text(encoding='utf-8'))
    if rec.get('status') != 'completed':
        return None
    origin = rec.get('platform') or 'colab'
    if REUSE_SAME_COMMIT_ONLY and rec.get('git_commit') != GIT_COMMIT:
        print('[' + job['job_id'] + '] Hub outputs come from commit ' + str(rec.get('git_commit'))[:10]
              + '; recomputing on ' + GIT_COMMIT[:10] + '.')
        return None
    if REUSE_SAME_PLATFORM_ONLY and origin != PLATFORM:
        print('[' + job['job_id'] + '] Hub outputs come from ' + origin + '; recomputing on ' + PLATFORM + '.')
        return None
    src = hub_download_prefix(prefix, HUB_CACHE)
    if jdir.exists():
        shutil.rmtree(jdir)
    shutil.copytree(src, jdir)
    if not (jdir / 'predictions.scored.jsonl').exists():
        return None
    return rec


def _time_left_for_job():
    elapsed_min = (time.time() - PIPELINE_T0) / 60
    return elapsed_min + MAX_JOB_MINUTES <= SESSION_BUDGET_MINUTES - RESERVE_MINUTES


def _predict(job, gpu, scope, echo=False):
    """Runs predict on one GPU and checks adapter provenance. Returns a context dict."""
    jid = job['job_id']
    jdir = EVAL_DIR / 'jobs' / jid
    if jdir.exists():
        shutil.rmtree(jdir)
    jdir.mkdir(parents=True)
    pred = jdir / 'predictions.jsonl'
    started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
    t0 = time.time()

    if job['backend'] == 'llamacpp':
        cmd = ('python app.py predict --config ' + job['config'] + ' --backend llamacpp --gguf '
               + str(job['gguf_local']) + ' --split ' + SPLIT + ' --limit ' + str(GGUF_BENCH_LIMIT)
               + ' --out ' + str(pred) + ' --tag ' + jid)
    else:
        cmd = ('python app.py predict --config ' + job['config'] + ' --backend hf --split ' + SPLIT
               + ' --limit ' + str(PREDICT_LIMIT) + ' --schema-mode ' + job['prompt_schema_mode']
               + ' --out ' + str(pred) + ' --tag ' + jid)
        if job['condition'] in ADAPTER_CONDITIONS:
            cmd += ' --adapter ' + str(job['adapter_dir'])
    run_step('predict-' + jid, cmd, log_path=jdir / 'predict.log', env={'CUDA_VISIBLE_DEVICES': gpu},
             echo=echo, scope=scope)

    # Adapter provenance (history/027.md): a fine-tuned score is only valid if the
    # verified loader confirmed every LoRA tensor was loaded.
    log_lines = (jdir / 'predict.log').read_text(encoding='utf-8', errors='replace').splitlines()
    verified = [ln.split('| ', 1)[-1] for ln in log_lines if 'Adapter verified:' in ln]
    remapped = [ln.split('| ', 1)[-1] for ln in log_lines if 'Remapped' in ln and 'adapter tensor' in ln]
    classes = [ln.split('Loaded model class:', 1)[-1].strip() for ln in log_lines if 'Loaded model class:' in ln]
    loaded_adapter = any('Loading LoRA adapter' in ln for ln in log_lines)
    if job['backend'] == 'hf' and job['condition'] in ADAPTER_CONDITIONS and not verified:
        raise RuntimeError(jid + ': the predict log has no Adapter verified line; refusing to score a '
                           'possibly unadapted model.')
    if job['condition'] == 'zero_shot' and loaded_adapter:
        raise RuntimeError(jid + ': an adapter was loaded for a zero-shot job.')

    return {'jdir': jdir, 'pred': pred, 'started': started, 't0': t0, 'predict_wall_s': round(time.time() - t0, 1),
            'gpu': gpu, 'verified': verified, 'remapped': remapped, 'classes': classes}


def _evaluate(job, ctx, scope, echo=False):
    """Scores a finished prediction file, writes job.json and uploads the job folder."""
    jid = job['job_id']
    jdir, pred = ctx['jdir'], ctx['pred']
    report = jdir / 'report.md'
    t_eval = time.time()
    flags = ((' --execute' if RUN_EXECUTION else '') + (' --validate-syntax' if RUN_SYNTAX_CHECK else '')
             + ' --dead-db-policy ' + DEAD_DB_POLICY)
    run_step('evaluate-' + jid, 'python app.py evaluate --predictions ' + str(pred) + flags + ' --out ' + str(report),
             log_path=jdir / 'evaluate.log', echo=echo, scope=scope)
    for required in ('predictions.scored.jsonl', 'report.json', 'predictions.meta.json'):
        if not (jdir / required).exists():
            raise RuntimeError(jid + ': missing ' + required + ' after evaluation.')

    meta = json.loads((jdir / 'predictions.meta.json').read_text(encoding='utf-8'))
    rec = _job_record(
        job, status='completed', started=ctx['started'],
        finished=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        wall_s=round(time.time() - ctx['t0'], 1), predict_wall_s=ctx['predict_wall_s'],
        evaluate_wall_s=round(time.time() - t_eval, 1),
        gpu=GPUS.get(ctx['gpu'], GPU_NAME), gpu_index=ctx['gpu'],
        n_rows=meta.get('n_instances'), seconds_per_item=meta.get('seconds_per_item'),
        generation_seconds=meta.get('total_seconds'), mean_prompt_tokens=meta.get('mean_prompt_tokens'),
        batch_size=meta.get('batch_size'), max_new_tokens=meta.get('max_new_tokens'),
        temperature=meta.get('temperature'), adapter_verified=ctx['verified'],
        adapter_remapped=ctx['remapped'], model_class=ctx['classes'],
        n_schema_trimmed=meta.get('n_schema_trimmed'), n_prompt_over_budget=meta.get('n_prompt_over_budget'),
        prompt_reserve_tokens=meta.get('prompt_reserve_tokens'),
        gguf_merge_metadata=job.get('gguf_merge_metadata'), python=platform.python_version(),
    )
    (jdir / 'job.json').write_text(json.dumps(rec, indent=2, default=str), encoding='utf-8')
    rec['uploaded'] = hub_upload_folder(jdir, HUB_EVAL_PREFIX + '/jobs/' + jid, '[final-eval] ' + EVAL_TAG + ' ' + jid)
    if UPLOAD_TO_HUB and not rec['uploaded']:
        print('WARNING: ' + jid + ' finished but its upload failed; it is only kept in this session.', flush=True)
    return rec


# --------------------------------------------------------------------------- #
# Reuse finished jobs (main thread), queue the rest
# --------------------------------------------------------------------------- #
print('Planned jobs (' + str(len(JOBS)) + '):')
for job in JOBS:
    print('  ' + job['job_id'])
print()

GPU_QUEUE = queue.Queue()
EVAL_QUEUE = queue.Queue()
for job in JOBS:
    jid = job['job_id']
    if job['condition'] in ADAPTER_CONDITIONS and not job['adapter_ok']:
        print('[' + jid + '] Expected job cannot run: no final adapter for ' + job['run_name'] + ' on the Hub.')
        _set_result(jid, _job_record(job, status='missing_adapter', error='no final adapter on the Hub'),
                    'missing_adapter')
        continue
    try:
        reused = _reuse_from_hub(job, EVAL_DIR / 'jobs' / jid)
    except Exception as exc:
        print('[' + jid + '] Could not check the Hub for completed outputs (' + str(exc) + '); recomputing.')
        reused = None
    if reused is not None:
        origin = reused.get('platform') or 'colab'
        print('[' + jid + '] Reusing completed outputs from the Hub (' + origin + ', commit '
              + str(reused.get('git_commit'))[:10] + ').')
        _set_result(jid, reused, 'completed')
    else:
        GPU_QUEUE.put(job)
print('Jobs to compute: ' + str(GPU_QUEUE.qsize()))


# --------------------------------------------------------------------------- #
# Workers
# --------------------------------------------------------------------------- #
def gpu_worker(gpu, stagger_s):
    scope = 'gpu' + gpu
    with _state_lock:
        WORKERS.setdefault(scope, {}).update(job=None, step=None, started=None, last_line='waiting to start')
    delay = stagger_s
    while delay > 0 and not STOP_EVENT.is_set():
        time.sleep(min(5, delay))
        delay -= 5
    while not STOP_EVENT.is_set():
        try:
            job = GPU_QUEUE.get_nowait()
        except queue.Empty:
            break
        jid = job['job_id']
        if not _time_left_for_job():
            with _results_lock:
                SKIPPED_FOR_TIME.append(jid)
            print('[' + scope + '] Not starting ' + jid + ': too little session time left.', flush=True)
            continue
        with _state_lock:
            WORKERS[scope]['job'] = jid
        try:
            ctx = _predict(job, gpu, scope)
            EVAL_QUEUE.put((job, ctx))
        except KeyboardInterrupt:
            _set_result(jid, _job_record(job, status='interrupted'), 'interrupted')
            break
        except Exception as exc:
            _set_result(jid, _job_record(job, status='failed', error=str(exc), gpu_index=gpu), 'failed')
            print('[' + scope + '] ' + jid + ' FAILED: ' + str(exc) + LF + traceback.format_exc(), flush=True)
            notify('job failed', jid + ' (predict, GPU ' + gpu + '): ' + str(exc)[-1500:], priority='high',
                   tags=['warning'])
    with _state_lock:
        WORKERS[scope].update(job=None, last_line='done')


def eval_worker(idx):
    scope = 'eval' + str(idx)
    with _state_lock:
        WORKERS.setdefault(scope, {}).update(job=None, step=None, started=None, last_line='idle')
    while True:
        item = EVAL_QUEUE.get()
        if item is None:
            break
        job, ctx = item
        jid = job['job_id']
        with _state_lock:
            WORKERS[scope]['job'] = jid
        try:
            rec = _evaluate(job, ctx, scope)
            _set_result(jid, rec, 'completed')
            print('[' + scope + '] ' + jid + ' completed.', flush=True)
        except KeyboardInterrupt:
            _set_result(jid, _job_record(job, status='interrupted'), 'interrupted')
        except Exception as exc:
            _set_result(jid, _job_record(job, status='failed', error=str(exc), gpu_index=ctx.get('gpu')), 'failed')
            print('[' + scope + '] ' + jid + ' FAILED: ' + str(exc) + LF + traceback.format_exc(), flush=True)
            notify('job failed', jid + ' (evaluate): ' + str(exc)[-1500:], priority='high', tags=['warning'])
        with _state_lock:
            WORKERS[scope].update(job=None, last_line='idle')


def _gpu_usage():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return {}
    usage = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) == 4:
            idx, used, total, util = parts
            try:
                usage[idx] = (format(float(used) / 1024, '.1f') + '/' + format(float(total) / 1024, '.1f')
                              + ' GiB ' + util + '%')
            except ValueError:
                pass
    return usage


def _ram_available_gb():
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / 1024 ** 2
    except Exception:
        pass
    return float('nan')


def heartbeat():
    usage = _gpu_usage()
    now = datetime.datetime.now(datetime.timezone.utc).strftime('%H:%M:%S')
    elapsed = (time.time() - PIPELINE_T0) / 60
    with _results_lock:
        n_done = sum(1 for v in STATUS['jobs'].values() if v == 'completed')
    lines = ['--- ' + now + ' UTC | session ' + format(elapsed, '.0f') + '/' + str(SESSION_BUDGET_MINUTES)
             + ' min | completed ' + str(n_done) + '/' + str(len(JOBS)) + ' | GPU queue ' + str(GPU_QUEUE.qsize())
             + ' | eval queue ' + str(EVAL_QUEUE.qsize()) + ' | RAM available '
             + format(_ram_available_gb(), '.1f') + ' GB']
    for gid in GPU_IDS:
        lines.append('  GPU' + gid + ' ' + usage.get(gid, '?'))
    with _state_lock:
        snapshot = {k: dict(v) for k, v in WORKERS.items() if k != 'main'}
    for scope in sorted(snapshot):
        st = snapshot[scope]
        if st.get('step'):
            started = st.get('started') or time.time()
            state = st['step'] + ' ' + format((time.time() - started) / 60, '.0f') + ' min'
        else:
            state = 'idle'
        last = (st.get('last_line') or '')[-110:]
        lines.append('  [' + scope + '] ' + state + ' | ' + last)
    print(LF.join(lines), flush=True)


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
eval_threads = [threading.Thread(target=eval_worker, args=(i,), name='eval-' + str(i), daemon=True)
                for i in range(N_EVAL_WORKERS)]
gpu_threads = [threading.Thread(target=gpu_worker, args=(gid, i * WORKER_STAGGER_S), name='gpu-' + gid, daemon=True)
               for i, gid in enumerate(GPU_IDS)]
for t in eval_threads + gpu_threads:
    t.start()

try:
    last_beat = time.time()
    while any(t.is_alive() for t in gpu_threads):
        time.sleep(5)
        if time.time() - last_beat >= HEARTBEAT_S:
            heartbeat()
            last_beat = time.time()
    for _ in eval_threads:
        EVAL_QUEUE.put(None)
    while any(t.is_alive() for t in eval_threads):
        time.sleep(5)
        if time.time() - last_beat >= HEARTBEAT_S:
            heartbeat()
            last_beat = time.time()
except KeyboardInterrupt:
    print('Interrupt received: stopping all workers...', flush=True)
    STOP_EVENT.set()
    _stop_all_procs()
    for _ in eval_threads:
        EVAL_QUEUE.put(None)
    for t in gpu_threads + eval_threads:
        t.join(timeout=120)
    STATUS['outcome'] = 'interrupted'
    notify('interrupted', 'Session ' + SESSION + ' was stopped by the user.', priority='high', tags=['stop_sign'])
    raise

heartbeat()
done = [j['job_id'] for j in JOBS if JOB_RESULTS.get(j['job_id'], {}).get('status') == 'completed']
not_completed = [(j['job_id'], JOB_RESULTS.get(j['job_id'], {}).get('status', 'not_started')) for j in JOBS
                 if JOB_RESULTS.get(j['job_id'], {}).get('status') != 'completed']
print()
print('Expected: ' + str(len(JOBS)) + ' | Completed: ' + str(len(done)) + ' | Not completed: ' + str(len(not_completed)))
for jid, status in not_completed:
    print('  ' + status + ': ' + jid)
if not_completed:
    print('Commit a new version to finish them; completed ones on this commit/platform are reused.')
notify('jobs finished', 'Completed ' + str(len(done)) + '/' + str(len(JOBS)) + '. Session ' + SESSION + '.',
       priority='high' if not_completed else 'default',
       tags=['warning'] if not_completed else ['white_check_mark'])
