# 7. Run the evaluation jobs: predict -> verify adapter -> evaluate -> upload
import platform

JOBS = []
for e in EXP:
    conditions = (['fine_tuned'] if e['adapter_ok'] else []) + (['zero_shot'] if RUN_ZERO_SHOT else [])
    for condition in conditions:
        JOBS.append({**e, 'condition': condition, 'backend': 'hf',
                     'job_id': e['run_name'] + '__' + condition})
# Fine-tuned jobs first: they are the primary results.
JOBS.sort(key=lambda j: j['condition'] != 'fine_tuned')


def _job_record(job, **extra):
    keep = ('job_id', 'model', 'variant', 'config', 'run_name', 'schema_mode', 'condition', 'backend',
            'fingerprint_now', 'fingerprint_trained', 'fingerprint_match')
    rec = {k: job.get(k) for k in keep}
    rec.update({'eval_tag': EVAL_TAG, 'split': SPLIT, 'limit': EVAL_LIMIT, 'sample_seed': SAMPLE_SEED,
                'git_commit': GIT_COMMIT, 'git_dirty': GIT_DIRTY, 'gpu': GPU_NAME, 'session': SESSION,
                'execution': RUN_EXECUTION, 'syntax_check': RUN_SYNTAX_CHECK})
    if job['condition'] == 'fine_tuned':
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
    src = hub_download_prefix(prefix, HUB_CACHE)
    if jdir.exists():
        shutil.rmtree(jdir)
    shutil.copytree(src, jdir)
    rec = json.loads((jdir / 'job.json').read_text(encoding='utf-8'))
    if rec.get('status') != 'completed' or not (jdir / 'predictions.scored.jsonl').exists():
        return None
    return rec


def run_job(job):
    jid = job['job_id']
    jdir = EVAL_DIR / 'jobs' / jid
    reused = _reuse_from_hub(job, jdir)
    if reused is not None:
        prev = str(reused.get('git_commit'))[:10]
        note = '' if reused.get('git_commit') == GIT_COMMIT else ' (produced at commit ' + prev + ')'
        print('[' + jid + '] Reusing completed outputs from the Hub' + note + '.', flush=True)
        return reused

    if jdir.exists():
        shutil.rmtree(jdir)
    jdir.mkdir(parents=True)
    pred = jdir / 'predictions.jsonl'
    report = jdir / 'report.md'
    job_config = job['config']
    started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
    t0 = time.time()

    if job['backend'] == 'llamacpp':
        gguf_path = job['gguf_local']
        cmd = (f'python app.py predict --config {job_config} --backend llamacpp --gguf {gguf_path} '
               f'--split {SPLIT} --limit {GGUF_BENCH_LIMIT} --out {pred} --tag {jid}')
    else:
        cmd = (f'python app.py predict --config {job_config} --backend hf '
               f'--split {SPLIT} --limit {PREDICT_LIMIT} --out {pred} --tag {jid}')
        if job['condition'] == 'fine_tuned':
            cmd += ' --adapter ' + str(job['adapter_dir'])
    run_step('predict-' + jid, cmd, log_path=jdir / 'predict.log')

    # Adapter provenance (history/027.md): a fine-tuned score is only valid if the
    # verified loader confirmed every LoRA tensor was loaded.
    log_lines = (jdir / 'predict.log').read_text(encoding='utf-8', errors='replace').splitlines()
    verified = [ln.split('| ', 1)[-1] for ln in log_lines if 'Adapter verified:' in ln]
    remapped = [ln.split('| ', 1)[-1] for ln in log_lines if 'Remapped' in ln and 'adapter tensor' in ln]
    classes = [ln.split('Loaded model class:', 1)[-1].strip() for ln in log_lines if 'Loaded model class:' in ln]
    loaded_adapter = any('Loading LoRA adapter' in ln for ln in log_lines)
    if job['backend'] == 'hf' and job['condition'] == 'fine_tuned' and not verified:
        raise RuntimeError(jid + ': the predict log has no Adapter verified line; refusing to score a '
                           'possibly unadapted model.')
    if job['condition'] == 'zero_shot' and loaded_adapter:
        raise RuntimeError(jid + ': an adapter was loaded for a zero-shot job.')

    flags = (' --execute' if RUN_EXECUTION else '') + (' --validate-syntax' if RUN_SYNTAX_CHECK else '')
    run_step('evaluate-' + jid, f'python app.py evaluate --predictions {pred}{flags} --out {report}',
             log_path=jdir / 'evaluate.log')
    for required in ('predictions.scored.jsonl', 'report.json', 'predictions.meta.json'):
        if not (jdir / required).exists():
            raise RuntimeError(jid + ': missing ' + required + ' after evaluation.')

    meta = json.loads((jdir / 'predictions.meta.json').read_text(encoding='utf-8'))
    rec = _job_record(
        job, status='completed', started=started,
        finished=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        wall_s=round(time.time() - t0, 1), n_rows=meta.get('n_instances'),
        seconds_per_item=meta.get('seconds_per_item'), generation_seconds=meta.get('total_seconds'),
        mean_prompt_tokens=meta.get('mean_prompt_tokens'), batch_size=meta.get('batch_size'),
        max_new_tokens=meta.get('max_new_tokens'), temperature=meta.get('temperature'),
        adapter_verified=verified, adapter_remapped=remapped, model_class=classes,
        gguf_merge_metadata=job.get('gguf_merge_metadata'), python=platform.python_version(),
    )
    (jdir / 'job.json').write_text(json.dumps(rec, indent=2, default=str), encoding='utf-8')
    rec['uploaded'] = hub_upload_folder(jdir, HUB_EVAL_PREFIX + '/jobs/' + jid,
                                        '[final-eval] ' + EVAL_TAG + ' ' + jid)
    if UPLOAD_TO_HUB and not rec['uploaded']:
        print('WARNING: ' + jid + ' finished but its upload failed; it is only kept in this runtime.')
    return rec


JOB_RESULTS = {}
print('Planned jobs (' + str(len(JOBS)) + '):')
for job in JOBS:
    print('  ' + job['job_id'])

for i, job in enumerate(JOBS, 1):
    jid = job['job_id']
    elapsed_min = (time.time() - T0) / 60
    if elapsed_min > MAX_SESSION_MINUTES:
        print(f'Session time {elapsed_min:.0f} min exceeds MAX_SESSION_MINUTES; not starting more jobs. '
              'Run all again in a fresh runtime to continue.')
        break
    print(LF + '#' * 72 + LF + f'# Job {i}/{len(JOBS)}: {jid}' + LF + '#' * 72, flush=True)
    try:
        JOB_RESULTS[jid] = run_job(job)
        STATUS['jobs'][jid] = 'completed'
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        STATUS['jobs'][jid] = 'failed'
        JOB_RESULTS[jid] = _job_record(job, status='failed', error=str(exc))
        print('[' + jid + '] FAILED: ' + str(exc), flush=True)
        notify('job failed', jid + ': ' + str(exc)[-1500:], priority='high', tags=['warning'])

done = [j for j, r in JOB_RESULTS.items() if r.get('status') == 'completed']
failed = [j for j, r in JOB_RESULTS.items() if r.get('status') == 'failed']
pending = [j['job_id'] for j in JOBS if j['job_id'] not in JOB_RESULTS]
print()
print('Completed: ' + str(len(done)) + ' | Failed: ' + str(len(failed)) + ' | Not started: ' + str(len(pending)))
for name, group in (('Failed', failed), ('Not started', pending)):
    for jid in group:
        print('  ' + name + ': ' + jid)
notify('jobs finished', 'Completed ' + str(len(done)) + ', failed ' + str(len(failed)) + ', not started '
       + str(len(pending)) + '. Session ' + SESSION + '.',
       priority='high' if failed or pending else 'default',
       tags=['warning'] if failed or pending else ['white_check_mark'])
