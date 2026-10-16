# 7. Plan jobs, benchmarks and exports; reuse finished work from the Hub
QUANT_ORDER = ['F16', 'Q8_0', 'Q6_K', 'Q5_K_M', 'Q5_K_S', 'Q4_K_M', 'Q4_K_S', 'Q4_0', 'Q3_K_M', 'Q2_K']


def _qkey(q):
    return QUANT_ORDER.index(q) if q in QUANT_ORDER else len(QUANT_ORDER)


BY_VARIANT = {(e['model'], e['variant']): e for e in EXP}


def _job(e, condition, quant, prompt_mode, priority):
    ref_id = e['run_name'] + '__fine_tuned' + ('' if condition == 'fine_tuned' else '__' + prompt_mode + '_prompts')
    return {**e, 'condition': condition, 'quant': quant, 'schema_mode': prompt_mode,
            'prompt_schema_mode': prompt_mode, 'train_schema_mode': e['schema_mode'],
            'reference_job_id': ref_id, 'job_id': ref_id + '__gguf_' + quant.lower() + '_cpu', 'priority': priority}


JOBS = [_job(e, 'fine_tuned', ACCURACY_QUANT, e['schema_mode'], 0) for e in EXP]
if RUN_CROSS_SCHEMA:
    for (m, v), e in BY_VARIANT.items():
        pruned = BY_VARIANT.get((m, 'pruned'))
        if v == 'enhanced' and pruned is not None:
            JOBS.append(_job(e, 'cross_schema', ACCURACY_QUANT, pruned['schema_mode'], 1))
for q in EXTRA_ACCURACY_QUANTS:
    if q == ACCURACY_QUANT:
        continue
    for e in EXP:
        if e['variant'] in EXTRA_QUANT_VARIANTS:
            JOBS.append(_job(e, 'fine_tuned', q, e['schema_mode'], 2))
JOBS.sort(key=lambda j: j['priority'])

# One run per model for throughput/footprint: speed depends on the architecture, not the adapter.
BENCH_PLAN = {}
for e in sorted(EXP, key=lambda x: (not x['adapter_ok'], x['variant'] != 'enhanced')):
    BENCH_PLAN.setdefault(e['model'], e)


def _provenance():
    # SERVER_MMAP is defined in cell 9; records written before it exist carry None (history/036.md).
    return {'edge_tag': EDGE_TAG, 'edge_ids_sha': EDGE_IDS_SHA, 'n_planned': len(EDGE_IDS),
            'reference_eval_tag': REFERENCE_EVAL_TAG, 'git_ref': GIT_REF, 'git_commit': GIT_COMMIT,
            'git_dirty': GIT_DIRTY, 'llama_cpp_commit': LLAMA_COMMIT, 'llama_cpp_version': LLAMA_VERSION,
            'cpu_model': CPU_MODEL, 'logical_cpus': LOGICAL_CPUS, 'physical_cores': PHYSICAL_CORES,
            'simd_flags': SIMD_FLAGS, 'cgroup_cpu_limit': CGROUP_CPU_LIMIT,
            'ram_total_gb': round(RAM_TOTAL_GB, 1), 'platform': PLATFORM, 'session': SESSION,
            'server_mmap_mode': (globals().get('SERVER_MMAP') or {}).get('mode')}


def _reuse_mismatch(rec):
    checks = [('edge_ids_sha', EDGE_IDS_SHA), ('llama_cpp_commit', LLAMA_COMMIT), ('cpu_model', CPU_MODEL)]
    if REUSE_SAME_COMMIT_ONLY:
        checks.append(('git_commit', GIT_COMMIT))
    for key, want in checks:
        if rec.get(key) != want:
            return key + ' differs (' + str(rec.get(key))[:24] + ' vs ' + str(want)[:24] + ')'
    return None


def _bench_rows_complete(rec):
    rows = rec.get('bench_rows') or []
    return bool(rows) and not any(r.get('error') for r in rows)


def _footprint_missing(rec):
    """Latency quantisations without a successful llama-server footprint (history/036.md)."""
    ok = {f.get('quant') for f in rec.get('footprint') or [] if not f.get('error')}
    return sorted(set(LATENCY_QUANTS) - ok, key=_qkey)


def _job_record(job, **extra):
    keep = ('job_id', 'reference_job_id', 'model', 'variant', 'config', 'run_name', 'schema_mode', 'condition',
            'quant', 'prompt_schema_mode', 'train_schema_mode', 'priority', 'fingerprint_now',
            'fingerprint_trained', 'fingerprint_match')
    rec = {k: job.get(k) for k in keep}
    rec.update(_provenance())
    rec['backend'] = 'llamacpp'
    rec.update(extra)
    return rec


JOB_RESULTS = {}
BENCH_RESULTS = {}
REUSED_BENCH_ROWS = {}  # model -> Hub bench record whose llama-bench rows are kept; footprint re-measured
EXPORT_RECORDS = {}
SKIPPED_FOR_TIME = []
files = hub_files(refresh=True)

# Export metadata from earlier sessions (sizes, hashes, merge verification).
for path in sorted(f for f in files if f.startswith(EDGE_PREFIX + '/exports/') and f.endswith('.json')):
    try:
        rec = json.loads(hub_download_file(path).read_text(encoding='utf-8'))
        rec['from_hub'] = True
        EXPORT_RECORDS[rec['run_name']] = rec
    except Exception as exc:
        print('Could not read ' + path + ': ' + str(exc))

PENDING_JOBS = []
for job in JOBS:
    jid = job['job_id']
    if not job['adapter_ok']:
        JOB_RESULTS[jid] = _job_record(job, status='missing_adapter', error='no final adapter on the Hub')
        print('[' + jid + '] cannot run: no final adapter.')
        continue
    prefix = EDGE_PREFIX + '/jobs/' + jid
    reused = None
    if SKIP_COMPLETED_JOBS and (prefix + '/job.json') in files:
        rec = json.loads(hub_download_file(prefix + '/job.json').read_text(encoding='utf-8'))
        why = _reuse_mismatch(rec) if rec.get('status') == 'completed' else 'status ' + str(rec.get('status'))
        if why:
            print('[' + jid + '] Hub outputs not reused: ' + why + '.')
        else:
            hub_download_patterns([prefix + '/*'], HUB_CACHE)
            jdir = EDGE_DIR / 'jobs' / jid
            if jdir.exists():
                shutil.rmtree(jdir)
            shutil.copytree(HUB_CACHE / prefix, jdir)
            if (jdir / 'predictions.scored.jsonl').exists():
                reused = rec
    if reused is not None:
        JOB_RESULTS[jid] = reused
        STATUS['jobs'][jid] = 'completed'
        print('[' + jid + '] reusing completed outputs from the Hub.')
    else:
        PENDING_JOBS.append(job)

PENDING_BENCH = []
for model, e in BENCH_PLAN.items():
    if not e['adapter_ok']:
        BENCH_RESULTS[model] = {'model': model, 'run_name': e['run_name'], 'status': 'missing_adapter'}
        continue
    path = EDGE_PREFIX + '/bench/' + model + '/bench.json'
    if SKIP_COMPLETED_JOBS and path in files:
        rec = json.loads(hub_download_file(path).read_text(encoding='utf-8'))
        why = _reuse_mismatch(rec)
        if why:
            print('[bench ' + model + '] Hub results not reused: ' + why + '.')
        elif not _bench_rows_complete(rec):
            print('[bench ' + model + '] Hub results not reused: llama-bench rows missing or failed.')
        else:
            missing = _footprint_missing(rec)
            if not missing:
                BENCH_RESULTS[model] = rec
                print('[bench ' + model + '] reusing results from the Hub.')
                continue
            # An earlier session measured throughput but its footprint failed (e.g. the
            # --no-mmap rejection, history/036.md): keep the llama-bench rows, redo the footprint.
            REUSED_BENCH_ROWS[model] = rec
            print('[bench ' + model + '] reusing llama-bench rows; footprint incomplete for '
                  + ', '.join(missing) + ', re-measuring it.')
    PENDING_BENCH.append(model)

NEEDED_QUANTS = {}
for job in PENDING_JOBS:
    NEEDED_QUANTS.setdefault(job['run_name'], set()).add(job['quant'])
for model in PENDING_BENCH:
    NEEDED_QUANTS.setdefault(BENCH_PLAN[model]['run_name'], set()).update(LATENCY_QUANTS + [ACCURACY_QUANT])

print()
print('Jobs planned : ' + str(len(JOBS)) + ' | reused ' + str(sum(1 for r in JOB_RESULTS.values()
                                                                   if r.get('status') == 'completed'))
      + ' | pending ' + str(len(PENDING_JOBS)))
for job in PENDING_JOBS:
    print('  [p' + str(job['priority']) + '] ' + job['job_id'])
print('Benchmarks pending: ' + (', '.join(m + (' (footprint only)' if m in REUSED_BENCH_ROWS else '')
                                          for m in PENDING_BENCH) or 'none'))
for rn, qs in NEEDED_QUANTS.items():
    print('  export ' + rn + ': ' + ', '.join(sorted(qs, key=_qkey)))
