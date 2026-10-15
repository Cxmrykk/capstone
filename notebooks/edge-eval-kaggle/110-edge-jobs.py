# 10. Accuracy and latency jobs on the CPU: llama-server -> predictions -> app.py evaluate -> upload
EST_S_PER_ITEM = {}


def _s_per_item(job):
    key = (job['model'], job['quant'])
    if key in EST_S_PER_ITEM:
        return EST_S_PER_ITEM[key]
    base = (BENCH_RESULTS.get(job['model']) or {}).get('est_s_per_item')
    if base:
        return base * (1.0 if job['quant'] == ACCURACY_QUANT else 1.6)
    return DEFAULT_S_PER_ITEM


def run_edge_job(job):
    jid, rn, q = job['job_id'], job['run_name'], job['quant']
    ex = EXPORT_RECORDS.get(rn) or {}
    info = (ex.get('quants') or {}).get(q)
    jdir = EDGE_DIR / 'jobs' / jid
    if jdir.exists():
        shutil.rmtree(jdir)
    jdir.mkdir(parents=True)

    cfg = edge_config(job['config'], job['prompt_schema_mode'])
    tok, fitted, reserve = build_prompts(cfg, EDGE_RECORDS)
    stop = stop_strings(cfg, tok)
    threads = THREADS_FOR.get(job['model'], LOGICAL_CPUS)
    ctx = cfg.data.max_seq_length + cfg.generation.max_new_tokens + 64
    max_new = cfg.generation.max_new_tokens
    started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
    t_job = time.time()

    rows = []
    srv = LlamaServer(info['path'], threads, ctx, jdir / 'server.log')
    try:
        srv.start()
        print('[' + jid + '] server ready in ' + format(srv.load_s, '.1f') + ' s, RSS ' + str(srv.rss_after_load_mb)
              + ' MB, threads ' + str(threads), flush=True)
        tw = time.perf_counter()
        srv.complete(strip_leading_bos(tok, fitted[0].prompt), 1, stop)  # untimed warm-up
        warmup_ms = (time.perf_counter() - tw) * 1000
        t_loop = time.time()
        for i, (rec, f) in enumerate(zip(EDGE_RECORDS, fitted), 1):
            prompt = strip_leading_bos(tok, f.prompt)
            resp, err, attempts = {}, None, 0
            t0 = time.perf_counter()
            for attempts in range(1, 4):
                try:
                    resp = srv.complete(prompt, max_new, stop)
                    err = None
                    break
                except Exception as exc:
                    err = type(exc).__name__ + ': ' + str(exc)[:300]
                    if not srv.alive():
                        raise RuntimeError('llama-server died during the job: ' + err)
                    time.sleep(2 * attempts)
            wall_ms = (time.perf_counter() - t0) * 1000
            raw = resp.get('content', '') if resp else ''
            tm = resp.get('timings') or {}
            rows.append({
                'instance_id': rec.get('instance_id'), 'question': rec.get('question'),
                'gold_cypher': (rec.get('cypher') or '').strip(), 'predicted_raw': raw,
                'predicted_cypher': extract_cypher(raw),
                'database_reference': rec.get('database_reference'), 'data_source': rec.get('data_source'),
                'schema_mode': cfg.data.schema_mode, 'schema_chars': len(f.schema_text),
                'schema_chars_untrimmed': f.schema_chars_untrimmed, 'schema_trimmed': f.schema_trimmed,
                'prompt_over_budget': f.over_budget, 'prompt_tokens': len(f.prompt_ids),
                'edge_wall_ms': round(wall_ms, 1), 'edge_attempts': attempts, 'edge_request_error': err,
                'edge_tokens_evaluated': resp.get('tokens_evaluated'), 'edge_tokens_cached': resp.get('tokens_cached'),
                'edge_prompt_n': tm.get('prompt_n'), 'edge_prompt_ms': tm.get('prompt_ms'),
                'edge_prompt_tps': tm.get('prompt_per_second'), 'edge_predicted_n': tm.get('predicted_n'),
                'edge_predicted_ms': tm.get('predicted_ms'), 'edge_predicted_tps': tm.get('predicted_per_second'),
                'edge_stopped_limit': stopped_at_limit(resp) if resp else None,
                'edge_truncated': resp.get('truncated') if resp else None,
            })
            if i % 10 == 0 or i == len(EDGE_RECORDS):
                rate = (time.time() - t_loop) / i
                print('[' + jid + '] ' + str(i) + '/' + str(len(EDGE_RECORDS)) + ' | ' + format(rate, '.1f')
                      + ' s/item | ETA ' + format(rate * (len(EDGE_RECORDS) - i) / 60, '.0f') + ' min', flush=True)
        peak_rss = srv.rss_mb('VmHWM')
        load_s, rss_load = srv.load_s, srv.rss_after_load_mb
    finally:
        srv.stop()

    elapsed = time.time() - t_loop
    pred = jdir / 'predictions.jsonl'
    with pred.open('w', encoding='utf-8') as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + LF)
    meta = {'run_name': cfg.run_name, 'tag': jid, 'model_key': cfg.model.key, 'model_path': cfg.model.resolved_path(),
            'adapter': str(job['adapter_dir']), 'gguf': info['path'], 'backend': 'llamacpp',
            'backend_info': {'threads': threads, 'n_ctx': ctx, 'cache_prompt': CACHE_PROMPT, 'stop': stop,
                             'llama_cpp_commit': LLAMA_COMMIT},
            'split': SPLIT, 'n_instances': len(rows), 'schema_mode': cfg.data.schema_mode,
            'max_seq_length': cfg.data.max_seq_length, 'prompt_reserve_tokens': reserve,
            'n_schema_trimmed': sum(1 for f in fitted if f.schema_trimmed),
            'n_prompt_over_budget': sum(1 for f in fitted if f.over_budget),
            'max_new_tokens': max_new, 'temperature': 0.0, 'batch_size': 1,
            'total_seconds': round(elapsed, 2), 'seconds_per_item': round(elapsed / max(1, len(rows)), 3),
            'mean_prompt_tokens': round(sum(len(f.prompt_ids) for f in fitted) / max(1, len(fitted)), 1),
            'host': platform.node(), 'platform': platform.platform(),
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')}
    (jdir / 'predictions.meta.json').write_text(json.dumps(meta, indent=2, default=str), encoding='utf-8')

    flags = ((' --execute' if RUN_EXECUTION else '') + (' --validate-syntax' if RUN_SYNTAX_CHECK else '')
             + ' --dead-db-policy ' + DEAD_DB_POLICY)
    run_step('evaluate-' + jid, 'python app.py evaluate --predictions ' + str(pred) + flags + ' --out '
             + str(jdir / 'report.md'), log_path=jdir / 'evaluate.log', echo=False)
    for required in ('predictions.scored.jsonl', 'report.json'):
        if not (jdir / required).exists():
            raise RuntimeError(jid + ': missing ' + required + ' after evaluation.')

    EST_S_PER_ITEM[(job['model'], q)] = meta['seconds_per_item']
    rec = _job_record(
        job, status='completed', started=started,
        finished=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        wall_s=round(time.time() - t_job, 1), n_rows=len(rows), seconds_per_item=meta['seconds_per_item'],
        generation_seconds=meta['total_seconds'], threads=threads, server_ctx=ctx,
        server_load_s=round(load_s, 2), rss_after_load_mb=rss_load, peak_rss_mb=peak_rss,
        warmup_ms=round(warmup_ms, 1), gguf_file=info['file'], gguf_size_bytes=info['size_bytes'],
        gguf_sha256=info['sha256'], stop_strings=stop, edge_stop_mode=EDGE_STOP_MODE, cache_prompt=CACHE_PROMPT,
        prompt_reserve_tokens=reserve, n_schema_trimmed=meta['n_schema_trimmed'],
        n_prompt_over_budget=meta['n_prompt_over_budget'],
        n_request_errors=sum(1 for r in rows if r['edge_request_error']),
        n_stopped_limit=sum(1 for r in rows if r['edge_stopped_limit']),
        merge_adapter_verified=ex.get('adapter_verified'),
        merge_model_class=(ex.get('merge_metadata') or {}).get('model_class'),
        python=platform.python_version(),
    )
    (jdir / 'job.json').write_text(json.dumps(rec, indent=2, default=str), encoding='utf-8')
    rec['uploaded'] = hub_upload_folder(jdir, EDGE_PREFIX + '/jobs/' + jid, '[edge-eval] ' + EDGE_TAG + ' ' + jid)
    if UPLOAD_TO_HUB and not rec['uploaded']:
        print('WARNING: ' + jid + ' finished but its upload failed; it is only kept in this session.')
    return rec


for i, job in enumerate(PENDING_JOBS, 1):
    jid = job['job_id']
    info = ((EXPORT_RECORDS.get(job['run_name']) or {}).get('quants') or {}).get(job['quant'])
    if not info or not Path(info['path']).exists():
        status = (EXPORT_RECORDS.get(job['run_name']) or {}).get('status', 'not_exported')
        JOB_RESULTS[jid] = _job_record(job, status='export_' + str(status),
                                       error=(EXPORT_RECORDS.get(job['run_name']) or {}).get('error'))
        print('[' + jid + '] not run: no local ' + job['quant'] + ' GGUF (export ' + str(status) + ').')
        continue
    need = len(EDGE_RECORDS) * _s_per_item(job) / 60 + 5
    if minutes_left() < need:
        SKIPPED_FOR_TIME.append(jid)
        JOB_RESULTS[jid] = _job_record(job, status='not_started',
                                       error='needs ~' + format(need, '.0f') + ' min, '
                                             + format(minutes_left(), '.0f') + ' min left')
        print('[' + jid + '] not started: needs ~' + format(need, '.0f') + ' min.')
        continue
    print(LF + '#' * 72 + LF + '# Job ' + str(i) + '/' + str(len(PENDING_JOBS)) + ': ' + jid + ' (~'
          + format(need, '.0f') + ' min)' + LF + '#' * 72, flush=True)
    try:
        JOB_RESULTS[jid] = run_edge_job(job)
        STATUS['jobs'][jid] = 'completed'
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        STATUS['jobs'][jid] = 'failed'
        JOB_RESULTS[jid] = _job_record(job, status='failed', error=str(exc)[-2000:])
        print('[' + jid + '] FAILED: ' + str(exc), flush=True)
        notify('job failed', jid + ': ' + str(exc)[-1500:], priority='high', tags=['warning'])

done = [j['job_id'] for j in JOBS if JOB_RESULTS.get(j['job_id'], {}).get('status') == 'completed']
not_done = [(j['job_id'], JOB_RESULTS.get(j['job_id'], {}).get('status', 'not_started')) for j in JOBS
            if JOB_RESULTS.get(j['job_id'], {}).get('status') != 'completed']
print()
print('Expected: ' + str(len(JOBS)) + ' | Completed: ' + str(len(done)) + ' | Not completed: ' + str(len(not_done)))
for jid, status in not_done:
    print('  ' + status + ': ' + jid)
notify('jobs finished', 'Completed ' + str(len(done)) + '/' + str(len(JOBS)) + '. Session ' + SESSION + '.',
       priority='high' if not_done else 'default', tags=['warning'] if not_done else ['white_check_mark'])
