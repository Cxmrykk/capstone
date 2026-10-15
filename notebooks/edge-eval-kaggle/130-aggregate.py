# 12. Aggregate the CPU jobs into thesis tables (CIs, paired tests vs GPU, latency, footprint, integrity)
OUT_DIR = REPO_DIR / 'artifacts' / 'thesis-edge' / EDGE_TAG
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
(OUT_DIR / 'tables').mkdir(parents=True)


def _flat(value):
    if isinstance(value, (list, tuple, set)):
        return '; '.join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, default=str)
    if isinstance(value, Path):
        return str(value)
    return value


def _g(row, key):
    value = row.get(key) if row is not None else None
    try:
        return float(value) if value is not None and pd.notna(value) else NAN
    except (TypeError, ValueError):
        return NAN


def latency_stats(df):
    out = {}
    for col, name in (('edge_wall_ms', 'wall_ms'), ('edge_prompt_ms', 'prefill_ms'), ('edge_predicted_ms', 'decode_ms')):
        s = _num(df, col)
        if len(s):
            out[name + '_mean'] = float(s.mean())
            out[name + '_p50'] = float(s.quantile(0.5))
            out[name + '_p90'] = float(s.quantile(0.9))
            out[name + '_p95'] = float(s.quantile(0.95))
            out[name + '_max'] = float(s.max())
    pn, gn = _num(df, 'edge_tokens_evaluated'), _num(df, 'edge_predicted_n')
    if len(pn):
        out['prompt_tokens_server_mean'] = float(pn.mean())
    if len(gn):
        out['generated_tokens_mean'] = float(gn.mean())
    pp, tg = _num(df, 'edge_prompt_tps'), _num(df, 'edge_predicted_tps')
    if len(pp):
        out['prefill_tps_median'] = float(pp.median())
    if len(tg):
        out['decode_tps_median'] = float(tg.median())
    wall, pre = _num(df, 'edge_wall_ms'), _num(df, 'edge_prompt_ms')
    if len(wall) and wall.sum() > 0:
        out['prefill_share_of_wall'] = float(pre.sum() / wall.sum()) if len(pre) else NAN
        out['queries_per_hour'] = float(3.6e6 / wall.mean())
    out['n_stopped_limit'] = int(df['edge_stopped_limit'].eq(True).sum()) if 'edge_stopped_limit' in df else 0
    out['n_request_errors'] = int(df['edge_request_error'].notna().sum()) if 'edge_request_error' in df else 0
    return out


def _error_category(r):
    if not str(r.get('predicted_cypher') or '').strip():
        return 'empty_output'
    status = r.get('execution_status')
    if status is None or (isinstance(status, float) and math.isnan(status)):
        return 'not_executed'
    if status in ('connection_failed', 'excluded_dead_db', 'gold_failed'):
        return status
    if r.get('pred_executed_ok') is not True:
        if r.get('syntax_valid') is False:
            return 'invalid_query'
        if r.get('syntax_valid') is True:
            return 'runtime_error'
        return 'execution_error'
    return 'correct' if r.get('execution_match') is True else 'wrong_result'


# --------------------------------------------------------------------------- #
# Per-job overview, with the GPU reference restricted to the same rows
# --------------------------------------------------------------------------- #
overall, long_frames, err_rows = [], [], []
for jid in EDGE_JOB_IDS:
    rec, df = ALL_JOBS[jid]['job'], ALL_JOBS[jid]['rows']
    row = {k: rec.get(k) for k in ('job_id', 'reference_job_id', 'run_name', 'model', 'variant', 'condition', 'quant',
                                   'prompt_schema_mode', 'train_schema_mode', 'threads', 'server_load_s',
                                   'rss_after_load_mb', 'peak_rss_mb', 'warmup_ms', 'seconds_per_item',
                                   'gguf_size_bytes', 'gguf_sha256', 'git_commit', 'llama_cpp_commit', 'cpu_model',
                                   'session')}
    row['job_id'] = jid
    row['gguf_size_mb'] = _g(rec, 'gguf_size_bytes') / 1e6
    row.update(job_metrics(df))
    row.update(latency_stats(df))
    row['prompt_tokens_mean'] = float(_num(df, 'prompt_tokens').mean()) if len(_num(df, 'prompt_tokens')) else NAN
    ref = ALL_JOBS.get(rec.get('reference_job_id'))
    tagged = df.copy()
    if ref is not None:
        rr = ref['rows'][ref['rows']['instance_id'].isin(set(df['instance_id']))]
        row.update({'gpu_' + k: v for k, v in job_metrics(rr).items()})
        rref = rr.drop_duplicates('instance_id').set_index('instance_id')
        for col in ('predicted_cypher', 'exact_match', 'correct', 'judged'):
            tagged['gpu_' + col] = tagged['instance_id'].map(rref[col])
    overall.append(row)
    for k in ('run_name', 'model', 'variant', 'condition', 'quant', 'prompt_schema_mode', 'train_schema_mode',
              'reference_job_id'):
        tagged[k] = rec.get(k)
    tagged['job_id'] = jid
    tagged['error_category'] = [_error_category(r) for r in tagged.to_dict('records')]
    long_frames.append(tagged)
    counts = tagged['error_category'].value_counts().to_dict()
    err_rows.append({'job_id': jid, **counts})

OVERALL = pd.DataFrame(overall)
EDGE_LONG = pd.concat(long_frames, ignore_index=True) if long_frames else pd.DataFrame()
ERRORS = pd.DataFrame(err_rows).fillna(0)
if not OVERALL.empty:
    OVERALL = OVERALL.sort_values(['model', 'variant', 'condition', 'quant']).reset_index(drop=True)
print('CPU jobs aggregated: ' + str(len(OVERALL)))


# --------------------------------------------------------------------------- #
# Paired comparisons on identical rows
# --------------------------------------------------------------------------- #
def edge_jid(run_name, condition, quant):
    for jid in EDGE_JOB_IDS:
        r = ALL_JOBS[jid]['job']
        if r.get('run_name') == run_name and r.get('condition') == condition and r.get('quant') == quant:
            return jid
    return None


def paired_compare(A, B, a_id, b_id, label, family):
    A = A.drop_duplicates('instance_id').set_index('instance_id')
    B = B.drop_duplicates('instance_id').set_index('instance_id')
    common = A.index.intersection(B.index)
    A, B = A.loc[common], B.loc[common]
    if not len(common):
        return []
    base = {'comparison': label, 'family': family, 'job_a': a_id, 'job_b': b_id,
            'identical_prediction_share': float((A['predicted_cypher'] == B['predicted_cypher']).mean()),
            'identical_normalised_share': float((A['predicted_cypher'].map(normalise_cypher)
                                                 == B['predicted_cypher'].map(normalise_cypher)).mean())}
    out = []
    for metric in ('exact_match', 'normalised_exact_match', 'execution_match'):
        if metric == 'execution_match':
            mask = (A['judged'] & B['judged']).to_numpy()
            a_ok, b_ok = A['correct'].to_numpy()[mask], B['correct'].to_numpy()[mask]
        else:
            a_ok, b_ok = A[metric].eq(True).to_numpy(), B[metric].eq(True).to_numpy()
        if not len(a_ok):
            continue
        stats = paired_diff_ci(a_ok, b_ok)
        stats.pop('bootstrap_p', None)
        a_only, b_only = int((a_ok & ~b_ok).sum()), int((b_ok & ~a_ok).sum())
        out.append({**base, 'metric': metric, 'n_pairs': len(a_ok), 'a_value': float(a_ok.mean()),
                    'b_value': float(b_ok.mean()), **stats, 'a_only_correct': a_only, 'b_only_correct': b_only,
                    'test': 'exact McNemar (two-sided)', 'p_value': mcnemar_exact(a_only, b_only),
                    'p_value_is_bound': False})
    ga = pd.to_numeric(A['gleu'], errors='coerce').fillna(0).to_numpy(float)
    gb = pd.to_numeric(B['gleu'], errors='coerce').fillna(0).to_numpy(float)
    stats = paired_diff_ci(ga, gb)
    p_raw = stats.pop('bootstrap_p')
    floor = 1.0 / BOOTSTRAP_SAMPLES
    out.append({**base, 'metric': 'sentence_gleu_mean', 'n_pairs': len(ga), 'a_value': float(ga.mean()),
                'b_value': float(gb.mean()), **stats, 'test': 'paired bootstrap (percentile)',
                'p_value': max(p_raw, floor), 'p_value_is_bound': p_raw < floor})
    for wall_col in ('edge_wall_ms', 'edge_prompt_ms'):
        if wall_col in A.columns and wall_col in B.columns:
            wa = pd.to_numeric(A[wall_col], errors='coerce')
            wb = pd.to_numeric(B[wall_col], errors='coerce')
            ok = (wa.notna() & wb.notna()).to_numpy()
            if ok.any():
                stats = paired_diff_ci(wa.to_numpy()[ok], wb.to_numpy()[ok])
                p_raw = stats.pop('bootstrap_p')
                for key in ('tost_ci_low', 'tost_ci_high', 'equivalent_within_margin'):
                    stats.pop(key, None)
                out.append({**base, 'metric': wall_col.replace('edge_', 'mean_'), 'n_pairs': int(ok.sum()),
                            'a_value': float(wa[ok].mean()), 'b_value': float(wb[ok].mean()), **stats,
                            'test': 'paired bootstrap (percentile)', 'p_value': max(p_raw, floor),
                            'p_value_is_bound': p_raw < floor})
    return out


COMPARISONS = []
for jid in EDGE_JOB_IDS:
    rec = ALL_JOBS[jid]['job']
    ref = ALL_JOBS.get(rec.get('reference_job_id'))
    if ref is not None:
        ids = set(ALL_JOBS[jid]['rows']['instance_id'])
        COMPARISONS.append((ref['rows'][ref['rows']['instance_id'].isin(ids)], ALL_JOBS[jid]['rows'],
                            rec['reference_job_id'], jid, jid + ': 4-bit HF (GPU) vs GGUF (CPU)', 'edge_vs_gpu'))
for e in EXP:
    for cond in ('fine_tuned', 'cross_schema'):
        a = edge_jid(e['run_name'], cond, ACCURACY_QUANT)
        for q in EXTRA_ACCURACY_QUANTS:
            b = edge_jid(e['run_name'], cond, q)
            if a and b:
                COMPARISONS.append((ALL_JOBS[a]['rows'], ALL_JOBS[b]['rows'], a, b,
                                    e['run_name'] + ' ' + cond + ': ' + ACCURACY_QUANT + ' vs ' + q, 'quant_fidelity'))
MODELS = list(dict.fromkeys(e['model'] for e in EXP))
for m in MODELS:
    enh, pru = BY_VARIANT.get((m, 'enhanced')), BY_VARIANT.get((m, 'pruned'))
    E = edge_jid(enh['run_name'], 'fine_tuned', ACCURACY_QUANT) if enh else None
    P = edge_jid(pru['run_name'], 'fine_tuned', ACCURACY_QUANT) if pru else None
    X = edge_jid(enh['run_name'], 'cross_schema', ACCURACY_QUANT) if enh else None
    for a, b, label, fam in ((E, P, m + ': enhanced vs exact-match (trained and prompted)', 'schema_filtering_edge'),
                             (E, X, m + ': enhanced-trained, enhanced vs exact-match prompts', 'schema_inference_effect_edge'),
                             (X, P, m + ': exact-match prompts, enhanced- vs exact-match-trained', 'schema_training_effect_edge')):
        if a and b:
            COMPARISONS.append((ALL_JOBS[a]['rows'], ALL_JOBS[b]['rows'], a, b, label, fam))
edge_enh = [(m, edge_jid(BY_VARIANT[(m, 'enhanced')]['run_name'], 'fine_tuned', ACCURACY_QUANT))
            for m in MODELS if (m, 'enhanced') in BY_VARIANT]
edge_enh = [x for x in edge_enh if x[1]]
for i, (ma, a) in enumerate(edge_enh):
    for mb, b in edge_enh[i + 1:]:
        COMPARISONS.append((ALL_JOBS[a]['rows'], ALL_JOBS[b]['rows'], a, b,
                            'CPU ' + ACCURACY_QUANT + ' enhanced: ' + ma + ' vs ' + mb, 'model_comparison_edge'))

paired_rows = []
for A, B, a, b, label, family in COMPARISONS:
    paired_rows.extend(paired_compare(A, B, a, b, label, family))
PAIRED = pd.DataFrame(paired_rows)
if not PAIRED.empty:
    PAIRED['p_holm'] = PAIRED.groupby(['family', 'metric'])['p_value'].transform(lambda s: holm(s.to_numpy()))
    PAIRED['significant_holm'] = PAIRED['p_holm'] < ALPHA
    floor_txt = '< ' + format(1.0 / BOOTSTRAP_SAMPLES, '.2g')
    PAIRED['p_value_display'] = [floor_txt if bound else format(p, '.3g')
                                 for p, bound in zip(PAIRED['p_value'], PAIRED['p_value_is_bound'])]

# --------------------------------------------------------------------------- #
# Fidelity summary, schema trade-off on the CPU, throughput, footprint, exports
# --------------------------------------------------------------------------- #
fid = []
if not PAIRED.empty:
    g = PAIRED[PAIRED['family'] == 'edge_vs_gpu']
    for jid, sub in g.groupby('job_b'):
        r = {'job_id': jid, 'reference_job_id': sub['job_a'].iloc[0],
             'identical_prediction_share': sub['identical_prediction_share'].iloc[0],
             'identical_normalised_share': sub['identical_normalised_share'].iloc[0]}
        for metric in ('exact_match', 'execution_match', 'sentence_gleu_mean'):
            m = sub[sub['metric'] == metric]
            if len(m):
                m = m.iloc[0]
                r.update({metric + '_gpu': m['a_value'], metric + '_cpu': m['b_value'],
                          metric + '_delta': m['difference_b_minus_a'], metric + '_delta_ci_low': m.get('diff_ci_low'),
                          metric + '_delta_ci_high': m.get('diff_ci_high'), metric + '_n': m['n_pairs'],
                          metric + '_p': m['p_value'], metric + '_p_holm': m['p_holm']})
                if metric == 'execution_match':
                    r['ea_equivalent_within_margin'] = m.get('equivalent_within_margin')
        fid.append(r)
FIDELITY = pd.DataFrame(fid)

trade = []
OV = OVERALL.set_index('job_id') if not OVERALL.empty else pd.DataFrame()
for m in MODELS:
    enh = BY_VARIANT.get((m, 'enhanced'))
    E = edge_jid(enh['run_name'], 'fine_tuned', ACCURACY_QUANT) if enh else None
    if not E:
        continue
    for label, other in (('exact_match_trained_and_prompted',
                          edge_jid(BY_VARIANT[(m, 'pruned')]['run_name'], 'fine_tuned', ACCURACY_QUANT)
                          if (m, 'pruned') in BY_VARIANT else None),
                         ('enhanced_trained_exact_match_prompts', edge_jid(enh['run_name'], 'cross_schema', ACCURACY_QUANT))):
        if not other:
            continue
        a, b = OV.loc[E], OV.loc[other]
        p = PAIRED[(PAIRED['job_a'] == E) & (PAIRED['job_b'] == other)] if not PAIRED.empty else PAIRED
        pe = p[p['metric'] == 'execution_match'].iloc[0] if len(p[p['metric'] == 'execution_match']) else None
        pw = p[p['metric'] == 'mean_wall_ms'].iloc[0] if len(p[p['metric'] == 'mean_wall_ms']) else None
        trade.append({
            'model': m, 'comparison': label, 'job_enhanced': E, 'job_filtered': other,
            'prompt_tokens_enhanced': _g(a, 'prompt_tokens_mean'), 'prompt_tokens_filtered': _g(b, 'prompt_tokens_mean'),
            'prefill_ms_p50_enhanced': _g(a, 'prefill_ms_p50'), 'prefill_ms_p50_filtered': _g(b, 'prefill_ms_p50'),
            'wall_ms_p50_enhanced': _g(a, 'wall_ms_p50'), 'wall_ms_p50_filtered': _g(b, 'wall_ms_p50'),
            'wall_ms_p95_enhanced': _g(a, 'wall_ms_p95'), 'wall_ms_p95_filtered': _g(b, 'wall_ms_p95'),
            'prefill_reduction_pct': 100 * (1 - _g(b, 'prefill_ms_mean') / _g(a, 'prefill_ms_mean'))
            if _g(a, 'prefill_ms_mean') else NAN,
            'wall_reduction_pct': 100 * (1 - _g(b, 'wall_ms_mean') / _g(a, 'wall_ms_mean')) if _g(a, 'wall_ms_mean') else NAN,
            'paired_wall_ms_delta': pw['difference_b_minus_a'] if pw is not None else NAN,
            'paired_wall_ms_delta_ci_low': pw['diff_ci_low'] if pw is not None else NAN,
            'paired_wall_ms_delta_ci_high': pw['diff_ci_high'] if pw is not None else NAN,
            'ea_enhanced': _g(a, 'execution_accuracy'), 'ea_filtered': _g(b, 'execution_accuracy'),
            'paired_ea_delta': pe['difference_b_minus_a'] if pe is not None else NAN,
            'paired_ea_delta_ci_low': pe['diff_ci_low'] if pe is not None else NAN,
            'paired_ea_delta_ci_high': pe['diff_ci_high'] if pe is not None else NAN,
            'paired_ea_n': int(pe['n_pairs']) if pe is not None else 0,
        })
SCHEMA_EDGE = pd.DataFrame(trade)

bench_rows = []
for model, rec in BENCH_RESULTS.items():
    for r in rec.get('bench_rows') or []:
        if r.get('error'):
            bench_rows.append({'model': model, 'quant': r.get('quant'), 'error': r['error']})
            continue
        size_b, params = r.get('model_size'), r.get('model_n_params')
        test = ('pp' + str(r.get('n_prompt'))) if r.get('n_prompt') and not r.get('n_gen') else ('tg' + str(r.get('n_gen')))
        bench_rows.append({'model': model, 'run_name': r.get('run_name'), 'quant': r.get('quant'), 'test': test,
                           'threads': r.get('n_threads'), 'tokens_per_s': r.get('avg_ts'),
                           'tokens_per_s_sd': r.get('stddev_ts'), 'model_size_mb': (size_b or 0) / 1e6 or NAN,
                           'model_n_params': params,
                           'bits_per_weight': 8 * size_b / params if size_b and params else NAN,
                           'gguf_size_mb': (r.get('gguf_size_bytes') or 0) / 1e6 or NAN,
                           'n_batch': r.get('n_batch'), 'n_ubatch': r.get('n_ubatch'), 'cpu_info': r.get('cpu_info'),
                           'build_commit': r.get('build_commit'), 'test_time': r.get('test_time')})
BENCH = pd.DataFrame(bench_rows)
FOOTPRINT = pd.DataFrame([{**fp, 'gguf_size_mb': (fp.get('gguf_size_bytes') or 0) / 1e6 or NAN}
                          for rec in BENCH_RESULTS.values() for fp in rec.get('footprint') or []])
THREADS_DF = pd.DataFrame([{'model': m, 'bench_run': rec.get('run_name'), 'status': rec.get('status'),
                            'threads_chosen': THREADS_FOR.get(m), 'thread_estimates_s': json.dumps(rec.get('thread_estimates_s')),
                            'typical_prompt_tokens': rec.get('typical_prompt_tokens'),
                            'typical_gen_tokens': rec.get('typical_gen_tokens'), 'est_s_per_item': rec.get('est_s_per_item')}
                           for m, rec in BENCH_RESULTS.items()])

exp_rows = []
for rn, rec in EXPORT_RECORDS.items():
    base = {'run_name': rn, 'model': rec.get('model'), 'status': rec.get('status'), 'merge_dtype': rec.get('merge_dtype'),
            'merge_s': rec.get('merge_s'), 'convert_s': rec.get('convert_s'),
            'merged_size_mb': (rec.get('merged_size_bytes') or 0) / 1e6 or NAN,
            'f16_size_mb': (rec.get('f16_size_bytes') or 0) / 1e6 or NAN,
            'adapter_verified': _flat(rec.get('adapter_verified')), 'adapter_remapped': _flat(rec.get('adapter_remapped')),
            'model_class': (rec.get('merge_metadata') or {}).get('model_class'),
            'llama_cpp_commit': rec.get('llama_cpp_commit'), 'session': rec.get('session'), 'error': rec.get('error')}
    if rec.get('quants'):
        for q, info in rec['quants'].items():
            exp_rows.append({**base, 'quant': q, 'file': info.get('file'), 'size_mb': info.get('size_bytes', 0) / 1e6,
                             'sha256': info.get('sha256'), 'quantize_s': info.get('quantize_s')})
    else:
        exp_rows.append(base)
EXPORTS_DF = pd.DataFrame(exp_rows)

parity = []
for jid in EDGE_JOB_IDS:
    df = ALL_JOBS[jid]['rows']
    srv = pd.to_numeric(df.get('edge_tokens_evaluated'), errors='coerce')
    hf = pd.to_numeric(df.get('prompt_tokens'), errors='coerce')
    diff = (srv - hf).dropna()
    parity.append({'job_id': jid, 'model': ALL_JOBS[jid]['job'].get('model'), 'n': len(diff),
                   'share_equal': float(diff.eq(0).mean()) if len(diff) else NAN,
                   'diff_counts': json.dumps({str(int(k)): int(v) for k, v in diff.value_counts().items()}),
                   'mean_diff': float(diff.mean()) if len(diff) else NAN})
TOKENIZER_PARITY = pd.DataFrame(parity)

# --------------------------------------------------------------------------- #
# Integrity checks
# --------------------------------------------------------------------------- #
INTEGRITY = []
INTEGRITY.append({'check': 'reference rows re-derived identically', 'subject': 'all',
                  'passed': not REF_ROWS_MISMATCH, 'detail': ', '.join(REF_ROWS_MISMATCH) or 'all match'})
INTEGRITY.append({'check': 'reference jobs from the current code commit', 'subject': 'all',
                  'passed': REF_COMMITS == [GIT_COMMIT], 'detail': ', '.join(c[:10] for c in REF_COMMITS)})
for jid in EDGE_JOB_IDS:
    ids = set(ALL_JOBS[jid]['rows']['instance_id'])
    INTEGRITY.append({'check': 'edge rows equal the planned subset', 'subject': jid,
                      'passed': ids == set(EDGE_IDS), 'detail': str(len(ids)) + ' rows'})
for rn, rec in EXPORT_RECORDS.items():
    if rec.get('status') == 'ok':
        INTEGRITY.append({'check': 'merged adapter verified (history/027.md)', 'subject': rn,
                          'passed': bool(rec.get('adapter_verified')), 'detail': _flat(rec.get('adapter_verified'))})
        INTEGRITY.append({'check': 'merge metadata records model_class', 'subject': rn,
                          'passed': bool((rec.get('merge_metadata') or {}).get('model_class')),
                          'detail': str((rec.get('merge_metadata') or {}).get('model_class'))})
for _, r in TOKENIZER_PARITY.iterrows():
    INTEGRITY.append({'check': 'GGUF prompt tokens equal training-tokenizer tokens (>=95% of rows)',
                      'subject': r['job_id'], 'passed': bool(r['share_equal'] >= 0.95) if r['share_equal'] == r['share_equal'] else None,
                      'detail': 'diff counts ' + r['diff_counts']})
for _, r in OVERALL.iterrows():
    INTEGRITY.append({'check': 'no llama-server request errors', 'subject': r['job_id'],
                      'passed': int(r.get('n_request_errors', 0)) == 0, 'detail': str(r.get('n_request_errors'))})
    INTEGRITY.append({'check': 'generation hit max_new_tokens in <5% of rows', 'subject': r['job_id'],
                      'passed': r.get('n_stopped_limit', 0) < 0.05 * r['n'], 'detail': str(r.get('n_stopped_limit'))})
    INTEGRITY.append({'check': 'no Neo4j connection failures', 'subject': r['job_id'],
                      'passed': int(r.get('n_connection_failed', 0)) == 0, 'detail': str(r.get('n_connection_failed'))})
recs = [ALL_JOBS[j]['job'] for j in EDGE_JOB_IDS] + [r for r in BENCH_RESULTS.values() if r.get('status') == 'completed']
for key, label in (('cpu_model', 'one CPU model across all edge measurements'),
                   ('llama_cpp_commit', 'one llama.cpp commit across all edge measurements'),
                   ('git_commit', 'one code commit across all edge measurements')):
    values = sorted({str(r.get(key)) for r in recs})
    INTEGRITY.append({'check': label, 'subject': 'all', 'passed': len(values) <= 1, 'detail': '; '.join(values)})
INTEGRITY.append({'check': 'code version pinned with GIT_REF', 'subject': 'all', 'passed': bool(GIT_REF),
                  'detail': GIT_REF or '(blank)'})
INTEGRITY.append({'check': 'llama.cpp pinned (LLAMA_CPP_REF or Hub pin)', 'subject': 'all',
                  'passed': bool(LLAMA_REF_USED), 'detail': str(LLAMA_REF_USED or 'latest at session start')})
for job in JOBS:
    status = JOB_RESULTS.get(job['job_id'], {}).get('status', 'not_started')
    INTEGRITY.append({'check': 'expected job completed', 'subject': job['job_id'], 'passed': status == 'completed',
                      'detail': status + ((': ' + str(JOB_RESULTS[job['job_id']].get('error'))[:200])
                                          if JOB_RESULTS.get(job['job_id'], {}).get('error') else '')})
for model, rec in BENCH_RESULTS.items():
    INTEGRITY.append({'check': 'throughput benchmark completed', 'subject': model,
                      'passed': rec.get('status') == 'completed', 'detail': str(rec.get('status'))})
INTEGRITY_DF = pd.DataFrame(INTEGRITY)


# --------------------------------------------------------------------------- #
# Headline
# --------------------------------------------------------------------------- #
def _ci(row, metric):
    v = _g(row, metric)
    if v != v:
        return ''
    lo, hi = _g(row, metric + '_ci_low'), _g(row, metric + '_ci_high')
    return format(v, '.3f') if lo != lo else format(v, '.3f') + ' [' + format(lo, '.3f') + ', ' + format(hi, '.3f') + ']'


def _fmt(v, spec='.3f'):
    return '' if v is None or (isinstance(v, float) and math.isnan(v)) or pd.isna(v) else format(float(v), spec)


FID = FIDELITY.set_index('job_id') if not FIDELITY.empty else pd.DataFrame()
COND_LABELS = {'fine_tuned': 'fine-tuned', 'cross_schema': 'fine-tuned (trained enhanced, exact-match prompts)'}
HEADLINE = pd.DataFrame([{
    'Model': r['model'], 'Schema': r['prompt_schema_mode'], 'Condition': COND_LABELS.get(r['condition'], r['condition']),
    'Quant': r['quant'], 'N': int(r['n']),
    'Exec. acc. CPU': _ci(r, 'execution_accuracy'), 'Exec. acc. GPU (same rows)': _fmt(r.get('gpu_execution_accuracy')),
    'Delta EA [CI]': (_fmt(FID.loc[r['job_id'], 'execution_match_delta']) + ' [' + _fmt(FID.loc[r['job_id'], 'execution_match_delta_ci_low'])
                      + ', ' + _fmt(FID.loc[r['job_id'], 'execution_match_delta_ci_high']) + ']')
    if r['job_id'] in FID.index and 'execution_match_delta' in FID.columns else '',
    'Exec. N': _fmt(r.get('n_judged'), '.0f'), 'EM CPU': _fmt(r.get('exact_match')),
    'EM GPU': _fmt(r.get('gpu_exact_match')), 'GLEU CPU': _fmt(r.get('google_bleu_corpus')),
    'Identical to GPU': _fmt(FID.loc[r['job_id'], 'identical_normalised_share'], '.1%')
    if r['job_id'] in FID.index else '',
    'Median s/query': _fmt(_g(r, 'wall_ms_p50') / 1000, '.1f'), 'p95 s/query': _fmt(_g(r, 'wall_ms_p95') / 1000, '.1f'),
    'Prefill share': _fmt(r.get('prefill_share_of_wall'), '.0%'),
    'Prompt tokens': _fmt(r.get('prompt_tokens_mean'), '.0f'), 'GGUF MB': _fmt(r.get('gguf_size_mb'), '.0f'),
    'Peak RSS MB': _fmt(r.get('peak_rss_mb'), '.0f'), 'Threads': _fmt(r.get('threads'), '.0f'),
} for _, r in OVERALL.iterrows()]) if not OVERALL.empty else pd.DataFrame()

print('Edge headline (' + str(int(CI_LEVEL * 100)) + '% CI in brackets):')
display(HEADLINE)
if not SCHEMA_EDGE.empty:
    print('Schema filtering on the CPU:')
    display(SCHEMA_EDGE)
failed_checks = INTEGRITY_DF[INTEGRITY_DF['passed'].eq(False)]
print('Integrity checks: ' + str(len(INTEGRITY_DF)) + ' run, ' + str(len(failed_checks)) + ' failed')
if len(failed_checks):
    display(failed_checks)
