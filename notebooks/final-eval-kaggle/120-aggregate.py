# 10. Aggregate every completed job into thesis tables (CIs, paired tests, integrity checks)
import itertools
import math
from statistics import NormalDist

import numpy as np
from src.evaluation.metrics import _ngram_counts, tokenize_cypher

Z = NormalDist().inv_cdf(0.5 + CI_LEVEL / 2)
ALPHA = 1 - CI_LEVEL
RNG = np.random.default_rng(STATS_SEED)
KEYS = ['job_id', 'run_name', 'model', 'variant', 'schema_mode', 'condition', 'backend']
COMPLEXITY_ORDER = ['easy', 'medium', 'hard', 'extra_hard']
OUT_DIR = REPO_DIR / 'artifacts' / 'thesis' / EVAL_TAG
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
(OUT_DIR / 'tables').mkdir(parents=True)


def _flat(value):
    if isinstance(value, (list, tuple)):
        return '; '.join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, default=str)
    if isinstance(value, Path):
        return str(value)
    return value


def _g(series, key):
    value = series.get(key) if series is not None else None
    try:
        return float(value) if value is not None and pd.notna(value) else float('nan')
    except (TypeError, ValueError):
        return float('nan')


def _error_category(r):
    if not str(r.get('predicted_cypher') or '').strip():
        return 'empty_output'
    status = r.get('execution_status')
    if status is None:
        return 'not_executed'
    if status in ('connection_failed', 'excluded_dead_db', 'gold_failed'):
        return status
    if not r.get('pred_executed_ok'):
        if r.get('syntax_valid') is False:
            return 'invalid_query'
        if r.get('syntax_valid') is True:
            return 'runtime_error'
        return 'execution_error'
    return 'correct' if r.get('execution_match') else 'wrong_result'


def _gleu_counts(ref, hyp):
    rc = _ngram_counts(tokenize_cypher(ref), 1, 4)
    hc = _ngram_counts(tokenize_cypher(hyp), 1, 4)
    return sum((rc & hc).values()), sum(hc.values()), sum(rc.values())


# --------------------------------------------------------------------------- #
# Per-row long table
# --------------------------------------------------------------------------- #
row_frames = []
for jid, rec in JOB_RESULTS.items():
    if rec.get('status') != 'completed':
        continue
    scored = EVAL_DIR / 'jobs' / jid / 'predictions.scored.jsonl'
    if not scored.exists():
        print('WARNING: ' + jid + ' has no scored predictions locally; skipped.')
        continue
    out = []
    for line in scored.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        gold = r.get('gold_cypher') or ''
        pred = r.get('predicted_cypher') or ''
        overlap, hyp_n, ref_n = _gleu_counts(gold, pred)
        out.append({
            'job_id': jid, 'run_name': rec.get('run_name'), 'model': rec.get('model'),
            'variant': rec.get('variant'), 'schema_mode': rec.get('schema_mode'),
            'condition': rec.get('condition'), 'backend': rec.get('backend'),
            'prompt_schema_mode': rec.get('prompt_schema_mode'), 'train_schema_mode': rec.get('train_schema_mode'),
            'instance_id': r.get('instance_id'), 'data_source': r.get('data_source'),
            'database_reference': r.get('database_reference'),
            'has_database': bool(str(r.get('database_reference') or '').strip()),
            'complexity': r.get('complexity'),
            'gold_hops': r.get('gold_hops'), 'prompt_tokens': r.get('prompt_tokens'),
            'schema_chars': r.get('schema_chars'),
            'schema_chars_untrimmed': r.get('schema_chars_untrimmed', r.get('schema_chars')),
            'schema_trimmed': r.get('schema_trimmed'), 'prompt_over_budget': r.get('prompt_over_budget'),
            'dead_database': r.get('dead_database'),
            'gleu': r.get('gleu'),
            'gleu_overlap': overlap, 'gleu_hyp_ngrams': hyp_n, 'gleu_ref_ngrams': ref_n,
            'exact_match': bool(r.get('exact_match')),
            'normalised_exact_match': bool(r.get('normalised_exact_match')),
            'empty_output': not pred.strip(),
            'execution_status': r.get('execution_status'),
            'gold_executed_ok': r.get('gold_executed_ok'),
            'pred_executed_ok': r.get('pred_executed_ok'),
            'execution_match': r.get('execution_match'),
            'syntax_valid': r.get('syntax_valid'),
            'gold_row_count': r.get('gold_row_count'), 'pred_row_count': r.get('pred_row_count'),
            'order_sensitive': r.get('order_sensitive'),
            'error_category': _error_category(r),
            'pred_error': (r.get('pred_error') or '')[:300] or None,
            'gold_error': (r.get('gold_error') or '')[:300] or None,
            'question': r.get('question'), 'gold_cypher': gold, 'predicted_cypher': pred,
            'predicted_raw': r.get('predicted_raw'),
        })
    row_frames.append(pd.DataFrame(out))

if not row_frames:
    raise RuntimeError('No completed jobs to aggregate. Run cell 7 first.')
ROWS = pd.concat(row_frames, ignore_index=True)
ROWS['prompt_tokens'] = pd.to_numeric(ROWS['prompt_tokens'], errors='coerce')
ROWS['schema_chars'] = pd.to_numeric(ROWS['schema_chars'], errors='coerce')
ROWS['schema_chars_untrimmed'] = pd.to_numeric(ROWS['schema_chars_untrimmed'], errors='coerce')
JOBS_DF = pd.DataFrame([{k: _flat(v) for k, v in rec.items()} for rec in JOB_RESULTS.values()])
print('Rows: ' + str(len(ROWS)) + ' across ' + str(ROWS['job_id'].nunique()) + ' jobs')


# --------------------------------------------------------------------------- #
# Metrics with confidence intervals
# --------------------------------------------------------------------------- #
def wilson(k, n):
    if n == 0:
        return float('nan'), float('nan')
    p = k / n
    denom = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / denom
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / denom
    lo = max(0.0, centre - half)
    hi = min(1.0, centre + half)
    # At p = 0 or p = 1 the exact bound equals p, but float rounding can leave it a hair
    # past p (e.g. lo = 1e-17 when p = 0). Keep p inside its own interval.
    return min(lo, p), max(hi, p)


def corpus_gleu_counts(o, h, r):
    return min(o / h, o / r) if h > 0 and r > 0 else 0.0


def bootstrap_corpus_gleu(df):
    n = len(df)
    if n == 0:
        return float('nan'), float('nan')
    o = df['gleu_overlap'].to_numpy(float)
    h = df['gleu_hyp_ngrams'].to_numpy(float)
    r = df['gleu_ref_ngrams'].to_numpy(float)
    idx = RNG.integers(0, n, size=(BOOTSTRAP_SAMPLES, n))
    so, sh, sr = o[idx].sum(axis=1), h[idx].sum(axis=1), r[idx].sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        vals = np.nan_to_num(np.minimum(so / sh, so / sr))
    lo, hi = np.quantile(vals, [ALPHA / 2, 1 - ALPHA / 2])
    return float(lo), float(hi)


def compute_metrics(df):
    n = len(df)
    out = {'n': n}
    if n == 0:
        return out
    out['google_bleu_corpus'] = corpus_gleu_counts(df['gleu_overlap'].sum(), df['gleu_hyp_ngrams'].sum(),
                                                   df['gleu_ref_ngrams'].sum())
    out['google_bleu_corpus_ci_low'], out['google_bleu_corpus_ci_high'] = bootstrap_corpus_gleu(df)
    out['google_bleu_mean'] = float(pd.to_numeric(df['gleu'], errors='coerce').mean())
    for name in ('exact_match', 'normalised_exact_match'):
        k = int(df[name].sum())
        out[name] = k / n
        out[name + '_ci_low'], out[name + '_ci_high'] = wilson(k, n)
    out['empty_output_rate'] = float(df['empty_output'].mean())
    NOT_SCORED = ['connection_failed', 'excluded_dead_db']
    out['n_has_database'] = int(df['has_database'].sum())
    out['n_connection_failed'] = int(df['execution_status'].eq('connection_failed').sum())
    out['n_excluded_dead_db'] = int(df['execution_status'].eq('excluded_dead_db').sum())
    trimmed = df['schema_trimmed'].dropna()
    if len(trimmed):
        out['n_schema_trimmed'] = int(trimmed.astype(bool).sum())
        out['schema_trimmed_rate'] = float(trimmed.astype(bool).mean())
    out['n_prompt_over_budget'] = int(df['prompt_over_budget'].eq(True).sum())
    executed = df[df['execution_status'].notna() & ~df['execution_status'].isin(NOT_SCORED)]
    gold_ok = executed[executed['gold_executed_ok'].eq(True)]
    out['n_executed'] = len(executed)
    out['n_gold_ok'] = len(gold_ok)
    out['n_gold_failed'] = len(executed) - len(gold_ok)
    if len(executed):
        out['execution_accuracy_toolkit'] = float(executed['execution_match'].eq(True).mean())
        out['executable_rate'] = float(executed['pred_executed_ok'].eq(True).mean())
    if len(gold_ok):
        k = int(gold_ok['execution_match'].eq(True).sum())
        out['execution_accuracy'] = k / len(gold_ok)
        out['execution_accuracy_ci_low'], out['execution_accuracy_ci_high'] = wilson(k, len(gold_ok))
        out['executable_rate_gold_ok'] = float(gold_ok['pred_executed_ok'].eq(True).mean())
    validated = df[df['syntax_valid'].notna()]
    if len(validated):
        out['syntax_valid_rate'] = float(validated['syntax_valid'].eq(True).mean())
    tok = df['prompt_tokens'].dropna()
    tok = tok[tok >= 0]
    if len(tok):
        out['prompt_tokens_mean'] = float(tok.mean())
        out['prompt_tokens_median'] = float(tok.median())
        out['prompt_tokens_p95'] = float(tok.quantile(0.95))
        out['prompt_tokens_max'] = float(tok.max())
    return out


def table_by(extra):
    recs = []
    for keys, sub in ROWS.groupby(KEYS + extra, dropna=False, sort=False):
        rec = dict(zip(KEYS + extra, keys))
        rec.update(compute_metrics(sub))
        recs.append(rec)
    return pd.DataFrame(recs)


OVERALL = table_by([])
job_cols = [c for c in ('job_id', 'seconds_per_item', 'generation_seconds', 'batch_size', 'max_new_tokens',
                        'gpu', 'gpu_index', 'platform', 'concurrent_gpu_workers', 'git_commit',
                        'adapter_verified') if c in JOBS_DF.columns]
OVERALL = OVERALL.merge(JOBS_DF[job_cols], on='job_id', how='left')
OVERALL = OVERALL.sort_values(['model', 'schema_mode', 'backend', 'condition']).reset_index(drop=True)

COND_LABELS = {'fine_tuned': 'fine-tuned', 'zero_shot': 'zero-shot',
               'cross_schema': 'fine-tuned (trained enhanced, exact-match prompts)'}


def condition_label(model, condition):
    label = COND_LABELS.get(condition, str(condition))
    if condition == 'zero_shot' and PLAIN_TEXT_PROMPT.get(model):
        label += ' (base checkpoint, plain-text prompt)'
    return label


OVERALL['condition_label'] = [condition_label(m, c) for m, c in zip(OVERALL['model'], OVERALL['condition'])]

BY_COMPLEXITY = table_by(['complexity'])
BY_COMPLEXITY['_rank'] = BY_COMPLEXITY['complexity'].map({c: i for i, c in enumerate(COMPLEXITY_ORDER)})
BY_COMPLEXITY = BY_COMPLEXITY.sort_values(['job_id', '_rank']).drop(columns='_rank').reset_index(drop=True)
BY_SOURCE = table_by(['data_source'])
BY_DATABASE = table_by(['database_reference'])
BY_DB_ACCESS = table_by(['has_database'])
BY_COMPLEXITY_DB_ACCESS = table_by(['has_database', 'complexity'])
ERRORS = ROWS.groupby(KEYS + ['error_category']).size().unstack(fill_value=0).reset_index()

# --------------------------------------------------------------------------- #
# Paired comparisons on identical rows
# --------------------------------------------------------------------------- #
JOB_KEYS = ROWS[KEYS].drop_duplicates().set_index('job_id')
MODELS = list(dict.fromkeys(e['model'] for e in EXP))
SCHEMA_FAMILIES = ('schema_filtering', 'schema_inference_effect', 'schema_training_effect')
JUDGED = ['compared', 'pred_failed']


def jid_for(model, variant, condition, backend='hf'):
    m = JOB_KEYS[(JOB_KEYS['model'] == model) & (JOB_KEYS['variant'] == variant)
                 & (JOB_KEYS['condition'] == condition) & (JOB_KEYS['backend'] == backend)]
    return m.index[0] if len(m) else None


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
    return min(1.0, 2 * tail / 2 ** n)


def paired_diff_ci(a_vals, b_vals):
    """Paired bootstrap of mean(b - a): CI, a (1 - 2*alpha) CI for TOST, and a two-sided p."""
    d = np.asarray(b_vals, float) - np.asarray(a_vals, float)
    idx = RNG.integers(0, len(d), size=(BOOTSTRAP_SAMPLES, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [ALPHA / 2, 1 - ALPHA / 2])
    lo90, hi90 = np.quantile(means, [ALPHA, 1 - ALPHA])
    return {'difference_b_minus_a': float(d.mean()), 'diff_ci_low': float(lo), 'diff_ci_high': float(hi),
            'tost_ci_low': float(lo90), 'tost_ci_high': float(hi90),
            'equivalent_within_margin': bool(lo90 > -EQUIV_MARGIN and hi90 < EQUIV_MARGIN),
            'bootstrap_p': min(1.0, 2 * min(float((means <= 0).mean()), float((means >= 0).mean())))}


def paired_compare(a, b, label, family):
    A = ROWS[ROWS['job_id'] == a].drop_duplicates('instance_id').set_index('instance_id')
    B = ROWS[ROWS['job_id'] == b].drop_duplicates('instance_id').set_index('instance_id')
    common = A.index.intersection(B.index)
    A, B = A.loc[common], B.loc[common]
    base = {'comparison': label, 'family': family, 'job_a': a, 'job_b': b}
    results = []
    for metric in ('exact_match', 'normalised_exact_match', 'execution_match'):
        if metric == 'execution_match':
            # Only rows judged on both sides: gold ran, and neither side hit an outage or a dead database.
            mask = (A['gold_executed_ok'].eq(True) & B['gold_executed_ok'].eq(True)
                    & A['execution_status'].isin(JUDGED) & B['execution_status'].isin(JUDGED)).to_numpy()
        else:
            mask = np.ones(len(common), dtype=bool)
        n = int(mask.sum())
        if n == 0:
            continue
        a_ok = A[metric].eq(True).to_numpy()[mask]
        b_ok = B[metric].eq(True).to_numpy()[mask]
        a_only, b_only = int((a_ok & ~b_ok).sum()), int((b_ok & ~a_ok).sum())
        stats = paired_diff_ci(a_ok, b_ok)
        stats.pop('bootstrap_p')
        results.append({**base, 'metric': metric, 'n_pairs': n, 'a_value': float(a_ok.mean()),
                        'b_value': float(b_ok.mean()), **stats,
                        'a_only_correct': a_only, 'b_only_correct': b_only,
                        'test': 'exact McNemar (two-sided)', 'p_value': mcnemar_exact(a_only, b_only),
                        'p_value_is_bound': False})
    ga = pd.to_numeric(A['gleu'], errors='coerce').fillna(0).to_numpy(float)
    gb = pd.to_numeric(B['gleu'], errors='coerce').fillna(0).to_numpy(float)
    if len(ga):
        stats = paired_diff_ci(ga, gb)
        p_raw = stats.pop('bootstrap_p')
        floor = 1.0 / BOOTSTRAP_SAMPLES
        results.append({**base, 'metric': 'sentence_gleu_mean', 'n_pairs': len(ga), 'a_value': float(ga.mean()),
                        'b_value': float(gb.mean()), **stats,
                        'test': 'paired bootstrap (percentile)', 'p_value': max(p_raw, floor),
                        'p_value_is_bound': p_raw < floor})
    return results


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


COMPARISONS = []
for e in EXP:
    m, v = e['model'], e['variant']
    a, b = jid_for(m, v, 'zero_shot'), jid_for(m, v, 'fine_tuned')
    if a and b:
        COMPARISONS.append((a, b, m + ' ' + v + ': zero-shot vs fine-tuned', 'fine_tuning_effect'))
for m in MODELS:
    for cond in ('fine_tuned', 'zero_shot'):
        a, b = jid_for(m, 'enhanced', cond), jid_for(m, 'pruned', cond)
        if a and b:
            COMPARISONS.append((a, b, m + ' ' + cond + ': enhanced vs exact-match schema', 'schema_filtering'))
ft_enhanced = [(m, jid_for(m, 'enhanced', 'fine_tuned')) for m in MODELS]
ft_enhanced = [x for x in ft_enhanced if x[1]]
for (ma, a), (mb, b) in itertools.combinations(ft_enhanced, 2):
    COMPARISONS.append((a, b, 'fine-tuned enhanced: ' + ma + ' vs ' + mb, 'model_comparison'))
for e in EXP:
    a = jid_for(e['model'], e['variant'], 'fine_tuned', 'hf')
    b = jid_for(e['model'], e['variant'], 'fine_tuned', 'llamacpp')
    if a and b:
        COMPARISONS.append((a, b, e['run_name'] + ': 4-bit HF (GPU) vs Q4_K_M GGUF (CPU)', 'quantisation'))
for m in MODELS:
    ee = jid_for(m, 'enhanced', 'fine_tuned')
    ex = jid_for(m, 'enhanced', 'cross_schema')
    pp = jid_for(m, 'pruned', 'fine_tuned')
    if ee and ex:
        COMPARISONS.append((ee, ex, m + ': enhanced-trained, enhanced vs exact-match prompts', 'schema_inference_effect'))
    if ex and pp:
        COMPARISONS.append((ex, pp, m + ': exact-match prompts, enhanced-trained vs exact-match-trained',
                            'schema_training_effect'))

paired_rows = []
for a, b, label, family in COMPARISONS:
    paired_rows.extend(paired_compare(a, b, label, family))
PAIRED = pd.DataFrame(paired_rows)
if not PAIRED.empty:
    PAIRED['p_holm'] = PAIRED.groupby(['family', 'metric'])['p_value'].transform(lambda s: holm(s.to_numpy()))
    PAIRED['significant_holm'] = PAIRED['p_holm'] < ALPHA
    floor_txt = '< ' + format(1.0 / BOOTSTRAP_SAMPLES, '.2g')
    PAIRED['p_value_display'] = [floor_txt if bound else format(p, '.3g')
                                 for p, bound in zip(PAIRED['p_value'], PAIRED['p_value_is_bound'])]
    non_schema = ~PAIRED['family'].isin(SCHEMA_FAMILIES)
    PAIRED.loc[non_schema, ['tost_ci_low', 'tost_ci_high']] = np.nan
    PAIRED['equivalent_within_margin'] = PAIRED['equivalent_within_margin'].astype(object)
    PAIRED.loc[non_schema, 'equivalent_within_margin'] = None

# --------------------------------------------------------------------------- #
# Schema filtering: token efficiency vs accuracy (sub-question 1)
# --------------------------------------------------------------------------- #
OV = OVERALL.set_index('job_id')
sf_rows = []
for m in MODELS:
    for cond in ('fine_tuned', 'zero_shot'):
        a, b = jid_for(m, 'enhanced', cond), jid_for(m, 'pruned', cond)
        if not (a and b):
            continue
        oa, ob = OV.loc[a], OV.loc[b]
        A = ROWS[ROWS['job_id'] == a].drop_duplicates('instance_id').set_index('instance_id')
        B = ROWS[ROWS['job_id'] == b].drop_duplicates('instance_id').set_index('instance_id')
        common = A.index.intersection(B.index)
        # Filter effect only: untrimmed schema sizes (training-budget trimming excluded).
        ratio = (B.loc[common, 'schema_chars_untrimmed']
                 / A.loc[common, 'schema_chars_untrimmed'].replace(0, np.nan)).astype(float)
        p_ex = PAIRED[(PAIRED['job_a'] == a) & (PAIRED['job_b'] == b)
                      & (PAIRED['metric'] == 'execution_match')] if not PAIRED.empty else PAIRED
        ex = p_ex.iloc[0] if len(p_ex) else None
        pruned_ids = ratio.index[(ratio < PRUNED_RATIO_THRESHOLD).to_numpy()]
        judged_a = A.loc[pruned_ids, 'execution_status'].isin(JUDGED)
        judged_b = B.loc[pruned_ids, 'execution_status'].isin(JUDGED)
        both = (judged_a & judged_b).to_numpy()
        ids = pruned_ids[both]
        a_ok = A.loc[ids, 'execution_match'].eq(True).to_numpy()
        b_ok = B.loc[ids, 'execution_match'].eq(True).to_numpy()
        pr = paired_diff_ci(a_ok, b_ok) if len(ids) else {}
        tok_a, tok_b = _g(oa, 'prompt_tokens_mean'), _g(ob, 'prompt_tokens_mean')
        nan = float('nan')
        sf_rows.append({
            'model': m, 'condition': cond, 'job_enhanced': a, 'job_exact_match': b,
            'prompt_tokens_mean_enhanced': tok_a, 'prompt_tokens_mean_exact_match': tok_b,
            'prompt_token_reduction_pct': 100 * (1 - tok_b / tok_a) if tok_a else nan,
            'prompt_tokens_p95_enhanced': _g(oa, 'prompt_tokens_p95'),
            'prompt_tokens_p95_exact_match': _g(ob, 'prompt_tokens_p95'),
            'n_schema_trimmed_enhanced': _g(oa, 'n_schema_trimmed'),
            'n_schema_trimmed_exact_match': _g(ob, 'n_schema_trimmed'),
            'schema_chars_median_ratio': float(ratio.median()),
            'share_rows_le5pct_schema_reduction': float((ratio >= PRUNED_RATIO_THRESHOLD).mean()),
            'execution_accuracy_enhanced': _g(oa, 'execution_accuracy'),
            'execution_accuracy_exact_match': _g(ob, 'execution_accuracy'),
            'paired_ea_delta': float(ex['difference_b_minus_a']) if ex is not None else nan,
            'paired_ea_delta_ci_low': float(ex['diff_ci_low']) if ex is not None else nan,
            'paired_ea_delta_ci_high': float(ex['diff_ci_high']) if ex is not None else nan,
            'paired_ea_n': int(ex['n_pairs']) if ex is not None else 0,
            'execution_match_p_value': float(ex['p_value']) if ex is not None else nan,
            'tost_margin': EQUIV_MARGIN,
            'tost_ci_low': float(ex['tost_ci_low']) if ex is not None else nan,
            'tost_ci_high': float(ex['tost_ci_high']) if ex is not None else nan,
            'equivalent_within_margin': ex['equivalent_within_margin'] if ex is not None else None,
            'pruned_rows_n': int(len(pruned_ids)), 'pruned_rows_n_judged': int(len(ids)),
            'pruned_rows_ea_enhanced': float(a_ok.mean()) if len(ids) else nan,
            'pruned_rows_ea_exact_match': float(b_ok.mean()) if len(ids) else nan,
            'pruned_rows_ea_delta': pr.get('difference_b_minus_a', nan),
            'pruned_rows_delta_ci_low': pr.get('diff_ci_low', nan),
            'pruned_rows_delta_ci_high': pr.get('diff_ci_high', nan),
            'pruned_rows_mcnemar_p': (mcnemar_exact(int((a_ok & ~b_ok).sum()), int((b_ok & ~a_ok).sum()))
                                      if len(ids) else nan),
            'google_bleu_enhanced': _g(oa, 'google_bleu_corpus'),
            'google_bleu_exact_match': _g(ob, 'google_bleu_corpus'),
            'exact_match_enhanced': _g(oa, 'exact_match'), 'exact_match_exact_match': _g(ob, 'exact_match'),
            'seconds_per_item_enhanced': _g(oa, 'seconds_per_item'),
            'seconds_per_item_exact_match': _g(ob, 'seconds_per_item'),
        })
SCHEMA_FILTER = pd.DataFrame(sf_rows)

# --------------------------------------------------------------------------- #
# Integrity checks
# --------------------------------------------------------------------------- #
INTEGRITY = []
hf_jobs = sorted(ROWS.loc[ROWS['backend'] == 'hf', 'job_id'].unique())
id_sets = {j: frozenset(ROWS.loc[ROWS['job_id'] == j, 'instance_id']) for j in hf_jobs}
INTEGRITY.append({'check': 'identical evaluation rows across all GPU jobs', 'subject': 'all',
                  'passed': len(set(id_sets.values())) <= 1,
                  'detail': ', '.join(j + '=' + str(len(s)) for j, s in id_sets.items())})
for jid, rec in JOB_RESULTS.items():
    if rec.get('status') == 'completed' and rec.get('condition') in ADAPTER_CONDITIONS and rec.get('backend') == 'hf':
        ver = rec.get('adapter_verified') or []
        INTEGRITY.append({'check': 'adapter tensors verified at load (history/027.md)', 'subject': jid,
                          'passed': bool(ver), 'detail': _flat(ver)})
for a, b, label, family in COMPARISONS:
    if family != 'fine_tuning_effect':
        continue
    A = ROWS[ROWS['job_id'] == a].drop_duplicates('instance_id').set_index('instance_id')['predicted_cypher']
    B = ROWS[ROWS['job_id'] == b].drop_duplicates('instance_id').set_index('instance_id')['predicted_cypher']
    common = A.index.intersection(B.index)
    identical = float((A.loc[common] == B.loc[common]).mean()) if len(common) else float('nan')
    INTEGRITY.append({'check': 'fine-tuned outputs differ from zero-shot outputs', 'subject': b,
                      'passed': identical < 0.9, 'detail': format(identical, '.1%') + ' of predictions identical to ' + a})
for _, row in OVERALL.iterrows():
    n_exec = _g(row, 'n_executed')
    if n_exec and n_exec == n_exec:
        share = _g(row, 'n_gold_failed') / n_exec
        INTEGRITY.append({'check': 'gold queries execute on the demo server (<5% failures)', 'subject': row['job_id'],
                          'passed': share < 0.05, 'detail': format(share, '.1%') + ' of executed rows had a failing gold query'})
for e in EXP:
    INTEGRITY.append({'check': 'training config fingerprint matches current config', 'subject': e['run_name'],
                      'passed': e['fingerprint_match'],
                      'detail': str(e['fingerprint_trained']) + ' vs ' + str(e['fingerprint_now'])})
for job in JOBS:
    rec = JOB_RESULTS.get(job['job_id'], {})
    status = rec.get('status', 'not_started')
    INTEGRITY.append({'check': 'expected job completed', 'subject': job['job_id'],
                      'passed': status == 'completed',
                      'detail': status + ((': ' + str(rec.get('error'))) if rec.get('error') else '')})
completed_recs = [r for r in JOB_RESULTS.values() if r.get('status') == 'completed']
BUNDLE_COMMITS = sorted({str(r.get('git_commit')) for r in completed_recs})
BUNDLE_PLATFORMS = sorted({r.get('platform') or 'colab' for r in completed_recs})
INTEGRITY.append({'check': 'all jobs from one code commit', 'subject': 'all',
                  'passed': len(BUNDLE_COMMITS) <= 1, 'detail': ', '.join(c[:10] for c in BUNDLE_COMMITS)})
INTEGRITY.append({'check': 'all jobs from one platform', 'subject': 'all',
                  'passed': len(BUNDLE_PLATFORMS) <= 1, 'detail': ', '.join(BUNDLE_PLATFORMS)})
INTEGRITY.append({'check': 'code version pinned with GIT_REF', 'subject': 'all',
                  'passed': bool(GIT_REF), 'detail': GIT_REF or '(blank)'})
for _, row in OVERALL.iterrows():
    ncf = _g(row, 'n_connection_failed')
    if ncf == ncf:
        INTEGRITY.append({'check': 'no Neo4j connection failures', 'subject': row['job_id'],
                          'passed': ncf == 0, 'detail': format(ncf, '.0f') + ' rows connection_failed'})

gold_rows = (ROWS[ROWS['execution_status'].notna() & ROWS['execution_status'].ne('connection_failed')]
             .drop_duplicates(['database_reference', 'instance_id']))
if len(gold_rows):
    GOLD_HEALTH = (gold_rows.groupby('database_reference')
                   .agg(n=('instance_id', 'size'),
                        n_gold_failed=('gold_executed_ok', lambda s: int((~s.eq(True)).sum())),
                        sample_gold_error=('gold_error', lambda s: next((x for x in s if x), None)))
                   .reset_index())
    GOLD_HEALTH['gold_failure_rate'] = GOLD_HEALTH['n_gold_failed'] / GOLD_HEALTH['n']
    GOLD_HEALTH = GOLD_HEALTH[GOLD_HEALTH['n_gold_failed'] > 0].sort_values('gold_failure_rate', ascending=False)
else:
    GOLD_HEALTH = pd.DataFrame(columns=['database_reference', 'n', 'n_gold_failed', 'sample_gold_error',
                                        'gold_failure_rate'])
DEAD_DBS = GOLD_HEALTH.loc[GOLD_HEALTH['gold_failure_rate'].eq(1.0), 'database_reference'].tolist()
INTEGRITY.append({'check': 'databases where every gold query fails (outside execution_accuracy by construction)',
                  'subject': 'all', 'passed': None, 'detail': ', '.join(DEAD_DBS) or 'none'})
if DEAD_DBS:
    print('Dead databases: ' + ', '.join(DEAD_DBS))
    display(GOLD_HEALTH)
INTEGRITY_DF = pd.DataFrame(INTEGRITY)


# --------------------------------------------------------------------------- #
# Headline table and published baselines
# --------------------------------------------------------------------------- #
def _ci(row, metric):
    v = _g(row, metric)
    if v != v:
        return ''
    lo, hi = _g(row, metric + '_ci_low'), _g(row, metric + '_ci_high')
    if lo != lo:
        return format(v, '.3f')
    return format(v, '.3f') + ' [' + format(lo, '.3f') + ', ' + format(hi, '.3f') + ']'


def _fmt(v, spec='.3f'):
    return '' if v is None or pd.isna(v) else format(float(v), spec)


HEADLINE = pd.DataFrame([{
    'Model': r['model'], 'Schema': r['schema_mode'], 'Condition': r['condition_label'],
    'Backend': r['backend'], 'N': int(r['n']),
    'GLEU': _ci(r, 'google_bleu_corpus'), 'EM': _ci(r, 'exact_match'),
    'Norm. EM': _fmt(r.get('normalised_exact_match')),
    'Exec. acc.': _ci(r, 'execution_accuracy'), 'Exec. N': _fmt(r.get('n_gold_ok'), '.0f'),
    'Executable': _fmt(r.get('executable_rate_gold_ok')),
    'Trimmed': _fmt(r.get('n_schema_trimmed'), '.0f'),
    'Prompt tokens': _fmt(r.get('prompt_tokens_mean'), '.0f'),
    's/item': _fmt(r.get('seconds_per_item'), '.2f'), 'Batch': _fmt(r.get('batch_size'), '.0f'),
    'Platform': r.get('platform') if isinstance(r.get('platform'), str) else 'colab',
    'Commit': str(r.get('git_commit') or '')[:7],
} for _, r in OVERALL.iterrows()])

PUBLISHED = (pd.DataFrame(PUBLISHED_BASELINES) if PUBLISHED_BASELINES
             else pd.DataFrame(columns=['source', 'model', 'metric', 'value', 'notes']))

print('Headline (' + str(int(CI_LEVEL * 100)) + '% CI in brackets):')
display(HEADLINE)
if not PAIRED.empty:
    print('Paired tests:')
    display(PAIRED[['comparison', 'metric', 'n_pairs', 'a_value', 'b_value', 'difference_b_minus_a',
                    'p_value_display', 'p_holm', 'significant_holm']])
if not SCHEMA_FILTER.empty:
    print('Schema filtering:')
    display(SCHEMA_FILTER)
failed_checks = INTEGRITY_DF[INTEGRITY_DF['passed'].eq(False)]
print('Integrity checks: ' + str(len(INTEGRITY_DF)) + ' run, ' + str(len(failed_checks)) + ' failed')
if len(failed_checks):
    display(failed_checks)
