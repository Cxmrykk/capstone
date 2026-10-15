# 11. Shared statistics, row preparation, and supplementary analyses (GPU reference + CPU jobs)
import math
import re
from statistics import NormalDist

import numpy as np
from IPython.display import display
from src.data.schema_filter import parse_schema
from src.evaluation.metrics import _ngram_counts, normalise_cypher, tokenize_cypher

Z = NormalDist().inv_cdf(0.5 + CI_LEVEL / 2)
ALPHA = 1 - CI_LEVEL
RNG = np.random.default_rng(STATS_SEED)
JUDGED_STATUSES = ('compared', 'pred_failed')
FUNCTIONAL_SOURCE = 'neo4jLabs_functional_cypher'
NAN = float('nan')


def wilson(k, n):
    if n == 0:
        return NAN, NAN
    p = k / n
    denom = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / denom
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / denom
    return min(max(0.0, centre - half), p), max(min(1.0, centre + half), p)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def paired_diff_ci(a_vals, b_vals):
    d = np.asarray(b_vals, float) - np.asarray(a_vals, float)
    if len(d) == 0:
        return {}
    means = d[RNG.integers(0, len(d), size=(BOOTSTRAP_SAMPLES, len(d)))].mean(axis=1)
    lo, hi = np.quantile(means, [ALPHA / 2, 1 - ALPHA / 2])
    lo90, hi90 = np.quantile(means, [ALPHA, 1 - ALPHA])
    return {'difference_b_minus_a': float(d.mean()), 'diff_ci_low': float(lo), 'diff_ci_high': float(hi),
            'tost_ci_low': float(lo90), 'tost_ci_high': float(hi90),
            'equivalent_within_margin': bool(lo90 > -EQUIV_MARGIN and hi90 < EQUIV_MARGIN),
            'bootstrap_p': min(1.0, 2 * min(float((means <= 0).mean()), float((means >= 0).mean())))}


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def _gleu_counts(ref, hyp):
    rc = _ngram_counts(tokenize_cypher(ref), 1, 4)
    hc = _ngram_counts(tokenize_cypher(hyp), 1, 4)
    return sum((rc & hc).values()), sum(hc.values()), sum(rc.values())


def _num(df, col):
    if df is None or len(df) == 0 or col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors='coerce').dropna()


def prep_rows(df):
    df = df.copy()
    for col in ('execution_status', 'gold_executed_ok', 'pred_executed_ok', 'execution_match', 'syntax_valid',
                'gold_row_count', 'pred_row_count', 'exact_match', 'normalised_exact_match', 'gleu',
                'database_reference', 'data_source', 'complexity', 'question', 'predicted_raw'):
        if col not in df.columns:
            df[col] = None
    df['instance_id'] = df['instance_id'].astype(str)
    df['gold_cypher'] = df['gold_cypher'].fillna('').astype(str)
    df['predicted_cypher'] = df['predicted_cypher'].fillna('').astype(str)
    df['has_database'] = df['database_reference'].fillna('').astype(str).str.strip().ne('')
    counts = [_gleu_counts(g, p) for g, p in zip(df['gold_cypher'], df['predicted_cypher'])]
    df['gleu_overlap'] = [c[0] for c in counts]
    df['gleu_hyp_ngrams'] = [c[1] for c in counts]
    df['gleu_ref_ngrams'] = [c[2] for c in counts]
    df['empty_output'] = df['predicted_cypher'].str.strip().eq('')
    df['judged'] = df['execution_status'].isin(JUDGED_STATUSES) & df['gold_executed_ok'].eq(True)
    df['correct'] = df['judged'] & df['execution_match'].eq(True)
    df['gold_rows_n'] = pd.to_numeric(df['gold_row_count'], errors='coerce')
    df['pred_rows_n'] = pd.to_numeric(df['pred_row_count'], errors='coerce')
    return df


def bootstrap_gleu(df):
    n = len(df)
    if n == 0:
        return NAN, NAN
    o, h, r = (df[c].to_numpy(float) for c in ('gleu_overlap', 'gleu_hyp_ngrams', 'gleu_ref_ngrams'))
    idx = RNG.integers(0, n, size=(BOOTSTRAP_SAMPLES, n))
    so, sh, sr = o[idx].sum(axis=1), h[idx].sum(axis=1), r[idx].sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        vals = np.nan_to_num(np.minimum(so / sh, so / sr))
    lo, hi = np.quantile(vals, [ALPHA / 2, 1 - ALPHA / 2])
    return float(lo), float(hi)


def job_metrics(df):
    n = len(df)
    out = {'n': n}
    if n == 0:
        return out
    o, h, r = df['gleu_overlap'].sum(), df['gleu_hyp_ngrams'].sum(), df['gleu_ref_ngrams'].sum()
    out['google_bleu_corpus'] = min(o / h, o / r) if h and r else 0.0
    out['google_bleu_corpus_ci_low'], out['google_bleu_corpus_ci_high'] = bootstrap_gleu(df)
    for name in ('exact_match', 'normalised_exact_match'):
        k = int(df[name].eq(True).sum())
        out[name] = k / n
        out[name + '_ci_low'], out[name + '_ci_high'] = wilson(k, n)
    out['empty_output_rate'] = float(df['empty_output'].mean())
    out['n_has_database'] = int(df['has_database'].sum())
    out['n_connection_failed'] = int(df['execution_status'].eq('connection_failed').sum())
    j = df[df['judged']]
    out['n_judged'] = len(j)
    if len(j):
        k = int(j['correct'].sum())
        out['execution_accuracy'] = k / len(j)
        out['execution_accuracy_ci_low'], out['execution_accuracy_ci_high'] = wilson(k, len(j))
        out['executable_rate_gold_ok'] = float(j['pred_executed_ok'].eq(True).mean())
        nonempty = j[j['gold_rows_n'].fillna(0) > 0]
        out['n_judged_nonempty_gold'] = len(nonempty)
        if len(nonempty):
            out['execution_accuracy_nonempty_gold'] = float(nonempty['correct'].mean())
        out['share_correct_with_empty_gold'] = float((j['correct'] & j['gold_rows_n'].fillna(0).eq(0)).sum()
                                                     / max(1, k))
    validated = df[df['syntax_valid'].notna()]
    if len(validated):
        out['syntax_valid_rate'] = float(validated['syntax_valid'].eq(True).mean())
    return out


# --------------------------------------------------------------------------- #
# Every job with scored rows: GPU references and CPU jobs
# --------------------------------------------------------------------------- #
ALL_JOBS = {}
for jid, v in REF.items():
    ALL_JOBS[jid] = {'source': 'gpu_reference', 'job': v['job'], 'rows': prep_rows(v['rows'])}
for jid, rec in JOB_RESULTS.items():
    scored = EDGE_DIR / 'jobs' / jid / 'predictions.scored.jsonl'
    if rec.get('status') != 'completed' or not scored.exists():
        continue
    df = pd.DataFrame([json.loads(ln) for ln in scored.read_text(encoding='utf-8').splitlines() if ln.strip()])
    ALL_JOBS[jid] = {'source': 'edge_cpu', 'job': rec, 'rows': prep_rows(df)}
EDGE_JOB_IDS = [jid for jid, v in ALL_JOBS.items() if v['source'] == 'edge_cpu']
print('Jobs with scored rows: ' + str(len(ALL_JOBS)) + ' (' + str(len(EDGE_JOB_IDS)) + ' on the CPU)')

# --------------------------------------------------------------------------- #
# Cypher element extraction (heuristic; documented in the README)
# --------------------------------------------------------------------------- #
_STR_RE = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")
_NODE_LABELS_RE = re.compile(r'\(\s*(?:[A-Za-z_]\w*)?\s*:\s*([^(){}\[\]]+?)\s*(?=\{|\))')
_REL_TYPES_RE = re.compile(r'\[\s*(?:[A-Za-z_]\w*)?\s*:\s*([^\[\]{}*]+?)\s*(?=\*|\{|\])')
_PROP_ACCESS_RE = re.compile(r'\b([A-Za-z_]\w*)\s*\.\s*`?([A-Za-z_]\w*)\b`?(?!\s*[(.])')
_PATTERN_MAP_RE = re.compile(r'[(\[]\s*(?:[A-Za-z_]\w*)?\s*(?::[^(){}\[\]]*)?\{([^{}]*)\}\s*[)\]]')
_PROJ_RE = re.compile(r'[{,]\s*\.\s*([A-Za-z_]\w*)')
_NAMESPACES = {'apoc', 'gds', 'db', 'dbms', 'point', 'date', 'datetime', 'localdatetime', 'localtime', 'time',
               'duration', 'math', 'text', 'coll', 'spatial', 'genai', 'vector'}


def _split_names(text):
    out = set()
    for part in re.split(r'[:|&]', text):
        name = part.strip().strip('`').lstrip('!').strip()
        if re.fullmatch(r'[A-Za-z_]\w*', name):
            out.add(name)
    return out


def cypher_elements(query):
    q = _STR_RE.sub("''", query or '')
    labels, rels, props = set(), set(), set()
    for m in _NODE_LABELS_RE.finditer(q):
        labels |= _split_names(m.group(1))
    for m in _REL_TYPES_RE.finditer(q):
        rels |= _split_names(m.group(1))
    bound = (set(re.findall(r'\(\s*([A-Za-z_]\w*)', q)) | set(re.findall(r'\[\s*([A-Za-z_]\w*)\s*[:\]*{]', q))
             | set(re.findall(r'(?i)\bAS\s+([A-Za-z_]\w*)', q)))
    for m in _PROP_ACCESS_RE.finditer(q):
        if m.group(1) in bound and m.group(1).lower() not in _NAMESPACES:
            props.add(m.group(2))
    for m in _PATTERN_MAP_RE.finditer(q):
        props |= set(re.findall(r'([A-Za-z_]\w*)\s*:', m.group(1)))
    props |= set(_PROJ_RE.findall(q))
    return labels, rels, props


def schema_elements(text):
    s = parse_schema(text or '')
    if not s.parsed:
        return None
    labels = set(s.nodes) | {a for a, _r, _b in s.patterns if a} | {b for _a, _r, b in s.patterns if b}
    rels = set(s.rel_props) | {r for _a, r, _b in s.patterns}
    props = set()
    for group in list(s.nodes.values()) + list(s.rel_props.values()):
        props |= set(group)
    return labels, rels, props


FULL_EL = {iid: schema_elements(r.get('schema') or '') for iid, r in REF_BY_ID.items()}
_PROMPT_EL = {}


def prompt_elements(iid, mode):
    key = (iid, mode)
    if key not in _PROMPT_EL:
        full = FULL_EL.get(iid)
        if full is None or mode == 'enhanced':
            _PROMPT_EL[key] = full
        else:
            rec = REF_BY_ID[iid]
            text = apply_schema_mode(rec.get('schema') or '', rec.get('question') or '', mode, **FILTER_KW)
            _PROMPT_EL[key] = schema_elements(text) or (set(), set(), set())
    return _PROMPT_EL[key]


def _job_prompt_mode(rec):
    return rec.get('prompt_schema_mode') or rec.get('schema_mode') or 'enhanced'


EMPTY_DF = pd.DataFrame()
SUBSET_METRICS = HALLUCINATION = RESULT_BREAKDOWN = COVERAGE_ROWS = COVERAGE = COVERAGE_EA = EMPTY_DF
AGREEMENT = COMPOSITION = EMPTY_DF

if RUN_SUPPLEMENTARY:
    # (a) Metrics on informative subsets: functional_cypher is near-saturated and inflates text metrics.
    subsets = {
        'all': lambda d: d,
        'excl_functional_cypher': lambda d: d[d['data_source'] != FUNCTIONAL_SOURCE],
        'with_database': lambda d: d[d['has_database']],
        'without_database': lambda d: d[~d['has_database']],
    }
    rows = []
    for jid, v in ALL_JOBS.items():
        rec = v['job']
        for name, fn in subsets.items():
            rows.append({'job_id': jid, 'source': v['source'], 'run_name': rec.get('run_name'),
                         'model': rec.get('model'), 'condition': rec.get('condition'),
                         'prompt_schema_mode': _job_prompt_mode(rec), 'quant': rec.get('quant'),
                         'subset': name, **job_metrics(fn(v['rows']))})
    SUBSET_METRICS = pd.DataFrame(rows)

    # (b) Exact-match filter coverage: are the elements the gold query needs still in the pruned schema?
    edge_id_set = set(EDGE_IDS)
    cov = []
    for iid, rec in REF_BY_ID.items():
        full = FULL_EL.get(iid)
        gl, gr, gp = cypher_elements(rec.get('cypher') or '')
        row = {'instance_id': iid, 'data_source': rec.get('data_source'),
               'has_database': bool(rec.get('database_reference')), 'schema_parsed': full is not None,
               'in_edge_subset': iid in edge_id_set}
        if full is not None:
            pl, pr, pp = prompt_elements(iid, FILTER_MODE)
            fl, fr, fp = full
            known = (gl & fl, gr & fr, gp & fp)
            missing = (known[0] - pl, known[1] - pr, known[2] - pp)
            pruned_text = apply_schema_mode(rec.get('schema') or '', rec.get('question') or '', FILTER_MODE,
                                            **FILTER_KW)
            row.update({
                'gold_labels': len(gl), 'gold_rel_types': len(gr), 'gold_properties': len(gp),
                'gold_unknown_labels': '; '.join(sorted(gl - fl)), 'gold_unknown_rel_types': '; '.join(sorted(gr - fr)),
                'gold_unknown_properties': '; '.join(sorted(gp - fp)),
                'missing_labels': '; '.join(sorted(missing[0])), 'missing_rel_types': '; '.join(sorted(missing[1])),
                'missing_properties': '; '.join(sorted(missing[2])),
                'fully_covered': not any(missing),
                'schema_chars_full': len(rec.get('schema') or ''), 'schema_chars_pruned': len(pruned_text),
                'filter_fell_back': pruned_text.strip() == (rec.get('schema') or '').strip(),
            })
        cov.append(row)
    COVERAGE_ROWS = pd.DataFrame(cov)
    parsed = COVERAGE_ROWS[COVERAGE_ROWS['schema_parsed']].copy()
    parsed['chars_ratio'] = parsed['schema_chars_pruned'] / parsed['schema_chars_full'].replace(0, np.nan)
    cov_rows = []
    for group, sub in [('all', parsed)] + [(src, g) for src, g in parsed.groupby('data_source')]:
        if not len(sub):
            continue
        cov_rows.append({'group': group, 'n_parsed': len(sub),
                         'share_fully_covered': float(sub['fully_covered'].astype(bool).mean()),
                         'share_missing_label': float(sub['missing_labels'].ne('').mean()),
                         'share_missing_rel_type': float(sub['missing_rel_types'].ne('').mean()),
                         'share_missing_property': float(sub['missing_properties'].ne('').mean()),
                         'share_gold_uses_unknown_element': float((sub['gold_unknown_labels'].ne('')
                                                                   | sub['gold_unknown_rel_types'].ne('')
                                                                   | sub['gold_unknown_properties'].ne('')).mean()),
                         'share_filter_fell_back': float(sub['filter_fell_back'].astype(bool).mean()),
                         'median_pruned_chars_ratio': float(sub['chars_ratio'].median())})
    COVERAGE = pd.DataFrame(cov_rows)
    # Plain Python bools: with a bool dtype column, Series.get returns numpy.bool_, and
    # `numpy.bool_(True) is True` is False, which emptied every group below (history/035.md).
    covered = {str(i): bool(v) for i, v in zip(parsed['instance_id'], parsed['fully_covered'])}

    # Coverage vs execution accuracy (GPU references): does the exact-match loss sit where elements were dropped?
    ea_rows = []
    for m in dict.fromkeys(e['model'] for e in EXP):
        enh, pru = BY_VARIANT.get((m, 'enhanced')), BY_VARIANT.get((m, 'pruned'))
        if not (enh and pru):
            continue
        ids = {'enhanced_trained_enhanced_prompts': enh['run_name'] + '__fine_tuned',
               'exact_match_trained_exact_match_prompts': pru['run_name'] + '__fine_tuned',
               'enhanced_trained_exact_match_prompts': enh['run_name'] + '__fine_tuned__' + pru['schema_mode'] + '_prompts'}
        frames = {k: ALL_JOBS[j]['rows'].drop_duplicates('instance_id').set_index('instance_id')
                  for k, j in ids.items() if j in ALL_JOBS}
        if len(frames) < 2:
            continue
        common = None
        for f in frames.values():
            judged_ids = set(f.index[f['judged']])
            common = judged_ids if common is None else common & judged_ids
        for flag in (True, False):
            group_ids = sorted(i for i in common if i in covered and covered[i] == flag)
            row = {'model': m, 'fully_covered': flag, 'n_judged_all': len(group_ids)}
            for k, f in frames.items():
                row['ea_' + k] = float(f.loc[group_ids, 'correct'].mean()) if group_ids else NAN
            ea_rows.append(row)
    COVERAGE_EA = pd.DataFrame(ea_rows)

    # (c) Schema hallucination: predicted labels / relationship types / properties absent from the schema.
    hal = []
    for jid, v in ALL_JOBS.items():
        rec, df = v['job'], v['rows']
        mode = _job_prompt_mode(rec)
        flags = []
        for _, r in df.iterrows():
            full = FULL_EL.get(r['instance_id'])
            if full is None or not r['predicted_cypher'].strip():
                continue
            pl, pr, pp = cypher_elements(r['predicted_cypher'])
            fl, fr, fp = full
            ql, qr, qp = prompt_elements(r['instance_id'], mode)
            unk = (pl - fl, pr - fr, pp - fp)
            outside = ((pl & fl) - ql) | ((pr & fr) - qr) | ((pp & fp) - qp)
            flags.append({'any': any(unk), 'label': bool(unk[0]), 'rel': bool(unk[1]), 'prop': bool(unk[2]),
                          'outside_prompt': bool(outside), 'judged': bool(r['judged']), 'correct': bool(r['correct'])})
        f = pd.DataFrame(flags)
        if f.empty:
            continue
        hal.append({'job_id': jid, 'source': v['source'], 'run_name': rec.get('run_name'), 'model': rec.get('model'),
                    'condition': rec.get('condition'), 'prompt_schema_mode': mode, 'quant': rec.get('quant'),
                    'n_checked': len(f), 'share_any_unknown_element': float(f['any'].mean()),
                    'share_unknown_label': float(f['label'].mean()), 'share_unknown_rel_type': float(f['rel'].mean()),
                    'share_unknown_property': float(f['prop'].mean()),
                    'share_uses_element_outside_prompt_schema': float(f['outside_prompt'].mean()),
                    'ea_rows_with_unknown': float(f.loc[f['any'] & f['judged'], 'correct'].mean())
                    if (f['any'] & f['judged']).any() else NAN,
                    'ea_rows_without_unknown': float(f.loc[~f['any'] & f['judged'], 'correct'].mean())
                    if (~f['any'] & f['judged']).any() else NAN})
    HALLUCINATION = pd.DataFrame(hal)

    # (d) Execution-result breakdown: matches on empty gold results are weak evidence of a correct query.
    br = []
    for jid, v in ALL_JOBS.items():
        j = v['rows'][v['rows']['judged']]
        if not len(j):
            continue
        g, p = j['gold_rows_n'].fillna(-1), j['pred_rows_n'].fillna(-1)
        cats = np.select(
            [j['correct'] & (g > 0), j['correct'] & (g == 0), ~j['pred_executed_ok'].eq(True),
             (p == 0) & (g > 0), p != g],
            ['correct_nonempty', 'correct_empty_gold', 'pred_failed', 'wrong_pred_empty', 'wrong_row_count'],
            default='wrong_same_row_count')
        counts = pd.Series(cats).value_counts()
        rec = v['job']
        br.append({'job_id': jid, 'source': v['source'], 'model': rec.get('model'), 'condition': rec.get('condition'),
                   'quant': rec.get('quant'), 'n_judged': len(j), 'share_gold_empty': float((g == 0).mean()),
                   **{'n_' + c: int(counts.get(c, 0)) for c in ('correct_nonempty', 'correct_empty_gold', 'pred_failed',
                                                                 'wrong_pred_empty', 'wrong_row_count',
                                                                 'wrong_same_row_count')}})
    RESULT_BREAKDOWN = pd.DataFrame(br)

    # (e) Cross-model agreement among the fine-tuned enhanced GPU references.
    ft = {e['model']: e['run_name'] + '__fine_tuned' for e in EXP if e['variant'] == 'enhanced'}
    ft = {m: j for m, j in ft.items() if j in ALL_JOBS}
    if len(ft) >= 2:
        frames = {m: ALL_JOBS[j]['rows'].drop_duplicates('instance_id').set_index('instance_id') for m, j in ft.items()}
        common = sorted(set.intersection(*[set(f.index[f['judged']]) for f in frames.values()]))
        correct = pd.DataFrame({m: f.loc[common, 'correct'].astype(int) for m, f in frames.items()})
        k = correct.sum(axis=1)
        row = {'models': '; '.join(ft), 'n_judged_all': len(common), 'oracle_ea': float((k > 0).mean()),
               'all_correct': float((k == len(ft)).mean()), 'majority_ea': float((k > len(ft) / 2).mean())}
        for m in ft:
            row['ea_' + m] = float(correct[m].mean())
        for kk in range(len(ft) + 1):
            row['share_correct_by_' + str(kk)] = float((k == kk).mean())
        all_ids = sorted(set.intersection(*[set(f.index) for f in frames.values()]))
        for a, b in [(a, b) for i, a in enumerate(ft) for b in list(ft)[i + 1:]]:
            na = frames[a].loc[all_ids, 'predicted_cypher'].map(normalise_cypher)
            nb = frames[b].loc[all_ids, 'predicted_cypher'].map(normalise_cypher)
            row['identical_norm_' + a + '_vs_' + b] = float((na == nb).mean())
        AGREEMENT = pd.DataFrame([row])

    # (f) Sample composition (explains what the 500-row and edge samples contain).
    base_id = next((j for j in ALL_JOBS if ALL_JOBS[j]['source'] == 'gpu_reference'), None)
    if base_id:
        base = ALL_JOBS[base_id]['rows'].copy()
        base['in_edge_subset'] = base['instance_id'].isin(edge_id_set)
        COMPOSITION = (base.groupby(['data_source', 'has_database', 'complexity'], dropna=False)
                       .agg(n_reference=('instance_id', 'size'), n_edge=('in_edge_subset', 'sum'))
                       .reset_index().sort_values('n_reference', ascending=False))

    print('Filter coverage (exact-match prompts):')
    display(COVERAGE)
    print('Coverage vs execution accuracy:')
    display(COVERAGE_EA)
    print('Schema hallucination:')
    display(HALLUCINATION)
else:
    print('RUN_SUPPLEMENTARY is False; supplementary tables are empty.')
