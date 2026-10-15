# 6. Runs and adapters, the GPU reference jobs, and the edge row subset
import random

import pandas as pd
import yaml
from src.config import load_config
from src.data.dataset import load_raw_split, subsample

EXP = []
for model, variant, config in EXPERIMENTS:
    raw = yaml.safe_load(Path(config).read_text(encoding='utf-8'))
    cfg = load_config(config)
    EXP.append({'model': model, 'variant': variant, 'config': config, 'run_name': raw['run_name'],
                'schema_mode': cfg.data.schema_mode, 'seed': cfg.data.seed,
                'fingerprint_now': cfg.fingerprint(),
                'adapter_dir': REPO_DIR / 'artifacts' / 'runs' / raw['run_name'] / 'final'})
seeds = sorted({e['seed'] for e in EXP})
if len(seeds) != 1:
    raise RuntimeError('Configs use different data seeds ' + str(seeds) + '; rows would not match the reference.')
SAMPLE_SEED = seeds[0]

for e in EXP:
    adir = e['adapter_dir']
    if not (adir / 'adapter_config.json').exists():
        rc = run_step('pull-' + e['run_name'], 'python app.py checkpoint pull --config ' + e['config'] + ' --step final',
                      allow_fail=True)
        if rc not in (0, 1):
            raise RuntimeError(e['run_name'] + ': fetching the final adapter failed with exit ' + str(rc)
                               + ' (Hub error, not "not found"). Rerun when the Hub is reachable.')
    e['adapter_ok'] = (adir / 'adapter_config.json').exists() and any(adir.glob('adapter_model.*'))
    meta_path = adir / 'run_metadata.json'
    e['train_meta'] = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
    fp = e['train_meta'].get('fingerprint')
    e['fingerprint_trained'] = fp
    e['fingerprint_match'] = (fp == e['fingerprint_now']) if fp else None
    print(e['run_name'].ljust(30) + ('adapter OK' if e['adapter_ok'] else 'adapter MISSING').ljust(18)
          + 'fingerprint ' + {True: 'match', False: 'DIFFERS', None: 'n/a'}[e['fingerprint_match']])

# Schema-filter parameters used by the exact-match runs (for the supplementary analyses).
_pruned_cfg = next((load_config(c) for _m, v, c in EXPERIMENTS if v == 'pruned'), load_config(EXPERIMENTS[0][2]))
FILTER_MODE = _pruned_cfg.data.schema_mode
FILTER_KW = dict(min_nodes=_pruned_cfg.data.filter_min_nodes,
                 keep_patterns_for_kept_nodes=_pruned_cfg.data.filter_keep_all_patterns_for_kept_nodes,
                 similarity_threshold=_pruned_cfg.data.similarity_threshold,
                 similarity_model=_pruned_cfg.data.similarity_model)

# --------------------------------------------------------------------------- #
# GPU reference jobs (final-eval-kaggle)
# --------------------------------------------------------------------------- #
REF_PREFIX = 'final-eval/' + REFERENCE_EVAL_TAG
HUB_CACHE = REPO_DIR / 'artifacts' / 'hub_cache'
files = hub_files(refresh=True)
ref_ids = sorted({f.split('/')[3] for f in files
                  if f.startswith(REF_PREFIX + '/jobs/') and f.endswith('/job.json') and f.count('/') == 4})
if not ref_ids:
    raise RuntimeError('No reference jobs under ' + REF_PREFIX + '/jobs/ on the Hub. Run final-eval-kaggle first.')
hub_download_patterns([REF_PREFIX + '/jobs/*/job.json', REF_PREFIX + '/jobs/*/predictions.scored.jsonl'], HUB_CACHE)

REF = {}
for jid in ref_ids:
    d = HUB_CACHE / REF_PREFIX / 'jobs' / jid
    try:
        rec = json.loads((d / 'job.json').read_text(encoding='utf-8'))
    except Exception as exc:
        print('Skipping reference ' + jid + ': ' + str(exc))
        continue
    scored = d / 'predictions.scored.jsonl'
    if rec.get('status') != 'completed' or not scored.exists():
        continue
    rows = [json.loads(ln) for ln in scored.read_text(encoding='utf-8').splitlines() if ln.strip()]
    df = pd.DataFrame(rows)
    df['instance_id'] = df['instance_id'].astype(str)
    REF[jid] = {'job': rec, 'rows': df}
REF_COMMITS = sorted({str(v['job'].get('git_commit')) for v in REF.values()})
print()
print('Reference jobs loaded: ' + str(len(REF)) + ' (commits ' + ', '.join(c[:10] for c in REF_COMMITS) + ')')
if GIT_COMMIT not in REF_COMMITS:
    print('NOTE: the reference jobs ran on a different code commit. Prompt construction must be identical for the '
          'paired GPU/CPU comparison; prefer GIT_REF = the reference commit.')
for e in EXP:
    if e['run_name'] + '__fine_tuned' not in REF:
        print('WARNING: no GPU reference job ' + e['run_name'] + '__fine_tuned; its CPU job will not be paired.')

# --------------------------------------------------------------------------- #
# Rows: re-derive the reference sample, then a seeded subset for the CPU
# --------------------------------------------------------------------------- #
TEST_RECORDS = load_raw_split(SPLIT)
REF_RECORDS = subsample(TEST_RECORDS, REFERENCE_LIMIT, SAMPLE_SEED)
REF_IDS = [str(r.get('instance_id')) for r in REF_RECORDS]
REF_BY_ID = {str(r.get('instance_id')): r for r in REF_RECORDS}
REF_ROWS_MISMATCH = [jid for jid, v in REF.items()
                     if v['job'].get('backend', 'hf') == 'hf' and set(v['rows']['instance_id']) != set(REF_IDS)]
if REF_ROWS_MISMATCH:
    print('WARNING: re-derived rows differ from reference jobs: ' + ', '.join(REF_ROWS_MISMATCH))

rng = random.Random(EDGE_SUBSET_SEED)
k = min(EDGE_LIMIT, len(REF_RECORDS))
if EDGE_SUBSET == 'stratified':
    with_db = [i for i, r in enumerate(REF_RECORDS) if r.get('database_reference')]
    without_db = [i for i, r in enumerate(REF_RECORDS) if not r.get('database_reference')]
    k_db = min(len(with_db), round(k * len(with_db) / max(1, len(REF_RECORDS))))
    idx = rng.sample(with_db, k_db) + rng.sample(without_db, min(len(without_db), k - k_db))
else:
    idx = rng.sample(range(len(REF_RECORDS)), k)
EDGE_RECORDS = [REF_RECORDS[i] for i in sorted(idx)]
EDGE_IDS = [str(r.get('instance_id')) for r in EDGE_RECORDS]
EDGE_IDS_SHA = hashlib.sha256(LF.join(EDGE_IDS).encode('utf-8')).hexdigest()[:16]

EDGE_TAG = SPLIT + '-n' + str(len(EDGE_IDS)) + 'of' + str(REFERENCE_LIMIT) + '-' + EDGE_SUBSET + '-' + EDGE_TAG_SUFFIX
EDGE_PREFIX = 'edge-eval/' + EDGE_TAG
EDGE_DIR = REPO_DIR / 'artifacts' / 'edge-eval' / EDGE_TAG
EDGE_DIR.mkdir(parents=True, exist_ok=True)

n_db_ref = sum(1 for r in REF_RECORDS if r.get('database_reference'))
n_db_edge = sum(1 for r in EDGE_RECORDS if r.get('database_reference'))
print('Edge tag     : ' + EDGE_TAG + ' (rows sha ' + EDGE_IDS_SHA + ')')
print('Edge rows    : ' + str(len(EDGE_IDS)) + ' (' + str(n_db_edge) + ' with a database; reference '
      + str(n_db_ref) + '/' + str(len(REF_RECORDS)) + ')')
