# 6. Resolve runs, pull the final adapters from the Hub and check their provenance
import yaml
from src.config import load_config

EXP = []
for model, variant, config in EXPERIMENTS:
    raw = yaml.safe_load(Path(config).read_text(encoding='utf-8'))
    cfg = load_config(config)
    run_name = raw['run_name']
    EXP.append({
        'model': model, 'variant': variant, 'config': config, 'run_name': run_name,
        'schema_mode': cfg.data.schema_mode, 'seed': cfg.data.seed,
        'fingerprint_now': cfg.fingerprint(),
        'adapter_dir': REPO_DIR / 'artifacts' / 'runs' / run_name / 'final',
    })

seeds = sorted({e['seed'] for e in EXP})
if len(seeds) != 1:
    raise RuntimeError(f'Configs use different data seeds {seeds}; jobs would be scored on different rows.')
SAMPLE_SEED = seeds[0]
LIMIT_LABEL = str(EVAL_LIMIT) if EVAL_LIMIT else 'all'
EVAL_TAG = f'{SPLIT}-n{LIMIT_LABEL}-seed{SAMPLE_SEED}'
# predict falls back to data.max_eval_samples (1000) when no limit is passed, so
# the full split needs an explicit limit larger than the split.
PREDICT_LIMIT = EVAL_LIMIT if EVAL_LIMIT else 10 ** 7
HUB_EVAL_PREFIX = 'final-eval/' + EVAL_TAG
EVAL_DIR = REPO_DIR / 'artifacts' / 'final-eval' / EVAL_TAG
HUB_CACHE = REPO_DIR / 'artifacts' / 'hub_cache'
EVAL_DIR.mkdir(parents=True, exist_ok=True)
print('Evaluation tag: ' + EVAL_TAG)

for e in EXP:
    rn = e['run_name']
    cfg_path = e['config']
    adir = e['adapter_dir']
    if not (adir / 'adapter_config.json').exists():
        run_step('pull-' + rn, f'python app.py checkpoint pull --config {cfg_path} --step final',
                 allow_fail=True)
    e['adapter_ok'] = (adir / 'adapter_config.json').exists() and any(adir.glob('adapter_model.*'))
    meta_path = adir / 'run_metadata.json'
    e['train_meta'] = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
    fp = e['train_meta'].get('fingerprint')
    e['fingerprint_trained'] = fp
    e['fingerprint_match'] = (fp == e['fingerprint_now']) if fp else None

print()
print('Run'.ljust(30) + 'Adapter'.ljust(9) + 'Step'.ljust(7) + 'Fingerprint')
for e in EXP:
    tm = e['train_meta']
    fp_state = {True: 'match', False: 'DIFFERS', None: 'n/a'}[e['fingerprint_match']]
    print(e['run_name'].ljust(30) + ('OK' if e['adapter_ok'] else 'MISSING').ljust(9)
          + str(tm.get('global_step', '-')).ljust(7) + fp_state)

if any(e['fingerprint_match'] is False for e in EXP):
    print()
    print('NOTE: a differing fingerprint usually means a config file or RunConfig field changed after '
          'training. Adapter loading is still verified tensor-by-tensor in every fine-tuned job.')
missing = [e['run_name'] for e in EXP if not e['adapter_ok']]
if missing:
    print()
    print('WARNING: no final adapter on the Hub for: ' + ', '.join(missing) + '. Their fine-tuned jobs are skipped.')
