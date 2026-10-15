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
    raise RuntimeError('Configs use different data seeds ' + str(seeds) + '; jobs would be scored on different rows.')
SAMPLE_SEED = seeds[0]
LIMIT_LABEL = str(EVAL_LIMIT) if EVAL_LIMIT else 'all'
EVAL_TAG = SPLIT + '-n' + LIMIT_LABEL + '-seed' + str(SAMPLE_SEED) + ('-' + EVAL_TAG_SUFFIX if EVAL_TAG_SUFFIX else '')
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
    adir = e['adapter_dir']
    if not (adir / 'adapter_config.json').exists():
        rc = run_step('pull-' + rn, 'python app.py checkpoint pull --config ' + e['config'] + ' --step final',
                      allow_fail=True)
        if rc not in (0, 1):
            raise RuntimeError(rn + ': fetching the final adapter failed with exit ' + str(rc)
                               + ' (Hub error, not "not found"). Rerun when the Hub is reachable.')
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
    print('WARNING: no final adapter on the Hub for: ' + ', '.join(missing) + '. Their adapter jobs are recorded as '
          'missing_adapter and fail an integrity check.')

# Which models have no chat template (prompt falls back to plain text + BOS).
from src.training.model_loader import load_tokenizer
PLAIN_TEXT_PROMPT = {}
for e in EXP:
    if e['model'] not in PLAIN_TEXT_PROMPT:
        tok = load_tokenizer(load_config(e['config']))
        PLAIN_TEXT_PROMPT[e['model']] = not bool(getattr(tok, 'chat_template', None))
    e['plain_text_prompt'] = PLAIN_TEXT_PROMPT[e['model']]
print('Plain-text prompt (no chat template): ' + str(PLAIN_TEXT_PROMPT))
