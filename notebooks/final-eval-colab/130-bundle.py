# 12. Write the structured results bundle, upload it and finish
import importlib.metadata as importlib_metadata


def _records(df):
    if df is None or df.empty:
        return []
    return json.loads(df.to_json(orient='records', date_format='iso', default_handler=str))


def _version(dist):
    try:
        return importlib_metadata.version(dist)
    except Exception:
        return None


created = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
MANIFEST = {
    'title': 'Text2Cypher LoRA fine-tuning: final evaluation results',
    'created_utc': created, 'session': SESSION, 'eval_tag': EVAL_TAG, 'split': SPLIT,
    'eval_limit': EVAL_LIMIT, 'sample_seed': SAMPLE_SEED, 'gguf_bench_limit': GGUF_BENCH_LIMIT,
    'git_repo': GITHUB_REPO, 'git_commit': GIT_COMMIT, 'git_dirty': GIT_DIRTY,
    'hub_repo': HUB_REPO, 'hub_eval_prefix': HUB_EVAL_PREFIX,
    'gpu': GPU_NAME, 'python': platform.python_version(), 'platform': platform.platform(),
    'packages': {d: _version(d) for d in ('torch', 'transformers', 'peft', 'bitsandbytes', 'accelerate',
                                          'huggingface_hub', 'neo4j', 'pandas', 'numpy')},
    'decoding': 'greedy (temperature 0.0); max_new_tokens and batch size from each config',
    'execution_provider': 'Neo4j Labs demo server (neo4j+s://demo.neo4jlabs.com:7687)',
    'statistics': {
        'ci_level': CI_LEVEL, 'bootstrap_samples': BOOTSTRAP_SAMPLES, 'stats_seed': STATS_SEED,
        'proportion_ci': 'Wilson score interval',
        'gleu_ci': 'percentile bootstrap over rows',
        'paired_tests': 'exact McNemar for binary metrics; paired bootstrap for mean sentence GLEU; '
                        'Holm correction within each family x metric',
    },
    'experiments': [{k: _flat(e.get(k)) for k in ('model', 'variant', 'config', 'run_name', 'schema_mode',
                                                  'adapter_ok', 'fingerprint_trained', 'fingerprint_now',
                                                  'fingerprint_match')} for e in EXP],
    'jobs': {jid: rec.get('status') for jid, rec in JOB_RESULTS.items()},
    'jobs_not_run': [j['job_id'] for j in JOBS if j['job_id'] not in JOB_RESULTS],
}

TABLES = {
    'headline': HEADLINE, 'overall_metrics': OVERALL, 'metrics_by_complexity': BY_COMPLEXITY,
    'metrics_by_data_source': BY_SOURCE, 'metrics_by_database': BY_DATABASE,
    'error_categories': ERRORS, 'paired_tests': PAIRED, 'schema_filtering': SCHEMA_FILTER,
    'training_summary': TRAINING, 'training_curves': CURVES, 'integrity_checks': INTEGRITY_DF,
    'jobs': JOBS_DF, 'published_baselines': PUBLISHED,
}
for name, df in TABLES.items():
    if df is not None:
        df.to_csv(OUT_DIR / 'tables' / (name + '.csv'), index=False)
try:
    ROWS.to_parquet(OUT_DIR / 'predictions_long.parquet', index=False)
except Exception as exc:
    print('Parquet export failed (' + str(exc) + '); the gzipped CSV is still written.')
ROWS.drop(columns=['predicted_raw']).to_csv(OUT_DIR / 'predictions_long.csv.gz', index=False, compression='gzip')
try:
    HEADLINE.to_latex(OUT_DIR / 'tables' / 'headline.tex', index=False)
except Exception as exc:
    print('LaTeX export skipped: ' + str(exc))

results = {'manifest': MANIFEST}
results.update({name: _records(df) for name, df in TABLES.items() if name != 'training_curves'})
(OUT_DIR / 'results.json').write_text(json.dumps(results, indent=2, default=str), encoding='utf-8')
(OUT_DIR / 'manifest.json').write_text(json.dumps(MANIFEST, indent=2, default=str), encoding='utf-8')

jobs_dir = OUT_DIR / 'jobs'
jobs_dir.mkdir(exist_ok=True)
for jid in JOB_RESULTS:
    src = EVAL_DIR / 'jobs' / jid
    for fname in ('job.json', 'report.md', 'predict.log', 'evaluate.log'):
        if (src / fname).exists():
            (jobs_dir / jid).mkdir(exist_ok=True)
            shutil.copy2(src / fname, jobs_dir / jid / fname)

readme = [
    '# Text2Cypher final evaluation: ' + EVAL_TAG,
    '',
    'Generated ' + created + ' (session ' + SESSION + ', commit ' + GIT_COMMIT[:10] + ', ' + GPU_NAME + ').',
    '',
    '## Files',
    '- `tables/headline.csv` (+ `headline.tex`): one row per job, formatted with confidence intervals.',
    '- `tables/overall_metrics.csv`: one row per job, all metrics, CI bounds, latency and provenance.',
    '- `tables/metrics_by_complexity.csv`, `metrics_by_data_source.csv`, `metrics_by_database.csv`: breakdowns.',
    '- `tables/error_categories.csv`: row counts per outcome category per job.',
    '- `tables/paired_tests.csv`: paired comparisons on identical rows (raw and Holm-adjusted p-values).',
    '- `tables/schema_filtering.csv`: enhanced vs exact-match schema, token efficiency and accuracy.',
    '- `tables/training_summary.csv`, `training_curves.csv`: training telemetry merged across sessions.',
    '- `tables/integrity_checks.csv`: sample consistency, adapter verification, gold-query health.',
    '- `tables/jobs.csv`: job registry; `tables/published_baselines.csv`: hand-entered literature numbers.',
    '- `predictions_long.parquet` / `predictions_long.csv.gz`: one row per job and instance.',
    '- `results.json`: all tables (except curves) plus the manifest; `manifest.json`: provenance.',
    '- `figures/`: PNG (300 dpi) and PDF figures; `jobs/`: per-job logs, reports and metadata.',
    '',
    '## Metric definitions',
    '- `google_bleu_corpus`: corpus Google-BLEU over 1-4 grams of Cypher-aware tokens; CI by percentile bootstrap over rows.',
    '- `exact_match`: string equality after trimming and dropping a trailing semicolon. `normalised_exact_match` also ignores whitespace and keyword case. Wilson CIs.',
    '- `execution_accuracy`: share of rows (with a demo database and an executable gold query) whose predicted result set equals the gold result set. Row order is compared only when the gold query has ORDER BY. Wilson CI.',
    '- `execution_accuracy_toolkit`: same numerator over every executed row including failing gold queries (matches `app.py evaluate` reports).',
    '- `executable_rate`: share of executed rows whose prediction ran without error. `syntax_valid_rate`: share passing EXPLAIN.',
    '- Error categories: correct, wrong_result (ran, different result), invalid_query (fails EXPLAIN: syntax or schema), runtime_error (passes EXPLAIN, fails at run time), execution_error (syntax not checked), empty_output, gold_failed, not_executed (no database).',
    '- Paired tests: exact McNemar for binary metrics; paired bootstrap for mean sentence GLEU; Holm correction within each family x metric.',
    '',
    '## Caveats',
    '- Every GPU job uses the same seeded sample of the ' + SPLIT + ' split (n=' + LIMIT_LABEL + ', seed ' + str(SAMPLE_SEED) + '). GGUF CPU jobs use a smaller subset of it.',
    '- Execution accuracy depends on the Neo4j Labs demo server; rows without a database are excluded from it.',
    '- Gemma 4 E2B is a base (pre-trained) checkpoint without a chat template, so its zero-shot prompt uses the plain-text fallback.',
    '- `logged_training_hours` is the span of logged steps per session; it excludes model loading, tokenisation and uploads.',
    '- `trainer_train_loss_final_session` is reported by the Trainer for the final session only.',
    '- Published baselines come from other papers with different samples and protocols; compare them descriptively only.',
]
(OUT_DIR / 'README.md').write_text(LF.join(readme) + LF, encoding='utf-8')

archive = shutil.make_archive(str(OUT_DIR.parent / ('thesis-results-' + EVAL_TAG)), 'zip', root_dir=OUT_DIR)
uploaded = hub_upload_folder(OUT_DIR, HUB_EVAL_PREFIX + '/thesis/' + SESSION,
                             '[final-eval] thesis bundle ' + EVAL_TAG + ' ' + SESSION)
print('Results bundle : ' + str(OUT_DIR))
print('Zip archive    : ' + archive)
if UPLOAD_TO_HUB:
    print('Hub upload     : ' + ('OK' if uploaded else 'FAILED') + ' (https://huggingface.co/' + HUB_REPO
          + '/tree/main/' + HUB_EVAL_PREFIX + '/thesis/' + SESSION + ')')

if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    dest = Path('/content/drive/MyDrive/text2cypher-thesis') / EVAL_TAG / SESSION
    shutil.copytree(OUT_DIR, dest, dirs_exist_ok=True)
    shutil.copy2(archive, dest.parent / Path(archive).name)
    print('Drive copy     : ' + str(dest))

if DOWNLOAD_ZIP:
    try:
        from google.colab import files as colab_files
        colab_files.download(archive)
    except Exception as exc:
        print('Browser download unavailable: ' + str(exc))

summary = ['Session ' + SESSION + ' finished (' + EVAL_TAG + ').']
for _, r in OVERALL.iterrows():
    summary.append(str(r['job_id']) + ': EA ' + _fmt(r.get('execution_accuracy')) + ', GLEU '
                   + _fmt(r.get('google_bleu_corpus')) + ', EM ' + _fmt(r.get('exact_match')))
if len(failed_checks):
    summary.append(str(len(failed_checks)) + ' integrity check(s) failed; see integrity_checks.csv.')
STATUS['outcome'] = 'completed'
notify('finished', LF.join(summary), tags=['warning'] if len(failed_checks) else ['white_check_mark'])
display(HEADLINE)
