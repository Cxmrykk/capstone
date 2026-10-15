# 12. Write the structured results bundle, upload it, save it with the version and finish
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
    'git_repo': GITHUB_REPO, 'git_ref': GIT_REF, 'git_commit': GIT_COMMIT, 'git_dirty': GIT_DIRTY,
    'hub_repo': HUB_REPO, 'hub_eval_prefix': HUB_EVAL_PREFIX,
    'platform_name': PLATFORM, 'kaggle_run_type': RUN_TYPE,
    'gpu': GPU_LABEL, 'gpus': GPUS, 'gpu_workers': GPU_IDS, 'eval_workers': N_EVAL_WORKERS,
    'python': platform.python_version(), 'platform': platform.platform(),
    'packages': {d: _version(d) for d in ('torch', 'transformers', 'peft', 'bitsandbytes', 'accelerate',
                                          'huggingface_hub', 'neo4j', 'pandas', 'numpy')},
    'decoding': 'greedy (temperature 0.0); max_new_tokens and batch size from each config',
    'execution_provider': 'Neo4j Labs demo server (neo4j+s://demo.neo4jlabs.com:7687)',
    'latency_note': ('s/item is NOT comparable across platforms (Colab: one T4 alone; Kaggle: '
                     + str(len(GPU_IDS)) + ' concurrent predictions on 4 vCPUs) or across batch sizes '
                     '(Gemma/Granite 4, Qwen 8). Compare latency only within one platform and batch size.'),
    'prompt_budget': 'max_seq_length minus generation.prompt_target_reserve (default max_new_tokens); '
                     'schema property lines trimmed as in training',
    'dead_db_policy': DEAD_DB_POLICY, 'dead_databases': DEAD_DBS,
    'equivalence_margin': EQUIV_MARGIN, 'pruned_ratio_threshold': PRUNED_RATIO_THRESHOLD,
    'statistics': {
        'ci_level': CI_LEVEL, 'bootstrap_samples': BOOTSTRAP_SAMPLES, 'stats_seed': STATS_SEED,
        'proportion_ci': 'Wilson score interval',
        'gleu_ci': 'percentile bootstrap over rows',
        'paired_tests': 'exact McNemar for binary metrics; paired bootstrap for mean sentence GLEU; '
                        'Holm correction within each family x metric',
        'equivalence': 'TOST via the (1 - 2 alpha) paired bootstrap CI, schema families only',
    },
    'experiments': [{k: _flat(e.get(k)) for k in ('model', 'variant', 'config', 'run_name', 'schema_mode',
                                                  'adapter_ok', 'fingerprint_trained', 'fingerprint_now',
                                                  'fingerprint_match')} for e in EXP],
    'jobs': {jid: rec.get('status') for jid, rec in JOB_RESULTS.items()},
    'jobs_expected': [j['job_id'] for j in JOBS],
    'jobs_not_run': [j['job_id'] for j in JOBS if JOB_RESULTS.get(j['job_id'], {}).get('status') != 'completed'],
    'jobs_skipped_for_time': SKIPPED_FOR_TIME,
    'job_platforms': {jid: rec.get('platform') or 'colab' for jid, rec in JOB_RESULTS.items()},
    'job_commits': {jid: rec.get('git_commit') for jid, rec in JOB_RESULTS.items()},
    'mixed_commits': len(BUNDLE_COMMITS) > 1, 'mixed_platforms': len(BUNDLE_PLATFORMS) > 1,
}

TABLES = {
    'headline': HEADLINE, 'overall_metrics': OVERALL, 'metrics_by_complexity': BY_COMPLEXITY,
    'metrics_by_data_source': BY_SOURCE, 'metrics_by_database': BY_DATABASE,
    'metrics_by_database_access': BY_DB_ACCESS,
    'metrics_by_complexity_and_database_access': BY_COMPLEXITY_DB_ACCESS,
    'gold_query_health': GOLD_HEALTH,
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
    'Generated ' + created + ' on Kaggle (session ' + SESSION + ', commit ' + GIT_COMMIT[:10] + ', ' + GPU_LABEL + ').',
    '',
    '## Files',
    '- `tables/headline.csv` (+ `headline.tex`): one row per job, formatted with confidence intervals.',
    '- `tables/overall_metrics.csv`: one row per job, all metrics, CI bounds, latency and provenance (platform, GPU).',
    '- `tables/metrics_by_complexity.csv`, `metrics_by_data_source.csv`, `metrics_by_database.csv`: breakdowns.',
    '- `tables/metrics_by_database_access.csv`, `metrics_by_complexity_and_database_access.csv`: rows with and without a demo database.',
    '- `tables/gold_query_health.csv`: databases where some or all gold queries fail to execute.',
    '- `tables/error_categories.csv`: row counts per outcome category per job.',
    '- `tables/paired_tests.csv`: paired comparisons on identical rows (raw and Holm-adjusted p-values, CIs, TOST for schema families).',
    '- `tables/schema_filtering.csv`: enhanced vs exact-match schema, token efficiency and accuracy (all rows and pruned rows only).',
    '- `tables/training_summary.csv`, `training_curves.csv`: training telemetry merged across sessions.',
    '- `tables/integrity_checks.csv`: sample consistency, adapter verification, commit/platform consistency, gold-query health.',
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
    '- `executable_rate`: share of executed rows whose prediction ran without error; `executable_rate_gold_ok` restricts it to rows whose gold query runs. `syntax_valid_rate`: share passing EXPLAIN.',
    '- Error categories: correct, wrong_result (ran, different result), invalid_query (fails EXPLAIN: syntax or schema), runtime_error (passes EXPLAIN, fails at run time), execution_error (syntax not checked), empty_output, gold_failed, excluded_dead_db, connection_failed (Neo4j unreachable; not scored), not_executed (no database).',
    '- Paired tests: exact McNemar for binary metrics; paired bootstrap for mean sentence GLEU; Holm correction within each family x metric.',
    '',
    '## Caveats',
    '- Every GPU job uses the same seeded sample of the ' + SPLIT + ' split (n=' + LIMIT_LABEL + ', seed ' + str(SAMPLE_SEED) + '). GGUF CPU jobs use a smaller subset of it.',
    '- s/item is not comparable across platforms or batch sizes (Gemma/Granite use batch 4, Qwen 8). Compare latency only within one platform (see the `platform` column).',
    '- Prompts are fitted to max_seq_length minus the generation reserve by trimming schema property lines, as in training; `Trimmed` counts affected rows per job.',
    '- `Executable` in the headline is computed over rows whose gold query runs (`executable_rate_gold_ok`).',
    '- Databases where every gold query fails (' + (', '.join(DEAD_DBS) or 'none') + ') cannot contribute to execution accuracy; see `gold_query_health.csv`.',
    '- Execution accuracy depends on the Neo4j Labs demo server; rows without a database are excluded from it.',
    '- Gemma 4 E2B is a base (pre-trained) checkpoint without a chat template, so its zero-shot prompt uses the plain-text fallback (labelled in the Condition column).',
    '- Bootstrap p-values below 1/BOOTSTRAP_SAMPLES are reported as < ' + format(1 / BOOTSTRAP_SAMPLES, '.2g') + ' and floored at that value before Holm adjustment, so their Holm p is an upper bound.',
    '- Schema-filtering equivalence: TOST at margin +/-' + str(EQUIV_MARGIN) + ' via the (1 - 2 alpha) paired bootstrap CI.',
    '- `cross_schema` jobs score the enhanced-trained adapter with exact-match prompts (inference-time effect of schema filtering).',
    '- Percentile bootstrap intervals for corpus GLEU are reported as computed and need not contain the point estimate; the figures clip error bars at the estimate.',
    '- `logged_training_hours` is the span of logged steps per session; it excludes model loading, tokenisation and uploads.',
    '- `trainer_train_loss_final_session` is reported by the Trainer for the final session only.',
    '- Published baselines come from other papers with different samples and protocols; compare them descriptively only.',
]
(OUT_DIR / 'README.md').write_text(LF.join(readme) + LF, encoding='utf-8')

if MANIFEST['mixed_commits'] or MANIFEST['mixed_platforms']:
    print('WARNING: this bundle mixes commits ' + str([c[:10] for c in BUNDLE_COMMITS]) + ' and/or platforms '
          + str(BUNDLE_PLATFORMS) + '. It is not a single-protocol result; follow the re-run plan.')

# Bundle and zip go to /kaggle/working so they are saved with the notebook version.
SAVED_DIR = Path(OUTPUT_DIR) / 'thesis-results' / EVAL_TAG / SESSION
shutil.copytree(OUT_DIR, SAVED_DIR, dirs_exist_ok=True)
archive = shutil.make_archive(str(Path(OUTPUT_DIR) / ('thesis-results-' + EVAL_TAG + '-' + SESSION)), 'zip',
                              root_dir=OUT_DIR)
uploaded = hub_upload_folder(OUT_DIR, HUB_EVAL_PREFIX + '/thesis/' + SESSION,
                             '[final-eval] thesis bundle ' + EVAL_TAG + ' ' + SESSION)
print('Results bundle : ' + str(SAVED_DIR) + ' (saved with the version)')
print('Zip archive    : ' + archive)
print('Logs           : ' + str(LOG_DIR))
if UPLOAD_TO_HUB:
    print('Hub upload     : ' + ('OK' if uploaded else 'FAILED') + ' (https://huggingface.co/' + HUB_REPO
          + '/tree/main/' + HUB_EVAL_PREFIX + '/thesis/' + SESSION + ')')

summary = ['Session ' + SESSION + ' finished (' + EVAL_TAG + ').']
for _, r in OVERALL.iterrows():
    summary.append(str(r['job_id']) + ': EA ' + _fmt(r.get('execution_accuracy')) + ', GLEU '
                   + _fmt(r.get('google_bleu_corpus')) + ', EM ' + _fmt(r.get('exact_match')))
if MANIFEST['jobs_not_run']:
    summary.append('Not run: ' + ', '.join(MANIFEST['jobs_not_run']) + '. Commit a new version to finish them.')
if len(failed_checks):
    summary.append(str(len(failed_checks)) + ' integrity check(s) failed; see integrity_checks.csv.')
STATUS['outcome'] = 'completed'
notify('finished', LF.join(summary),
       tags=['warning'] if len(failed_checks) or MANIFEST['jobs_not_run'] else ['white_check_mark'])
display(HEADLINE)
