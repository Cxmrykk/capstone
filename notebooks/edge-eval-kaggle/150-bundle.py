# 14. Write the structured results bundle, upload it, save it with the version and finish
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
    'title': 'Text2Cypher LoRA fine-tuning: edge (CPU) evaluation and supplementary analyses',
    'created_utc': created, 'session': SESSION, 'edge_tag': EDGE_TAG, 'edge_prefix': EDGE_PREFIX,
    'reference_eval_tag': REFERENCE_EVAL_TAG, 'reference_commits': REF_COMMITS,
    'split': SPLIT, 'reference_limit': REFERENCE_LIMIT, 'edge_limit': len(EDGE_IDS), 'edge_subset': EDGE_SUBSET,
    'edge_subset_seed': EDGE_SUBSET_SEED, 'sample_seed': SAMPLE_SEED, 'edge_ids_sha': EDGE_IDS_SHA,
    'edge_rows_with_database': n_db_edge,
    'git_repo': GITHUB_REPO, 'git_ref': GIT_REF, 'git_commit': GIT_COMMIT, 'git_dirty': GIT_DIRTY,
    'llama_cpp_repo': LLAMA_CPP_REPO, 'llama_cpp_ref': LLAMA_REF_USED, 'llama_cpp_commit': LLAMA_COMMIT,
    'llama_cpp_version': LLAMA_VERSION,
    'llama_cpp_build': 'CPU only, GGML_NATIVE=ON (kernels tuned to the host CPU), Release',
    'hub_repo': HUB_REPO, 'platform_name': PLATFORM, 'kaggle_run_type': RUN_TYPE,
    'cpu': {'model': CPU_MODEL, 'logical_cpus': LOGICAL_CPUS, 'physical_cores': PHYSICAL_CORES,
            'simd_flags': SIMD_FLAGS, 'cgroup_cpu_limit': CGROUP_CPU_LIMIT, 'ram_total_gb': round(RAM_TOTAL_GB, 1),
            'lscpu': {k: LSCPU.get(k) for k in ('Architecture', 'Model name', 'Socket(s)', 'Core(s) per socket',
                                                'Thread(s) per core', 'L2 cache', 'L3 cache', 'Hypervisor vendor')}},
    'gpu_attached_but_unused': GPUS_PRESENT,
    'python': platform.python_version(), 'platform': platform.platform(),
    'packages': {d: _version(d) for d in ('torch', 'transformers', 'peft', 'safetensors', 'huggingface_hub', 'gguf',
                                          'neo4j', 'pandas', 'numpy', 'requests')},
    'export': {'merge_dtype': MERGE_DTYPE, 'intermediate': 'f16 GGUF (convert_hf_to_gguf.py)',
               'quantiser': 'llama-quantize from the same build'},
    'serving': {'server': 'llama-server', 'command_template': server_cmd_template(),
                'mmap': SERVER_MMAP['mode'], 'mmap_env': SERVER_MMAP['env'],
                'mmap_help_lines': SERVER_MMAP['help_lines'], 'rejected_options': sorted(SERVER_REJECTED),
                'smoke_test': SERVER_SMOKE, 'help_file': str(LOG_DIR / 'llama-server-help.txt'),
                'threads': THREADS_FOR, 'server_threads_setting': SERVER_THREADS, 'cache_prompt': CACHE_PROMPT,
                'edge_stop_mode': EDGE_STOP_MODE,
                'decoding': 'greedy (temperature 0, top_k 1); max_new_tokens from each config',
                'warmup': 'one untimed 1-token request after load, excluded from latency statistics'},
    'benchmark': {'tool': 'llama-bench', 'prompt_tokens': BENCH_PROMPT_TOKENS, 'gen_tokens': BENCH_GEN_TOKENS,
                  'reps': BENCH_REPS, 'threads': BENCH_THREADS, 'thread_sweep_quants': THREAD_SWEEP_QUANTS,
                  'latency_quants': LATENCY_QUANTS, 'mmap': 'llama-bench default'},
    'execution_provider': 'Neo4j Labs demo server (neo4j+s://demo.neo4jlabs.com:7687)',
    'dead_db_policy': DEAD_DB_POLICY,
    'statistics': {'ci_level': CI_LEVEL, 'bootstrap_samples': BOOTSTRAP_SAMPLES, 'stats_seed': STATS_SEED,
                   'proportion_ci': 'Wilson score interval', 'gleu_ci': 'percentile bootstrap over rows',
                   'paired_tests': 'exact McNemar for binary metrics; paired bootstrap for mean sentence GLEU and '
                                   'latency; Holm correction within each family x metric',
                   'equivalence': 'TOST via the (1 - 2 alpha) paired bootstrap CI, margin ' + str(EQUIV_MARGIN)},
    'experiments': [{k: _flat(e.get(k)) for k in ('model', 'variant', 'config', 'run_name', 'schema_mode', 'adapter_ok',
                                                  'fingerprint_trained', 'fingerprint_now', 'fingerprint_match')}
                    for e in EXP],
    'jobs': {jid: rec.get('status') for jid, rec in JOB_RESULTS.items()},
    'jobs_expected': [j['job_id'] for j in JOBS],
    'jobs_not_run': [j['job_id'] for j in JOBS if JOB_RESULTS.get(j['job_id'], {}).get('status') != 'completed'],
    'jobs_skipped_for_time': SKIPPED_FOR_TIME,
    'bench': {m: rec.get('status') for m, rec in BENCH_RESULTS.items()},
    'exports': {rn: rec.get('status') for rn, rec in EXPORT_RECORDS.items()},
    'supplementary': RUN_SUPPLEMENTARY, 'filter_mode': FILTER_MODE, 'filter_params': FILTER_KW,
}

TABLES = {
    'edge_headline': HEADLINE, 'edge_overall_metrics': OVERALL, 'edge_fidelity_vs_gpu': FIDELITY,
    'edge_paired_tests': PAIRED, 'edge_schema_tradeoff': SCHEMA_EDGE, 'edge_error_categories': ERRORS,
    'edge_tokenizer_parity': TOKENIZER_PARITY, 'throughput_llama_bench': BENCH, 'footprint': FOOTPRINT,
    'serving_threads': THREADS_DF, 'exports': EXPORTS_DF, 'integrity_checks': INTEGRITY_DF,
    'jobs': pd.DataFrame([{k: _flat(v) for k, v in rec.items()} for rec in JOB_RESULTS.values()]),
    'supp_subset_metrics': SUBSET_METRICS, 'supp_filter_coverage': COVERAGE,
    'supp_filter_coverage_rows': COVERAGE_ROWS, 'supp_filter_coverage_vs_ea': COVERAGE_EA,
    'supp_schema_hallucination': HALLUCINATION, 'supp_execution_result_breakdown': RESULT_BREAKDOWN,
    'supp_model_agreement': AGREEMENT, 'supp_sample_composition': COMPOSITION,
}
for name, df in TABLES.items():
    if df is not None:
        df.to_csv(OUT_DIR / 'tables' / (name + '.csv'), index=False)
if not EDGE_LONG.empty:
    try:
        EDGE_LONG.to_parquet(OUT_DIR / 'edge_predictions_long.parquet', index=False)
    except Exception as exc:
        print('Parquet export failed (' + str(exc) + '); the gzipped CSV is still written.')
    EDGE_LONG.drop(columns=['predicted_raw'], errors='ignore').to_csv(
        OUT_DIR / 'edge_predictions_long.csv.gz', index=False, compression='gzip')
for name, df in (('edge_headline', HEADLINE), ('edge_schema_tradeoff', SCHEMA_EDGE)):
    try:
        if not df.empty:
            df.to_latex(OUT_DIR / 'tables' / (name + '.tex'), index=False)
    except Exception as exc:
        print('LaTeX export of ' + name + ' skipped: ' + str(exc))

results = {'manifest': MANIFEST}
results.update({name: _records(df) for name, df in TABLES.items() if name != 'supp_filter_coverage_rows'})
(OUT_DIR / 'results.json').write_text(json.dumps(results, indent=2, default=str), encoding='utf-8')
(OUT_DIR / 'manifest.json').write_text(json.dumps(MANIFEST, indent=2, default=str), encoding='utf-8')

# Per-job logs, reports and records; bench and export records.
for sub_name, src_root, names in (('jobs', EDGE_DIR / 'jobs', ('job.json', 'report.md', 'evaluate.log', 'server.log')),
                                  ('bench', EDGE_DIR / 'bench', ('bench.json',)),
                                  ('exports', EDGE_DIR / 'exports', None)):
    if not src_root.exists():
        continue
    for item in sorted(src_root.iterdir()):
        dest = OUT_DIR / sub_name
        dest.mkdir(exist_ok=True)
        if item.is_dir():
            for fname in names or ():
                if (item / fname).exists():
                    (dest / item.name).mkdir(exist_ok=True)
                    shutil.copy2(item / fname, dest / item.name / fname)
        elif names is None:
            shutil.copy2(item, dest / item.name)

cpu_short = CPU_MODEL + ', ' + str(LOGICAL_CPUS) + ' logical CPUs'
if server_mmap_off():
    load_desc = ('`load_s` is process start to a healthy /health endpoint with memory mapping off ('
                 + SERVER_MMAP['mode'] + '; weights are read into memory during load)')
else:
    load_desc = ('`load_s` is process start to a healthy /health endpoint with memory mapping ON ('
                 + SERVER_MMAP['mode'] + '): weights are paged in lazily, so `load_s` understates the cost of '
                 'reading them; `first_request_ms` and `rss_after_first_request_mb` carry it')
readme = [
    '# Text2Cypher edge (CPU) evaluation: ' + EDGE_TAG,
    '',
    'Generated ' + created + ' on Kaggle (session ' + SESSION + ', commit ' + GIT_COMMIT[:10] + ', llama.cpp '
    + LLAMA_COMMIT[:10] + ', ' + cpu_short + ').',
    'GPU reference: `final-eval/' + REFERENCE_EVAL_TAG + '` (commits ' + ', '.join(c[:10] for c in REF_COMMITS) + ').',
    'Server command: `' + server_cmd_template() + '`.',
    '',
    '## Files',
    '- `tables/edge_headline.csv` (+ `.tex`): one row per CPU job: accuracy with CIs, GPU accuracy on the same rows, '
    'paired delta, share of identical predictions, latency, size and memory.',
    '- `tables/edge_overall_metrics.csv`: all metrics, CI bounds, latency percentiles and provenance per CPU job; '
    '`gpu_*` columns are the reference job restricted to the same rows.',
    '- `tables/edge_fidelity_vs_gpu.csv`: CPU vs GPU per job (EM, execution match, sentence GLEU deltas with CIs, '
    'TOST equivalence, identical-prediction shares).',
    '- `tables/edge_paired_tests.csv`: every paired comparison (families edge_vs_gpu, quant_fidelity, '
    'schema_filtering_edge, schema_inference_effect_edge, schema_training_effect_edge, model_comparison_edge).',
    '- `tables/edge_schema_tradeoff.csv`: enhanced vs exact-match schema on the CPU (prompt tokens, prefill and total '
    'latency vs execution accuracy).',
    '- `tables/throughput_llama_bench.csv`: llama-bench tokens/s per model, quantisation, test (ppN/tgN) and threads, '
    'with bits per weight.',
    '- `tables/footprint.csv`: llama-server load time, RSS after load and after the first request, peak RSS, '
    'first-request time and memory-mapping mode per quantisation.',
    '- `tables/serving_threads.csv`: thread count used for serving and the estimates it was chosen from.',
    '- `tables/exports.csv`: merge/convert/quantise timings, sizes, SHA-256, adapter verification and model class.',
    '- `tables/edge_tokenizer_parity.csv`: GGUF prompt tokens minus training-tokenizer tokens per job.',
    '- `tables/edge_error_categories.csv`, `tables/integrity_checks.csv`, `tables/jobs.csv`.',
    '- `tables/supp_*.csv`: supplementary analyses of GPU reference and CPU jobs (see below).',
    '- `edge_predictions_long.parquet` / `.csv.gz`: one row per CPU job and instance, with server timings and the '
    'matching GPU prediction and correctness.',
    '- `results.json` (all tables except per-row coverage, plus the manifest), `manifest.json`, `figures/`, '
    '`jobs/`, `bench/`, `exports/`.',
    '',
    '## Metric definitions',
    '- `execution_accuracy`: share of judged rows (a demo database, an executable gold query, no outage or excluded '
    'database) whose result set equals the gold result. Wilson CI. `execution_accuracy_nonempty_gold` restricts it to '
    'rows whose gold query returns at least one row; `share_correct_with_empty_gold` is the share of matches where the '
    'gold result is empty.',
    '- `exact_match`, `normalised_exact_match`, `google_bleu_corpus`: as in final-eval (Wilson CIs; percentile '
    'bootstrap for GLEU).',
    '- Latency: `wall_ms` is client-measured time per request (one request at a time, after one untimed warm-up); '
    '`prefill_ms` and `decode_ms` are llama-server timings for prompt processing and generation.',
    '- `prefill_share_of_wall`: summed prefill time over summed wall time. `queries_per_hour`: 3.6e6 / mean wall ms.',
    '- Footprint: ' + load_desc + '; `peak_rss_mb` is VmHWM of the server process (includes mapped weight pages '
    'once touched).',
    '- Throughput: llama-bench `avg_ts` (tokens/s) for ppN (prompt processing) and tgN (generation), ' + str(BENCH_REPS)
    + ' repetitions. `bits_per_weight` = 8 * model_size / model_n_params as reported by llama-bench.',
    '- Tokenizer parity: `tokens_evaluated` reported by llama-server minus `prompt_tokens` from the training tokenizer. '
    '0 means the GGUF model saw the same token count as in training (BOS handled identically).',
    '- Paired tests: exact McNemar for binary metrics on identical rows; paired bootstrap for sentence GLEU and latency; '
    'Holm correction within each family x metric. TOST equivalence at +/-' + str(EQUIV_MARGIN) + ' for accuracy deltas.',
    '',
    '## Supplementary analyses',
    '- `supp_subset_metrics`: metrics on all rows, without `' + FUNCTIONAL_SOURCE + '` (near-saturated EM inflates '
    'text metrics), with and without a database.',
    '- `supp_filter_coverage*`: whether the ' + FILTER_MODE + ' filter keeps every label, relationship type and '
    'property the gold query uses (heuristic extraction; name-level for properties; label predicates in WHERE are not '
    'parsed). `supp_filter_coverage_vs_ea` splits execution accuracy by fully covered vs not.',
    '- `supp_schema_hallucination`: predicted labels/relationship types/properties absent from the full schema, and use '
    'of real elements that were not in the prompt schema (prompt schema computed before budget trimming).',
    '- `supp_execution_result_breakdown`: judged rows split into correct (non-empty / empty gold), prediction failed, '
    'empty prediction result, row-count mismatch and same row count with different values.',
    '- `supp_model_agreement`: oracle, majority and all-correct execution accuracy across the fine-tuned enhanced GPU '
    'models on rows judged for all of them; pairwise identical-prediction shares.',
    '- `supp_sample_composition`: data source x database access x complexity counts for the 500 reference rows and the '
    'edge subset.',
    '',
    '## Caveats',
    '- CPU jobs score ' + str(len(EDGE_IDS)) + ' of the ' + str(REFERENCE_LIMIT) + ' reference rows (' + EDGE_SUBSET
    + ', seed ' + str(EDGE_SUBSET_SEED) + '); only ' + str(n_db_edge) + ' have a database, so CPU execution-accuracy '
    'CIs are wide. GPU comparisons always use the same rows.',
    '- Timings are for ' + cpu_short + ' (Kaggle VM), not a specific edge device; the thread sweep indicates scaling '
    'to fewer cores. The llama.cpp build is tuned to this CPU (GGML_NATIVE).',
    '- Adapters were trained against an NF4-quantised base and merged into the ' + MERGE_DTYPE + ' base before GGUF '
    'quantisation; CPU-vs-GPU differences combine the merge and the GGUF quantisation.',
    '- Stop condition: ' + ('EOS only, matching the GPU jobs' if EDGE_STOP_MODE == 'hf'
                            else 'EOS plus generation.stop_strings (differs from the GPU jobs)') + '.',
    '- Prompt caching was ' + ('on' if CACHE_PROMPT else 'off') + ': each query pays the full prefill.',
    '- llama-server memory mapping: ' + SERVER_MMAP['mode'] + '. Options probed from the build\'s --help'
    + ((' (rejected at start-up and dropped: ' + ', '.join(sorted(SERVER_REJECTED)) + ')') if SERVER_REJECTED else '')
    + '; llama-bench used its own defaults.',
    '- Load times may benefit from the OS page cache because GGUFs were written shortly before loading.',
    '- Element extraction for coverage and hallucination is regex-based and approximate.',
]
(OUT_DIR / 'README.md').write_text(LF.join(readme) + LF, encoding='utf-8')

SAVED_DIR = Path(OUTPUT_DIR) / 'edge-results' / EDGE_TAG / SESSION
shutil.copytree(OUT_DIR, SAVED_DIR, dirs_exist_ok=True)
archive = shutil.make_archive(str(Path(OUTPUT_DIR) / ('edge-results-' + EDGE_TAG + '-' + SESSION)), 'zip',
                              root_dir=OUT_DIR)
uploaded = hub_upload_folder(OUT_DIR, EDGE_PREFIX + '/thesis/' + SESSION,
                             '[edge-eval] thesis bundle ' + EDGE_TAG + ' ' + SESSION)
print('Results bundle : ' + str(SAVED_DIR) + ' (saved with the version)')
print('Zip archive    : ' + archive)
print('Logs           : ' + str(LOG_DIR))
if UPLOAD_TO_HUB:
    print('Hub upload     : ' + ('OK' if uploaded else 'FAILED') + ' (https://huggingface.co/' + HUB_REPO
          + '/tree/main/' + EDGE_PREFIX + '/thesis/' + SESSION + ')')

summary = ['Session ' + SESSION + ' finished (' + EDGE_TAG + ', ' + CPU_MODEL[:40] + ').']
for _, r in OVERALL.iterrows():
    summary.append(str(r['job_id']) + ': EA ' + _fmt(r.get('execution_accuracy')) + ' (GPU '
                   + _fmt(r.get('gpu_execution_accuracy')) + '), median ' + _fmt(_g(r, 'wall_ms_p50') / 1000, '.1f')
                   + ' s/query, peak ' + _fmt(r.get('peak_rss_mb'), '.0f') + ' MB')
if MANIFEST['jobs_not_run']:
    summary.append('Not run: ' + ', '.join(MANIFEST['jobs_not_run']) + '. Commit a new version to finish them.')
if len(failed_checks):
    summary.append(str(len(failed_checks)) + ' integrity check(s) failed; see integrity_checks.csv.')
STATUS['outcome'] = 'completed'
notify('finished', LF.join(summary),
       tags=['warning'] if len(failed_checks) or MANIFEST['jobs_not_run'] else ['white_check_mark'])
display(HEADLINE)
