# 8. Export: merge (verified) -> f16 GGUF -> quantise, only for runs with unfinished work
GGUF_DIR = WORK_ROOT / 'gguf'
MERGED_ROOT = WORK_ROOT / 'merged'
QUANTIZE_BIN = LLAMA_BIN / 'llama-quantize'


def _dir_bytes(path):
    return sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())


def export_run(e, quants):
    rn = e['run_name']
    out_dir = GGUF_DIR / rn
    out_dir.mkdir(parents=True, exist_ok=True)
    merged = MERGED_ROOT / rn
    f16 = out_dir / (rn + '-f16.gguf')
    rec = {'run_name': rn, 'model': e['model'], 'variant': e['variant'], 'merge_dtype': MERGE_DTYPE,
           'requested_quants': sorted(quants, key=_qkey), 'status': 'started', 'quants': {}, **_provenance()}
    EXPORT_RECORDS[rn] = rec
    try:
        print('Free disk before export: ' + format(_free_gb(WORK_ROOT), '.0f') + ' GB')
        t0 = time.time()
        merge_log = LOG_DIR / ('merge-' + rn + '.log')
        run_step('merge-' + rn, 'python app.py export merge --config ' + e['config'] + ' --adapter '
                 + str(e['adapter_dir']) + ' --out ' + str(merged) + ' --dtype ' + MERGE_DTYPE, log_path=merge_log)
        rec['merge_s'] = round(time.time() - t0, 1)
        log_lines = merge_log.read_text(encoding='utf-8', errors='replace').splitlines()
        rec['adapter_verified'] = [ln.split('| ', 1)[-1] for ln in log_lines if 'Adapter verified:' in ln]
        rec['adapter_remapped'] = [ln.split('| ', 1)[-1] for ln in log_lines
                                   if 'Remapped' in ln and 'adapter tensor' in ln]
        if not rec['adapter_verified']:
            raise RuntimeError('The merge log has no Adapter verified line; refusing to export a possibly '
                               'unadapted model (history/027.md).')
        rec['merge_metadata'] = json.loads((merged / 'merge_metadata.json').read_text(encoding='utf-8'))
        rec['merged_size_bytes'] = _dir_bytes(merged)

        t0 = time.time()
        run_step('convert-' + rn, 'python app.py export gguf --merged ' + str(merged) + ' --out-dir '
                 + str(out_dir) + ' --quant F16', log_path=LOG_DIR / ('convert-' + rn + '.log'))
        rec['convert_s'] = round(time.time() - t0, 1)
        if not f16.exists():
            raise RuntimeError('Conversion finished without ' + f16.name + '.')
        rec['f16_size_bytes'] = f16.stat().st_size
        if not KEEP_MERGED:
            shutil.rmtree(merged, ignore_errors=True)

        for q in sorted(quants, key=_qkey):
            if q == 'F16':
                path, qs = f16, 0.0
            else:
                path = out_dir / (rn + '-' + q.lower() + '.gguf')
                t0 = time.time()
                run_step('quantize-' + rn + '-' + q.lower(),
                         str(QUANTIZE_BIN) + ' ' + str(f16) + ' ' + str(path) + ' ' + q + ' ' + str(LOGICAL_CPUS),
                         log_path=LOG_DIR / ('quantize-' + rn + '-' + q.lower() + '.log'), echo=False)
                qs = round(time.time() - t0, 1)
            rec['quants'][q] = {'path': str(path), 'file': path.name, 'size_bytes': path.stat().st_size,
                                'sha256': sha256_file(path), 'quantize_s': qs}
            print('  ' + q.ljust(8) + format(path.stat().st_size / 1e6, '9.0f') + ' MB  ' + path.name)
            if UPLOAD_GGUF and q in UPLOAD_GGUF_QUANTS:
                hub_upload_file(path, EDGE_PREFIX + '/gguf/' + rn + '/' + path.name, '[edge-eval] ' + path.name)
        if 'F16' not in quants and not KEEP_F16:
            f16.unlink(missing_ok=True)
        rec['status'] = 'ok'
    except KeyboardInterrupt:
        rec['status'] = 'interrupted'
        raise
    except Exception as exc:
        rec['status'] = 'failed'
        rec['error'] = str(exc)[-2000:]
        print('[' + rn + '] EXPORT FAILED: ' + rec['error'], flush=True)
        notify('export failed', rn + ': ' + rec['error'][-1500:], priority='high', tags=['warning'])
    finally:
        if not KEEP_MERGED:
            shutil.rmtree(merged, ignore_errors=True)
        rec_file = EDGE_DIR / 'exports' / (rn + '.json')
        rec_file.parent.mkdir(parents=True, exist_ok=True)
        rec_file.write_text(json.dumps(rec, indent=2, default=str), encoding='utf-8')
        if rec['status'] in ('ok', 'failed'):
            hub_upload_file(rec_file, EDGE_PREFIX + '/exports/' + rn + '.json', '[edge-eval] export record ' + rn)
    return rec


for e in EXP:
    rn = e['run_name']
    if rn not in NEEDED_QUANTS:
        continue
    if minutes_left() < EXPORT_MINUTES_ESTIMATE:
        print('[' + rn + '] Not exporting: ' + format(minutes_left(), '.0f') + ' min left before the reserve.')
        EXPORT_RECORDS[rn] = {'run_name': rn, 'model': e['model'], 'status': 'skipped_for_time', 'quants': {}}
        continue
    print(LF + '#' * 72 + LF + '# Export ' + rn + ': ' + ', '.join(sorted(NEEDED_QUANTS[rn], key=_qkey))
          + LF + '#' * 72, flush=True)
    export_run(e, NEEDED_QUANTS[rn])

print()
for rn, rec in EXPORT_RECORDS.items():
    print(rn.ljust(30) + str(rec.get('status')).ljust(18) + ', '.join(sorted(rec.get('quants', {}), key=_qkey)))
