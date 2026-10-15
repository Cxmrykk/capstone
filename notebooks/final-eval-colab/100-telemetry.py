# 9. Training telemetry from the Hub (merged across every Colab/Kaggle session of each run)
import pandas as pd
from IPython.display import display


def _num(df, col):
    if df is None or df.empty or col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors='coerce').dropna()


files = hub_files(refresh=True)
curve_rows = []
for e in EXP:
    rn = e['run_name']
    key = e['model'] + '-' + e['variant']
    paths = sorted(f for f in files if f.startswith('results/' + key + '/') and f.endswith('/metrics.jsonl'))
    print(rn + ': ' + str(len(paths)) + ' session metric file(s) on the Hub')
    for path in paths:
        session = path.split('/')[2]
        try:
            text = hub_download_file(path).read_text(encoding='utf-8')
        except Exception as exc:
            print('  could not read ' + path + ': ' + str(exc))
            continue
        for line in text.splitlines():
            if line.strip():
                row = json.loads(line)
                row.update(run_name=rn, model=e['model'], variant=e['variant'], session=session)
                curve_rows.append(row)

RAW_CURVES = pd.DataFrame(curve_rows)
CURVES = RAW_CURVES.copy()
if not CURVES.empty:
    CURVES['kind'] = 'other'
    for col, kind in (('loss', 'train'), ('eval_loss', 'eval'), ('train_runtime', 'summary')):
        if col in CURVES.columns:
            CURVES.loc[CURVES[col].notna(), 'kind'] = kind
    # A resumed session repeats the steps after its checkpoint; keep the latest session's value.
    CURVES = (CURVES.sort_values(['run_name', 'wall_time'])
              .drop_duplicates(['run_name', 'kind', 'step'], keep='last')
              .sort_values(['run_name', 'step']).reset_index(drop=True))

train_rows = []
for e in EXP:
    rn = e['run_name']
    tm = e['train_meta'] or {}
    c = CURVES[CURVES['run_name'] == rn] if not CURVES.empty else pd.DataFrame()
    raw = RAW_CURVES[RAW_CURVES['run_name'] == rn] if not RAW_CURVES.empty else pd.DataFrame()
    tr = c[c['kind'] == 'train'] if not c.empty else c
    ev = c[c['kind'] == 'eval'] if not c.empty else c
    hours, n_sessions = float('nan'), 0
    if not raw.empty and 'wall_time' in raw.columns:
        spans = raw.groupby('session')['wall_time'].agg(['min', 'max'])
        hours = float((spans['max'] - spans['min']).sum()) / 3600
        n_sessions = len(spans)
    tr_loss = _num(tr, 'loss')
    ev_loss = _num(ev, 'eval_loss')
    metrics = tm.get('metrics') or {}
    train_rows.append({
        'run_name': rn, 'model': e['model'], 'variant': e['variant'], 'schema_mode': e['schema_mode'],
        'adapter_available': e['adapter_ok'],
        'global_step': tm.get('global_step'), 'epoch': tm.get('epoch'),
        'n_train': tm.get('n_train'), 'n_val': tm.get('n_val'),
        'trainable_parameters': tm.get('trainable_parameters'),
        'target_modules': json.dumps(tm.get('target_modules'), default=str),
        'dtype': tm.get('dtype'), 'device': tm.get('device'),
        'target_only_loss': tm.get('target_only_loss'), 'unsloth': tm.get('unsloth'),
        'train_sessions_logged': n_sessions,
        'logged_training_hours': round(hours, 2) if hours == hours else None,
        'first_train_loss': float(tr_loss.iloc[0]) if len(tr_loss) else None,
        'final_train_loss_mean_last5': float(tr_loss.tail(5).mean()) if len(tr_loss) else None,
        'final_eval_loss': float(ev_loss.iloc[-1]) if len(ev_loss) else None,
        'min_eval_loss': float(ev_loss.min()) if len(ev_loss) else None,
        'vram_peak_gb_max': float(_num(c, 'vram_peak_gb').max()) if len(_num(c, 'vram_peak_gb')) else tm.get('vram_peak_gb'),
        'trainer_train_loss_final_session': metrics.get('train_loss'),
        'fingerprint_trained': e['fingerprint_trained'], 'fingerprint_now': e['fingerprint_now'],
        'fingerprint_match': e['fingerprint_match'],
    })
TRAINING = pd.DataFrame(train_rows)
display(TRAINING)
