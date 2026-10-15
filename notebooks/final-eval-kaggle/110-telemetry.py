# 9. Training telemetry: loss curves from the newest checkpoint's trainer_state.json
#    log_history on the Hub (survives resumes and killed sessions); metrics.jsonl only
#    fills steps after the last checkpoint and supplies VRAM / wall time.
import re
import numpy as np
import pandas as pd
from IPython.display import display
from safetensors import safe_open


def _num(df, col):
    if df is None or df.empty or col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors='coerce').dropna()


def _adapter_modules(adir):
    """Counts adapted modules from the saved tensors; fixes the Unsloth leaf-name report."""
    st = Path(adir) / 'adapter_model.safetensors'
    if not st.exists():
        return None, None
    with safe_open(str(st), framework='pt') as f:
        mods = {k.split('.lora_A')[0] for k in f.keys() if '.lora_A' in k}
    return len(mods), sorted({m.split('.')[-1] for m in mods})


files = hub_files(refresh=True)
CKPT_STATE_RE = re.compile(r'^runs/([^/]+)/checkpoint-(\d+)/trainer_state\.json$')
curve_rows, jsonl_rows, state_steps = [], [], {}
for e in EXP:
    rn, key = e['run_name'], e['model'] + '-' + e['variant']
    steps = sorted(int(m.group(2)) for f in files for m in [CKPT_STATE_RE.match(f)] if m and m.group(1) == rn)
    if steps:
        state_steps[rn] = steps[-1]
        state = json.loads(hub_download_file('runs/' + rn + '/checkpoint-' + str(steps[-1])
                                             + '/trainer_state.json').read_text(encoding='utf-8'))
        for entry in state.get('log_history') or []:
            curve_rows.append({**entry, 'run_name': rn, 'model': e['model'], 'variant': e['variant'],
                               'session': None, 'source': 'trainer_state'})
    paths = sorted(f for f in files if f.endswith('/metrics.jsonl')
                   and (f.startswith('results/' + key + '/') or f.startswith('runs/' + rn + '/telemetry/')))
    print(rn + ': trainer_state at step ' + str(state_steps.get(rn, '-')) + ', ' + str(len(paths))
          + ' metrics.jsonl file(s)')
    for path in paths:
        session = path.split('/')[-2]
        try:
            text = hub_download_file(path).read_text(encoding='utf-8')
        except Exception as exc:
            print('  could not read ' + path + ': ' + str(exc))
            continue
        for line in text.splitlines():
            if line.strip():
                row = json.loads(line)
                row.update(run_name=rn, model=e['model'], variant=e['variant'], session=session,
                           source='metrics_jsonl')
                jsonl_rows.append(row)

RAW_CURVES = pd.DataFrame(jsonl_rows)
CURVES = pd.DataFrame(curve_rows + jsonl_rows)
if not CURVES.empty:
    CURVES['kind'] = 'other'
    for col, kind in (('loss', 'train'), ('eval_loss', 'eval'), ('train_runtime', 'summary')):
        if col in CURVES.columns:
            CURVES.loc[CURVES[col].notna(), 'kind'] = kind
    wall = pd.to_numeric(CURVES['wall_time'], errors='coerce') if 'wall_time' in CURVES.columns \
        else pd.Series(np.nan, index=CURVES.index)
    # trainer_state wins; among metrics.jsonl rows the latest session wins.
    CURVES['_order'] = np.where(CURVES['source'] == 'trainer_state', -np.inf, -wall.fillna(0))
    CURVES = (CURVES.sort_values(['run_name', 'kind', 'step', '_order'])
              .drop_duplicates(['run_name', 'kind', 'step'], keep='first')
              .drop(columns='_order').sort_values(['run_name', 'step']).reset_index(drop=True))

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
    tr_loss, ev_loss = _num(tr, 'loss'), _num(ev, 'eval_loss')
    logging_steps = ((tm.get('config') or {}).get('train') or {}).get('logging_steps') or 10
    expected_logs = (tm.get('global_step') or 0) // logging_steps
    n_mod, leaves = _adapter_modules(e['adapter_dir'])
    metrics = tm.get('metrics') or {}
    train_rows.append({
        'run_name': rn, 'model': e['model'], 'variant': e['variant'], 'schema_mode': e['schema_mode'],
        'adapter_available': e['adapter_ok'],
        'global_step': tm.get('global_step'), 'epoch': tm.get('epoch'),
        'n_train': tm.get('n_train'), 'n_val': tm.get('n_val'),
        'trainable_parameters': tm.get('trainable_parameters'),
        'adapted_modules': n_mod, 'adapted_leaf_names': '; '.join(leaves or []),
        'target_modules_reported': json.dumps(tm.get('target_modules'), default=str),
        'dtype': tm.get('dtype'), 'device': tm.get('device'),
        'target_only_loss': tm.get('target_only_loss'), 'unsloth': tm.get('unsloth'),
        'curve_source_step': state_steps.get(rn),
        'train_loss_points': int(len(tr_loss)),
        'train_loss_coverage': round(len(tr_loss) / expected_logs, 3) if expected_logs else None,
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
