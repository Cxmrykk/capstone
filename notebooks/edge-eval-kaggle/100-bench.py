# 9. Prompt helpers, llama-server client, throughput (llama-bench) and memory footprint
import re

import requests
from src.data.dataset import fit_prompt_to_budget
from src.data.prompts import extract_cypher, strip_leading_bos, template_supports_system
from src.data.schema_filter import apply_schema_mode
from src.inference.predict import prompt_reserve_tokens
from src.training.model_loader import load_tokenizer

# --------------------------------------------------------------------------- #
# Prompts: identical construction to app.py predict (history/030.md)
# --------------------------------------------------------------------------- #
_TOKENIZERS = {}


def edge_config(config_path, prompt_mode):
    cfg = load_config(config_path)
    cfg.data.schema_mode = prompt_mode
    return cfg


def get_tokenizer(cfg):
    key = cfg.model.key
    if key not in _TOKENIZERS:
        _TOKENIZERS[key] = load_tokenizer(cfg, padding_side='left')
    return _TOKENIZERS[key]


def build_prompts(cfg, records):
    tok = get_tokenizer(cfg)
    supports_system = cfg.model.spec.supports_system_role
    if supports_system is None:
        supports_system = template_supports_system(tok)
    reserve = prompt_reserve_tokens(cfg)
    fitted = []
    for rec in records:
        question = rec.get('question') or ''
        schema_text = apply_schema_mode(
            rec.get('schema') or '', question, cfg.data.schema_mode,
            min_nodes=cfg.data.filter_min_nodes,
            keep_patterns_for_kept_nodes=cfg.data.filter_keep_all_patterns_for_kept_nodes,
            similarity_threshold=cfg.data.similarity_threshold,
            similarity_model=cfg.data.similarity_model,
        )
        fitted.append(fit_prompt_to_budget(tok, question, schema_text, cfg, supports_system, reserve_tokens=reserve))
    return tok, fitted, reserve


def stop_strings(cfg, tok):
    stop = [] if EDGE_STOP_MODE == 'hf' else list(cfg.generation.stop_strings)
    eos = getattr(tok, 'eos_token', None)
    if eos and eos not in stop:
        stop.append(eos)
    return stop


# --------------------------------------------------------------------------- #
# llama-server options supported by THIS build (history/036.md)
# The pinned llama.cpp build rejected --no-mmap ("error: invalid argument"), which
# failed every job and every footprint measurement. Options are now taken from the
# build's own --help; optional ones are dropped if absent, and an option the server
# still rejects at start-up is removed and the start retried.
# --------------------------------------------------------------------------- #
def _server_help():
    try:
        proc = subprocess.run([str(LLAMA_BIN / 'llama-server'), '--help'], capture_output=True, text=True,
                              timeout=120)
    except Exception as exc:
        raise RuntimeError('Could not run llama-server --help: ' + str(exc)) from exc
    text = (proc.stdout or '') + LF + (proc.stderr or '')
    if proc.returncode != 0 and '--port' not in text:
        raise RuntimeError('llama-server --help exited ' + str(proc.returncode) + ':' + LF + text[-1500:])
    return text


SERVER_HELP = _server_help()
(LOG_DIR / 'llama-server-help.txt').write_text(SERVER_HELP, encoding='utf-8')
SERVER_FLAGS = set(re.findall(r'(?<![\w-])(--?[A-Za-z][\w-]*)', SERVER_HELP))
SERVER_REJECTED = set()


def _first_supported(*flags):
    return next((f for f in flags if f in SERVER_FLAGS and f not in SERVER_REJECTED), None)


def _optional(flags, *values):
    flag = _first_supported(*flags)
    return ([flag] + [str(v) for v in values]) if flag else []


_missing = [g for g in (('-m', '--model'), ('-c', '--ctx-size'), ('-t', '--threads'), ('--host',), ('--port',))
            if not _first_supported(*g)]
if _missing:
    raise RuntimeError('This llama-server build does not list the required options ' + str(_missing)
                       + '. See ' + str(LOG_DIR / 'llama-server-help.txt'))
SERVER_MODEL_FLAG = _first_supported('-m', '--model')
SERVER_CTX_FLAG = _first_supported('-c', '--ctx-size')
SERVER_THREADS_FLAG = _first_supported('-t', '--threads')
SERVER_REQUIRED = {SERVER_MODEL_FLAG, SERVER_CTX_FLAG, SERVER_THREADS_FLAG, '--host', '--port'}

# Memory mapping: prefer --no-mmap (weights read at load, so load time and RSS are
# comparable across quantisations); otherwise the matching LLAMA_ARG_* environment
# variable named in --help; otherwise llama.cpp's default (mmap on), recorded as such.
_envs = sorted(set(re.findall(r'LLAMA_ARG_\w*MMAP\w*', SERVER_HELP)))
_neg = [v for v in _envs if 'NO_MMAP' in v]
_pos = [v for v in _envs if 'NO_MMAP' not in v]
MMAP_ENV_CANDIDATE = {_neg[0]: '1'} if _neg else ({_pos[0]: '0'} if _pos else {})
SERVER_MMAP = {
    'args': ['--no-mmap'] if '--no-mmap' in SERVER_FLAGS else [],
    'env': {} if '--no-mmap' in SERVER_FLAGS else dict(MMAP_ENV_CANDIDATE),
    'mode': None,
    'help_lines': [ln.strip() for ln in SERVER_HELP.splitlines() if 'mmap' in ln.lower()][:8],
}


def _refresh_mmap_mode():
    if SERVER_MMAP['args']:
        SERVER_MMAP['mode'] = 'off (' + ' '.join(SERVER_MMAP['args']) + ')'
    elif SERVER_MMAP['env']:
        k, v = next(iter(SERVER_MMAP['env'].items()))
        SERVER_MMAP['mode'] = 'off (env ' + k + '=' + v + ')'
    else:
        SERVER_MMAP['mode'] = 'on (llama.cpp default; no working option to disable it in this build)'


_refresh_mmap_mode()


def server_mmap_off():
    return bool(SERVER_MMAP['args'] or SERVER_MMAP['env'])


def server_cmd(gguf, ctx, threads, port):
    cmd = [str(LLAMA_BIN / 'llama-server'), SERVER_MODEL_FLAG, str(gguf), SERVER_CTX_FLAG, str(ctx),
           SERVER_THREADS_FLAG, str(threads)]
    cmd += _optional(('-tb', '--threads-batch'), threads)
    cmd += ['--host', '127.0.0.1', '--port', str(port)]
    cmd += _optional(('--parallel', '-np'), 1)
    cmd += [a for a in SERVER_MMAP['args'] if a not in SERVER_REJECTED]
    cmd += _optional(('--no-warmup',))
    return cmd


def server_cmd_template():
    return ' '.join(['llama-server'] + server_cmd('<gguf>', '<ctx>', '<threads>', '<port>')[1:])


class ServerExitedError(RuntimeError):
    """llama-server exited while loading (bad option, unsupported model, ...)."""


def _handle_start_failure(text):
    """Adjusts the server options after a failed start. Returns True if a retry makes sense."""
    rejected = re.findall(r'invalid argument:\s*(\S+)', text)
    if rejected:
        flag = rejected[-1]
        if flag in SERVER_REQUIRED or flag in SERVER_REJECTED:
            return False
        SERVER_REJECTED.add(flag)
        if flag in SERVER_MMAP['args']:
            SERVER_MMAP['args'] = []
            SERVER_MMAP['env'] = dict(MMAP_ENV_CANDIDATE)
        _refresh_mmap_mode()
        print('llama-server rejected ' + flag + '; retrying without it (memory mapping: ' + SERVER_MMAP['mode'] + ').',
              flush=True)
        return True
    if SERVER_MMAP['env']:
        print('llama-server failed with ' + str(SERVER_MMAP['env']) + '; retrying with the default (mmap on).',
              flush=True)
        SERVER_MMAP['env'] = {}
        MMAP_ENV_CANDIDATE.clear()
        _refresh_mmap_mode()
        return True
    return False


print('llama-server command : ' + server_cmd_template())
print('Memory mapping       : ' + SERVER_MMAP['mode'])
for _ln in SERVER_MMAP['help_lines']:
    print('  help: ' + _ln[:160])

# --------------------------------------------------------------------------- #
# llama-server client (records load time, memory and per-request timings)
# --------------------------------------------------------------------------- #
_PORT = {'next': 8790}


class LlamaServer:
    def __init__(self, gguf, threads, ctx, log_path):
        self.gguf, self.threads, self.ctx = str(gguf), int(threads), int(ctx)
        self.log_path = Path(log_path)
        self.port = _PORT['next']
        _PORT['next'] += 1
        self.proc, self._fh, self.url = None, None, None
        self.load_s, self.rss_after_load_mb = None, None
        self.cmd = None
        self.session = requests.Session()

    def _start_once(self, timeout_s, log_mode):
        env_extra = dict(SERVER_MMAP['env'])
        cmd = server_cmd(self.gguf, self.ctx, self.threads, self.port)
        self.cmd = cmd
        self._fh = open(self.log_path, log_mode, encoding='utf-8')
        self._fh.write('$ ' + ' '.join(cmd) + (('  # env ' + json.dumps(env_extra)) if env_extra else '') + LF)
        self._fh.flush()
        t0 = time.perf_counter()
        self.proc = subprocess.Popen(cmd, stdout=self._fh, stderr=subprocess.STDOUT, start_new_session=True,
                                     env={**os.environ, **env_extra})
        url = 'http://127.0.0.1:' + str(self.port)
        while time.perf_counter() - t0 < timeout_s:
            if self.proc.poll() is not None:
                self._close_log()
                tail = LF.join(self.log_path.read_text(encoding='utf-8', errors='replace').splitlines()[-20:])
                raise ServerExitedError('llama-server exited with code ' + str(self.proc.returncode)
                                        + ' while loading:' + LF + tail)
            try:
                r = self.session.get(url + '/health', timeout=5)
                if r.status_code == 200:
                    self.load_s = time.perf_counter() - t0
                    self.url = url
                    self.rss_after_load_mb = self.rss_mb('VmRSS')
                    return self
            except requests.RequestException:
                pass
            time.sleep(0.25)
        self.stop()
        raise RuntimeError('llama-server did not become healthy within ' + str(timeout_s) + ' s.')

    def start(self, timeout_s=900):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        mode = 'w'
        for attempt in range(4):
            try:
                return self._start_once(timeout_s, mode)
            except ServerExitedError as exc:
                mode = 'a'
                if attempt == 3 or not _handle_start_failure(str(exc)):
                    raise

    def rss_mb(self, field='VmRSS'):
        try:
            for line in Path('/proc/' + str(self.proc.pid) + '/status').read_text().splitlines():
                if line.startswith(field + ':'):
                    return round(int(line.split()[1]) / 1024, 1)
        except Exception:
            return None
        return None

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def complete(self, prompt, n_predict, stop, timeout_s=REQUEST_TIMEOUT_S):
        payload = {'prompt': prompt, 'n_predict': int(n_predict), 'temperature': 0.0, 'top_k': 1, 'top_p': 1.0,
                   'stop': stop, 'cache_prompt': CACHE_PROMPT, 'stream': False}
        r = self.session.post(self.url + '/completion', json=payload, timeout=timeout_s)
        r.raise_for_status()
        return r.json()

    def _close_log(self):
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=30)
            except Exception:
                try:
                    self.proc.kill()
                    self.proc.wait(timeout=10)
                except Exception:
                    pass
        self._close_log()


def stopped_at_limit(resp):
    return bool(resp.get('stopped_limit')) or resp.get('stop_type') == 'limit'


# --------------------------------------------------------------------------- #
# Smoke test: fail in seconds, not after hours of benchmarks, if the server cannot start
# --------------------------------------------------------------------------- #
def _smoke_test_server():
    cands = []
    for rec in EXPORT_RECORDS.values():
        for info in (rec.get('quants') or {}).values():
            p = Path(info.get('path') or '')
            if info.get('path') and p.exists():
                cands.append((p.stat().st_size, str(p)))
    if not cands:
        print('No local GGUF to smoke-test llama-server with; skipping.')
        return None
    _size, path = min(cands)
    srv = LlamaServer(path, LOGICAL_CPUS, 512, LOG_DIR / 'server-smoke.log')
    try:
        srv.start(timeout_s=600)
        resp = srv.complete('Hello', 1, [], timeout_s=300)
        return {'gguf': Path(path).name, 'load_s': round(srv.load_s, 2), 'response_ok': 'content' in resp,
                'command': ' '.join(srv.cmd[1:]), 'mmap': SERVER_MMAP['mode']}
    finally:
        srv.stop()


SERVER_SMOKE = None
if PENDING_JOBS or PENDING_BENCH:
    try:
        SERVER_SMOKE = _smoke_test_server()
    except Exception as exc:
        raise RuntimeError('llama-server cannot start with this build, so every CPU job and footprint measurement '
                           'would fail. See ' + str(LOG_DIR / 'server-smoke.log') + ' and '
                           + str(LOG_DIR / 'llama-server-help.txt') + '.' + LF + str(exc)[-1500:]) from exc
    print('llama-server smoke test: ' + str(SERVER_SMOKE))

# --------------------------------------------------------------------------- #
# llama-bench (synthetic ppN / tgN throughput) and footprint
# --------------------------------------------------------------------------- #
BENCH_THREADS = sorted({t for t in THREAD_SWEEP if 1 <= t <= LOGICAL_CPUS} | {LOGICAL_CPUS})
BENCH_KEYS = ('n_prompt', 'n_gen', 'n_threads', 'n_batch', 'n_ubatch', 'avg_ns', 'stddev_ns', 'avg_ts', 'stddev_ts',
              'model_size', 'model_n_params', 'model_type', 'cpu_info', 'build_commit', 'type_k', 'type_v',
              'test_time')


def run_llama_bench(gguf, threads, log_path):
    cmd = [str(LLAMA_BIN / 'llama-bench'), '-m', str(gguf), '-p', str(BENCH_PROMPT_TOKENS),
           '-n', str(BENCH_GEN_TOKENS), '-t', ','.join(str(t) for t in threads), '-r', str(BENCH_REPS), '-o', 'json']
    print('$ ' + ' '.join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=BENCH_TIMEOUT_S)
    Path(log_path).write_text('$ ' + ' '.join(cmd) + LF + proc.stderr + LF + '# stdout' + LF + proc.stdout,
                              encoding='utf-8')
    if proc.returncode != 0:
        raise RuntimeError('llama-bench exited ' + str(proc.returncode) + ': ' + proc.stderr[-800:])
    out = proc.stdout
    data = json.loads(out[out.find('['):out.rfind(']') + 1])
    return data, round(time.time() - t0, 1)


def choose_threads(rows, prompt_tokens, gen_tokens):
    by_t = {}
    for r in rows:
        if r.get('error') or r.get('quant') != ACCURACY_QUANT:
            continue
        t = int(r['n_threads'])
        if r.get('n_prompt') and not r.get('n_gen'):
            by_t.setdefault(t, {})['pp'] = r['avg_ts']
        elif r.get('n_gen') and not r.get('n_prompt'):
            by_t.setdefault(t, {})['tg'] = r['avg_ts']
    est = {t: prompt_tokens / v['pp'] + gen_tokens / v['tg'] for t, v in by_t.items() if v.get('pp') and v.get('tg')}
    if not est:
        return LOGICAL_CPUS, {}
    return min(est, key=est.get), est


def measure_footprint(gguf, threads, ctx, prompt, stop, log_path):
    srv = LlamaServer(gguf, threads, ctx, log_path)
    try:
        srv.start()
        t0 = time.perf_counter()
        resp = srv.complete(prompt, 32, stop)
        first_ms = (time.perf_counter() - t0) * 1000
        tm = resp.get('timings') or {}
        return {'threads': threads, 'load_s': round(srv.load_s, 2), 'rss_after_load_mb': srv.rss_after_load_mb,
                'rss_after_first_request_mb': srv.rss_mb('VmRSS'),
                'peak_rss_mb': srv.rss_mb('VmHWM'), 'first_request_ms': round(first_ms, 1),
                'first_prompt_tokens': resp.get('tokens_evaluated') or tm.get('prompt_n'),
                'first_prompt_ms': tm.get('prompt_ms'), 'first_predicted_n': tm.get('predicted_n'),
                'mmap_mode': SERVER_MMAP['mode'], 'server_command': ' '.join(srv.cmd[1:])}
    finally:
        srv.stop()


for model in PENDING_BENCH:
    e = BENCH_PLAN[model]
    rn = e['run_name']
    ex = EXPORT_RECORDS.get(rn) or {}
    if ex.get('status') != 'ok':
        BENCH_RESULTS[model] = {'model': model, 'run_name': rn, 'status': 'export_' + str(ex.get('status')),
                                'error': ex.get('error'), **_provenance()}
        print('[bench ' + model + '] skipped: export status ' + str(ex.get('status')))
        continue
    print(LF + '#' * 72 + LF + '# Benchmark ' + model + ' (' + rn + ')' + LF + '#' * 72, flush=True)
    cfg = edge_config(e['config'], e['schema_mode'])
    tok, fitted, _reserve = build_prompts(cfg, EDGE_RECORDS)
    lens = sorted(len(f.prompt_ids) for f in fitted)
    typical_prompt = lens[len(lens) // 2]
    median_prompt = strip_leading_bos(tok, fitted[sorted(range(len(fitted)),
                                                         key=lambda i: len(fitted[i].prompt_ids))[len(fitted) // 2]].prompt)
    ctx = cfg.data.max_seq_length + cfg.generation.max_new_tokens + 64
    stop = stop_strings(cfg, tok)

    reused = REUSED_BENCH_ROWS.get(model)
    if reused is not None:
        bench_rows = [dict(r) for r in reused.get('bench_rows') or []]
        print('[bench ' + model + '] llama-bench rows reused from session ' + str(reused.get('session'))
              + '; measuring the llama-server footprint only.')
    else:
        bench_rows = []
        for q in sorted(set(LATENCY_QUANTS) | {ACCURACY_QUANT}, key=_qkey):
            info = ex['quants'].get(q)
            if not info or not Path(info['path']).exists():
                continue
            if minutes_left() < 20:
                print('[bench ' + model + '] stopping: too little session time left.')
                break
            threads = BENCH_THREADS if q in THREAD_SWEEP_QUANTS else [LOGICAL_CPUS]
            try:
                data, secs = run_llama_bench(info['path'], threads,
                                             LOG_DIR / ('bench-' + model + '-' + q.lower() + '.log'))
            except Exception as exc:
                bench_rows.append({'model': model, 'run_name': rn, 'quant': q, 'error': str(exc)[-800:]})
                print('[bench ' + model + ' ' + q + '] FAILED: ' + str(exc)[-400:])
                continue
            for d in data:
                bench_rows.append({'model': model, 'run_name': rn, 'quant': q, 'gguf_size_bytes': info['size_bytes'],
                                   'bench_wall_s': secs, **{k: d.get(k) for k in BENCH_KEYS}})
            for d in data:
                kind = 'pp' if d.get('n_prompt') else 'tg'
                print('  ' + q.ljust(8) + kind + str(d.get('n_prompt') or d.get('n_gen')).ljust(5) + ' t='
                      + str(d.get('n_threads')).ljust(3) + format(d.get('avg_ts') or 0, '8.2f') + ' tok/s')

    chosen, estimates = choose_threads(bench_rows, typical_prompt, TYPICAL_GEN_TOKENS)
    if isinstance(SERVER_THREADS, int):
        chosen = SERVER_THREADS

    footprint = []
    for q in sorted(LATENCY_QUANTS, key=_qkey):
        info = ex['quants'].get(q)
        if not info or not Path(info['path']).exists() or minutes_left() < 10:
            continue
        try:
            fp = measure_footprint(info['path'], chosen, ctx, median_prompt, stop,
                                   LOG_DIR / ('server-footprint-' + model + '-' + q.lower() + '.log'))
            fp.update(model=model, run_name=rn, quant=q, gguf_size_bytes=info['size_bytes'])
            footprint.append(fp)
            print('  footprint ' + q.ljust(8) + 'load ' + format(fp['load_s'], '.1f') + ' s, peak RSS '
                  + str(fp['peak_rss_mb']) + ' MB, first request ' + format(fp['first_request_ms'] / 1000, '.1f') + ' s')
        except Exception as exc:
            footprint.append({'model': model, 'run_name': rn, 'quant': q, 'error': str(exc)[-800:]})
            print('  footprint ' + q.ljust(8) + 'FAILED: ' + str(exc)[-400:])

    rec = {'model': model, 'run_name': rn, 'status': 'completed', 'bench_rows': bench_rows, 'footprint': footprint,
           'bench_rows_session': reused.get('session') if reused is not None else SESSION,
           'bench_threads': BENCH_THREADS, 'threads_chosen': chosen,
           'thread_estimates_s': {str(k): round(v, 2) for k, v in estimates.items()},
           'est_s_per_item': round(estimates.get(chosen, 0.0), 2) or None,
           'typical_prompt_tokens': typical_prompt, 'typical_gen_tokens': TYPICAL_GEN_TOKENS,
           'bench_prompt_tokens': BENCH_PROMPT_TOKENS, 'bench_gen_tokens': BENCH_GEN_TOKENS,
           'bench_reps': BENCH_REPS, 'server_ctx': ctx, 'server_command': server_cmd_template(),
           **_provenance()}
    bdir = EDGE_DIR / 'bench' / model
    bdir.mkdir(parents=True, exist_ok=True)
    (bdir / 'bench.json').write_text(json.dumps(rec, indent=2, default=str), encoding='utf-8')
    hub_upload_folder(bdir, EDGE_PREFIX + '/bench/' + model, '[edge-eval] bench ' + model)
    BENCH_RESULTS[model] = rec
    print('[bench ' + model + '] threads chosen for serving: ' + str(chosen) + ' (estimates ' + str(rec['thread_estimates_s']) + ')')

THREADS_FOR = {}
for model in BENCH_PLAN:
    if isinstance(SERVER_THREADS, int):
        THREADS_FOR[model] = SERVER_THREADS
    else:
        THREADS_FOR[model] = int((BENCH_RESULTS.get(model) or {}).get('threads_chosen') or LOGICAL_CPUS)
print('Serving threads: ' + str(THREADS_FOR))
