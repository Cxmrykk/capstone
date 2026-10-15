# 7. Run the lanes in parallel: train -> predict -> evaluate -> export, one GPU each
import traceback

# Merge + GGUF run one lane at a time: merged fp16 weights and f16 GGUF
# intermediates are large, and two lanes building llama.cpp in the same
# checkout would collide.
_export_lock = threading.Lock()
LLAMA_QUANTIZE = REPO_DIR / "vendor" / "llama.cpp" / "build" / "bin" / "llama-quantize"


def _lane_step(lane, name, cmd, allow_fail=False, env=None):
    lane_env = {"CUDA_VISIBLE_DEVICES": lane["gpu"], **(env or {})}
    return run_step(name, cmd, lane=lane["key"], allow_fail=allow_fail,
                    env=lane_env, cwd=str(lane["workdir"]), echo=False)


def _report_lines(lane):
    report_json = Path(lane["report_path"]).with_suffix(".json")
    if not report_json.exists():
        return []
    try:
        overall = json.loads(report_json.read_text(encoding="utf-8"))[0].get("overall", {})
    except Exception as exc:
        return [f"(could not read evaluation report: {exc})"]
    out = []
    for key, label in (("n", "N"), ("google_bleu_corpus", "GLEU"),
                       ("exact_match", "Exact match"),
                       ("normalised_exact_match", "Norm. EM"),
                       ("execution_accuracy", "Exec. acc.")):
        if key in overall:
            out.append(f"{label}: {overall[key]}")
    return out


def _train(lane, st):
    """Returns True when a final adapter is available, False when the lane halted."""
    key = lane["key"]
    marker = lane["adapter_dir"] / "adapter_config.json"
    if SKIP_TRAINING_IF_FINAL_EXISTS and not marker.exists():
        # Exit 1 = no final adapter on the Hub yet. Exit 6 = Hub error: refuse to
        # train, otherwise a finished run would be retrained (history/030.md).
        rc = _lane_step(lane, "pull-final",
                        f"{APP} checkpoint pull --config {lane['config_path']} --step final",
                        allow_fail=True)
        if rc not in (0, 1):
            raise RuntimeError(f"[{key}] Could not check the Hub for a final adapter (exit {rc}). "
                               "Refusing to train; see the pull-final log.")
    if SKIP_TRAINING_IF_FINAL_EXISTS and marker.exists():
        print(f"[{key}] Final adapter found; skipping training.", flush=True)
        return True

    elapsed_min = (time.time() - PIPELINE_T0) / 60
    available_min = SESSION_BUDGET_MINUTES - elapsed_min - RESERVE_MINUTES
    train_limit = int(min(TIME_LIMIT_MINUTES, available_min))
    st["train_limit"] = train_limit
    print(f"[{key}] Session time used: {elapsed_min:.0f} min. Training limit: {train_limit} min.",
          flush=True)

    if train_limit < 15:
        st["outcome"] = "no-time"
        save_status()
        notify(_title(key, "not enough session time"),
               f"Only {available_min:.0f} min left for training in session {SESSION}. "
               "Commit a new version to continue.", priority="high", tags=["hourglass"])
        return False

    stop_at = (datetime.datetime.now(datetime.timezone.utc)
               + datetime.timedelta(minutes=train_limit))
    notify(_title(key, "training started"),
           f"Run {lane['run_name']} on GPU {lane['gpu']}, session {SESSION}, limit {train_limit} min. "
           f"The training clock starts after model loading and tokenization, so if the schedule "
           f"does not finish first, expect a 'paused at time limit' alert roughly 10-30 min after "
           f"{stop_at:%H:%M} UTC. Silence well past that means the session was killed.",
           priority="low", tags=["hourglass_flowing_sand"])
    try:
        _lane_step(lane, "train", f"{APP} train --config {lane['config_path']} --resume auto "
                                  f"--time-limit {train_limit}",
                   env={"T2C_SESSION": SESSION})  # Hub telemetry folder = this session
    except RuntimeError as exc:
        if getattr(exc, "halt", None) == "paused":
            print(f"[{key}] {exc}", flush=True)
            return False
        if getattr(exc, "exit_code", None) == 5:
            rc = _lane_step(lane, "push-final",
                            f"{APP} checkpoint push-final --config {lane['config_path']}",
                            allow_fail=True)
            if rc == 0:
                notify(_title(key, "final adapter uploaded on retry"),
                       f"push-final succeeded after the train step exited 5. Session {SESSION}.",
                       tags=["white_check_mark"])
                return True
        raise
    finally:
        archive_results(key, [
            lane["run_dir"] / "metrics.jsonl",
            lane["run_dir"] / "run_config.json",
            lane["run_dir"] / "final" / "run_metadata.json",
        ])
    return True


def _predict_and_evaluate(lane):
    key = lane["key"]
    pred = lane["pred_path"]
    report = lane["report_path"]
    _lane_step(lane, "predict",
               f"{APP} predict --config {lane['config_path']} --adapter {lane['adapter_dir']} "
               f"--split test --limit {PREDICT_LIMIT} --out {pred} --tag {lane['run_name']}")
    archive_results(key, [pred, pred.with_suffix(".meta.json")])

    eval_flags = "--execute --validate-syntax" if RUN_EXECUTION_EVAL else ""
    _lane_step(lane, "evaluate",
               f"{APP} evaluate --predictions {pred} {eval_flags} --out {report}")
    archive_results(key, [
        pred.with_name(pred.stem + ".scored.jsonl"),
        report,
        report.with_suffix(".json"),
    ])


def _export(lane, st):
    """Best-effort merge + GGUF conversion. Never raises for tool failures."""
    key = lane["key"]
    if not RUN_EXPORT:
        st["export"] = "disabled"
        return
    print(f"[{key}] Waiting for the export slot...", flush=True)
    with _export_lock:
        if STOP_EVENT.is_set():
            return
        merged = lane["merged_dir"]
        ok = _lane_step(lane, "merge", f"{APP} export merge --config {lane['config_path']} "
                                       f"--adapter {lane['adapter_dir']} --out {merged}",
                        allow_fail=True) == 0
        if ok and not LLAMA_QUANTIZE.exists():
            # A CPU build is enough for convert + quantize and compiles far faster than CUDA.
            ok = _lane_step(lane, "build-llama-cpp", f"bash {REPO_DIR / 'scripts' / 'build_llama_cpp.sh'}",
                            allow_fail=True, env={"BUILD_CUDA": "0"}) == 0
        if ok:
            ok = _lane_step(lane, "gguf", f"{APP} export gguf --merged {merged} --quant Q4_K_M",
                            allow_fail=True) == 0
        if ok:
            for gguf in sorted((ARTIFACTS / "gguf").glob(f"{lane['run_name']}-*.gguf*")):
                if gguf.suffix == ".json":
                    archive_results(key, [gguf])
                elif UPLOAD_GGUF:
                    # The GGUF stays in scratch storage (not saved output); the Hub is its only copy.
                    print(f"[{key}] Uploading {gguf.name} ({gguf.stat().st_size / 1e9:.2f} GB)...",
                          flush=True)
                    hub_upload(gguf, f"gguf/{lane['run_name']}/{gguf.name}", f"[{key}] {gguf.name}")
        st["export"] = "ok" if ok else "failed"
        if not KEEP_MERGED and merged.exists():
            shutil.rmtree(merged, ignore_errors=True)
            print(f"[{key}] Removed merged weights to free disk space.", flush=True)
        save_status()


def lane_pipeline(lane):
    key = lane["key"]
    st = STATUS["lanes"][key]
    try:
        delay = lane.get("stagger_s", 0)
        while delay > 0 and not STOP_EVENT.is_set():
            time.sleep(min(5, delay))
            delay -= 5
        if STOP_EVENT.is_set():
            raise KeyboardInterrupt("Stop requested before the lane started.")

        if _train(lane, st):
            _predict_and_evaluate(lane)
            _export(lane, st)
            st["outcome"] = "completed"
            save_status()
            lines = [f"Run {lane['run_name']} finished in session {SESSION}."] + _report_lines(lane)
            if st.get("export") == "failed":
                lines.append("GGUF export failed (best-effort); see the lane logs.")
            notify(_title(key, "lane finished"), LF.join(lines),
                   tags=["warning"] if st.get("export") == "failed" else ["white_check_mark"],
                   click=(f"https://huggingface.co/{HUB_REPO}/tree/main/results/{key}/{SESSION}"
                          if UPLOAD_RESULTS_TO_HUB and HUB_REPO else None))
    except KeyboardInterrupt:
        st["outcome"] = st.get("outcome") or "interrupted"
    except Exception as exc:
        st["outcome"] = "failed"
        tb_text = traceback.format_exc()
        print(f"[{key}] Lane failed:" + LF + tb_text, flush=True)
        if not getattr(exc, "ntfy_sent", False):
            notify(_title(key, "lane error"),
                   f"{type(exc).__name__}: {exc}" + LF + f"Session {SESSION}." + LF + tb_text[-1500:],
                   priority="urgent", tags=["rotating_light"])
    finally:
        save_status()
        publish_results(key)


def _gpu_usage():
    """Returns {index: 'used/total GiB, util%'} from nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        ).stdout
    except Exception:
        return {}
    usage = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4:
            idx, used, total, util = parts
            try:
                usage[idx] = f"{float(used) / 1024:.1f}/{float(total) / 1024:.1f} GiB {util}%"
            except ValueError:
                pass
    return usage


def _ram_available_gb():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 ** 2
    except Exception:
        pass
    return float("nan")


def heartbeat():
    usage = _gpu_usage()
    now = datetime.datetime.now(datetime.timezone.utc)
    elapsed = (time.time() - PIPELINE_T0) / 60
    lines = [f"--- {now:%H:%M:%S} UTC | session {elapsed:.0f}/{SESSION_BUDGET_MINUTES} min | "
             f"RAM available {_ram_available_gb():.1f} GB | disk free {_free_gb(WORK_ROOT):.0f} GB"]
    for lane in LANE_SPECS:
        st = STATUS["lanes"][lane["key"]]
        step = st.get("current_step")
        if step:
            started = st.get("step_started") or time.time()
            state = f"{step} {(time.time() - started) / 60:.0f} min"
        else:
            state = st.get("outcome") or "waiting"
        last = (st.get("last_line") or "")[-110:]
        lines.append(f"  [{lane['key']}] GPU{lane['gpu']} {usage.get(lane['gpu'], '?')} | {state} | {last}")
    print(LF.join(lines), flush=True)


threads = []
for i, lane in enumerate(LANE_SPECS):
    lane["stagger_s"] = i * LANE_STAGGER_S
    thread = threading.Thread(target=lane_pipeline, args=(lane,), name=f"lane-{lane['key']}", daemon=True)
    thread.start()
    threads.append(thread)

try:
    last_beat = time.time()
    while any(t.is_alive() for t in threads):
        time.sleep(5)
        if time.time() - last_beat >= HEARTBEAT_S:
            heartbeat()
            last_beat = time.time()
except KeyboardInterrupt:
    print("Interrupt received: stopping all lanes (training writes emergency checkpoints)...",
          flush=True)
    STOP_EVENT.set()
    _stop_all_procs()
    for t in threads:
        t.join(timeout=300)
    STATUS["outcome"] = "interrupted"
    save_status()
    notify(_title(INSTANCE_LABEL, "interrupted"),
           f"Session {SESSION} was stopped by the user.", priority="high", tags=["stop_sign"])
    raise

heartbeat()
