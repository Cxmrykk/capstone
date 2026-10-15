# 8. Summary
from IPython.display import Markdown, display

print(f"Session {SESSION}: instance {INSTANCE_LABEL} ({RUN_TYPE})")
print(f"Total session time: {(time.time() - PIPELINE_T0) / 60:.0f} min")
print(f"Local logs: {LOG_DIR}")

summary_lines = [f"Session {SESSION} finished."]
any_failed = False
for lane in LANE_SPECS:
    st = STATUS["lanes"][lane["key"]]
    outcome = st.get("outcome") or "unknown"
    any_failed = any_failed or outcome in ("failed", "unknown")
    print()
    print(f"[{lane['key']}] run_name={lane['run_name']} GPU {lane['gpu']} outcome={outcome} "
          f"export={st.get('export') or '-'}")
    for step in st["steps"]:
        print(f"  [{step['index']:02d}] {step['name']:<16} exit={step['exit_code']:<4} "
              f"{step['elapsed_s'] / 60:7.1f} min")
    metrics = _report_lines(lane)
    for line in metrics:
        print(f"  {line}")
    summary_lines.append(f"{lane['key']}: {outcome}" + (f" ({'; '.join(metrics)})" if metrics else ""))
    if UPLOAD_RESULTS_TO_HUB and HUB_REPO:
        print(f"  Results: https://huggingface.co/{HUB_REPO}/tree/main/results/{lane['key']}/{SESSION}")

STATUS["outcome"] = "completed"
save_status()
# Per-lane failures and pauses were already alerted; this is the end-of-session digest.
notify(_title(INSTANCE_LABEL, "pipeline finished"), LF.join(summary_lines),
       priority="high" if any_failed else "default",
       tags=["warning"] if any_failed else ["white_check_mark"])

for lane in LANE_SPECS:
    if Path(lane["report_path"]).exists():
        display(Markdown(Path(lane["report_path"]).read_text(encoding="utf-8")))
