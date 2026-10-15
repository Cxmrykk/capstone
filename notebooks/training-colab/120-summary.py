# @title 11. Summary
from IPython.display import Markdown, display

print(f"Session {SESSION}: {RUN_KEY} (run_name={RUN_NAME})")
for step in STATUS["steps"]:
    print(f"  [{step['index']:02d}] {step['name']:<16} exit={step['exit_code']:<4} "
          f"{step['elapsed_s'] / 60:7.1f} min")
print(f"Local logs: {LOG_DIR}")
if HUB_REPO:
    print(f"Hub logs  : https://huggingface.co/{HUB_REPO}/tree/main/{HUB_LOG_PREFIX}")
    print(f"Results   : https://huggingface.co/{HUB_REPO}/tree/main/{HUB_RESULTS_PREFIX}")

summary_lines = [f"Session {SESSION} finished ({RUN_NAME})."]
report_json = Path(REPORT_PATH).with_suffix(".json")
if report_json.exists():
    try:
        overall = json.loads(report_json.read_text(encoding="utf-8"))[0].get("overall", {})
        for key, label in (("n", "N"), ("google_bleu_corpus", "GLEU"),
                           ("exact_match", "Exact match"),
                           ("normalised_exact_match", "Norm. EM"),
                           ("execution_accuracy", "Exec. acc.")):
            if key in overall:
                summary_lines.append(f"{label}: {overall[key]}")
    except Exception as exc:
        summary_lines.append(f"(could not read evaluation report: {exc})")
# pull-final exiting 1 only means no adapter was on the Hub yet, so it is not listed.
soft_failures = [s["name"] for s in STATUS["steps"]
                 if s["exit_code"] != 0 and s["name"] != "pull-final"]
if soft_failures:
    summary_lines.append(f"Best-effort steps with errors: {', '.join(soft_failures)}")

STATUS["outcome"] = "completed"
save_status()
notify(f"{NTFY_PREFIX}: pipeline finished", LF.join(summary_lines),
       tags=["warning"] if soft_failures else ["white_check_mark"],
       click=f"https://huggingface.co/{HUB_REPO}/tree/main/{HUB_RESULTS_PREFIX}" if HUB_REPO else None)

if Path(REPORT_PATH).exists():
    display(Markdown(Path(REPORT_PATH).read_text(encoding="utf-8")))
