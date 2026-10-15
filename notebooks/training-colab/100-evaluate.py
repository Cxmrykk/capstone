# @title 9. Evaluation (GLEU, exact match, execution accuracy)
eval_flags = "--execute --validate-syntax" if RUN_EXECUTION_EVAL else ""
run_step("evaluate",
         f"python app.py evaluate --predictions {PRED_PATH} {eval_flags} --out {REPORT_PATH}")

pred = Path(PRED_PATH)
report = Path(REPORT_PATH)
archive_results([
    pred.with_name(pred.stem + ".scored.jsonl"),
    report,
    report.with_suffix(".json"),
])
