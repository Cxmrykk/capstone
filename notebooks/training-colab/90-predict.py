# @title 8. Inference on the test split
run_step("predict",
         f"python app.py predict --config {CONFIG} --adapter {ADAPTER_DIR} --split test "
         f"--limit {PREDICT_LIMIT} --out {PRED_PATH} --tag {RUN_NAME}")

pred = Path(PRED_PATH)
archive_results([pred, pred.with_suffix(".meta.json")])
