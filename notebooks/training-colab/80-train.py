# @title 7. LoRA fine-tuning (resumes from the Hub automatically)
adapter_marker = Path(ADAPTER_DIR) / "adapter_config.json"

if SKIP_TRAINING_IF_FINAL_EXISTS and not adapter_marker.exists():
    # Exit code 1 simply means no finished adapter exists on the Hub yet.
    run_step("pull-final", f"python app.py checkpoint pull --config {CONFIG} --step final",
             allow_fail=True)

if SKIP_TRAINING_IF_FINAL_EXISTS and adapter_marker.exists():
    print(f"Final adapter found at {ADAPTER_DIR}; skipping training.")
else:
    stop_at = (datetime.datetime.now(datetime.timezone.utc)
               + datetime.timedelta(minutes=TIME_LIMIT_MINUTES))
    notify(f"{NTFY_PREFIX}: training started",
           f"Run {RUN_NAME}, session {SESSION}. If the schedule does not finish first, expect a "
           f"'paused at time limit' alert shortly after {stop_at:%H:%M} UTC (plus checkpoint "
           f"upload time). Silence well past that means the runtime was reclaimed.",
           priority="low", tags=["hourglass_flowing_sand"])
    try:
        run_step("train", f"python app.py train --config {CONFIG} --resume auto "
                          f"--time-limit {TIME_LIMIT_MINUTES}")
    finally:
        run_dir = Path(f"artifacts/runs/{RUN_NAME}")
        archive_results([
            run_dir / "metrics.jsonl",
            run_dir / "run_config.json",
            run_dir / "final" / "run_metadata.json",
        ])
