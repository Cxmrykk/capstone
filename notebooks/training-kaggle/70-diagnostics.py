# 6. Diagnostics and data previews (shared, sequential)
run_step("doctor", f"{APP} doctor", allow_fail=True)
for lane in LANE_SPECS:
    run_step(f"data-preview-{lane['key']}",
             f"{APP} data preview --config {lane['config_path']} --index 5", allow_fail=True)
    run_step(f"token-stats-{lane['key']}",
             f"{APP} data token-stats --config {lane['config_path']} --samples 500", allow_fail=True)
