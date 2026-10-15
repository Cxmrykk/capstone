# @title 6. Diagnostics and data preview
run_step("doctor", "python app.py doctor", allow_fail=True)
run_step("data-preview", f"python app.py data preview --config {CONFIG} --index 5", allow_fail=True)
run_step("token-stats", f"python app.py data token-stats --config {CONFIG} --samples 500", allow_fail=True)
