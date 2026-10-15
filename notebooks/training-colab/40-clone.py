# @title 3. Clone or update the repository
if (REPO_DIR / ".git").exists():
    run_step("git-pull", f"git -C {REPO_DIR} pull --ff-only", allow_fail=True, cwd="/content")
else:
    run_step("git-clone", f"git clone {GITHUB_REPO} {REPO_DIR}", cwd="/content")
os.chdir(REPO_DIR)

import yaml

RUN_NAME = yaml.safe_load(Path(CONFIG).read_text(encoding="utf-8"))["run_name"]
STATUS["run_name"] = RUN_NAME
save_status()

ADAPTER_DIR = f"artifacts/runs/{RUN_NAME}/final"
PRED_PATH = f"artifacts/predictions/{RUN_NAME}-test.jsonl"
REPORT_PATH = f"artifacts/reports/{RUN_NAME}-eval.md"
MERGED_DIR = f"artifacts/merged/{RUN_NAME}"

print(f"Run name    : {RUN_NAME}")
print(f"Adapter     : {ADAPTER_DIR}")
print(f"Predictions : {PRED_PATH}")
print(f"Report      : {REPORT_PATH}")
