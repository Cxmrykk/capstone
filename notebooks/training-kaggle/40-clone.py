# 3. Clone or update the repository and resolve per-lane paths
if (REPO_DIR / ".git").exists():
    run_step("git-pull", f"git -C {REPO_DIR} pull --ff-only", allow_fail=True, cwd=str(WORK_ROOT))
else:
    run_step("git-clone", f"git clone {GITHUB_REPO} {REPO_DIR}", cwd=str(WORK_ROOT))
os.chdir(REPO_DIR)

import yaml

# Lanes run from their own cwd, so every path passed to app.py is absolute.
APP = f"python {REPO_DIR / 'app.py'}"
ARTIFACTS = REPO_DIR / "artifacts"

for lane in LANE_SPECS:
    config_path = REPO_DIR / lane["config"]
    run_name = yaml.safe_load(config_path.read_text(encoding="utf-8"))["run_name"]
    lane["config_path"] = str(config_path)
    lane["run_name"] = run_name
    lane["run_dir"] = ARTIFACTS / "runs" / run_name
    lane["adapter_dir"] = ARTIFACTS / "runs" / run_name / "final"
    lane["pred_path"] = ARTIFACTS / "predictions" / f"{run_name}-test.jsonl"
    lane["report_path"] = ARTIFACTS / "reports" / f"{run_name}-eval.md"
    lane["merged_dir"] = ARTIFACTS / "merged" / run_name
    lane["workdir"] = LANES_DIR / lane["key"]
    lane["workdir"].mkdir(parents=True, exist_ok=True)
    STATUS["lanes"][lane["key"]]["run_name"] = run_name

if len({lane["run_name"] for lane in LANE_SPECS}) != len(LANE_SPECS):
    raise RuntimeError("Two lanes resolve to the same run_name; their artifacts would collide.")
save_status()

for lane in LANE_SPECS:
    print(f"[{lane['key']}] run_name={lane['run_name']}")
    print(f"    adapter : {lane['adapter_dir']}")
    print(f"    report  : {lane['report_path']}")
    print(f"    cwd     : {lane['workdir']}")
