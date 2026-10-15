# 3. Clone the repository and pin the code version
if (REPO_DIR / '.git').exists():
    run_step('git-fetch', 'git -C ' + str(REPO_DIR) + ' fetch --all --tags --prune', cwd=str(WORK_ROOT),
             allow_fail=True)
    if not GIT_REF:
        run_step('git-pull', 'git -C ' + str(REPO_DIR) + ' pull --ff-only', cwd=str(WORK_ROOT), allow_fail=True)
else:
    run_step('git-clone', 'git clone ' + GITHUB_REPO + ' ' + str(REPO_DIR), cwd=str(WORK_ROOT))
if GIT_REF:
    run_step('git-checkout', 'git -C ' + str(REPO_DIR) + ' checkout ' + GIT_REF, cwd=str(WORK_ROOT))

os.chdir(REPO_DIR)
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))


def _git(*args):
    return subprocess.run(['git', *args], cwd=REPO_DIR, capture_output=True, text=True).stdout.strip()


GIT_COMMIT = _git('rev-parse', 'HEAD')
GIT_DIRTY = bool(_git('status', '--porcelain'))
if not GIT_REF:
    print('WARNING: GIT_REF is blank, so this bundle is not pinned to a tag. Set GIT_REF for thesis runs.')

# The merge must use the verified adapter loader (history/027.md) and prompts must
# follow the training budget (history/030.md), exactly as in the GPU reference jobs.
loader_src = (REPO_DIR / 'src' / 'training' / 'model_loader.py').read_text(encoding='utf-8')
dataset_src = (REPO_DIR / 'src' / 'data' / 'dataset.py').read_text(encoding='utf-8')
neo4j_src = (REPO_DIR / 'src' / 'evaluation' / 'neo4j_client.py').read_text(encoding='utf-8')
if 'def attach_adapter' not in loader_src or 'Adapter verified' not in loader_src:
    raise RuntimeError('This checkout lacks the verified adapter loader (history/027.md).')
if 'def fit_prompt_to_budget' not in dataset_src or 'connection_failed' not in neo4j_src:
    raise RuntimeError('This checkout lacks the history/030.md changes (prompt budget, Neo4j fixes).')
print('Commit : ' + GIT_COMMIT + (' (uncommitted changes)' if GIT_DIRTY else ''))
