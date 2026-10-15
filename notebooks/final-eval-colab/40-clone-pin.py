# 3. Clone the repository and pin the code version
if (REPO_DIR / '.git').exists():
    run_step('git-fetch', f'git -C {REPO_DIR} fetch --all --tags --prune', cwd='/content', allow_fail=True)
    if not GIT_REF:
        run_step('git-pull', f'git -C {REPO_DIR} pull --ff-only', cwd='/content', allow_fail=True)
else:
    run_step('git-clone', f'git clone {GITHUB_REPO} {REPO_DIR}', cwd='/content')
if GIT_REF:
    run_step('git-checkout', f'git -C {REPO_DIR} checkout {GIT_REF}', cwd='/content')

os.chdir(REPO_DIR)
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))


def _git(*args):
    return subprocess.run(['git', *args], cwd=REPO_DIR, capture_output=True, text=True).stdout.strip()


GIT_COMMIT = _git('rev-parse', 'HEAD')
GIT_DIRTY = bool(_git('status', '--porcelain'))

# Predictions made before history/027.md silently evaluated the base Qwen model.
# Refuse to run on code that lacks the verified adapter loader.
loader_src = (REPO_DIR / 'src' / 'training' / 'model_loader.py').read_text(encoding='utf-8')
if 'def attach_adapter' not in loader_src or 'Adapter verified' not in loader_src:
    raise RuntimeError('This checkout lacks the verified adapter loader (history/027.md). '
                       'Pull a newer commit or clear GIT_REF.')
print('Commit : ' + GIT_COMMIT + (' (uncommitted changes)' if GIT_DIRTY else ''))
