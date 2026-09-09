## 1. Project Setup

- [x] 1.1 Add `pydantic>=2` and `cyclopts>=4` with `uv add`, and verify `uv run python -c "import pydantic, cyclopts, mlflow"` succeeds
- [x] 1.2 Add `pytest` as a dev dependency with `uv add --dev pytest`, and verify `uv run pytest --version` reports a version
- [x] 1.3 Create the module skeleton `_tags.py`, `_git.py`, `_capture.py`, `_params.py`, `_cli.py` under `src/exp_track/`, and verify `uv run python -c "import exp_track"` succeeds
- [x] 1.4 Define tag names, artifact paths (`code/state.patch`, `hparams/<name>.json`), the default patch size ceiling, and the `EXP_TRACK_CAPTURE_CODE` / `EXP_TRACK_MAX_PATCH_BYTES` environment variable names in `_tags.py`, and verify the constants import cleanly

## 2. Git Primitives

- [x] 2.1 Add a pytest fixture that builds a throwaway git repository with a configurable working-tree state, and verify a smoke test using it passes
- [x] 2.2 Implement repository discovery and state reading in `_git.py` (repo root, HEAD SHA, branch name, remote URL with credentials stripped), and verify tests covering a normal branch, a detached HEAD, and a repository with no commits
- [x] 2.3 Implement patch construction via a copied index with `GIT_INDEX_FILE`, `git add -A -N .`, and `git diff HEAD --binary`, and verify a test asserts the source repository's `git status` output is byte-identical before and after
- [x] 2.4 Add round-trip tests applying the constructed patch to a clean checkout of the base commit, covering staged-then-re-edited files, untracked additions, deletions, executable-bit changes, binary files, and `.gitignore`d files being excluded, and verify all pass
- [x] 2.5 Implement worktree creation, in-place branch checkout, and patch application helpers in `_git.py`, and verify tests that reconstruct a repository state and assert the resulting tree matches the original
- [x] 2.6 Make every `_git.py` entry point raise a single typed error carrying git's stderr, and verify tests for a missing `git` binary, a non-repository directory, and a failing `git apply`

## 3. Code State Capture

- [x] 3.1 Implement per-process cached git state resolution in `_capture.py`, and verify a test asserts git is invoked once across several runs
- [x] 3.2 Implement tag writing (commit, branch, dirty, repo_url, patch_sha256) and `code/state.patch` artifact logging for a run, and verify a test using a local MLflow tracking store reads the tags and downloads the artifact
- [x] 3.3 Skip the patch artifact on clean trees and set `dirty` to `false`, and verify a test asserts no artifact is present
- [x] 3.4 Detect nesting and, for nested runs, write `exp_track.git.patch_run_id` instead of duplicating the artifact, and verify a test with a parent and two children asserts the artifact exists once
- [x] 3.5 Enforce the patch size ceiling, setting `exp_track.git.capture_error` and emitting a warning instead of uploading, and verify a test with a lowered ceiling
- [x] 3.6 Mirror commit, branch, repo URL, and dirty into `mlflow.source.git.*` only when unset, never writing `mlflow.source.git.diff`, and verify tests for the absent and already-set cases
- [x] 3.7 Wrap the whole capture so no exception escapes into user code, warning and recording `capture_error` where possible, and verify tests for a non-repository directory, an absent `git` binary, a repository with no commits, and an injected unexpected git failure
- [x] 3.8 Implement `exp_track.start_run()` as a pass-through to `mlflow.start_run()` that triggers capture, honouring `capture_code=False` and `EXP_TRACK_CAPTURE_CODE=0`, and verify tests asserting argument forwarding, context-manager behaviour, that the returned object is MLflow's run, and that both opt-outs suppress capture

## 4. Pydantic Hyperparameters

- [x] 4.1 Implement the flattener in `_params.py` (`model_dump(mode="json")`, dotted keys for nested mappings, strings verbatim, all other leaves `json.dumps`-encoded, sequences and empty containers logged whole), and verify unit tests for nested models, three-level nesting, each scalar type, sequences, and empty containers
- [x] 4.2 Implement `exp_track.log_params()` writing flattened params plus the `hparams/<name>.json` artifact containing the serialized model and its class's fully qualified name, and verify a test reading both back from a run
- [x] 4.3 Reject non-Pydantic input with `TypeError`, support the `name` and `prefix` arguments, and warn when an encoded param value exceeds MLflow's param length limit, and verify tests for each
- [x] 4.4 Implement `exp_track.load_params()` downloading and validating the artifact against a caller-supplied class, warning on class-name mismatch and raising a clear error when the artifact is absent, and verify a round-trip test plus tests for mismatch, validation failure, and missing artifact

## 5. CLI

- [x] 5.1 Create the cyclopts app in `_cli.py`, repoint the `exp-track` entry point in `pyproject.toml` away from the placeholder, and verify `uv run exp-track --help` lists `checkout` and `show`
- [x] 5.2 Implement run reference resolution accepting a 32-character hex run id or a run name resolved via `search_runs`, honouring `--experiment`, and verify tests for id lookup, unique name lookup, ambiguous names listing candidates, and no match
- [x] 5.3 Implement tracking URI resolution from the MLflow environment with a `--tracking-uri` override, and verify a test asserting the override wins
- [x] 5.4 Implement `exp-track show`, printing commit, branch, dirty flag, patch digest, and the files the patch touches without modifying the repository, and verify a test asserting the output fields and an unchanged `git status`
- [x] 5.5 Implement `exp-track checkout` in its default worktree mode, following `patch_run_id` for nested runs, leaving the patch uncommitted, and printing the created path, and verify a test that reconstructs a run and compares the resulting tree against the original working-tree state
- [x] 5.6 Add the `--no-worktree`, `--path`, `--branch`, `--commit`, and `--force` flags with the documented semantics, and verify tests for in-place checkout, the dirty-tree refusal and its `--force` override, an explicit branch name, and `--commit` producing one extra commit with a clean tree
- [x] 5.7 Implement precondition checks that change nothing on failure — missing base commit, run with no recorded code state, existing target branch, failing patch application, and not being in a git repository — and verify tests asserting each error message and that no branch or worktree is left behind

## 6. Final Pass

- [x] 6.1 Run the full suite with `uv run pytest` and verify it passes
- [x] 6.2 Run `uv run ruff check`, `uv run ruff format`, and `uv run ty check`, and verify all report clean
- [x] 6.3 Write `README.md` covering the `start_run` and `log_params` usage, the recorded tag and artifact schema, the `exp-track checkout` and `show` commands, and the note that a patch can contain uncommitted secrets from tracked files, and verify the documented commands run as written
