## Why

runsnap's whole premise is that a run's tags and patch describe exactly the code
that produced it. Three defects break that promise without any visible failure:
a process-wide cache makes every run after the first record the *first* run's
code, `runsnap show` silently omits any patched file whose name git quotes, and
`add_record` on a multi-element PyTorch tensor throws a bare `RuntimeError` into
the training loop instead of naming the offending field. Wrong data reaches the
tracking server and stays there, and the test suite is green throughout.

Four narrower defects and a set of accumulated duplications ride along in the
same modules, so they are corrected together rather than reopening those files
later.

## What Changes

Capture fidelity:

- **BREAKING (internal):** `resolve_code_state()` and `resolve_repo_root()` lose
  their `lru_cache`, so every `start_run()` reads the repository as it stands.
  The test-only `reset_code_state_cache()` hook is removed with them.
- `patch_files()` reports every path a patch touches, including the C-quoted
  form git emits for names holding non-ASCII bytes, quotes, or backslashes.
- `flatten_metrics()` raises `ValueError` naming the field for any multi-element
  leaf regardless of array library, and treats a multi-number sequence the same
  way instead of dropping it in silence.

Narrower corrections:

- A failed `checkout --no-worktree` rolls back to the commit HEAD actually stood
  at, rather than to the run's commit when HEAD was detached.
- The TensorBoard writer locates its event files without reaching through
  `tensorboardX` private attributes, and `tensorboardx` gains an upper version
  bound.
- `assemble_logdir()` never fails on a run whose name collides with a
  disambiguated name it generated, and `fetch_run()` repairs a stale or
  non-symlink cache entry rather than raising.
- `record_continues()` warns on a tracking store that refuses the tag, matching
  every other tag write in the package.

Simplifications, no behavior change:

- `resolve_code_state()` calls `read_state()` instead of re-inlining it.
- `capture()` collapses its two `try` blocks into one.
- `flatten_model()` drops the special case that makes an empty model's result
  depend on whether a prefix was passed.
- `start_run()` is annotated `-> LifecycleRun`.
- `.gitignore` covers `mlruns/`, `.pytest_cache/`, and `.ruff_cache/`.

Out of scope: unhandled `GitError`/`ValueError` reaching the terminal as a
traceback from `main()`.

## Capabilities

`openspec/specs/` is currently empty, so every capability below is written as a
new spec. The names come from the archived changes that established them, so
these deltas extend that vocabulary rather than forking it.

### New Capabilities
- `code-state-capture`: what a run records about the repository that produced it.
  This change covers freshness and the rule for reusing a parent run's patch.
- `code-state-restore`: inspecting and reconstructing a run's recorded code
  state. This change covers the patch's file listing and rollback on failure.
- `scalar-records`: how a record of numbers becomes named scalars, including
  which leaves are dropped and which are an error.
- `run-lifecycle`: how a run's ending, failure cause, and continued attempt are
  recorded.
- `tensorboard-logging`: how TensorBoard events are written and uploaded.
- `tensorboard-viewing`: how runs are fetched and assembled for viewing.

### Modified Capabilities
<!-- None: openspec/specs/ holds no capabilities yet. -->

## Impact

- `src/runsnap/_capture.py`, `_git.py`, `_metrics.py`, `_lifecycle.py`,
  `_tensorboard.py`, `_tb_fetch.py`, `_cli.py`, `__init__.py`
- `pyproject.toml` (tensorboardx upper bound), `.gitignore`
- `tests/conftest.py` loses the `reset_code_state_cache()` teardown; tests that
  relied on a cached code state across runs in one process need rewriting.
- No public API removal. `runsnap.start_run`, `log_params`, `load_params`,
  `tensorboard`, `flatten_metrics`, and `attempt_chain` keep their signatures.
- Per-run cost of `start_run()` rises by one `git add -N` plus one `git diff`
  for callers that start many runs in a single process.
