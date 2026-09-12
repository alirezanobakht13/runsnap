## 1. Code-state freshness

- [x] 1.1 Remove the `lru_cache` from `resolve_code_state()` and
  `resolve_repo_root()` in `_capture.py`, and delete `reset_code_state_cache()`
  and its `tests/conftest.py` teardown; verify the suite still passes
- [x] 1.2 Rewrite `_patch_holders` to map a run id to `(holder_run_id, digest)`
  and make `_inherited_patch_holder()` inherit only on a digest match; verify a
  nested run whose tree changed after its parent started uploads its own patch
  and carries its own digest
- [x] 1.3 Add a test that a process starting two runs across an edit records two
  different patches, and one that it records a new commit made between them

## 2. Patch file listing

- [x] 2.1 Add a C-unquoting helper in `_git.py` covering `\n \t \r \" \\` and
  octal escapes, decoding with `surrogateescape`; verify unit tests round-trip a
  non-ASCII name and a name holding a double quote
- [x] 2.2 Rewrite `patch_files()` to parse each record's `diff --git` header by
  the three rules in design.md (quoted pair, matching ` b/` split, `rename to`
  fallback); verify tests cover a non-ASCII name, an embedded quote, a path
  containing ` b/`, a rename, a deletion, and a binary add
- [x] 2.3 Verify `runsnap show` lists a binary file added to the working tree,
  which the current implementation already handles, has not regressed

## 3. Metric leaves

- [x] 3.1 Rewrite `_leaf()` in `_metrics.py` so a multi-element leaf raises
  `ValueError` naming the key whether it fails with `ValueError`, `RuntimeError`
  or `TypeError`, and so a non-string sequence of more than one number raises
  the same way; verify tests cover a numpy array, a stub object raising
  `RuntimeError` from `.item()`, and a plain list
- [x] 3.2 Verify a one-element sequence yields its element, an empty sequence is
  omitted, and `str`/`bytes`/`None` leaves stay omitted
- [x] 3.3 Update the `flatten_metrics()` docstring and the README's
  `add_record` paragraph to state the sequence rule

## 4. TensorBoard writer and viewer

- [x] 4.1 Replace `_event_path()`'s private-attribute walk with a glob of the
  writer's `get_logdir()` for `events.out.tfevents.*<filename_suffix>`; verify
  `light_path` and `media_path` still resolve and shard rolling still works
- [x] 4.2 Add `tensorboardx<3` to `pyproject.toml` via `uv add` and verify
  `uv sync` resolves
- [x] 4.3 Rewrite `assemble_logdir()` name assignment as two passes per
  design.md; verify a test with two runs named `base` plus one named
  `base-<that run's short id>` produces three distinct names without raising
- [x] 4.4 Make `fetch_run()` replace a cache entry that is not already a symlink
  to the intended target; verify tests cover a stale symlink and a regular file
  occupying the link path

## 5. Narrower corrections

- [x] 5.1 In `_cli.py` `_reconstruct()`, capture the pre-run position as
  `current_branch(root) or head_commit(root)` and roll back to it; verify a test
  that a failed `--no-worktree` patch from a detached `HEAD` returns to the
  original commit and removes the created branch
- [x] 5.2 Make `record_continues()` warn instead of raising when the tracking
  store refuses the tag, mirroring `record_cause()`; verify a test with a client
  that raises on `set_tag`

## 6. Simplifications

- [ ] 6.1 Have `resolve_code_state()` call `_git.read_state()` instead of
  re-inlining it, and collapse `capture()`'s two `try` blocks into one that
  keeps both warning messages distinguishable; verify existing capture tests pass
- [ ] 6.2 Remove the `if not prefix and not data` special case from
  `flatten_model()` and settle empty-model behavior one way in the docstring;
  verify the params tests cover an empty model both with and without a prefix
- [ ] 6.3 Annotate `start_run()` as `-> LifecycleRun` and verify `ty check` passes
- [ ] 6.4 Add `mlruns/`, `.pytest_cache/`, and `.ruff_cache/` to `.gitignore` and
  verify `git status --porcelain` is clean after a test run

## 7. Final pass

- [ ] 7.1 Run `uv run pytest`, `uv run ruff check`, `uv run ruff format`, and
  `uv run ty check`; verify all pass
- [ ] 7.2 Run `openspec validate --changes fix-capture-fidelity --strict` and
  verify it reports no errors
