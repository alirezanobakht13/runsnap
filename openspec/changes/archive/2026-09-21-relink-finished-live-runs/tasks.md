## 1. Watch live links in `assemble_logdir`

- [x] 1.1 Add `LIVE_POLL_SECONDS` to `src/runsnap/_tb_fetch.py` and a `poll_interval` keyword on `assemble_logdir` defaulting to it; verify `test_tb_fetch.py` still passes with no other change
- [x] 1.2 Add a `_LiveWatch` thread (start/stop on a `threading.Event`, like `_Sync`) that polls the live-linked directories and, when one is gone, calls `fetch_run` with the assembly's `cache_dir` and `media` and re-points the link of the same name; verify with a test that removes a live run's directory while the context is open and, within a deadline, sees the link resolve into the cache with the uploaded scalar file
- [x] 1.3 Start the thread only when a run was linked live and stop it before `TemporaryDirectory` cleans up; verify with a test that a context of only cached runs starts no thread (`threading.active_count()` unchanged) and that leaving a context with live runs joins the thread and removes the directory
- [x] 1.4 Warn and stop watching a run whose fetch raises after its directory disappears, leaving its link as is; verify with a test that patches `download_artifacts` to raise, sees exactly one `UserWarning`, and sees a second live run in the same context still switch correctly

## 2. End-to-end behaviour

- [x] 2.1 Add a test alongside `test_assembled_logdir_follows_a_run_still_being_written` that opens `assemble_logdir` on a run inside `runsnap.tensorboard()`, leaves the writer block, and asserts that an `EventAccumulator` on the link reads every scalar written before exit from the cached file; verify it passes with a small `poll_interval`
- [x] 2.2 Add a test that a live run whose directory is left in place is never fetched (patch `download_artifacts` and assert it is not called after several polls); verify it passes
- [x] 2.3 Check `--media` parity: with `media=True` the re-pointed link resolves to the run's `events` directory, without it to `scalars`; verify with a parametrized test

## 3. Documentation and final pass

- [x] 3.1 Replace the README sentence "If a live writer block exits while the dashboard is open, rerun the command to view its final uploaded artifacts." with a description of the run switching to its cached artifacts on its own; verify by reading the TensorBoard section
- [x] 3.2 Update the `assemble_logdir` docstring to describe the watching; verify `uv run ruff check`, `uv run ruff format --check`, and `uv run ty check` pass
- [x] 3.3 Run `uv run pytest` and verify the whole suite passes
