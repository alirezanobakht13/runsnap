## 1. Logging side records the local directory

- [x] 1.1 Add `TB_TAG_LOCAL_HOST` (`runsnap.tb.local_host`), `TB_TAG_LOCAL_DIR` (`runsnap.tb.local_dir`), and `TB_FLUSH_SECONDS = 10` to `src/runsnap/_tags.py`; verify `uv run python -c "import runsnap._tags"` imports cleanly
- [x] 1.2 In `tensorboard()` in `src/runsnap/_tensorboard.py`, set the two tags to `socket.gethostname()` and the absolute scratch directory inside the existing warn-not-raise tagging block, before the writer is yielded; verify with a test beside `test_logdir_tag_is_set_on_entry` in `tests/test_tensorboard.py` asserting both tags are present, the directory exists, and it holds an event file while the block is open
- [x] 1.3 Add a test that the tags remain on the run after the block exits and the directory is gone; verify it passes
- [x] 1.4 Add a test that a tracking server refusing `set_tag` produces a warning and still yields a working writer; verify it passes
- [x] 1.5 In `tensorboard()`, inject `flush_secs=TB_FLUSH_SECONDS` into the writer keyword arguments only when the caller did not pass one; verify with a test that a scalar written without an explicit `flush()` is readable from the local event file well before 120 seconds, and a test that an explicit `flush_secs=60` reaches the underlying writers unchanged

## 2. Viewer prefers a live local directory

- [x] 2.1 In `assemble_logdir` in `src/runsnap/_tb_fetch.py`, add a helper that returns the run's tagged local directory when `runsnap.tb.local_host` equals `socket.gethostname()` and the directory exists, else `None`; use it per run so `fetch_run` is only called when the helper returns `None`; verify existing `tests/test_tb_fetch.py` still passes unchanged
- [x] 2.2 Add a test in `tests/test_tb_fetch.py` for a run tagged with this host and an existing directory: the assembled logdir links that directory under the run name, and `download_artifacts` is never called for it; verify it passes
- [x] 2.3 Add tests for the fallbacks: tagged directory missing, and tag naming another host while the path exists; both must fetch from artifacts; verify they pass
- [x] 2.4 Add a test that a live run and a cached run are assembled together under their names, using `--chain` through `tests/test_tb_cli.py`; verify it passes
- [x] 2.5 Add an end-to-end test in `tests/test_tb_fetch.py` or `tests/test_tb_cli.py`: open `runsnap.tensorboard()` in one run, write a scalar, assemble the logdir, write another scalar and flush, and read the assembled directory with `EventAccumulator` to see both steps; verify it passes

## 3. Documentation and final pass

- [x] 3.1 Update the TensorBoard section of `README.md`: replace the sentence that every invocation is a snapshot with the same-host live behavior, note that a live run shows its media regardless of `--media`, and state the 10 second flush default and the `flush_secs=` override; verify by reading the section
- [x] 3.2 Run `uv run pytest`, `uv run ruff check`, `uv run ruff format --check`, and `uv run ty check`; verify all pass
