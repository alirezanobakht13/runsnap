## 1. Tags and lifecycle module

- [x] 1.1 Add `TAG_FAILURE_CAUSE` (`runsnap.failure.cause`) and `TAG_CONTINUES` (`runsnap.continues`) to `_tags.py`, and verify the module imports cleanly under `uv run python -c "import runsnap._tags"`.
- [x] 1.2 Add `_lifecycle.py` with an `ActiveRun` subclass whose `__exit__` records the cause tag under the warn-instead-of-raise guard, ends the run as `KILLED` on `KeyboardInterrupt` and otherwise delegates to the parent class, and always propagates the exception; verify with tests covering `FINISHED` with no tag, `FAILED` with a cause naming the type and message, `KILLED` on `KeyboardInterrupt`, and a refused tag write that still ends the run and warns.
- [x] 1.3 Make `start_run` wrap MLflow's run in the subclass and accept `continues`, writing `runsnap.continues` before capture; verify with tests that the tag is present when given, absent otherwise, coexists with `mlflow.parentRunId` under `nested=True`, that `mlflow.active_run()` inside the block reports the same id, and that a run started without `with` ends normally through `mlflow.end_run()`.
- [x] 1.4 Add `attempt_chain(run_id)` returning ids newest first, raising `ValueError` on a revisited run, and export it from `runsnap`; verify with tests for a three-attempt chain, a first attempt, a cycle, and a missing predecessor.
- [x] 1.5 Verify a nested child that fails while its parent catches the exception leaves the child `FAILED` with a cause and the parent `FINISHED` without one, as a test in `tests/test_capture.py` or the new lifecycle test module.

## 2. CLI

- [x] 2.1 Make `runsnap show` print a `continues:` line when the tag is present and nothing otherwise; verify with two tests in `tests/test_cli.py`.
- [x] 2.2 Add `--chain` to `runsnap tb`, extending the selected runs with each run's attempt chain deduplicated by id; verify with tests that a named latest attempt pulls in its predecessors, that a filtered selection is extended, and that omitting the flag selects only the named run.

## 3. Documentation and final pass

- [x] 3.1 Update the README: the `KILLED` status and `runsnap.failure.cause`, `continues=` and `attempt_chain`, the `show` line, and `tb --chain`; verify the tag list in the README matches `_tags.py`.
- [x] 3.2 Run `uv run ruff check`, `uv run ruff format`, `uv run ty check`, and `uv run pytest`, and verify all pass.
