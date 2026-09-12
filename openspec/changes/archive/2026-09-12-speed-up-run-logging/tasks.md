## 1. Param batching

- [x] 1.1 Rewrite `log_params()` to build `Param` entities and call
  `log_batch()` in chunks of `MAX_PARAMS_TAGS_PER_BATCH`, keeping the per-key
  length warning; verify existing params tests pass unchanged
- [x] 1.2 Add a test with a client recording its calls that a model of more than
  one batch's worth of leaves produces the expected number of `log_batch` calls
  and that every param is present
- [x] 1.3 Verify a model smaller than one batch produces exactly one
  `log_batch` call

## 2. Scalar sharding

- [x] 2.1 Add a scalar shard threshold to `_tags.py` and a `tb_light_suffix(n)`
  helper alongside `tb_media_suffix(n)`; verify the suffix format matches
  `.scalars.<n>`
- [x] 2.2 Generalize `TensorBoardWriter` so both streams roll at their own
  threshold, with `_call_light` rolling the scalar shard the way `_call_media`
  rolls the media shard; verify a test writing past the scalar threshold
  produces several sealed scalar files
- [x] 2.3 Narrow `_Sync.sync()`'s skip to the open media shard only, so the open
  scalar shard uploads on every pass that finds it grown; verify a test that a
  second sync pass after more scalars uploads the open scalar shard and that a
  pass with no growth uploads nothing
- [x] 2.4 Widen the scalar/media classifier in `_tb_fetch.fetch_run()` to match
  `.scalars` and `.scalars.<n>`; verify tests cover a run with pre-change
  single-file scalars, a run with sharded scalars, and a directory holding both

## 3. Bounded patch capture

- [x] 3.1 Add `PatchTooLarge(GitError)` to `_git.py` and give `build_patch()` a
  byte ceiling that reads git's stdout incrementally, terminating the subprocess
  once the total passes the ceiling; verify a test that a tree far over a small
  ceiling raises without the process resident size growing with the tree
- [x] 3.2 Give `CodeState` an explicit `dirty` field instead of deriving it from
  the patch, sourced from what git reported; verify a clean tree still records
  `dirty=false`
- [x] 3.3 Pass `max_patch_bytes()` into patch building and handle
  `PatchTooLarge` in `capture()` so the run records the warning, the
  capture-error tag naming the ceiling, `dirty=true`, and no digest; verify a
  test asserting each of those four
- [x] 3.4 Remove the now-unreachable post-hoc size check from `_record_patch()`
  and verify the oversize-patch tests still pass

## 4. CLI query bounds

- [x] 4.1 Thread a resolved experiment-id list through `resolve_run()` and
  `_tb_runs()` so it is computed once per invocation; verify a test with a
  counting client that three run names enumerate experiments once
- [x] 4.2 Verify that naming an experiment never enumerates the full experiment
  list
- [x] 4.3 Add `attempt_runs(client, run_id) -> list[Run]` in `_lifecycle.py`,
  reimplement `attempt_chain()` on top of it, and have `_with_chains()` use it;
  verify a test with a counting client that a three-attempt chain fetches each
  run once and that `attempt_chain()`'s return value is unchanged

## 5. Concurrent fetch

- [x] 5.1 Fetch runs in `assemble_logdir()` through a `ThreadPoolExecutor`
  bounded at `min(8, len(runs))`; verify existing viewer tests pass and results
  are independent of completion order
- [x] 5.2 Make a failed fetch warn and drop that run from the selection instead
  of aborting; verify a test where one run's download raises and the others
  still appear

## 6. Final pass

- [x] 6.1 Update the README's TensorBoard section for scalar sharding and the
  capture section for the oversize-patch digest change
- [x] 6.2 Run `uv run pytest`, `uv run ruff check`, `uv run ruff format`, and
  `uv run ty check`; verify all pass
- [x] 6.3 Run `openspec validate --changes speed-up-run-logging --strict` and
  verify it reports no errors
