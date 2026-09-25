## 1. Move chain expansion into the assembly

- [x] 1.1 Move `_with_chains` from `src/runsnap/_cli.py` to `src/runsnap/_lifecycle.py` as `with_attempts(client, runs)`, unchanged in behaviour; verify the `--chain` tests in `tests/test_tb_cli.py` still pass
- [x] 1.2 Add a `chain: bool = False` keyword to `assemble_logdir` that expands the given runs with `with_attempts` before linking, and have `tb` pass `chain=chain` instead of expanding itself; verify `test_chain_pulls_in_earlier_attempts`, `test_chain_fetches_each_attempt_once`, `test_chain_shows_a_live_run_beside_a_cached_attempt`, and `test_chain_extends_a_filtered_selection` pass unchanged

## 2. Share linking and naming between startup and later passes

- [x] 2.1 Extract the naming in `assemble_logdir` into one helper that takes a batch's base names, the base names already shown, and the link names already taken, and assigns plain names first and then `_distinct_names` candidates not taken; verify `test_assembly_names_duplicates_and_unnamed_runs` and `test_assembly_handles_names_that_collide_with_generated_suffixes` pass unchanged
- [x] 2.2 Extract the "resolve live dirs, fetch the rest concurrently, keep those holding event data" body of `assemble_logdir` into a function over a batch of runs, used by startup; verify the whole of `tests/test_tb_fetch.py` passes unchanged

## 3. Follow the query in the watch thread

- [x] 3.1 Add a `follow: Callable[[], Iterable[Run]] | None = None` keyword to `assemble_logdir`; start the watch thread when there are live links or `follow` is set, and keep it looping while either is true; update the `LIVE_POLL_SECONDS` and `assemble_logdir` docstrings; verify `test_context_of_cached_runs_starts_no_watch_thread` still passes and a new test sees a thread for a context with only cached runs and a `follow` callable, joined on exit
- [x] 3.2 Keep a `handled` set seeded at startup with every run shown or carrying `runsnap.tb.local_dir`, and on each tick after the re-point step run a follow pass: call `follow()`, keep runs carrying `runsnap.tb.local_dir` and not in `handled`, expand them with `with_attempts` when `chain` is set, drop any in `handled`, process them with the batch function from 2.2, name them with the helper from 2.1 against the shown bases and taken names, create their links, add live ones to the watched links, and add every processed id to `handled`; verify with a test that a `follow` callable returning a `live_run` created after the context opened makes `logdir / <name>` appear within a deadline, resolve to its local directory, and switch to the cache after its directory is removed
- [x] 3.3 Verify the readiness gate with tests: a run returned by `follow` without the local-dir tag is never linked and `download_artifacts`/`list_artifacts` are not called for it over several polls; the same run is linked once the tag is set; a run passed at startup without the tag and without event data is linked once `follow` returns it tagged
- [x] 3.4 Verify late-run edge cases with tests: a run whose local directory is already gone when first ready is linked from the cache; a late run named like a shown run gets `<name>-<run id prefix>` and the shown link keeps its name and target; a late run whose base matches runs distinguished at startup is distinguished too; with `chain=True` a late run's earlier attempt is linked and each attempt is fetched once; a run already shown is never fetched again by a pass
- [x] 3.5 Wrap the follow pass so an exception from `follow()` warns only when the previous pass succeeded, leaves links untouched, and never ends the thread; verify with a test whose `follow` raises on several consecutive calls and then returns a new run: exactly one `UserWarning`, the shown run still linked, and the new run linked afterwards

## 4. Wire the CLI

- [x] 4.1 In `tb`, pass `follow=partial(_tb_runs, client, (), experiment, filter)` when `run_refs` is empty and `None` otherwise; verify with CLI tests (a `viewer` that waits for a link to appear, with `LIVE_POLL_SECONDS` patched small) that bare `runsnap tb` picks up a run tagged live after launch, including one in an experiment created after launch, that `--experiment` ignores a new run in another experiment, and that `runsnap tb <name>` never adds a new run
- [x] 4.2 Verify the query bound: record `search_experiments` calls across one follow pass of bare `runsnap tb` and assert one enumeration per pass; verify `test_several_names_enumerate_experiments_once` and `test_a_named_experiment_is_resolved_without_enumerating_them` still pass

## 5. Documentation and final pass

- [x] 5.1 Update the TensorBoard section of `README.md`: a query selection (bare, `--experiment`, `--filter`) adds runs that start writing TensorBoard events while the dashboard is open, named selections stay fixed, a late run sharing a shown name gets a suffix, and "rerun the command" remains only for runs logged on another host; verify by reading the section
- [x] 5.2 Manually run `uv run runsnap tb` with a real TensorBoard, start a short training with `runsnap.tensorboard()` afterwards, and confirm the run appears within about 10 s, updates live, and stays visible after the block exits; record the result in the task
  - Result (2026-09-25, TensorBoard 2.21.0 with its fast data loader, bare `runsnap tb`): a 40-step run started afterwards in an experiment created after launch was listed 1.8 s after entering its `runsnap.tensorboard()` block, gained 5 steps within 5 s while linked to its local directory, switched to the cache 1.2 s after the block exited, and stayed listed with steps 0..39; Ctrl-C exited 0 and removed the assembled directory.
- [x] 5.3 Run `uv run ruff check`, `uv run ruff format --check`, `uv run ty check`, and `uv run pytest`; verify all pass
