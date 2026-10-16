## 1. Copies instead of links under `--artifact-root`

- [x] 1.1 Replace `_link_scalars` with `_copy_scalars(folder, copies)` in `src/runsnap/_tb_fetch.py`, as in design.md:
  - a missing copy: stage, then move into place;
  - a larger source: stage, then `_extend_or_replace`;
  - an equal or smaller source: skip.

  Staging is a `shutil.copyfile` into a `TemporaryDirectory(prefix=".copy-", dir=copy.parent)`, under a name without `tfevents`. Call it from `_LiveWatch.link` and from the artifact-root branch of `_refresh_pass`.

  Verify in `tests/test_tb_fetch.py`:
  - the existing artifact-root tests are updated from `readlink()` / `is_symlink()` to ordinary files whose bytes equal the folder's (`test_a_stored_run_is_linked_from_its_folder_under_the_artifact_root`, `test_only_scalar_files_are_linked_from_the_artifact_root`, `test_a_new_scalar_shard_in_the_artifact_root_is_linked`, `test_a_run_that_ends_gets_its_last_shard_linked_and_no_more`, `test_a_run_live_on_this_host_is_linked_from_the_artifact_root`, `test_a_deleted_run_leaves_its_files_in_the_artifact_root`), and they pass;
  - new: a file the server replaces with a longer copy leaves the dashboard's copy with the same `st_ino` and the server's bytes after a pass;
  - new: a shorter source leaves the copy unchanged;
  - new: an earlier shard that grows after a later shard appeared is extended while the run is `RUNNING`;
  - new: during and after a refresh, no file or folder under the logdir other than the copies has `tfevents` in its name;
  - `test_a_scalar_file_replaced_by_a_longer_copy_shows_its_new_steps` passes, with a pass between the upload and `accumulator.Reload()`;
  - `test_viewing_from_the_artifact_root_neither_lists_nor_downloads_nor_caches` passes unchanged, and the folder's files keep their bytes and inodes.
- [x] 1.2 Make a failing copy a per-run failure, as in design.md:
  - in `_refresh_pass`, a run whose `_copy_scalars` raises warns unless it is in `_refresh_failing`, and is left out of `fetched`;
  - in `link`, such a run stays shown with what was copied, warns once, and is added to `_refreshing` and `_refresh_failing`.

  Verify with new tests where `_copy_scalars` (or `shutil.copyfile`) raises on several consecutive passes for one of two shown runs:
  - exactly one `UserWarning` names that run;
  - both runs stay in the logdir;
  - the other run keeps refreshing;
  - once copying succeeds again, the run's copy catches up, and a later failure warns again.

  Also verify that a failure on the first copy at startup does not raise out of `assemble_logdir`.

## 2. The default loader and the docs

- [ ] 2.1 In `src/runsnap/_cli.py`:
  - remove the `load_fast` parameter and its `--load_fast=false` branch from `_launch_tensorboard`;
  - stop passing it from `tb`;
  - drop the comment about the Python loader.

  Verify in `tests/test_tb_cli.py`: `test_artifact_root_shows_runs_from_the_folder_with_the_python_loader` becomes a test that:
  - `--load_fast` is absent from the arguments;
  - the logdir holds `baseline/<scalar file>` as an ordinary file with the folder's bytes;
  - the cache is not created.

  The other `tb` CLI tests pass.
- [ ] 2.2 Update the `--artifact-root` paragraph of `README.md`:
  - each run's scalar files are copied into the dashboard's temporary directory, costing disk space equal to the shown runs' scalar data;
  - new and grown files are copied while the run runs and once after it ends;
  - TensorBoard runs with its default loader;
  - remove `--load_fast=false` and the sentence that only a run's newest scalar shard is followed;
  - keep that nothing is downloaded, the cache is unused, MLflow's folder is never written, and `--media` is refused.

  Verify by reading the section against the spec delta.

## 3. Integration checks

- [ ] 3.1 Exercise the change on real data, outside the repo:
  - create runs in a scratch SQLite tracking store, in an experiment with an `mlflow-artifacts:` artifact location;
  - place a copy of the VPS's scalar files under a scratch artifact root, at those runs' folders;
  - run `runsnap tb --artifact-root <scratch root>`.

  Confirm that:
  - TensorBoard lists every run within 30 s;
  - the dashboard process tree settles well below 1 GiB;
  - appending events to a source shard by moving a longer copy over it shows the new points within a few passes.
- [ ] 3.2 Run `uv run ruff check`, `uv run ruff format --check`, `uv run ty check` and `uv run pytest`; verify all pass.

## 4. Deploy on the VPS

These tasks need VPS access and the user's explicit go-ahead before pushing to GitHub.

- [ ] 4.1 After the user approves the push, push the commits and run `./vps.sh update <commit>`. Verify that:
  - `systemctl is-active runsnap-tb` is `active`, and `mlflow` was not restarted (its `ActiveEnterTimestamp` is unchanged);
  - the service's command line has no `--load_fast`, and a `tensorboard_data_server` process is running.
- [ ] 4.2 Once the dashboard has loaded, record in this file:
  - `./vps.sh status`: the service's RAM and swap, and available memory;
  - the time for `/tb/data/plugin/scalars/tags`;
  - how long a running run's new upload takes to appear.

  Verify that memory settles below 1 GiB with no swap growth (`vmstat` `si`/`so` near 0), and the tag list responds within 1 s.

## Workflow follow-up

- Archive the change after review, with `/opsx:archive`.
