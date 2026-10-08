## 1. The cache extends grown files

- [x] 1.1 In `fetch_run` (`src/runsnap/_tb_fetch.py`), choose the action by comparing the listed remote size with the cached size, as in design.md:
  - equal or smaller: skip without downloading;
  - larger: download to the staging directory, then append only the tail when the download begins with the cached bytes, and otherwise move it over the cached file;
  - not cached: download and move into place, as today.

  Verify with new tests in `tests/test_tb_fetch.py`:
  - a grown remote file leaves the cached file with the same inode (`st_ino`) and the remote bytes;
  - a shorter remote file causes no `download_artifacts` call and leaves the cached bytes;
  - a remote file not beginning with the cached bytes replaces them;
  - a cached file truncated to a prefix of the remote file is completed by the next fetch;
  - `test_partial_cache_file_is_replaced`, `test_interrupted_download_preserves_previous_file` and `test_only_changed_and_new_files_are_downloaded` still pass.

## 2. Check watched runs every pass and remove deleted ones

- [x] 2.1 Add a helper that checks watched run ids with `search_runs(<their experiment ids>, filter_string="attributes.run_id IN (...)", run_view_type=ACTIVE_ONLY)`. It follows page tokens, sends ids in batches of at most 200, and returns each found run's status.

  Verify with tests:
  - a running and a finished run come back with their statuses;
  - a deleted run, and a run of a deleted experiment, are absent;
  - 450 ids cost exactly three searches (spy on `search_runs`).
- [x] 2.2 Make `_LiveWatch` always start while the context is open. Run the check as the first step of each pass over shown runs and, for a named selection, every named run. For each run that is missing:
  - unlink it;
  - drop it from `handled`, the shown names and the watched links;
  - delete `<cache>/<run_id>`.

  A failing check warns once per run of failures and changes nothing that pass. Verify with tests:
  - a shown run deleted via `MlflowClient.delete_run` loses its link within a deadline and its cache folder is gone;
  - deleting the experiment of shown runs removes them;
  - a run set to `FINISHED`, `FAILED` or `KILLED` stays linked;
  - a check raising on several consecutive passes gives exactly one `UserWarning` and keeps every link.

  Also replace `test_context_of_cached_runs_starts_no_watch_thread` with a test that the thread runs for cached runs and is joined on exit.
- [x] 2.3 Let removed runs come back. Verify with tests:
  - a removed run in a followed query is linked again after `restore_run`;
  - a removed named run is linked again after `restore_run`;
  - a returning run whose old name is now taken by a late run is distinguished.
- [x] 2.4 Update the TensorBoard section of `README.md`:
  - runs deleted in MLflow, directly or with their experiment, leave the dashboard within a few seconds and their cached files are removed;
  - restored runs return;
  - ended runs stay.

  Verify by reading the section.

## 3. Pending runs are retried until they have data

- [x] 3.1 Keep a `pending` set: selected runs carrying `runsnap.tb.local_dir` that were not linked because they have no event data. Seed it at startup instead of putting such runs in `handled`, and fill it from the follow pass. On each pass, try to link pending runs that the check reports `RUNNING`; give a run that has ended a last try and then move it to `handled`. Untagged runs never become pending.

  Verify with tests:
  - a run tagged with another host and with no artifacts when the context opens is linked once its event files are logged;
  - the same happens when `follow` returns it after the context opened;
  - a pending run that ends without event data is never linked, and `list_artifacts` is not called for it after the pass that saw it ended (spy);
  - an untagged run returned by `follow` is never listed, as in the existing readiness tests.
- [x] 3.2 Update `README.md`: a run starting on another host appears within a few seconds of its first upload, replacing the sentence saying it "stays absent". Verify by reading the section.

## 4. Running cached runs are refreshed

- [x] 4.1 As the last step of each pass, fetch again every run shown from the cache whose checked status is `RUNNING`, through `_fetch_runs`. Give a run that ended since the previous pass one final fetch and then drop it from the refresh set. A failing refresh warns once per run per run of failures and leaves its link.

  Verify with tests:
  - a shown remote run with status `RUNNING` gains steps after a longer event file is logged to its artifacts (`wait_for_steps`);
  - after `set_terminated` the run is fetched exactly once more and never again over several passes (spy on `list_artifacts`);
  - a run finished at startup is never fetched by a pass;
  - a refresh raising on several passes warns once.

  Rewrite `test_a_pass_never_fetches_a_shown_run_again` so it covers finished runs.
- [x] 4.2 Update `README.md`: shown runs logged on another host keep updating while they run, replacing "Rerun the command to fetch newer uploads". Verify by reading the section.

## 5. The CLI opens on an empty selection and takes a path prefix

- [x] 5.1 In `tb` (`src/runsnap/_cli.py`), decide whether to launch:
  - a followed query launches TensorBoard even when nothing is linked, printing that no runs were found yet;
  - a named selection launches when any named run is linked or pending, and otherwise reports and exits as today.

  Verify with CLI tests using `viewer`:
  - bare `runsnap tb` against an empty server launches and later shows a run that starts writing;
  - `runsnap tb <name>` for a pending remote run launches;
  - `runsnap tb <name>` for a run without TensorBoard data prints the no-runs message without launching.
- [x] 5.2 Add `path_prefix: str | None`, named `("--path-prefix", "--path_prefix")`, passed to TensorBoard as `--path_prefix`. Verify with tests on the `subprocess.run` arguments:
  - both spellings produce `--path_prefix /tb`;
  - nothing is passed when the option is omitted.
- [x] 5.3 Update `README.md` with the empty-selection behaviour and a `--path-prefix` example next to the `--host` and `--port` examples. Verify by reading the section.

## 6. Viewing straight from the artifact folder

- [x] 6.1 Add the `--artifact-root` mapping. A run's `mlflow-artifacts:` `artifact_uri` path, without leading slashes, is joined to the root and `tb` is added. A result outside the root, or any other scheme, is reported as a warning and the run is treated as handled.

  Verify with tests that create experiments with `artifact_location="mlflow-artifacts:/<name>"` and write event files directly under `<root>/<name>/<run_id>/artifacts/tb/`:
  - such a run maps to that folder;
  - a run of a `file:` experiment is warned about and left out while the others are linked;
  - a path that escapes the root is refused.
- [x] 6.2 Link artifact-root runs as a real directory per run holding one link per scalar file, mirroring subfolders. Add links for new scalar files on each pass while the run is `RUNNING`, plus a final pass after it ends. Never consult `_live_logdir` in this mode, and never call `list_artifacts` or `download_artifacts`.

  Verify with tests:
  - no artifact API calls (spy) and no cache folder created;
  - media files are not linked;
  - a new scalar shard written to the folder is linked within a deadline;
  - after replacing a scalar file with a longer copy through `os.replace`, `EventAccumulator` on the run's link directory shows the new steps;
  - a run with a live local directory on this host is linked to its artifact folder;
  - deleting a shown run removes its link and leaves its files in the artifact folder intact.
- [x] 6.3 Add `--artifact-root PATH` to `tb`. Refuse it with `--media` (a `CliError`) before any run is resolved. Pass the root to `assemble_logdir`, and launch TensorBoard with `--load_fast=false` when it is given.

  Verify with CLI tests:
  - the refusal message, with `make_client` never called;
  - `--load_fast=false` is present only with `--artifact-root`;
  - bare `runsnap tb --artifact-root <root>` links runs from the folder.
- [x] 6.4 Document `--artifact-root` in `README.md`:
  - for use on the tracking server's own machine, with its `--artifacts-destination`;
  - no downloads or cache;
  - scalars, records, text and HParams only, with `--media` refused;
  - deleted runs' files freed only by `mlflow gc`.

  Verify by reading the section.

## 7. Integration checks

- [x] 7.1 With `uv run --env-file .env runsnap tb --experiment vps-smoke --port 0` open on the laptop against the VPS, start a simulated training there with `runsnap.tensorboard()` and confirm:
  - the run appears within about 10 s of its first upload;
  - its curves grow without restarting, and it stays after finishing;
  - after deleting it in the MLflow UI it disappears and its `~/.cache/runsnap/tensorboard/<run_id>` folder is gone.

  Record the timings in the task.

  Recorded on 2026-10-08:
  - **Setup:**
    - The run was `smoke-sync` (`6f2012bf6bd04d5ebe7a3f873ae18998`): 150 steps at 1 step/s, logged from the laptop to the VPS.
    - Its hostname was faked as `vps-smoke-remote` so that the dashboard took the remote, cached path.
    - It was deleted with `MlflowClient.delete_run`, which the MLflow UI also uses.
  - **Uploads:** every upload was picked up without a restart.

    | Upload (remote `tb/` size) | Points shown in TensorBoard | Delay |
    |---|---|---|
    | 1st (1478 B) | 30 | 3.1 s |
    | 2nd (2918 B) | 60 | 10.4 s |
    | 3rd (4358 B) | 90 | 8.4 s |
    | 4th (5798 B) | 120 | 3.5 s |
    | Final, at writer exit (7300 B) | 150 | 8.2 s |
  - **After finishing:** the run was `FINISHED` in MLflow and still shown with 150 points 40 s after the writer exited.
  - **Deletion:** the run left TensorBoard 6.7 s after deletion, and its cache folder was gone.
  - **Shutdown:** Ctrl-C stopped the dashboard with no warnings and removed its assembled directory. `baseline` and `tuned` stayed shown throughout.
- [x] 7.2 Run `uv run ruff check`, `uv run ruff format --check`, `uv run ty check` and `uv run pytest`; verify all pass.

## 8. Deploy the dashboard on the VPS

These tasks need VPS access, and the user's explicit go-ahead before pushing to GitHub.

- [x] 8.1 After the user approves the push, push the commits, then install runsnap on the VPS with `uv tool install git+https://github.com/alirezanobakht13/runsnap@<commit>`, using `UV_TOOL_DIR=/opt/uv/tools`, `UV_TOOL_BIN_DIR=/usr/local/bin` and `UV_PYTHON_INSTALL_DIR=/opt/uv/python`. Verify that `runsnap tb --help` on the VPS lists `--artifact-root` and `--path-prefix`.
- [x] 8.2 Add `/etc/systemd/system/runsnap-tb.service` as in design.md's Migration Plan and enable it. Verify:
  - `systemctl is-active runsnap-tb`;
  - `curl -s http://127.0.0.1:6006/tb/data/runs` on the VPS lists `baseline` and `tuned`.
- [x] 8.3 Add `redir /tb /tb/`, `handle /tb/*` and `handle` blocks to the Caddyfile under the existing `basic_auth`, then validate and reload. Verify from the laptop:
  - `https://ae7pnfchd3dc.zozodogg.com/tb/` returns 401 without the login and 200 with it;
  - `/tb` redirects to `/tb/`;
  - the MLflow API still answers with the login.
- [x] 8.4 From the laptop, run a simulated training against the VPS and confirm on `https://ae7pnfchd3dc.zozodogg.com/tb/`:
  - it appears after its first upload, grows, stays after finishing, and disappears after being deleted in MLflow;
  - while it runs, `runsnap-tb` cgroup memory and CPU leave the VPS with free memory.

  Record the numbers in the task.
- [x] 8.5 Add `mlflow-gc.service` and a daily `mlflow-gc.timer` as in design.md, with `--older-than 7d`. Verify:
  - `systemctl list-timers` shows it;
  - before running gc by hand, confirm with the user that no deleted run other than a throwaway one exists;
  - then a manual run with `--older-than 0s` removes the throwaway run's folder under `/var/lib/mlflow/artifacts`.

### Deployment record, 2026-10-08

- **8.1:** runsnap 8ad8f5d is installed as a uv tool on the system `python3.12`; `/opt/uv/python` is unused because uv found no need for it. `runsnap tb --help` lists `--artifact-root` and `--path-prefix`.
- **8.2:** `runsnap-tb.service` is active. It follows design.md and adds `Group=mlflow` and `RestartSec=5` to match `mlflow.service`. `/tb/data/runs` listed `baseline` and `tuned`. Idle, it used 177 MiB.
- **8.3:** the old Caddyfile is kept at `/etc/caddy/Caddyfile.bak-2026-10-08`. Results:
  - `/tb/`: 401 without the login, 200 with it.
  - `/tb`: 302 to `/tb/`. Caddy runs `redir` before `basic_auth`, so the redirect is also answered without the login, and its target still asks for one.
  - MLflow API and UI: 200 with the login, 401 without it.
- **8.4:** `smoke-deploy`, 150 steps at 1 step/s, logged from the laptop.
  - **Appearance and growth:** the run appeared 4.2 s after its first upload. The later uploads were shown 2.1 s, 4.0 s, 4.3 s and 2.4 s after they landed: 30, 60, 90, 120, then 150 points.
  - **Finish and deletion:** after finishing, the run stayed `FINISHED` with 150 points. It left the dashboard 4.9 s after deletion, and its files stayed in the artifact folder until gc.
  - **Load:** `runsnap-tb` peaked at 177.4 MiB and used 1.29 s CPU over 208 s, 0.6% of one core. VPS available memory never fell below 807 MiB.
  - **Cache:** no download cache was created.
- **8.5:** `mlflow-gc.timer` is listed, with its next run at 00:00 UTC.
  - **Manual run:** the user agreed to remove all test data. The `vps-smoke` experiment was deleted, and its runs `baseline` and `tuned` left the dashboard about 6 s later.
  - **gc results:** `mlflow gc --older-than 0s` permanently deleted the 4 test runs and the experiment. It removed every file under `/var/lib/mlflow/artifacts`, but left the empty run folders, which were pruned by hand.
  - **End state:** the server holds only the empty `Default` experiment.

## Workflow follow-up

- Archive the change after review, with `/opsx:archive`.
