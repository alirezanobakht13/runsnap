## Why

`runsnap tb` cannot serve as a long-running dashboard next to a remote tracking
server. A run logged on another host is downloaded once and never refreshed. A
run that starts on another host after the dashboard opened is found before its
first upload, judged empty, and never retried. A deleted run stays shown, and
its cached files stay on disk. On the tracking server itself, the cache also
duplicates event files that already sit on the same disk. The new VPS hosting
MLflow has little storage to spare, so a dashboard there must stay current
without keeping a second copy of the data.

## What Changes

- While TensorBoard is open, every shown run is re-checked each pass:
  - A run deleted in MLflow, directly or through its experiment, leaves the
    dashboard. A run restored afterwards comes back if it is still selected.
  - A finished, failed, or killed run stays shown.
- A selected run that has begun writing TensorBoard events, but whose data is
  not available yet, is retried on every pass until it has data or ends. This
  covers runs starting on another host, both when they are found later by a
  followed query and when they are named at startup.
- Runs shown from the download cache stay current. While a run is `RUNNING`:
  - Each pass fetches event files that have grown.
  - Only the new bytes are appended to the cached file, which TensorBoard's
    default data loader can follow.
  - A remote copy shorter than the cached one is ignored.
  - A run gets one final fetch once it ends and is not fetched after that.
- Deleting a run from the dashboard also removes its files from the download
  cache.
- New `--artifact-root PATH`: `runsnap tb` links runs straight to the tracking
  server's artifact folder on this disk instead of downloading them.
  - Runs are never downloaded and no cache is used. New shards are linked as
    they appear.
  - Each upload replaces a file under its own name. TensorBoard runs with its
    Python data loader, which re-reads replaced files.
  - Scalars, records, text, and HParams are shown. `--media` is refused with
    this option for now. The design keeps a layout open for adding media later
    without copies.
- New `--path-prefix PREFIX`, passed to TensorBoard's `--path_prefix`, so a
  reverse proxy can serve the dashboard under a path such as `/tb/` next to
  MLflow.
- When following a query (no run ids or names given), `runsnap tb` opens
  TensorBoard even when nothing matches yet and waits for runs. A selection of
  named runs that matches nothing still exits as today.
- The README's TensorBoard section describes the new behaviour and options.
- Deployment on the VPS:
  - a `runsnap tb --artifact-root` service behind the existing Caddy login at
    `/tb/`
  - a scheduled `mlflow gc`, because deleting a run in MLflow frees no disk
    space by itself

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `tensorboard-viewing`:
  - Shown runs are kept in step with MLflow: added once they have data,
    refreshed while running, kept after they end, removed when deleted.
  - The cache appends to grown files and drops the files of deleted runs.
  - Runs can be viewed straight from a local artifact folder.
  - TensorBoard can be served under a path prefix.
  - A followed query opens TensorBoard before any run matches.
  - The query bound counts the per-pass check of shown runs.

## Impact

- `src/runsnap/_tb_fetch.py`:
  - `fetch_run` appends instead of replacing grown files.
  - `_LiveWatch` adds a per-pass check of shown runs, retries pending runs,
    refreshes running runs, and removes deleted runs.
  - A second link source covers a local artifact folder.
- `src/runsnap/_cli.py`:
  - `tb` gains `--artifact-root` and `--path-prefix` and refuses
    `--artifact-root` together with `--media`.
  - With `--artifact-root`, TensorBoard is launched with `--load_fast=false`.
  - A followed query no longer exits when the startup selection is empty.
- `tests/test_tb_fetch.py`, `tests/test_tb_cli.py`:
  - New tests for refresh, append, deletion, pending runs, and artifact-root
    links.
  - `test_context_of_cached_runs_starts_no_watch_thread` and
    `test_a_pass_never_fetches_a_shown_run_again` change, because the watch
    thread now always runs and running runs are fetched again.
- `README.md`: the TensorBoard section.
- No new dependencies. No change to the writer (`runsnap.tensorboard()`), its
  tags, or the artifact layout.
- Each pass adds one MLflow search covering all shown runs, plus artifact
  listings for runs still `RUNNING` and shown from the cache.
- `--artifact-root` reads MLflow's on-disk layout: `mlflow-artifacts:/<path>`
  under `--artifacts-destination`. It only applies on the tracking server's own
  machine.
- On the VPS:
  - runsnap is installed from GitHub.
  - A systemd unit, a Caddy `/tb/` route, and a `mlflow gc` timer are added.
- Pushing the implementation commits to GitHub needs the user's explicit
  go-ahead.
