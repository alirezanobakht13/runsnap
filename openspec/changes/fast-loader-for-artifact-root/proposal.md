## Why

The `runsnap tb --artifact-root` dashboard on the VPS has become unusable. With
`--load_fast=false`, TensorBoard's Python data loader holds every loaded point as
a Python object. On 2026-10-16, 109 runs, with 327 MiB of scalar files, grew the
process to about 5 GiB on a host with 2 GiB of RAM. The host swapped
constantly, the Scalars page waited 19.5 s for its tag list, and MLflow's
workers on the same host restarted every few minutes. Deleting runs only
slows the growth, because new runs keep arriving. The Python loader was chosen
only because MLflow stores each upload by replacing the file, and the default
(fast) loader does not see replaced files. It does see bytes appended to a file
it already reads, which the download cache already relies on.

## What Changes

- With `--artifact-root`, each shown run's folder in the dashboard holds the
  dashboard's own copy of each scalar file instead of a link to MLflow's file.
  - **Startup:** a run's scalar files are copied when the run is first shown.
  - **Refresh:** on the passes that refresh a run today (while MLflow reports
    it running, and once more after it ends), a grown file gains only its new
    bytes, so the copy grows in place. New shards are copied as they appear.
  - **Location:** copies live in the dashboard's temporary directory and are
    removed when the command exits or the run leaves the dashboard. MLflow's
    folder is still never written, and nothing is downloaded or cached.
  - **Failures:** a copy that fails is reported once while the failure lasts
    and retried on the next pass, and the run keeps the data it had.
- With `--artifact-root`, TensorBoard runs with its default data loader;
  `--load_fast=false` is no longer passed.
- Because the default loader reads every file in a run's folder, the stated
  limit that only a run's newest scalar shard is followed goes away.
- The README's `--artifact-root` paragraph describes the copies, their disk
  cost, and the default loader.
- Deployment: after the user approves pushing to GitHub, update runsnap on the
  VPS and record the dashboard's memory and response times.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `tensorboard-viewing`:
  - Runs viewed from an artifact folder are shown from the dashboard's own
    copies of their scalar files, read with TensorBoard's default loader.
  - Those copies stay current while a run runs, get one final refresh after it
    ends, and every scalar shard is shown.
  - A failing refresh of such a run is reported once and retried.

## Impact

- **`src/runsnap/_tb_fetch.py`:**
  - `_LiveWatch.link` and `_refresh_pass` copy scalar files under an
    artifact root instead of linking them.
  - A copy refresh can fail per run without stopping the watch thread.
- **`src/runsnap/_cli.py`:** `_launch_tensorboard` loses its `load_fast`
  parameter, and `tb` no longer passes it.
- **Tests:**
  - The artifact-root tests in `tests/test_tb_fetch.py` and
    `tests/test_tb_cli.py` assert copies instead of links.
  - New tests cover in-place growth, an earlier shard growing, and a failing
    refresh.
- **`README.md`:** the `--artifact-root` paragraph.
- **Unchanged:** no new dependencies. The writer (`runsnap.tensorboard()`),
  its tags, and the artifact layout stay as they are, so training projects
  using runsnap need no update. Only the VPS dashboard service is reinstalled.
  The MLflow server is not touched.
- **Disk:** the dashboard's temporary directory holds a copy of the shown runs'
  scalar files: 152 MiB on 2026-10-16, against 40 GB free. Media and other
  artifacts are still not copied.
