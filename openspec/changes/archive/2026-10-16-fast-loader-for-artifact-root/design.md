## Context

See proposal.md for why.

### The code today (`src/runsnap/_tb_fetch.py`, `src/runsnap/_cli.py`)

- **Showing a run:** with `artifact_root`, `_LiveWatch.link` makes
  `<logdir>/<name>/` and `_link_scalars` adds one link per scalar file in the
  run's folder, mirroring subfolders.
- **Refreshing:** `_refresh_pass` picks the runs that are due: those MLflow
  reports running, plus those in `_refreshing`, which were last refreshed while
  running or whose refresh failed. For each, it calls `_link_scalars` again,
  which only adds links for new shards.
- **Failures:** the artifact-root branch counts every due run as refreshed. A
  copy that raises would end the watch thread.
- **Download cache:** `fetch_run` already keeps grown files growing in place.
  It downloads to a staging folder, and `_extend_or_replace(target, downloaded)`
  appends the tail when the download begins with the cached bytes and moves the
  download over the target otherwise. Remote copies that are the same size or
  smaller are skipped.
- **Launch:** `tb` calls `_launch_tensorboard(..., load_fast=artifact_root is None)`.
  `load_fast=False` is its only use of that parameter.

### Loader behaviour, checked on 2026-10-08 (archived sync-tb-dashboard-with-mlflow design)

| Behaviour | Default (fast) loader | Python loader |
|---|---|---|
| File replaced by a longer copy under the same name | new points never shown | shown |
| Bytes appended to the same file | shown | shown |
| Several streams in one folder | all read | only the newest file by name |

### Memory, measured on 2026-10-16

- **Data:** the VPS's scalar files: 48 runs, 7,817 tags, 152 MiB.
- **Setup:** copied to a workstation, laid out as `--artifact-root` lays them
  out, and served by TensorBoard 2.21.0 with `tensorboard-data-server` 0.7.2.

| | Default loader | Python loader |
|---|---|---|
| Resident memory once loaded | 465 MiB (TensorBoard about 80, data server about 375) | 2,054 MiB |
| Time until every run is listed | under 10 s | about 240 s |
| Tag list request | 0.13 s | 0.3–0.7 s while loading |

On the VPS, the Python loader reached 1.0 GiB of RAM plus 4.1 GiB of swap with
109 runs.

### Disk

- **Location:** the dashboard's temporary directory is under `PrivateTmp`. On
  the VPS, `/tmp` is on the ext4 root filesystem (60 GB, 40 GB free), not
  tmpfs.
- **Effect:** copies cost disk space and reclaimable page cache, not process
  memory.

## Goals / Non-Goals

**Goals:**

- The VPS dashboard runs the default loader, using its documented growth
  behaviour, without touching MLflow's folder.
- Reuse the cache's extend-or-replace rule, so both views follow one rule for
  grown files.

**Non-Goals:**

- **Media under `--artifact-root`:** the default loader could read it, but
  nobody needs it on the VPS yet.
- **A `--samples_per_plugin` passthrough or other TensorBoard tuning:** only
  if memory becomes a problem again.
- **Live runs under `--artifact-root`:** each run keeps its single source, as
  today.
- **Changes to the download path, the writer, or the deployment units.**

## Decisions

### Each shown run gets a folder of copies instead of a folder of links

`<logdir>/<name>/` stays a real directory that mirrors the run's `tb/`
subfolders, as today. It now holds a copy of each scalar file in place of a
link. `_link_scalars` becomes `_copy_scalars(folder, copies)`, called from the
same two places: `link` and the artifact-root branch of `_refresh_pass`.

For each scalar file under `folder`:

| Source size vs copy | Action |
|---|---|
| copy missing | stage the source, then move it into place |
| larger | stage the source, then `_extend_or_replace(copy, staged)`: append the tail if the source begins with the copy's bytes, otherwise move the staged file over the copy |
| equal or smaller | skip |

- **Why staging:** `_extend_or_replace` moves its second argument when the
  prefix differs. Passing MLflow's own file would move it out of MLflow's
  folder, so the source is first copied (`shutil.copyfile`) into a
  `TemporaryDirectory(prefix=".copy-", dir=copy.parent)`.
- **Staging is invisible to TensorBoard:**
  - The staged file is named without `tfevents`, and both loaders take only
    names containing `tfevents` as event files. A test pins this down.
  - The staging folder is on the copy's filesystem, so moving the file into
    place is atomic.
- **Reading a file MLflow is replacing:** MLflow moves each upload over the
  final name, so a read sees either the whole old file or the whole new one.
  A source that is smaller than the copy is skipped, as the cache skips a
  shorter remote copy.
- **Cost:**
  - A grown shard is read in full once per pass while its run is due. Scalar
    shards roll at 1 MiB, so each pass reads at most about 1 MiB per running
    run, plus a `stat` per scalar file.
  - The first pass after startup copies every shown run's scalar data: 152 MiB
    on the VPS today, a few seconds of disk I/O.

Alternative: hard links. MLflow replaces files rather than rewriting them,
so a hard link keeps pointing at the old content. This is the same failure the
default loader has with symlinks.

Alternative: stage outside the logdir, for example in a sibling folder of the
`TemporaryDirectory`. This would also work, but changes `assemble_logdir`'s
layout for no gain once the staged name is not an event file name.

Alternative: rewrite the copy in place (`write_bytes`) when the prefix
differs. A reader partway through the old bytes would then read mismatched
bytes. A move gives it a clean old file to finish instead.

### A refresh failure is per run, reported once, and retried

`_refresh_pass` tracks failures through `_refresh_failing` and keeps a
failed run in `_refreshing`. The artifact-root branch now uses that machinery
too:

- **Each due run:** `_copy_scalars` runs inside a `try`. On failure:
  - a warning names the run, unless the run is already in `_refresh_failing`;
  - the run is left out of `fetched`.
- **What follows from that:** the existing bookkeeping keeps the run in
  `_refreshing`, so it is tried again on the next pass even after it ends, and
  keeps its warning quiet until a copy succeeds.
- **The first copy, in `link`:** it can fail too, for example when a file
  vanishes between listing and reading because gc removed a deleted run. The
  run is then shown with what was copied, and added to `_refreshing` and
  `_refresh_failing` after one warning. The next pass retries it like any
  failed refresh. This keeps a single unreadable run from ending startup or
  the watch thread.

Alternative: let the exception propagate, as `_link_scalars` effectively did.
Links could not fail on file contents, but copies can, and an exception in
`_pending_pass` would end the watch thread.

### The default loader with `--artifact-root`

`tb` stops passing `load_fast`. `_launch_tensorboard` loses the parameter and
its `--load_fast=false` branch, since nothing else uses them. The comment in
`tb` about the Python loader goes too.

- **What changes:** the README's limit that only a run's newest scalar shard
  is followed no longer applies. The default loader reads every event file in
  a folder.
- **What stays:** under `--artifact-root`, `--media` is still refused, as
  before.

### Removal and exit need no new code

- **Removal:** `_remove` already deletes the run's folder in the logdir with
  `shutil.rmtree`, which now removes copies instead of links. MLflow's folder
  is still never touched.
- **Exit:** the logdir is a `TemporaryDirectory`, so the copies go when the
  command exits. On the VPS, `PrivateTmp` also clears it if the process dies.

### Tests

- **Copies instead of links:** existing artifact-root tests that assert
  `readlink()` or `is_symlink()` change to assert ordinary files whose bytes
  match the folder's.
- **In-place growth:** after a refresh, a grown file's copy keeps its inode
  and equals the server's file. This is the property the default loader needs.
- **Replaced file:** `test_a_scalar_file_replaced_by_a_longer_copy_shows_its_new_steps`
  needs a refresh pass between the upload and the reload, since the copy no
  longer follows MLflow's file by itself.
- **New tests:**
  - an earlier shard growing after a later one appeared;
  - a failing copy warned once, the run kept, and the copy retried;
  - a staging folder that never shows up as an event file;
  - the CLI launching TensorBoard without `--load_fast` under
    `--artifact-root`.

## Risks / Trade-offs

- **[Memory still grows with what is shown]:**
  - The default loader's data server used about 375 MiB for 152 MiB of scalar data.
  - It keeps a bounded sample of points per series, so memory follows the
    number of tags more than run length.
  - With MLflow's share, the VPS leaves the dashboard about 1.2 GiB, a few
    times today's data.
  - → `gc` and deleting old runs still help. A sample-size option can follow
    if the dashboard nears that limit.
- **[A copy that MLflow rewrites with different leading bytes is replaced, not
  extended, so the default loader keeps the old file]:**
  - Event files are only appended to, so this needs a writer that rewrites a
    shard. The cache has the same limit today.
  - → The run shows its earlier data. A restart picks up the new file.
- **[Copies duplicate scalar data on disk]:**
  - The archived design avoided copies because the VPS had little storage to
    spare. Scalar data is now 152 MiB, against 40 GB free, and media and
    other artifacts are still not copied.
  - → Accepted.
- **[A run is no longer followed after its final refresh]:** a links view
  would show a late replacement of an ended run's file. The copies view
  matches the cache view instead.
  - → A restart, or the next startup, shows the late data.
- **[A staged file under a name TensorBoard might someday treat as events]:**
  → A test fails if the staged name is listed as a run or an event file.

## Migration Plan

1. Implement and test locally.
2. Push to GitHub only after the user's explicit go-ahead. `./vps.sh update`
   installs from GitHub.
3. Run `./vps.sh update <ref>`. It reinstalls runsnap and restarts only
   `runsnap-tb`. MLflow and Caddy keep running, and training clients keep
   uploading.
4. Check:
   - `/tb/data/runs` lists the active runs;
   - a running run's curves grow;
   - the service's memory settles, recorded with `./vps.sh status`;
   - the tag list responds within a second.
5. **Rollback:** run `./vps.sh update <previous commit>`.
