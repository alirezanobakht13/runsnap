## Context

See proposal.md for the motivation; specs/tensorboard-viewing/spec.md holds the
behaviour.

### How the code works today

- **The `tb` command** (`src/runsnap/_cli.py`) resolves runs once with
  `_tb_runs`. It passes `follow=` a query callable only when no run ids or
  names are given, and exits before launching TensorBoard when the assembled
  directory starts out empty.
- **`assemble_logdir`** (`src/runsnap/_tb_fetch.py`):
  - links each run into a temporary directory;
  - uses the writer's directory when `_live_logdir` finds one on this host;
  - otherwise fetches the run into the cache with `fetch_run`, through
    `_fetch_runs`;
  - leaves out runs without event files.
- **`_LiveWatch`**:
  - polls every `LIVE_POLL_SECONDS` (5 s);
  - re-points gone live links at the cache;
  - re-runs the followed query and links runs carrying
    `runsnap.tb.local_dir` that are not in `handled`;
  - only runs when there are live links or a query to follow.
- **Runs that are never re-checked:**
  - Startup seeds `handled` with every shown or tagged run.
  - A run gets into `handled` before it is linked, so a run with no data yet
    is never retried.
  - A cached run is never fetched again.
- **`fetch_run`:**
  - downloads every event file whose cached size differs from the remote
    size;
  - stages each download in the cache's filesystem and moves it over the
    cached file with `Path.replace`;
  - keeps a `scalars/` view of per-file links to the light files.

### The writer (`src/runsnap/_tensorboard.py`)

- It sets `runsnap.tb.logdir`, `runsnap.tb.local_host`, and
  `runsnap.tb.local_dir` on entry, then syncs every 30 s.
- The open scalar shard is uploaded whenever it has grown. A media shard is
  uploaded once it seals.
- Shard files are named `events.out.tfevents.<creation time>.<host>.scalars.N`
  and `.media.N`, so scalar shards sort in the order they were created.
- The first upload therefore lands about 30 s after the tags.

### MLflow 3.16 on the VPS

- **On disk:** with `--artifacts-destination /var/lib/mlflow/artifacts`, a
  run's `artifact_uri` is `mlflow-artifacts:/<experiment>/<run>/artifacts`,
  stored at `/var/lib/mlflow/artifacts/<experiment>/<run>/artifacts`. This was
  checked on the server.
- **Uploads:** each upload is written to a temporary file and moved over the
  final name (`os.replace`, `mlflow/store/artifact/local_artifact_repo.py:119`).
  A growing shard is therefore a new file under the same name after every
  upload.

### Checked on 2026-10-08

TensorBoard 2.21.0 with `tensorboard-data-server` 0.7.2, using scripts in a
scratch directory:

| Behaviour | Fast loader (default) | Python loader (`--load_fast=false`) |
|---|---|---|
| File replaced by a longer copy under the same name | new points never shown | shown |
| Bytes appended to the same file | shown | shown |
| Run link removed | run disappears | run disappears |
| Run link added | run appears | run appears |
| New file in a linked folder | read | read |
| Several streams in one folder | all read | only the newest file by name is followed |

MLflow 3.16, local SQLite store:
- `search_runs` with `filter_string="attributes.run_id IN (...)"` and
  `run_view_type=ACTIVE_ONLY` returns status for every listed run that is not
  deleted.
- Runs that were deleted directly, and runs of a deleted experiment, are
  absent, even when the deleted experiment's id is passed.
- Restoring the experiment brings its runs back.

### The VPS

- 2 vCPU, about 1 GiB of free RAM.
- MLflow listens on `127.0.0.1:5000` with no login of its own.
- Caddy provides TLS and a basic-auth login in front of it.

## Goals / Non-Goals

**Goals:**

- One watch pass keeps every way a run can be shown in step with MLflow: live
  directory, download cache, and artifact folder.
- `--artifact-root` writes no event bytes anywhere. Only links are created, in
  the assembled temporary directory.
- Server load per pass is bounded:
  - one search per batch of up to 200 watched runs;
  - the followed query, as today;
  - artifact listings only for runs still `RUNNING` and shown from the cache.

**Non-Goals:**

- Media under `--artifact-root`. The layout below leaves a place for it.
- Pruning cache folders of runs deleted while no dashboard showed them.
- Ranged or partial downloads. A grown file is downloaded whole, and its
  scalar shards are capped at 1 MiB.
- A configurable refresh interval or a flag to turn refreshing off.
- Deleting or changing anything in MLflow's artifact folder. Disk space there
  is freed by `mlflow gc`.
- Detecting runs that stay `RUNNING` after their process was killed with
  `SIGKILL`.

## Decisions

### Each pass of the watch thread runs five steps, in order

1. **Check the watched runs** in one search. A run missing from the results
   is deleted: remove it. Every other run gets its current status recorded.
2. **Re-point gone live links**, as today.
3. **Re-run the followed query**, if there is one. Runs carrying
   `runsnap.tb.local_dir` that are neither shown nor handled join `pending`.
4. **Try to link pending runs.** A run with data is linked. A run with no data
   stays pending while `RUNNING`. One that has ended gets a last try and is
   then dropped.
5. **Refresh `RUNNING` runs:**
   - cached runs are fetched again;
   - artifact-folder runs get links for any new scalar files;
   - a run that ended since the last pass gets one final refresh.

Removing deleted runs first means no pass fetches a run it is about to drop.
The status read in step 1 decides which runs steps 4 and 5 touch, so no
per-run `get_run` is needed.

The thread now always runs while TensorBoard is open, because deletion and
refresh apply to every selection. `test_context_of_cached_runs_starts_no_watch_thread`
is replaced by a test that the thread runs and is joined on exit.

Alternative: separate threads for checking, following, and refreshing. They
would share the link directory and need a lock. Rejected, as in the
follow-new-tb-runs change.

### Watched runs are checked with one search per batch

- **Which runs are watched:** shown runs, pending runs, and, for a named
  selection, every named run. That last part lets a restored named run come
  back.
- **The call:**
  `search_runs(<their experiment ids>, filter_string="attributes.run_id IN (...)", run_view_type=ACTIVE_ONLY)`,
  following page tokens.
- **Batches:** run ids are sent in batches of up to 200, which stays within
  SQLite's bound-parameter limits. Fifty shown runs cost one search.
- **Failure:** a failing search is reported once per run of failures, as the
  follow pass does, and changes nothing that pass.
- **Restored runs:** a restored run in a followed query matches the query
  again and is not handled, so step 3 finds it. A restored named run reappears
  in step 1's results and goes back to pending.

Alternative: `get_run` per watched run each pass. That is one request per run
every 5 s, which the bounded-queries requirement rules out.

### Removing a run frees its name and its cache

Removal unlinks the run's entry in the assembled directory. It drops the run
from `handled`, `pending`, the refresh set, and the shown names, and in cache
mode deletes `<cache>/<run_id>`. A run that comes back is named as a late run
would be. Removal never touches MLflow's artifact folder.

### `fetch_run` extends grown files instead of replacing them

For each event file, `fetch_run` compares the remote `file_size` from the
listing with the cached size:

| Remote size vs cached | Action |
|---|---|
| equal | skip, as today |
| smaller | skip; the remote copy was caught mid-upload |
| larger | download to the staging directory, as today, then: if the download begins with the cached bytes, append only the tail to the cached file; otherwise move it over the cached file, as today |
| not cached yet | download and move into place, as today |

Why this works:
- The fast loader keeps its open file and sees the appended bytes (see the
  checks above).
- An interrupted download never touches the cached file.
- An interrupted append leaves a prefix of the remote file, which the next
  pass completes because the prefix check passes.
- A reader that meets a partially appended record waits for the rest, as it
  does for a file a writer is still writing.

`fetch_run` already adds `scalars/` links for light files it has not seen, so
refreshing also links new shards.

Alternative: replace as today and launch TensorBoard with `--load_fast=false`.
Rejected for downloads: the Python loader follows only the newest file per
folder, and the `--media` view keeps both streams in one folder.

### `--artifact-root` links each run's scalar files where MLflow keeps them

- **Finding the folder:**
  - The run's `artifact_uri` must use the `mlflow-artifacts` scheme.
  - Its path, with leading slashes removed, is joined to the root, and `tb`
    is added.
  - The resolved folder must lie inside the root.
  - Any other run is reported as a warning and treated as handled.
- **The links:** `<logdir>/<name>/` is a real directory holding one link per
  scalar file. Subfolders are mirrored, so nested runs written by
  `add_scalars` keep their place. This is the cache's `scalars/` layout
  without the copies.
- **Updates:** steps 4 and 5 add links for scalar files that appear later.
  MLflow replaces files under the same name, so existing links keep pointing
  at the current data.
- **Live runs:** under `--artifact-root`, `_live_logdir` is not consulted.
  Each run has one source, and a live directory holds both streams, which
  the Python loader cannot follow.
- **Download cache:** never created or written in this mode.

Alternative: link the run's whole `tb/` folder. That would bring in media,
which the Python loader cannot read beside a growing scalar shard.

Alternative: point TensorBoard at the artifact root itself. Runs would be
named by experiment and run ids, every run would be shown regardless of the
selection, and media would be included.

### Room for media under `--artifact-root`

Media can later get its own subfolder per run, linking the `.media.N` files.
TensorBoard would show it as a separate run, `<name>/<subfolder>`, and each
folder would still hold a single stream for the Python loader. The subfolder
name must not clash with nested runs. Until then, `tb` refuses `--media` with
`--artifact-root` before resolving anything.

### The Python loader is used with `--artifact-root`

`tb` launches TensorBoard with `--load_fast=false` when `--artifact-root` is
given. The Python loader re-opens files by name, so it follows MLflow's
replacements. Each run folder holds only scalar shards created in order, so
"newest file per folder" is always the open shard. Without `--artifact-root`,
TensorBoard keeps its default loader.

### TensorBoard opens even when nothing is shown yet

Whether `tb` starts TensorBoard with nothing to show:

| Selection | Nothing shown at startup |
|---|---|
| followed query | launch anyway and print that no runs were found yet |
| named runs, some pending | launch |
| named runs, none pending | report and exit, as today |

A long-running service would otherwise exit on an empty server and stay down.

### `--path-prefix` is passed through

`path_prefix: str | None` is named `("--path-prefix", "--path_prefix")`, like
`--bind_all`, and passed to TensorBoard as `--path_prefix`. TensorBoard
validates the value.

## Risks / Trade-offs

- **[Lost tail with the Python loader]** The Python loader may move to a new
  scalar shard before reading the previous shard's last upload. The writer
  uploads in sorted order, so the old shard's final copy normally lands first.
  If that upload fails and is retried after the new shard lands, the tail
  stays hidden until the dashboard restarts. → Accept and document.
- **[Runs stuck as RUNNING]** A run killed with `SIGKILL` stays `RUNNING`, so
  it is listed (cache mode) or rescanned on disk (artifact-root mode) on every
  pass, forever. → Accept. It is one small request or directory walk per such
  run. Revisit if it shows.
- **[Listing load from a laptop]** Each `RUNNING` cached run costs an artifact
  listing every 5 s, while the writer uploads every 30 s. → Accept for now. A
  slower refresh cadence can be added without changing the specs.
- **[Python loader cost]** The Python loader is slower and heavier than the
  fast loader. → Measure RAM and CPU on the VPS during deployment, and stop if
  TensorBoard does not fit in the free memory.
- **[MLflow layout changes]** `--artifact-root` reads MLflow's on-disk layout.
  → It is an explicit option. A run that does not map into the root is
  reported, not guessed at.
- **[MLflow's soft delete]** A deleted run leaves the dashboard, but its files
  stay on the server until `mlflow gc` runs. → A daily gc timer is part of the
  deployment.

## Migration Plan

### Code

- No data migration. Existing cache folders keep working.
- Cached files are now appended to rather than replaced. That gives the same
  bytes for files that only grow.

### VPS deployment

This is done after the code is merged and pushed. Pushing needs the user's
go-ahead.

1. **Install runsnap** with
   `uv tool install git+https://github.com/alirezanobakht13/runsnap@<commit>`,
   using the same `UV_TOOL_DIR=/opt/uv/tools`, `UV_TOOL_BIN_DIR=/usr/local/bin`
   and `UV_PYTHON_INSTALL_DIR=/opt/uv/python` as MLflow.
2. **Add `runsnap-tb.service`:**
   - `User=mlflow`, `PrivateTmp=yes`;
   - `Environment=MLFLOW_TRACKING_URI=http://127.0.0.1:5000`;
   - `ExecStart=/usr/local/bin/runsnap tb --artifact-root /var/lib/mlflow/artifacts --host 127.0.0.1 --port 6006 --path-prefix /tb`;
   - `After=` and `Wants=mlflow.service`, `Restart=on-failure`.

   `PrivateTmp` means systemd removes the assembled directory whenever the
   service stops, including after a crash.
3. **Caddy:** in the existing site block, under the same `basic_auth`, add
   `redir /tb /tb/`, `handle /tb/* { reverse_proxy 127.0.0.1:6006 }`, and
   `handle { reverse_proxy 127.0.0.1:5000 }`. MLflow serves nothing under
   `/tb`; this was checked on 2026-10-08, when `/tb/` returned MLflow's 404.
4. **Add `mlflow-gc.service` and `mlflow-gc.timer`** (daily, `User=mlflow`,
   `MLFLOW_TRACKING_URI=http://127.0.0.1:5000`):
   `mlflow gc --backend-store-uri sqlite:////var/lib/mlflow/mlflow.db --artifacts-destination /var/lib/mlflow/artifacts --older-than 7d`.
   Verify on a throwaway run that gc removes its artifact folder.
5. **Rollback:** disable the two services and the timer, and remove the
   `/tb` handles from the Caddyfile. Nothing else depends on them.

## Open Questions

- **gc grace period:** 7 days is a proposed default. It can change at
  deployment without touching the specs.
- **Batch size:** 200 ids per search can be tuned once real runs show what
  the server handles comfortably.
