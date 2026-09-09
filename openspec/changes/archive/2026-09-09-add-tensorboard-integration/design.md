## Context

See proposal.md — Why. The constraints that shape the approach, measured in the installed environment:

- MLflow 3.16.0's TensorBoard support is two framework-locked paths. `mlflow/tensorflow/__init__.py:1403` uploads a log directory only for `tf.keras.Model.fit`; `mlflow/pytorch/_pytorch_autolog.py:29` monkeypatches `torch.utils.tensorboard.FileWriter.add_event` to copy scalars into MLflow metrics and never uploads the event file. Neither is reusable.
- `mlflow server` defaults to `--serve-artifacts` (`mlflow/utils/cli_args.py:225`), which makes the artifact root `mlflow-artifacts:/` (`mlflow/store/tracking/__init__.py:19`). TensorBoard's filesystem layer (`tensorboard/compat/tensorflow_stub/io/gfile.py:659`) registers local paths and `s3://` only, so it cannot read that URI. Event files must be fetched to local disk before TensorBoard sees them.
- MLflow artifact upload is a whole-file PUT. There is no append.
- `MlflowClient.list_artifacts` returns `file_size`, so remote growth is detectable without downloading.
- `tensorboardX` 2.6.5 costs 344 KB; its `numpy`, `protobuf`, and `packaging` requirements are already installed by MLflow, and its `protobuf>=3.20` is compatible with MLflow's `protobuf<8,>=3.12.0`.
- TensorBoard merges every `events.out.tfevents.*` in one directory into a single run, and picks up files added later on its reload interval.
- TensorBoard keeps 10 images and 500 histogram entries per tag by default (`DEFAULT_TENSOR_SIZE_GUIDANCE` in `tensorboard/backend/event_processing/data_ingester.py`), sampled across the whole run.
- Measured cost of a 100-epoch run: scalars 2.35 MB, histograms 39.79 MB at the writer's default bins, images 87–157 MB. Media dominates by two orders of magnitude.

## Goals / Non-Goals

**Goals:**

- One added public symbol, `exp_track.tensorboard()`, with the writer behind it being the standard summary-writer surface rather than a new API.
- Viewing curves across many runs costs kilobytes per run, not hundreds of megabytes.
- A preempted job keeps successfully uploaded data, with periodic background sync reducing unsynced data without signal handling.

**Non-Goals:**

- Copying metrics into MLflow. Runs stay queryable through params and whatever metrics the user chooses to log; deriving summary metrics from event files is not part of this change.
- Following a running experiment live. `exp-track tb` assembles a log directory once per invocation.
- Reading MLflow metric history into TensorBoard, or reading log directories written by MLflow's own autolog.
- Writing event files directly to a remote artifact store.

## Decisions

### Two writers in one directory, joined by attribute delegation

`exp_track.tensorboard()` yields a facade holding two `tensorboardX.SummaryWriter` instances over the same log directory, distinguished by `filename_suffix`. `add_scalar` and `add_text` go to the light writer; every other method goes to the media writer, resolved through `__getattr__` so that the full writer surface — present and future — is available without restating it.

`add_histogram` is the one method spelled out explicitly, to change the default `bins` from the writer's `'tensorflow'` (9947 bytes per entry, ~775 exponential buckets) to `30` (587 bytes per entry). It is a keyword with a different default, so callers override it per call as they would any other.

Alternatives considered. A single writer keeps the code shorter but makes the split impossible, and the split is the entire reason viewing 20 runs is affordable: measured, a run's scalar file is 2 KB against 4.1 MB of media in the same directory. Subclassing `SummaryWriter` would inherit the surface for free but would also inherit its constructor and file management, which is precisely what needs to differ.

`add_scalars` and `add_hparams` create nested run directories by the underlying writer's design. They are delegated like anything else and their files sync normally; they appear in TensorBoard as nested runs. `add_hparams` is redundant here in any case, since hyperparameters are already recorded by `log_params`.

### Media rolls on size, in the write path

The facade tracks the open media shard's size and, when it exceeds a threshold, closes it and opens the next one with an incremented index in the filename suffix. Rolling happens inside the delegated call, under a lock, so no write can land in a file that is being closed.

The shard index must be in the suffix: the writer names files `events.out.tfevents.<unix_seconds>.<hostname><suffix>` with no pid and no counter, so two shards sealed within the same second would otherwise overwrite each other.

An 8 MB threshold gives roughly 19 shards for a 150 MB run — negligible upload overhead. It controls individual shard size, not total data at risk: several sealed shards can await upload. Size is checked with `os.path.getsize`; the writer thread drains its queue eagerly, so on-disk size tracks written size closely enough to roll within a shard's tolerance.

### One background sync thread, one rule

A daemon thread waits for an interval (30 seconds by default) between sync passes and uploads every file in the log directory whose size differs from what was last successfully uploaded, **except the currently-open media shard**. That single rule produces the intended behavior: a successfully uploaded sealed shard is skipped thereafter because its size never changes again; the scalar file is re-uploaded whenever it grows; the open shard is never re-uploaded while it grows. Failed uploads are retried on later passes.

Sealing makes a shard eligible for a subsequent sync pass; it does not trigger an immediate upload or wait for one to succeed. Context-manager exit closes the writers and attempts all outstanding uploads. The sync interval is not a durability deadline: upload time, retries, and outages can extend the backlog beyond one interval.

Alternatives considered. Uploading opportunistically from the write path avoids a thread but blocks training on network I/O and stops syncing entirely when logging pauses. Two separate cadences for scalars and media would express the same policy with more machinery.

### No signal handling

`SIGINT` becomes `KeyboardInterrupt`, so the context manager's `__exit__` already flushes and uploads on Ctrl-C. `SIGTERM` runs neither `__exit__` nor `atexit`, so surviving preemption would require installing a handler — invasive for a library, prone to being overwritten by a launcher that installs its own, and in conflict with this project's explicit-opt-in contract.

No signal handler is installed. After an unannounced kill, successfully uploaded shards and scalar updates remain available, but the open shard, pending sealed shards, and unsynced scalar updates can be lost. There is no fixed bound on the unsynced backlog. A launcher that needs cleanup on `SIGTERM` must arrange for the context manager to exit; calling the writer's `flush()` alone only flushes to local disk.

### Whole-file download with a size-keyed cache

The viewer downloads whole event files into a cache keyed by run id, skipping any file already held at its current remote size. Sharding makes this efficient without further work: a sealed shard's size never changes, so it is fetched exactly once and never revisited.

Alternatives considered. The proxy does honour HTTP `Range` — a `Range: bytes=1000-` against `/api/2.0/mlflow-artifacts/artifacts/...` returns `206 Partial Content` with a correct `content-range`, and event files tolerate truncation, so appending fetched bytes would be safe. But immutable shards already remove the case that byte-range fetching exists to solve, leaving only the growing scalar file, which is ~2 MB. The offset bookkeeping is not worth it.

Downloads land in a temporary file that is renamed into place, so an interrupted invocation cannot leave a truncated file that a later invocation would trust.

### A directory of symlinks, named by run

The viewer assembles a temporary directory of symlinks into the cache, one per selected run, named by the run's MLflow run name; colliding names take a short run-id suffix. TensorBoard uses the directory name as the run name, so runs appear under the names they were given. TensorBoard is then launched on that directory as a subprocess in the foreground.

Alternatives considered. `--logdir_spec` maps names to paths without symlinks, but TensorBoard's own help calls it discouraged and warns that some features do not work with it. Copying instead of symlinking would duplicate the cache. Encoding hyperparameters into the directory name would make TensorBoard's regex run filter act as a hyperparameter filter, but it requires a naming-template mechanism that nothing else needs.

### Both packages are direct dependencies

`tensorboardx` and `tensorboard` are ordinary dependencies rather than an optional extra. The extra would save 134 MB in an environment that only trains and never views, at the cost of a second install mode, an error path, and documentation for both. Against MLflow and a framework already in the environment, the saving does not justify the concept.

## Risks / Trade-offs

- **An unannounced kill can lose all unsynced data** → Successfully uploaded data survives. The open media shard, sealed shards awaiting successful upload, and unsynced scalar updates remain at risk. The shard threshold and sync interval do not impose a fixed loss bound, especially during slow or failed uploads.
- **Media that dwarfs the shard threshold in a single write** — one very large image or mesh — produces a shard larger than the threshold → Accepted. Rolling is checked after each write, so a shard is never split mid-record; a single oversized entry simply makes one oversized shard.
- **MLflow's client retries 7 times with backoff (~4 minutes) on a failed request** → The sync runs on a background thread, so a stalled upload delays syncing rather than blocking training. Uploads are attempted again on the next tick.
- **TensorBoard shows only 10 images per tag** → Not addressed in code; logging 100 grids to see 10 is a cost the user controls. The scalar/media split at least keeps that cost off the common path.
- **A run's log directory grows to tens of files across many runs** → An 8 MB threshold keeps it near 20 files per run. TensorBoard opens them all, so a much smaller threshold would trade upload cost for open-file cost.
- **`tensorboardX` is an additional maintenance surface** → Its `SummaryWriter` is the package's public API and is mirrored by PyTorch's own writer; delegation means an added method needs no change here.

## Migration Plan

Additive. No existing behavior changes, no existing artifact or tag is touched, and runs that never call `exp_track.tensorboard()` are unaffected. Reverting means removing the module, the CLI command, and the two dependencies; runs that already carry `tb/` artifacts keep them as ordinary files.
