## Purpose

Lets a run log metrics, images, and histograms as TensorBoard event files that live in the run's own MLflow artifacts, so that MLflow stays the system of record while TensorBoard does the viewing. Logging works from any framework, and a job that is killed mid-training keeps everything already synced.

## Requirements

### Requirement: A writer bound to the active run

`runsnap.tensorboard()` SHALL return a context manager yielding a TensorBoard summary writer bound to the active MLflow run. The writer SHALL expose the standard TensorBoard summary-writer method surface (`add_scalar`, `add_image`, `add_histogram`, `add_text`, `add_figure`, and the rest), so that code written against any TensorBoard writer works unchanged. The writer SHALL NOT require, import, or assume any machine-learning framework.

Leaving the context manager SHALL flush and upload everything not yet uploaded, then close the writer.

#### Scenario: Writing from any framework

- **WHEN** a user opens `runsnap.tensorboard()` inside an active run and calls `add_scalar("train/loss", value, step)` with a plain Python float, and separately with a JAX or PyTorch array converted by the writer's normal array handling
- **THEN** the value is recorded in the run's event files in every case, and no framework package is imported by `runsnap`

#### Scenario: No active run

- **WHEN** `runsnap.tensorboard()` is entered with no active MLflow run and no run id given
- **THEN** it raises an error naming the problem, rather than silently discarding the data

#### Scenario: Clean exit uploads everything

- **WHEN** the context manager exits normally, or by exception, or by `KeyboardInterrupt`
- **THEN** all event data written up to that point is present in the run's artifacts

### Requirement: A record of scalars is charted in one call

The writer returned by `runsnap.tensorboard()` SHALL provide `add_record(prefix, record, global_step=None, walltime=None)`, which flattens `record` by the same rule as `runsnap.flatten_metrics()` and charts each entry as a scalar named `prefix/<key>`, or `<key>` when the prefix is empty, at the given step and wall time. Entries SHALL land in the scalar event file, and non-numeric fields SHALL be skipped.

#### Scenario: Training record

- **WHEN** a user calls `writer.add_record("train", {"loss": 0.5, "kind": "train", "actor": {"entropy": 1.2}}, 10)`
- **THEN** the run's scalar event file holds `train/loss` = `0.5` and `train/actor.entropy` = `1.2` at step `10`, and nothing for `kind`

#### Scenario: Wall time forwarded

- **WHEN** a user calls `add_record("eval", record, 10, walltime=1700000000.0)`
- **THEN** every charted event carries wall time `1700000000.0`

#### Scenario: Scalars stay downloadable without media

- **WHEN** a run logs records through `add_record` and images through `add_image`
- **THEN** downloading only the scalar event file shows every record's curves

#### Scenario: Empty prefix

- **WHEN** a user calls `add_record("", {"loss": 0.5}, 1)`
- **THEN** the scalar is named `loss`

### Requirement: Event files live in the run's artifacts

Event files SHALL be uploaded to the run's MLflow artifacts under the path `tb/`, and the run SHALL carry the tag `runsnap.tb.logdir` naming that path. The uploaded directory SHALL be a valid TensorBoard log directory: pointing TensorBoard at a local copy of it SHALL show the run's scalars, images, and histograms.

#### Scenario: Artifacts are a usable log directory

- **WHEN** a completed run's `tb/` artifacts are downloaded to a local directory and TensorBoard is pointed at its parent
- **THEN** TensorBoard shows one run containing every tag that was logged

#### Scenario: Run is discoverable

- **WHEN** a run has logged TensorBoard data
- **THEN** it carries tag `runsnap.tb.logdir` = `tb`, so runs with TensorBoard data can be found by an MLflow tag query

### Requirement: Scalars are separated from heavy media

Scalar and text data SHALL be written to a different event file from images, histograms, and other media, within the same log directory. The two SHALL merge into a single TensorBoard run when TensorBoard reads the directory. The separation SHALL make it possible to download a run's scalars without downloading its media.

#### Scenario: Scalar data can be fetched alone

- **WHEN** a run logs both scalars and images, and only its scalar event file is downloaded
- **THEN** TensorBoard shows the run's complete scalar curves, and the download is a small fraction of the run's total TensorBoard artifact size

#### Scenario: Both files form one run

- **WHEN** both event files are present in one directory and TensorBoard reads it
- **THEN** TensorBoard shows a single run whose tags include both the scalars and the media, not two separate runs

#### Scenario: Interleaved writing does not drop data

- **WHEN** scalars and media are written interleaved across the whole of training, so that step numbers in the two files overlap and restart relative to each other
- **THEN** every logged scalar point is present when TensorBoard reads the directory, with none discarded as out-of-order or orphaned data

### Requirement: Media is sharded and uploaded by periodic background sync

The media event file SHALL be rolled into successive shards, each closed once it exceeds a size threshold. Background sync SHALL periodically attempt to upload sealed shards and changed scalar files while the run is in progress, skipping the open media shard. A sealed shard SHALL NOT be uploaded again after a successful upload. Failed uploads SHALL remain eligible for later sync attempts.

The default interval between sync passes SHALL be 30 seconds. Sealing a shard SHALL make it eligible for a subsequent pass without waiting for an upload to succeed in the write path. The interval SHALL NOT be treated as a maximum durability delay: slow uploads, retries, or outages can extend the unsynced backlog. Neither the interval nor the shard size threshold guarantees a fixed bound on data lost on an unannounced kill.

Each shard SHALL have a distinct artifact name, including when several shards are sealed within the same second.

#### Scenario: Sealed shards upload during training

- **WHEN** a long run has sealed three media shards, is still writing a fourth, and a subsequent background sync successfully uploads the eligible files
- **THEN** the three sealed shards and the scalar data included in that upload are present in the run's artifacts, while the open fourth shard is skipped

#### Scenario: Sealed shards await sync

- **WHEN** a shard seals between sync passes
- **THEN** training continues without waiting for its upload, and the shard remains vulnerable to an unannounced kill until an upload succeeds

#### Scenario: Killed job keeps synced data

- **WHEN** a training process is killed without warning, by `SIGKILL` or by preemption after an unhandled `SIGTERM`
- **THEN** successfully uploaded shards and scalar updates remain in the run's artifacts; the open media shard, sealed shards awaiting successful upload, and unsynced scalar updates can be lost

#### Scenario: Shards sealed in quick succession

- **WHEN** two media shards are sealed within the same second
- **THEN** successful sync stores both as separate artifacts, and neither overwrites the other

### Requirement: Histograms default to a compact bin count

`add_histogram` SHALL default to a bin count that keeps a histogram entry small, rather than the several-hundred-bucket default of the underlying writer. The caller SHALL be able to override it per call.

#### Scenario: Default is compact

- **WHEN** a histogram is logged without specifying bins
- **THEN** the recorded entry is substantially smaller than the underlying writer's default would produce, and the histogram is still displayed by TensorBoard

#### Scenario: Caller overrides

- **WHEN** a caller passes an explicit `bins` value to `add_histogram`
- **THEN** that value is used unchanged

### Requirement: Logging never fails the run

No failure in writing, rolling, or uploading TensorBoard data SHALL raise into user code or fail the MLflow run. Failures SHALL be reported as warnings.

#### Scenario: Tracking server unreachable mid-run

- **WHEN** an upload fails because the tracking server is unreachable
- **THEN** a warning is emitted, training continues, and later uploads are still attempted

#### Scenario: Writer failure does not propagate

- **WHEN** an upload fails while the context manager is exiting
- **THEN** the block exits without raising, and the failure is reported as a warning
