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

### Requirement: A run records where its events are written locally

When `runsnap.tensorboard()` creates the local log directory it writes into, the run SHALL carry the tag `runsnap.tb.local_host` holding the name of the host the writer runs on and the tag `runsnap.tb.local_dir` holding the absolute path of that directory. The tags SHALL be set before the writer is handed to the caller, SHALL NOT be removed when the block exits, and a failure to set them SHALL be reported as a warning without failing the run.

#### Scenario: Tags describe the live directory

- **WHEN** a user enters `runsnap.tensorboard()` inside an active run on host `gpu-box`
- **THEN** the run carries `runsnap.tb.local_host` = `gpu-box` and `runsnap.tb.local_dir` naming a directory that exists and holds the run's event files while the block is open

#### Scenario: Tags outlive the directory

- **WHEN** the block exits normally and the local directory is removed
- **THEN** both tags remain on the run, and a viewer treats the missing directory as a signal to use the run's artifacts instead

#### Scenario: Tagging failure is a warning

- **WHEN** the tracking server refuses the tags
- **THEN** a warning is emitted, the writer is still handed to the caller, and logging proceeds

### Requirement: The writer flushes to disk promptly by default

`runsnap.tensorboard()` SHALL flush buffered events to the local event files at least every 10 seconds unless the caller passes an explicit `flush_secs`, so that a viewer reading the local directory or the uploaded artifacts lags the training loop by seconds rather than minutes.

#### Scenario: Default flush interval

- **WHEN** a user writes a scalar through `runsnap.tensorboard()` without passing `flush_secs`
- **THEN** the scalar is readable from the local event file within 10 seconds, without the user calling `flush()`

#### Scenario: Caller overrides

- **WHEN** a user enters `runsnap.tensorboard(flush_secs=60)`
- **THEN** the underlying writers use a 60 second flush interval

### Requirement: Writing TensorBoard data never fails the run that produced it

A failure anywhere in TensorBoard logging — writing an event, locating the file
an event was written to, rolling a shard, or uploading — SHALL be reported as a
warning and SHALL NOT raise into the training loop.

#### Scenario: The underlying writer's event file cannot be located

- **WHEN** the summary writer does not expose the file it is appending events to
  in the expected form
- **THEN** a warning is issued, the training loop continues, and events already
  written remain uploadable

### Requirement: Scalar events are sharded so a sync pass uploads a bounded amount

The scalar event stream SHALL roll into a new shard once the open shard passes a
size threshold. A sealed scalar shard SHALL be uploaded once and not re-uploaded;
the open scalar shard SHALL be uploaded on every sync pass that finds it changed,
so a dashboard watching the run keeps updating.

#### Scenario: A run logs scalars past the shard threshold

- **WHEN** a run logs enough scalars for the scalar stream to pass the shard
  threshold several times
- **THEN** each sealed shard is uploaded exactly once, and the data available to
  a viewer is the same as if the stream had been one file

#### Scenario: A dashboard watches a run still being written

- **WHEN** a sync pass runs while the open scalar shard has grown since the last
  pass
- **THEN** the open shard is uploaded, and a viewer fetching the run afterwards
  sees the newly logged scalars

#### Scenario: Nothing has changed since the last pass

- **WHEN** a sync pass runs and no event file has grown since the last pass
- **THEN** nothing is uploaded

### Requirement: A run's logged hyperparameters form its HParams session

`runsnap.tensorboard()` SHALL write, before handing the writer to the caller, one HParams session holding every leaf logged to the run through `runsnap.log_params()` in the same process, keyed as the run's MLflow params are keyed. The session SHALL be written to the run's scalar event file at the root of the log directory, so that TensorBoard shows it on the run itself rather than on a nested run, and so that a view fetching scalars without media includes it. No experiment summary SHALL be written, so that runs logging different keys are shown together under the union of their keys. No metric series SHALL be written for the session: the run's scalar tags are its metrics. A run with no params logged in the process SHALL get no session. A failure to write the session SHALL be reported as a warning, and the writer SHALL still be handed to the caller.

#### Scenario: Logged model appears on the run

- **WHEN** a user calls `runsnap.log_params(hp)` where `hp` has `seed = 42` and a nested `opt.lr = 0.001`, then enters `runsnap.tensorboard()`
- **THEN** the run's scalar event file at the log directory root holds a session with `seed` = `42` and `opt.lr` = `0.001`, and TensorBoard lists it under the run's own name

#### Scenario: Several models on one run

- **WHEN** a user calls `runsnap.log_params(config)` and `runsnap.log_params(meta, name="metadata", prefix="metadata")`, then enters `runsnap.tensorboard()`
- **THEN** the run has one session holding the keys of both models, the second under `metadata.`

#### Scenario: Runs logging different keys are viewed together

- **WHEN** one run logs `model.trunk.width` and another logs `task.name`, and both are shown in one TensorBoard
- **THEN** the HParams table has a column for each key, and no experiment summary is present in either run's event files

#### Scenario: Scalars are the session's metrics

- **WHEN** a run with a session charts `writer.add_record("evaluation", {"mean_return": 3.0}, 10)`
- **THEN** TensorBoard's HParams table shows `evaluation/mean_return` for that run at its latest value, and the run's event files hold no other series for it

#### Scenario: Scalars-only view shows the session

- **WHEN** a finished run with a session is opened with `runsnap tb` without `--media`
- **THEN** TensorBoard's HParams table lists the run with its hyperparameters

#### Scenario: Session is uploaded while the run is in progress

- **WHEN** a sync pass completes after the writer is handed to the caller
- **THEN** the run's uploaded scalar event file holds the session, so a job killed afterwards keeps it

#### Scenario: No params logged

- **WHEN** a user enters `runsnap.tensorboard()` on a run that no `runsnap.log_params()` call in the process logged to
- **THEN** the run's event files hold no HParams session

#### Scenario: Session cannot be written

- **WHEN** writing the session fails
- **THEN** a warning is emitted, the writer is handed to the caller, and logging proceeds

### Requirement: Hyperparameter values keep their types

In the session, boolean, integer, float, and string leaves SHALL be written as TensorBoard booleans, numbers, and strings. Sequences and empty containers SHALL be written as strings holding the same JSON text as the corresponding MLflow param. `None` leaves SHALL be omitted, so that a hyperparameter that is a number in some runs and `None` in others stays numeric when the runs are viewed together.

#### Scenario: Scalar types

- **WHEN** a logged model has `name = "resnet"`, `lr = 0.001`, `epochs = 10`, and `use_amp = True`
- **THEN** the session holds `name` as the string `resnet`, `lr` and `epochs` as the numbers `0.001` and `10`, and `use_amp` as the boolean `true`

#### Scenario: Sequence and empty container

- **WHEN** a logged model has `layers = [64, 128]` and `overrides = {}`
- **THEN** the session holds `layers` as the string `[64, 128]` and `overrides` as the string `{}`

#### Scenario: None omitted

- **WHEN** one run logs `target_kl = 0.01`, another logs `target_kl = None`, and both are shown in one TensorBoard
- **THEN** the second run's session holds no `target_kl`, and the HParams table treats `target_kl` as a number

### Requirement: Params logged after the session is written are reported

When `runsnap.log_params()` logs to a run for which `runsnap.tensorboard()` has already been entered in the same process, it SHALL still log the MLflow params and the artifact, and SHALL emit a warning stating that those values are missing from the run's TensorBoard hyperparameters and that they have to be logged before `runsnap.tensorboard()` is entered.

#### Scenario: Model logged inside the writer block

- **WHEN** a user enters `runsnap.tensorboard()` after logging `config`, then calls `runsnap.log_params(derived, name="derived", prefix="derived")`
- **THEN** a warning is emitted, the run has the `derived.` params and `hparams/derived.json`, and the session holds only the keys of `config`

#### Scenario: Models logged before the writer opens

- **WHEN** a user logs two models and then enters `runsnap.tensorboard()`
- **THEN** no such warning is emitted
