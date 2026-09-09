## Why

MLflow is the better system of record and TensorBoard is the better viewer, but
nothing connects them. MLflow's own TensorBoard support is framework-locked and
wrong in both directions: `mlflow.tensorflow.autolog()` uploads a tfevents
directory only for `tf.keras.Model.fit`, and `mlflow.pytorch.autolog()`
monkeypatches `torch.utils.tensorboard` to copy every scalar into MLflow metrics
while never uploading the tfevents at all. Neither works under JAX, and the
PyTorch path is exactly the duplicate-write it should avoid.

A run's metrics therefore live in whichever tool was chosen first, and comparing
curves across runs means either giving up MLflow's query engine or giving up
TensorBoard's viewer.

## What Changes

- A framework-agnostic writer, `exp_track.tensorboard()`, that returns a
  `tensorboardX.SummaryWriter` facade bound to the active MLflow run and
  uploads its tfevents to the run's artifacts under `tb/`. Any framework that
  can call `add_scalar`/`add_image` works; nothing imports torch, TF, or JAX.
- Scalars and heavy media are written to **separate event files in one
  directory**, so viewing curves across many runs downloads kilobytes instead
  of the hundreds of megabytes that images and histograms cost.
- Media files are rolled into sealed shards that upload during background
  syncs. A preempted job keeps successfully uploaded data; the open shard,
  pending sealed shards, and unsynced scalar updates can be lost.
- A new `exp-track tb` command that turns an MLflow run query into a
  TensorBoard log directory and launches TensorBoard on it, giving MLflow's
  query engine as TensorBoard's run selector.
- `exp_track.log_params` and `start_run` are unchanged.

## Capabilities

### New Capabilities

- `tensorboard-logging`: writing a run's TensorBoard event files and syncing
  them to the run's MLflow artifacts, including the scalar/media split, shard
  rolling, and the loss window a preempted run is left with.
- `tensorboard-viewing`: resolving an MLflow run query into a named TensorBoard
  log directory of cached event files and launching TensorBoard on it.

### Modified Capabilities

None. `code-state-capture`, `code-state-restore`, and `pydantic-params` keep
their current requirements.

## Impact

- **New modules**: `src/exp_track/_tensorboard.py` (writer and sync),
  `src/exp_track/_tb_fetch.py` (cache and log directory assembly); a `tb`
  command added to `src/exp_track/_cli.py`.
- **Public API**: one new symbol, `exp_track.tensorboard()`.
- **Tags**: one new run tag, `exp_track.tb.logdir`, naming the artifact path
  holding the event files. Existing `exp_track.git.*` tags are untouched.
- **Artifacts**: a new `tb/` artifact directory per run, alongside `code/` and
  `hparams/`.
- **Dependencies**: adds `tensorboardx` (~344 KB; its numpy, protobuf, and
  packaging requirements are already satisfied by MLflow) and `tensorboard`
  (~134 MB, needed to launch the viewer).
- **Not changed**: no MLflow monkeypatching, no autolog, no metric duplication
  into MLflow.
