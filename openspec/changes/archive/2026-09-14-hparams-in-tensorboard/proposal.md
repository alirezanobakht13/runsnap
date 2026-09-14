## Why

`runsnap.log_params()` records a Pydantic model as MLflow params and `hparams/<name>.json`, but nothing reaches TensorBoard, so runs can be compared by their settings in MLflow's table and not in TensorBoard's HParams view, where the settings would sit beside the curves runsnap already writes. runsnap's first consumer logs a configuration and a derived metadata model on every run and wants both in both tools without a TensorBoard-specific call in its trainer.

## What Changes

- Every model logged to a run through `log_params()` in the current process is written as that run's HParams session when `runsnap.tensorboard()` opens a writer for the run. Consumers that already call both need no change.
- The session lands in the run's scalar event file at the log directory root, so it is part of the run itself, is fetched by `runsnap tb` without `--media`, and is uploaded by the first sync pass. No experiment summary is written, so runs logging different keys combine into one table, and the run's existing scalar tags serve as the table's metrics.
- Hyperparameter values keep their types: booleans, numbers, and strings stay as they are; sequences and empty containers become their JSON text; `None` leaves are left out so a column that is numeric in some runs stays numeric.
- `log_params()` warns when it logs to a run whose writer has already written the run's session, since those values cannot reach TensorBoard.

## Capabilities

### New Capabilities

### Modified Capabilities

- `tensorboard-logging`: a run's logged hyperparameters are written as its HParams session when the writer opens, with typed values, and late params are reported.

## Impact

- `src/runsnap/_params.py`: leaves flattened once with their JSON types, encoded to strings for MLflow and kept typed for TensorBoard; an in-process record of each run's typed leaves.
- `src/runsnap/_tensorboard.py`: `runsnap.tensorboard()` writes the session into the scalar stream on entry.
- README and tests. No new dependencies: the session protos come from tensorboardX, which is already required.
- MLflow params, the `hparams/` artifact, and `load_params()` are unchanged.
