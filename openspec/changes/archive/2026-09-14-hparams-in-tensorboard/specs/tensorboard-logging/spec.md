## ADDED Requirements

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
