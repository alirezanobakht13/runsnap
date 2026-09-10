## MODIFIED Requirements

### Requirement: Run creation passes through to MLflow

`runsnap.start_run()` SHALL accept the same arguments as `mlflow.start_run()`, forward them unchanged, and return an object exposing the same interface as the MLflow `ActiveRun` it wraps, usable as a context manager and as a plain run ended by `mlflow.end_run()`. Code-state capture and lifecycle recording (status and failure cause on exit, and the `continues` link) SHALL be the only added behaviors; all subsequent interaction with the run — logging metrics, artifacts, models, tags — SHALL require no `runsnap` API.

#### Scenario: Existing MLflow code keeps working

- **WHEN** a user replaces `mlflow.start_run(run_name="x", nested=True)` with `runsnap.start_run(run_name="x", nested=True)` and continues to call `mlflow.log_metric` inside the block
- **THEN** the run is created with the same name and nesting, the metrics are logged to it, and the block exits with the same status MLflow would give it, except that a `KeyboardInterrupt` ends it as `KILLED`

#### Scenario: Return value is the MLflow run

- **WHEN** a user writes `with runsnap.start_run() as run:`
- **THEN** `run` exposes `run.info.run_id` and the rest of the MLflow `ActiveRun` interface, and `mlflow.active_run()` inside the block reports the same run id

#### Scenario: Run used without a context manager

- **WHEN** a user calls `run = runsnap.start_run()` and later `mlflow.end_run()`
- **THEN** the run ends as `FINISHED` exactly as with `mlflow.start_run()`
