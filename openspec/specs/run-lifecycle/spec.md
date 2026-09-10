## Purpose

Records how a run ended and which earlier attempt it continues, under tag names shared by every project, so failures are explained and resume chains stay recoverable without project-private conventions.

## Requirements

### Requirement: An interrupted run ends as killed

When the `with` block of a run started by `runsnap.start_run()` exits because a `KeyboardInterrupt` propagated, the run SHALL end with MLflow status `KILLED`. Any other exception SHALL end the run with status `FAILED`, and a normal exit SHALL end it with status `FINISHED`. The exception SHALL still propagate to the caller in every case.

#### Scenario: Ctrl-C during training

- **WHEN** a `KeyboardInterrupt` propagates out of the `with runsnap.start_run():` block
- **THEN** the run's status is `KILLED` and the `KeyboardInterrupt` reaches the caller

#### Scenario: Crash during training

- **WHEN** a `FloatingPointError` propagates out of the block
- **THEN** the run's status is `FAILED` and the error reaches the caller

#### Scenario: Normal completion

- **WHEN** the block exits without an exception
- **THEN** the run's status is `FINISHED` and it carries no `runsnap.failure.cause` tag

### Requirement: The cause of a failure is recorded

When the block exits by an exception, including `KeyboardInterrupt`, the run SHALL carry tag `runsnap.failure.cause` holding the exception's type name and message. Recording the cause SHALL NOT change which exception propagates, and a failure to write the tag SHALL be reported as a warning while the run still ends with the correct status.

#### Scenario: Cause names the exception

- **WHEN** the block exits by `FloatingPointError("nonfinite update 7")`
- **THEN** tag `runsnap.failure.cause` reads `FloatingPointError: nonfinite update 7`

#### Scenario: A failing child under a continuing parent

- **WHEN** a nested run's block exits by an exception that the enclosing parent run's code catches, and the parent then finishes normally
- **THEN** the child is `FAILED` with a `runsnap.failure.cause` tag, and the parent is `FINISHED` with no such tag

#### Scenario: Tracking server refuses the tag

- **WHEN** the block exits by an exception and the tag write fails
- **THEN** a warning is emitted, the run still ends with status `FAILED`, and the original exception reaches the caller

### Requirement: A run can name the attempt it continues

`runsnap.start_run()` SHALL accept a `continues` argument holding the id of an earlier run. When given, the new run SHALL carry tag `runsnap.continues` with that id. The relation SHALL be independent of MLflow nesting: a run can both continue an earlier attempt and be nested under a parent.

#### Scenario: Resumed attempt

- **WHEN** a user calls `runsnap.start_run(continues=first_id)`
- **THEN** the new run carries tag `runsnap.continues` = `first_id`

#### Scenario: Fresh attempt

- **WHEN** `continues` is not given
- **THEN** the run carries no `runsnap.continues` tag

#### Scenario: Continuing inside a sweep

- **WHEN** a user calls `runsnap.start_run(nested=True, continues=first_id)` inside a parent run
- **THEN** the run carries both `mlflow.parentRunId` and `runsnap.continues`

### Requirement: The chain of attempts is walkable

`runsnap.attempt_chain(run_id)` SHALL return the ids of the given run and every attempt it transitively continues, ordered from the given run back to the first attempt. A chain that revisits a run SHALL raise an error naming that run rather than looping. A `runsnap.continues` tag naming a run the tracking server does not have SHALL raise an error naming the missing run.

#### Scenario: Three attempts

- **WHEN** run `third` continues `second`, which continues `first`, and a user calls `attempt_chain(third)`
- **THEN** the result is `[third, second, first]`

#### Scenario: First attempt

- **WHEN** a user calls `attempt_chain(first)` on a run with no `runsnap.continues` tag
- **THEN** the result is `[first]`

#### Scenario: Cycle

- **WHEN** two runs each name the other as the attempt they continue
- **THEN** `attempt_chain` raises an error naming the run it met twice

### Requirement: Lineage is visible from the CLI

`runsnap show` SHALL print the id of the attempt a run continues when one is recorded.

#### Scenario: Inspecting a resumed attempt

- **WHEN** a user runs `runsnap show <ref>` on a run carrying `runsnap.continues`
- **THEN** the output contains a line naming that predecessor's run id

#### Scenario: Inspecting a first attempt

- **WHEN** the run carries no `runsnap.continues` tag
- **THEN** the output contains no lineage line
