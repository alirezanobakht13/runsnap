## ADDED Requirements

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
