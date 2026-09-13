## ADDED Requirements

### Requirement: A run records the command and directory that started it

Capture SHALL record the argument vector of the process that started the run and
the working directory it ran in, expressed relative to the repository root when
the working directory lies inside the repository and absolutely otherwise.
Recording SHALL be governed by the same switches that govern code capture, and
SHALL NOT raise into the caller.

#### Scenario: A run started from a script

- **WHEN** a run is started by a process invoked with arguments, from a
  subdirectory of the repository
- **THEN** the run records that argument vector and records the working
  directory as the path relative to the repository root

#### Scenario: A run started from outside the repository

- **WHEN** a run is started by a process whose working directory is not inside
  the repository
- **THEN** the run records the working directory as an absolute path

#### Scenario: Capture is switched off

- **WHEN** a run is started with code capture disabled
- **THEN** no invocation is recorded, alongside no code state

#### Scenario: The tracking store refuses the record

- **WHEN** the tracking store rejects the invocation record
- **THEN** a warning is issued and the run proceeds
