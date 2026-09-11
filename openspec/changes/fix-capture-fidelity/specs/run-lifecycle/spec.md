## Purpose

Records how a run ended and which earlier attempt it continues, under tag names
shared by every project, so failures are explained and resume chains stay
recoverable without project-private conventions.

## ADDED Requirements

### Requirement: Recording a continued attempt never fails the run

Recording that a run continues an earlier attempt SHALL NOT raise into the
caller. A tracking store that refuses the record SHALL be reported as a warning,
and the run SHALL proceed as if the record had been made.

#### Scenario: The tracking store refuses the record

- **WHEN** a run is started naming an earlier attempt it continues and the
  tracking store rejects the write
- **THEN** a warning is issued, no exception reaches the caller, and the run is
  usable

#### Scenario: The tracking store accepts the record

- **WHEN** a run is started naming an earlier attempt it continues
- **THEN** the run records that attempt and the chain of attempts walks back to
  the first one
