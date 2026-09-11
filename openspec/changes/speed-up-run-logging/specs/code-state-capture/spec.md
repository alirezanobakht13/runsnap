## ADDED Requirements

### Requirement: The patch size ceiling bounds the work done, not only the upload

Building a run's patch SHALL stop once the configured ceiling is passed, without
holding more than the ceiling's worth of patch in memory. A run whose patch
exceeds the ceiling SHALL record the same outcome whether the excess is
discovered during or after reading: a warning, a capture-error tag naming the
ceiling, no patch artifact, and a run that starts normally.

#### Scenario: The working tree's diff far exceeds the ceiling

- **WHEN** a run starts in a repository whose uncommitted changes would produce a
  patch many times the configured ceiling
- **THEN** the run starts, a warning reports the patch was skipped, the run
  carries a capture-error tag naming the ceiling, no patch artifact is uploaded,
  and no patch digest is recorded

#### Scenario: An oversize working tree is still marked dirty

- **WHEN** a run's patch is abandoned at the ceiling
- **THEN** the run is recorded as having a dirty working tree, distinguishing it
  from a run started from a clean tree

#### Scenario: The working tree's diff fits under the ceiling

- **WHEN** a run starts in a repository whose uncommitted changes fit under the
  configured ceiling
- **THEN** the complete patch is uploaded and its digest is recorded
