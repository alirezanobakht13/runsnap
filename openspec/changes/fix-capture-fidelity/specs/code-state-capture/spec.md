## Purpose

Records the exact state of the code that produced an MLflow run — the base
commit plus every uncommitted change in the working tree — so that any run can
later be reconstructed byte-for-byte. Capture happens at run start and never
interferes with the user's git working state or with normal MLflow usage.

## ADDED Requirements

### Requirement: Captured code state reflects the repository as it stands when the run starts

Each run SHALL record the repository state read at the moment that run starts. A
process that starts several runs SHALL record each run's own code state, and
SHALL NOT reuse state read for an earlier run.

#### Scenario: Code changes between two runs in one process

- **WHEN** a process starts a run, a tracked file is then modified, and the same
  process starts a second run
- **THEN** the second run's recorded patch contains the modification and the two
  runs carry different patch digests

#### Scenario: A commit is made between two runs in one process

- **WHEN** a process starts a run, the working tree is then committed, and the
  same process starts a second run
- **THEN** the second run records the new commit and a clean working tree

### Requirement: A nested run reuses its parent's patch only when the patches match

A nested run SHALL reference an ancestor's uploaded patch only when its own
patch is byte-identical to the ancestor's. When the two differ, the nested run
SHALL upload and record its own patch, as an unparented run would.

#### Scenario: The tree changed between parent and nested run

- **WHEN** a nested run starts after the working tree changed since its parent
  run started
- **THEN** the nested run records its own patch digest and its own patch
  artifact, not a reference to its parent's

#### Scenario: The tree is unchanged since the parent started

- **WHEN** a nested run starts with the working tree unchanged since its parent
  run started
- **THEN** the nested run references its parent's patch rather than uploading a
  second copy
