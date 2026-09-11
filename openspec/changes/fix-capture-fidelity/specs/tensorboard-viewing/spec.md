## Purpose

Turns a set of MLflow runs into a TensorBoard log directory and opens TensorBoard
on it, so that MLflow's query engine acts as TensorBoard's run selector and runs
appear under their own names rather than their ids.

## ADDED Requirements

### Requirement: Viewing tolerates cache entries left by earlier invocations

Assembling runs for viewing SHALL succeed whatever a previous invocation left in
the cache. A cache entry that does not point at the current data SHALL be
replaced rather than reported as an error.

#### Scenario: A cache entry points at data that has moved

- **WHEN** a run is viewed and its cache holds an entry pointing somewhere other
  than that run's current event file
- **THEN** the entry is replaced and the run is viewable

#### Scenario: A cache entry is not of the expected kind

- **WHEN** a run is viewed and its cache holds an ordinary file where a link
  belongs
- **THEN** the entry is replaced and the run is viewable

### Requirement: Every selected run with event data appears under a distinct name

Runs assembled for viewing SHALL each appear under a distinct name. When two
runs share a name, each SHALL be distinguished; a distinguished name SHALL NOT
collide with another selected run's own name.

#### Scenario: Two runs share a name

- **WHEN** two selected runs carry the same run name
- **THEN** both appear, each under a name that identifies which run it is

#### Scenario: A run is named like a distinguished name

- **WHEN** two selected runs share a name and a third selected run is named
  exactly as the pattern used to distinguish them would produce
- **THEN** all three appear under distinct names and no error is raised
