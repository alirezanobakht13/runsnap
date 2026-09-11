## ADDED Requirements

### Requirement: Runs selected for viewing are fetched concurrently

Assembling several runs for viewing SHALL fetch them concurrently. A run whose
fetch fails SHALL NOT prevent the remaining runs from being viewed.

#### Scenario: Several cached runs are selected

- **WHEN** several runs that are not being written on this host are selected for
  viewing
- **THEN** all of them are fetched and every run holding event data appears in
  the dashboard

### Requirement: Run selection queries the server a bounded number of times

Resolving the runs a command selects SHALL enumerate the server's experiments at
most once per invocation, however many run names are given. Expanding a run's
chain of earlier attempts SHALL fetch each attempt at most once.

#### Scenario: Several run names are given

- **WHEN** a command is invoked with several run names and no experiment
- **THEN** every named run is resolved and the server's experiments are
  enumerated once

#### Scenario: A chain of attempts is expanded

- **WHEN** a command expands a run whose chain holds several earlier attempts
- **THEN** every attempt appears in the selection and each is fetched once

#### Scenario: An experiment is named

- **WHEN** a command is invoked with an experiment name
- **THEN** only that experiment is resolved and the full experiment list is
  never enumerated
