## ADDED Requirements

### Requirement: A query selection follows runs that start later

When `runsnap tb` is given no run ids or names, it SHALL re-run its selection
query every few seconds while TensorBoard is open. Each matching run that is
not already shown SHALL be added to the dashboard once it has begun writing
TensorBoard events through `runsnap.tensorboard()`, without the command being
restarted. A run added this way SHALL be shown as a run selected at startup
would be: from its local directory while it is being written on this host,
including the switch to its cached artifacts when its writer exits, and from
its cached artifacts otherwise, with the same `--media` and `--chain` choices.
A run SHALL NOT be removed from the dashboard because it stops matching the
query. When run ids or names are given, the selection SHALL stay as it was
resolved at startup.

#### Scenario: A run starts on this host after the dashboard opened

- **WHEN** `runsnap tb` is open and a new run on this host enters
  `runsnap.tensorboard()`
- **THEN** within a few seconds TensorBoard shows the run under its name from
  its local directory, shows new scalar points as training continues, and
  shows the run from its cached artifacts after its writer exits, all without
  the command being restarted

#### Scenario: A run starts in an experiment created later

- **WHEN** `runsnap tb` is open without `--experiment` and a run starts writing
  TensorBoard events in an experiment created after the dashboard opened
- **THEN** the run is added to the dashboard

#### Scenario: A new run outside the query

- **WHEN** `runsnap tb --experiment ablations` is open and a run starts writing
  TensorBoard events in another experiment
- **THEN** the run is not added

#### Scenario: A run selected before it wrote events

- **WHEN** a run already matched the query when the dashboard opened but had
  not yet entered `runsnap.tensorboard()`, and enters it later
- **THEN** the run is added once it does

#### Scenario: A new run that never writes TensorBoard events

- **WHEN** a new run matches the query but never enters
  `runsnap.tensorboard()`
- **THEN** the run is never added and nothing is downloaded for it

#### Scenario: A short run ends between two checks

- **WHEN** a new run on this host enters and leaves its `runsnap.tensorboard()`
  block between two re-runs of the query
- **THEN** the run is added from its cached artifacts

#### Scenario: A late run with `--chain`

- **WHEN** `runsnap tb --filter "attributes.status = 'RUNNING'" --chain` is
  open and a new run starts that continues an earlier `KILLED` attempt
- **THEN** both the new run and the earlier attempt are added

#### Scenario: Named runs stay fixed

- **WHEN** `runsnap tb baseline` is open and another run starts writing
  TensorBoard events
- **THEN** the other run is not added

#### Scenario: A shown run stops matching

- **WHEN** a run shown under `runsnap tb --filter "attributes.status = 'RUNNING'"`
  finishes
- **THEN** the run stays in the dashboard

#### Scenario: The tracking server does not answer

- **WHEN** re-running the query fails
- **THEN** a warning is reported, every run already shown stays visible, and
  the query is tried again at the next check; while the failure persists it is
  reported once rather than at every check

#### Scenario: Following ends with the command

- **WHEN** the user interrupts TensorBoard while the query is being followed
- **THEN** the command exits and removes its assembled directory

## MODIFIED Requirements

### Requirement: Runs appear under readable names

Each run in the assembled log directory SHALL be named by its MLflow run name, not its run id. When two runs selected at startup share a name, each SHALL be disambiguated so that both remain visible and distinguishable.

#### Scenario: Run name used

- **WHEN** a run named `baseline-lr0.001` is shown
- **THEN** TensorBoard's run selector lists it as `baseline-lr0.001`

#### Scenario: Duplicate names

- **WHEN** two runs selected at startup are both named `baseline`
- **THEN** both appear, each distinguished by part of its run id, and neither is dropped

#### Scenario: Unnamed run

- **WHEN** a selected run has no run name
- **THEN** it appears under its run id rather than being skipped

### Requirement: Every selected run with event data appears under a distinct name

Runs assembled for viewing SHALL each appear under a distinct name. When two
runs selected at startup share a name, each SHALL be distinguished; a
distinguished name SHALL NOT collide with another selected run's own name. A
run added while the dashboard is open whose name is already shown SHALL appear
under a distinguished name, and runs already shown SHALL keep the names they
were shown under.

#### Scenario: Two runs share a name

- **WHEN** two selected runs carry the same run name
- **THEN** both appear, each under a name that identifies which run it is

#### Scenario: A run is named like a distinguished name

- **WHEN** two selected runs share a name and a third selected run is named
  exactly as the pattern used to distinguish them would produce
- **THEN** all three appear under distinct names and no error is raised

#### Scenario: A late run shares a shown run's name

- **WHEN** a run named `baseline` is shown and a new run also named `baseline`
  is added while the dashboard is open
- **THEN** the shown run keeps the name `baseline` and the new run appears
  under a name distinguished by part of its run id

#### Scenario: A late run shares the name of runs shown distinguished

- **WHEN** two runs named `baseline` were shown distinguished at startup and a
  third run named `baseline` is added while the dashboard is open
- **THEN** the third run also appears under a distinguished name, and the two
  shown runs keep theirs

### Requirement: Run selection queries the server a bounded number of times

Resolving the runs a command selects SHALL enumerate the server's experiments at
most once per selection pass, however many run names are given: once when the
command resolves its selection, and once each time a followed query is re-run
while the dashboard is open. Expanding a run's chain of earlier attempts SHALL
fetch each attempt at most once per selection pass, and a re-run of the query
SHALL expand chains only for the runs it adds.

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

#### Scenario: A followed query is re-run

- **WHEN** `runsnap tb --chain` without run names re-runs its query while the
  dashboard is open and one new run has begun writing TensorBoard events
- **THEN** the server's experiments are enumerated once for that re-run, only
  the new run's chain is expanded, and no run already shown is fetched again
