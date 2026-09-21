## Purpose

Turns a set of MLflow runs into a TensorBoard log directory and opens TensorBoard on it, so that MLflow's query engine acts as TensorBoard's run selector and runs appear under their own names rather than their ids.

## Requirements

### Requirement: Runs are selected by MLflow query

`runsnap tb` SHALL accept runs given as run ids or run names, as a whole experiment, and as an MLflow filter expression, and SHALL open TensorBoard on the runs that match. When nothing matches, it SHALL report that and exit without launching TensorBoard.

#### Scenario: Explicit runs

- **WHEN** a user runs `runsnap tb baseline ablation-nodropout`
- **THEN** TensorBoard opens showing those two runs

#### Scenario: Whole experiment

- **WHEN** a user runs `runsnap tb --experiment ablations`
- **THEN** TensorBoard opens showing every run in that experiment that has TensorBoard data

#### Scenario: Filtered query

- **WHEN** a user runs `runsnap tb --experiment ablations --filter "params.optimizer = 'adamw'"`
- **THEN** only the runs matching that filter are shown

#### Scenario: No matching runs

- **WHEN** the selection matches no runs, or matches only runs that logged no TensorBoard data
- **THEN** the command reports that no runs were found and exits without starting TensorBoard

### Requirement: Attempt chains can be viewed together

`runsnap tb` SHALL accept a flag that extends the selected runs with every attempt each selected run transitively continues, so a training that was interrupted and resumed is viewed as one set of curves. A predecessor already in the selection SHALL appear once.

#### Scenario: Resumed training

- **WHEN** run `third` continues `second`, which continues `first`, and a user runs `runsnap tb third --chain`
- **THEN** TensorBoard opens showing `third`, `second`, and `first`

#### Scenario: Chain over a query

- **WHEN** a user runs `runsnap tb --experiment ablations --filter "attributes.status = 'FINISHED'" --chain` and a matching run continues an earlier `KILLED` attempt
- **THEN** the earlier attempt is shown alongside the matching run even though it does not match the filter

#### Scenario: Without the flag

- **WHEN** a user runs `runsnap tb third` without the flag
- **THEN** only `third` is shown

### Requirement: Runs appear under readable names

Each run in the assembled log directory SHALL be named by its MLflow run name, not its run id. When two selected runs share a name, each SHALL be disambiguated so that both remain visible and distinguishable.

#### Scenario: Run name used

- **WHEN** a run named `baseline-lr0.001` is shown
- **THEN** TensorBoard's run selector lists it as `baseline-lr0.001`

#### Scenario: Duplicate names

- **WHEN** two selected runs are both named `baseline`
- **THEN** both appear, each distinguished by part of its run id, and neither is dropped

#### Scenario: Unnamed run

- **WHEN** a selected run has no run name
- **THEN** it appears under its run id rather than being skipped

### Requirement: Media is fetched only on request

By default the command SHALL download only each run's scalar data. Media SHALL be downloaded only when explicitly requested, and then only for the selected runs.

#### Scenario: Default is scalars only

- **WHEN** `runsnap tb --experiment ablations` selects runs that logged images and histograms
- **THEN** only the scalar event files are downloaded, and TensorBoard shows the scalar curves for every selected run

#### Scenario: Media requested

- **WHEN** the same command is given `--media`
- **THEN** the media event files are downloaded as well, and TensorBoard additionally shows the images and histograms

### Requirement: Downloads are cached and reused

Downloaded event files SHALL be cached on disk, keyed by run, and reused on later invocations. A file already held at its current remote size SHALL NOT be downloaded again.

#### Scenario: Second invocation is cheap

- **WHEN** `runsnap tb` is run twice over the same finished runs
- **THEN** the second invocation downloads nothing and starts TensorBoard from the cache

#### Scenario: In-progress run picks up new data

- **WHEN** a selected run is still training and has uploaded new event data since the last invocation
- **THEN** the new data is downloaded and the previously cached data is not re-downloaded unnecessarily

#### Scenario: Cache survives a partial download

- **WHEN** a previous invocation was interrupted, leaving a partially written cached file
- **THEN** the next invocation replaces it with the complete file rather than serving truncated data

### Requirement: TensorBoard is launched on the assembled directory

The command SHALL start TensorBoard against the assembled log directory and leave it running in the foreground until interrupted, passing through the address it is serving on. When TensorBoard is not installed, the command SHALL say so and how to install it, rather than failing obscurely.

#### Scenario: TensorBoard runs

- **WHEN** runs are selected successfully
- **THEN** TensorBoard starts, prints the URL it is serving, and keeps running until the user interrupts it

#### Scenario: TensorBoard missing

- **WHEN** TensorBoard is not available in the environment
- **THEN** the command reports that TensorBoard is required and names the package to install

### Requirement: A run being logged on this host is shown live

When a selected run carries `runsnap.tb.local_host` equal to the viewing host's name and `runsnap.tb.local_dir` naming a directory that exists, `runsnap tb` SHALL show that run from that directory rather than from a downloaded snapshot, and SHALL NOT download that run's artifacts. TensorBoard's own reload SHALL then reflect events the training process has flushed since the dashboard was opened. Any other selected run SHALL be shown from its cached artifacts as before, and the two kinds SHALL appear together in one dashboard under their usual names.

#### Scenario: Dashboard follows training

- **WHEN** a run is training on this host with `runsnap.tensorboard()` and a user runs `runsnap tb <run>` while it is in progress
- **THEN** TensorBoard opens on the run's local log directory, downloads nothing for it, and shows new scalar points as training continues without the command being restarted

#### Scenario: Finished run falls back to artifacts

- **WHEN** a selected run carries the local tags but its directory no longer exists
- **THEN** the run is fetched from its artifacts and shown from the cache exactly as a run without the tags would be

#### Scenario: Run logged on another host

- **WHEN** a selected run carries `runsnap.tb.local_host` naming a different host
- **THEN** the run is fetched from its artifacts, even if a directory with the tagged path happens to exist here

#### Scenario: Live and cached runs together

- **WHEN** a user runs `runsnap tb <run> --chain` while `<run>` trains on this host and continues a finished earlier attempt
- **THEN** the live run is shown from its local directory and the earlier attempt from the cache, both under their run names

#### Scenario: Local directory is shown whole

- **WHEN** a live run has logged images and the user does not pass `--media`
- **THEN** the run's images are visible anyway, because the local directory is shown as it is and nothing was downloaded to omit

#### Scenario: Run killed without cleanup

- **WHEN** a run on this host was killed without leaving its writer block, so its local directory still exists
- **THEN** the run is shown from that directory, which holds every event flushed before the kill, including data never uploaded

### Requirement: A live run stays visible after its writer exits

While TensorBoard is open, `runsnap tb` SHALL keep watching every run it is
showing from a local directory. When such a directory disappears, the command
SHALL fetch the run's uploaded artifacts into the cache and show the run from
there, exactly as it would a run whose directory was already gone when the
command started, without the command being restarted. Until the directory
disappears, the run SHALL continue to be shown from it.

#### Scenario: Writer exits while the dashboard is open

- **WHEN** a run is shown live from its local directory and its
  `runsnap.tensorboard()` block exits, uploading its final events and removing
  the directory
- **THEN** the run is fetched from its artifacts and shown from the cache, with
  every scalar written before the block exited, and TensorBoard shows it again
  without the command being restarted

#### Scenario: Cached view matches a finished run

- **WHEN** a run switches from its local directory to the cache while the
  dashboard is open and the user did not pass `--media`
- **THEN** the run is shown with scalars only, as a run fetched at startup
  would be; with `--media` its media is shown as well

#### Scenario: Run killed without cleanup keeps its local view

- **WHEN** a run shown live is killed without leaving its writer block, so its
  local directory remains
- **THEN** the run keeps being shown from that directory and nothing is fetched
  for it

#### Scenario: Fetch fails after the directory disappears

- **WHEN** a live run's directory disappears and the tracking server will not
  hand over its artifacts
- **THEN** the failure is reported as a warning, that run is left out of the
  dashboard, and every other run stays visible

#### Scenario: Watching ends with the command

- **WHEN** the user interrupts TensorBoard while live runs are still being
  watched
- **THEN** the command exits and removes its assembled directory as it does
  when no run was watched

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
