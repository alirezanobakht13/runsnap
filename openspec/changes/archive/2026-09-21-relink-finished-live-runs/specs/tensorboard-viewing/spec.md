## ADDED Requirements

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
