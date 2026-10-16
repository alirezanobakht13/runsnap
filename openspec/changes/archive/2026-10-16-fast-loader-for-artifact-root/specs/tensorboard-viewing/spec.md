## REMOVED Requirements

### Requirement: Runs can be viewed straight from a local artifact folder

**Reason**: Runs viewed from an artifact folder are now shown through the dashboard's own copies of their files, so its "Nothing is copied" scenario no longer holds. Replaced by "Runs can be viewed from copies of a local artifact folder", which keeps every other part of this requirement.

**Migration**: None for users. `--artifact-root` keeps its meaning and its restrictions.

## ADDED Requirements

### Requirement: Runs can be viewed from copies of a local artifact folder

Given `--artifact-root` naming the folder where the tracking server stores proxied artifacts, `runsnap tb` SHALL show each run from its files in that folder, through the dashboard's own temporary copies read with TensorBoard's default data loader. It SHALL NOT download event files, write to the download cache, or change anything in that folder. A run whose artifacts do not lie under that folder SHALL be reported as a warning and left out.

#### Scenario: Nothing is downloaded

- **WHEN** `runsnap tb --artifact-root /var/lib/mlflow/artifacts` shows runs
- **THEN** no event file is downloaded, the download cache is not written, and nothing in the folder changes

#### Scenario: The copies are temporary

- **WHEN** the command exits
- **THEN** the dashboard's copies of the shown runs' files are removed, and the files in the folder are left untouched

#### Scenario: TensorBoard's default loader

- **WHEN** TensorBoard is started on runs viewed from an artifact folder
- **THEN** it is started with its default data loader, as it is for runs viewed through downloads

#### Scenario: A run stored elsewhere

- **WHEN** a selected run's artifacts are not stored under the given folder
- **THEN** a warning names the run, the run is left out, and the other runs are shown

#### Scenario: A deleted run's files are left to MLflow

- **WHEN** a run shown from the folder is deleted in MLflow
- **THEN** the run leaves the dashboard, the dashboard's copies of its files are removed, and its files in the folder are left untouched

## MODIFIED Requirements

### Requirement: Runs viewed from an artifact folder stay current

When viewing from an artifact folder, TensorBoard SHALL show the events the tracking server stores for a shown run while MLflow reports it running, including events in a file the server replaced with a longer copy and events in any of the run's scalar shards, and SHALL show the run's final events once after it ends. A failure to read one run's files SHALL NOT stop the dashboard or affect the other runs.

#### Scenario: The server replaces a file with a longer copy

- **WHEN** the tracking server replaces a shown run's scalar file with a longer copy after an upload
- **THEN** TensorBoard shows the new points without the command being restarted

#### Scenario: A new shard appears

- **WHEN** a shown run's writer rolls to a new scalar shard and that shard reaches the folder
- **THEN** the points in the new shard appear in TensorBoard

#### Scenario: An earlier shard grows after a later one

- **WHEN** a running run's earlier scalar shard reaches the folder with more events after a later shard has already appeared
- **THEN** the earlier shard's new points appear in TensorBoard as well

#### Scenario: A run finishes while shown

- **WHEN** a shown run's status changes from `RUNNING` to `FINISHED`
- **THEN** its files are read once more, showing its final events, and are not read again while the dashboard stays open

#### Scenario: Reading a run's files fails

- **WHEN** reading a shown run's files from the folder fails during a check
- **THEN** a warning is reported, the run stays shown with the data it had, and the read is tried again at the next check; while the failure persists it is reported once
