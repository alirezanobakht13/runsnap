## Purpose

Records the exact state of the code that produced an MLflow run — the base commit plus every uncommitted change in the working tree — so that any run can later be reconstructed byte-for-byte. Capture happens at run start and never interferes with the user's git working state or with normal MLflow usage.

## Requirements

### Requirement: Run creation passes through to MLflow

`runsnap.start_run()` SHALL accept the same arguments as `mlflow.start_run()`, forward them unchanged, and return an object exposing the same interface as the MLflow `ActiveRun` it wraps, usable as a context manager and as a plain run ended by `mlflow.end_run()`. Code-state capture and lifecycle recording (status and failure cause on exit, and the `continues` link) SHALL be the only added behaviors; all subsequent interaction with the run — logging metrics, artifacts, models, tags — SHALL require no `runsnap` API.

#### Scenario: Existing MLflow code keeps working

- **WHEN** a user replaces `mlflow.start_run(run_name="x", nested=True)` with `runsnap.start_run(run_name="x", nested=True)` and continues to call `mlflow.log_metric` inside the block
- **THEN** the run is created with the same name and nesting, the metrics are logged to it, and the block exits with the same status MLflow would give it, except that a `KeyboardInterrupt` ends it as `KILLED`

#### Scenario: Return value is the MLflow run

- **WHEN** a user writes `with runsnap.start_run() as run:`
- **THEN** `run` exposes `run.info.run_id` and the rest of the MLflow `ActiveRun` interface, and `mlflow.active_run()` inside the block reports the same run id

#### Scenario: Run used without a context manager

- **WHEN** a user calls `run = runsnap.start_run()` and later `mlflow.end_run()`
- **THEN** the run ends as `FINISHED` exactly as with `mlflow.start_run()`

### Requirement: Base commit is recorded

When the process runs inside a git repository, capture SHALL record the full SHA of the commit at `HEAD`, the active branch name, and the repository's remote URL when one exists.

#### Scenario: Commit and branch recorded

- **WHEN** a run starts from a repository whose `HEAD` is commit `abc123…` on branch `main`
- **THEN** the run carries tag `runsnap.git.commit` = `abc123…` and tag `runsnap.git.branch` = `main`

#### Scenario: Remote URL recorded without credentials

- **WHEN** the repository's remote URL embeds credentials, such as `https://user:token@host/org/repo.git`
- **THEN** tag `runsnap.git.repo_url` records the URL with the credentials removed

#### Scenario: Detached HEAD

- **WHEN** a run starts while `HEAD` is detached
- **THEN** the commit SHA is still recorded and no branch tag is set

### Requirement: Uncommitted work is captured as an applicable patch

Capture SHALL record all differences between `HEAD` and the working tree as a single patch, stored as the run artifact `code/state.patch`. The patch SHALL be taken against a single base so that it applies cleanly to a checkout of the recorded commit, and SHALL include staged changes, unstaged changes, newly added untracked files, deleted files, file-mode changes, and binary file contents. Files excluded by git's ignore rules SHALL NOT appear in the patch.

#### Scenario: Staged and unstaged edits to one file

- **WHEN** a file has been edited, staged, and then edited again before the run starts
- **THEN** the recorded patch applies cleanly to a checkout of the recorded commit and reproduces the file's final working-tree content

#### Scenario: Untracked file included

- **WHEN** the working tree contains a new file that has never been added to git and is not ignored
- **THEN** applying the recorded patch to a checkout of the recorded commit recreates that file with its content

#### Scenario: Deletions and mode changes included

- **WHEN** a tracked file has been deleted and another has had its executable bit set
- **THEN** applying the recorded patch reproduces both the deletion and the mode change

#### Scenario: Binary content included

- **WHEN** the working tree contains a modified or newly added binary file
- **THEN** applying the recorded patch reproduces that file's exact bytes

#### Scenario: Ignored files excluded

- **WHEN** the working tree contains files matched by `.gitignore`, such as a virtual environment or a local database
- **THEN** those files do not appear in the recorded patch

### Requirement: Capture does not disturb the user's git state

Capture SHALL leave the repository's staging area, working tree, and `HEAD` exactly as they were. Running a capture SHALL NOT change what `git status` reports.

#### Scenario: Staging area preserved

- **WHEN** a user has staged one file and left another modified, and then starts a run
- **THEN** after the run starts, `git status` reports the same staged file and the same modified file as before

### Requirement: Clean working tree records no patch

When the working tree has no changes relative to `HEAD`, capture SHALL record that the tree was clean and SHALL NOT write a patch artifact.

#### Scenario: Clean tree

- **WHEN** a run starts from a repository with no uncommitted changes
- **THEN** tag `runsnap.git.dirty` is `false` and the run has no `code/state.patch` artifact

#### Scenario: Dirty tree

- **WHEN** a run starts from a repository with uncommitted changes
- **THEN** tag `runsnap.git.dirty` is `true` and the run has a `code/state.patch` artifact

### Requirement: Identical code states are discoverable

Capture SHALL record a `sha256` digest of the patch content as tag `runsnap.git.patch_sha256`, so that runs sharing an identical code state can be identified by searching on the commit and digest tags.

#### Scenario: Two runs from an unchanged working tree

- **WHEN** two runs are started from the same commit with the same uncommitted changes
- **THEN** both runs carry the same `runsnap.git.commit` and the same `runsnap.git.patch_sha256`

#### Scenario: Digest changes when code changes

- **WHEN** a file is edited between two runs started from the same commit
- **THEN** the two runs carry different `runsnap.git.patch_sha256` values

### Requirement: Nested runs reference their parent's patch

A run started while another `runsnap` run is active SHALL receive the same code-state tags but SHALL NOT duplicate the patch artifact. It SHALL instead carry tag `runsnap.git.patch_run_id` identifying the run that holds the artifact.

#### Scenario: Hyperparameter sweep

- **WHEN** a parent run starts and then launches fifty nested child runs
- **THEN** the patch artifact is stored once on the parent, and each child carries the code-state tags plus `runsnap.git.patch_run_id` pointing at the parent

### Requirement: Oversized patches are skipped, not uploaded

Capture SHALL enforce a configurable maximum patch size. When the patch exceeds it, capture SHALL skip the artifact, record the reason, and allow the run to proceed.

#### Scenario: Patch over the limit

- **WHEN** the working tree's changes produce a patch larger than the configured maximum
- **THEN** no `code/state.patch` artifact is written, tag `runsnap.git.capture_error` records that the patch was too large, the run starts normally, and a warning is emitted

### Requirement: Capture never fails the experiment

A failure to capture code state SHALL NOT prevent the run from starting or raise into user code. Every failure mode SHALL emit a warning and record what could be determined.

#### Scenario: Not a git repository

- **WHEN** a run starts from a directory that is not inside a git repository
- **THEN** the run is created normally, no code-state tags are set, and a warning is emitted

#### Scenario: Git binary unavailable

- **WHEN** `git` is not on `PATH`
- **THEN** the run is created normally and a warning is emitted

#### Scenario: Repository with no commits

- **WHEN** a run starts from an initialized repository that has no commit yet
- **THEN** the run is created normally, tag `runsnap.git.capture_error` records that there is no base commit, and no patch artifact is written

#### Scenario: Unexpected git failure

- **WHEN** a git command fails for any other reason during capture
- **THEN** the exception does not propagate to user code, the run proceeds, and a warning is emitted

### Requirement: Capture is configurable and can be disabled

Capture SHALL be enabled by default and SHALL be disableable per call and by environment variable. The maximum patch size SHALL be configurable by environment variable.

#### Scenario: Disabled per call

- **WHEN** a user calls `runsnap.start_run(capture_code=False)`
- **THEN** the run is created with no code-state tags and no patch artifact

#### Scenario: Disabled by environment

- **WHEN** the environment sets `RUNSNAP_CAPTURE_CODE=0`
- **THEN** no run started through `runsnap.start_run()` records code state

### Requirement: MLflow's standard git tags are populated when absent

Capture SHALL mirror the recorded commit, branch, repository URL, and dirty flag into MLflow's standard `mlflow.source.git.*` tags when MLflow has not already set them, so that MLflow's own source display works in contexts where MLflow cannot resolve the repository, such as notebooks and REPLs. Capture SHALL NOT write the patch into any tag.

#### Scenario: Notebook run

- **WHEN** a run is started from a Jupyter kernel inside a git repository, where MLflow's built-in git detection resolves nothing
- **THEN** the run carries `mlflow.source.git.commit` and `mlflow.source.git.branch` matching the captured values

#### Scenario: Existing MLflow tags are not overwritten

- **WHEN** MLflow has already resolved and set `mlflow.source.git.commit` for the run
- **THEN** capture leaves that tag's value unchanged

#### Scenario: Patch is never stored as a tag

- **WHEN** any run records a patch
- **THEN** the patch content appears only in the `code/state.patch` artifact and in no tag, so it is never subject to tag length truncation

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
