## Purpose

Reconstructs the code that produced a recorded MLflow run, placing it on a new branch at the run's base commit with its uncommitted changes reapplied, so the researcher can read, run, or diff the exact codebase that produced a result.

## Requirements

### Requirement: A run's code state can be reconstructed by reference

The `runsnap checkout` command SHALL take a reference to a recorded run, create a branch at that run's base commit, and reapply that run's recorded patch, producing a working tree matching the run's original code state.

#### Scenario: Reconstructing a dirty run

- **WHEN** a user runs `runsnap checkout <run-id>` for a run captured from a dirty working tree
- **THEN** a branch is created at the recorded commit, the recorded patch is applied on top, and the resulting working tree matches the state the run was started from

#### Scenario: Reconstructing a clean run

- **WHEN** a user checks out a run captured from a clean working tree
- **THEN** a branch is created at the recorded commit with no patch applied, and the command reports that the run's tree was clean

#### Scenario: Nested run resolves to its parent's patch

- **WHEN** a user checks out a child run whose patch is stored on its parent
- **THEN** the patch is retrieved from the run identified by the child's recorded pointer and applied normally

### Requirement: Reapplied changes are left uncommitted by default

Reconstruction SHALL leave the reapplied patch as uncommitted working-tree changes, faithfully reproducing the run's original dirty state and making the uncommitted portion visible to `git diff`. A flag SHALL commit the applied patch instead, for a durable single-commit snapshot.

#### Scenario: Default leaves the tree dirty

- **WHEN** a user checks out a run captured from a dirty tree without extra flags
- **THEN** `git status` in the reconstructed tree shows the same uncommitted changes the run was started with

#### Scenario: Committing the snapshot

- **WHEN** a user passes the flag that commits the applied patch
- **THEN** the reconstructed branch has one additional commit containing the patch, and its working tree is clean

### Requirement: Reconstruction targets a separate worktree by default

Reconstruction SHALL default to a separate git worktree, leaving the user's current checkout untouched. A flag SHALL instead switch the current working tree to the new branch, and another SHALL set the worktree's location. The command SHALL report the path of the reconstructed tree.

#### Scenario: Default worktree

- **WHEN** a user checks out a run while in the middle of unrelated work
- **THEN** a new worktree is created in a separate directory, the user's current checkout and its uncommitted changes are untouched, and the new directory's path is printed

#### Scenario: Explicit worktree path

- **WHEN** a user supplies a target path
- **THEN** the worktree is created there

#### Scenario: In-place checkout

- **WHEN** a user passes the flag requesting an in-place checkout
- **THEN** the current working tree switches to the new branch and the patch is applied there, with no worktree created

#### Scenario: In-place refuses to clobber uncommitted work

- **WHEN** a user requests an in-place checkout while the current working tree has uncommitted changes
- **THEN** the command fails with an error explaining the conflict and changes nothing, unless the user passes the flag that overrides this check

### Requirement: Branch names are derived and overridable

The created branch SHALL be named from the run so that it is recognizable and unlikely to collide. A flag SHALL override the name. An existing branch of the target name SHALL cause a clear failure rather than being reused or overwritten.

#### Scenario: Derived name

- **WHEN** a user checks out a run named `baseline-lr3` with id `a1b2c3…`
- **THEN** the created branch name incorporates both the run name and a short form of the run id

#### Scenario: Name already taken

- **WHEN** the target branch name already exists
- **THEN** the command fails, names the existing branch, and suggests supplying an explicit name

### Requirement: Runs can be referenced by id or by name

A run reference SHALL be accepted as either an MLflow run id or a run name. A name SHALL be resolved by searching runs, optionally narrowed to a named experiment. Ambiguous or unmatched names SHALL fail with an actionable message.

#### Scenario: Reference by run id

- **WHEN** the reference is a value in MLflow's run id form
- **THEN** it is treated as a run id and looked up directly

#### Scenario: Reference by run name

- **WHEN** the reference is not in run id form and exactly one run has that name
- **THEN** that run is used

#### Scenario: Ambiguous run name

- **WHEN** several runs share the given name
- **THEN** the command fails, lists the matching run ids with their experiments, and asks the user to specify one

#### Scenario: Unknown reference

- **WHEN** no run matches the reference
- **THEN** the command fails with an error naming the reference and the tracking server that was searched

### Requirement: Preconditions are checked with actionable errors

The command SHALL verify that it can complete before modifying anything, and SHALL fail with a message stating what to do next when it cannot.

#### Scenario: Base commit missing locally

- **WHEN** the run's recorded commit is not present in the local repository
- **THEN** the command fails, names the missing commit, and suggests fetching it

#### Scenario: Run has no recorded code state

- **WHEN** the referenced run carries no code-state tags
- **THEN** the command fails, explaining that the run was not captured with code-state tracking

#### Scenario: Patch does not apply

- **WHEN** the recorded patch fails to apply to the recorded commit
- **THEN** the command fails, reports git's reason, and leaves no partially reconstructed branch or worktree behind

#### Scenario: Not in a git repository

- **WHEN** the command is run outside a git repository and no repository path is supplied
- **THEN** the command fails, explaining that a repository is required and how to point at one

### Requirement: A run's recorded code state can be inspected without reconstruction

The `runsnap show` command SHALL print a run's recorded code state — base commit, branch, dirty flag, patch digest, and a summary of the files the patch touches — without creating branches or worktrees.

#### Scenario: Inspecting a run

- **WHEN** a user runs `runsnap show <run-ref>` for a captured run
- **THEN** the recorded commit, branch, dirty flag, patch digest, and the list of files the patch changes are printed, and the repository is not modified

#### Scenario: Comparing code across runs

- **WHEN** a user inspects two runs and compares the printed commit and patch digest
- **THEN** identical values across both runs indicate the runs used identical code

### Requirement: The tracking server is resolved the way MLflow resolves it

The CLI SHALL read the MLflow tracking server from the standard MLflow environment configuration, and SHALL accept a flag that overrides it for a single invocation.

#### Scenario: Environment configuration

- **WHEN** the MLflow tracking URI is set in the environment
- **THEN** the CLI queries that server without further configuration

#### Scenario: Per-invocation override

- **WHEN** the user supplies a tracking URI flag
- **THEN** the CLI queries that server instead

### Requirement: A recorded patch's file listing names every file the patch touches

The file listing reported for a run's recorded patch SHALL name every path the
patch adds, modifies, renames, or deletes, including paths the version control
system emits in quoted form because they hold non-ASCII bytes, double quotes, or
backslashes, and including files whose content is binary. A quoted path SHALL be
reported as the literal path on disk, unescaped.

#### Scenario: A patched file has a non-ASCII name

- **WHEN** a run's patch modifies a file whose name contains non-ASCII characters
- **THEN** the file listing includes that file under its literal name

#### Scenario: A patched file's name contains a double quote

- **WHEN** a run's patch adds a file whose name contains a double quote
- **THEN** the file listing includes that file, with the quote unescaped

#### Scenario: A patched path contains the patch header's own separator text

- **WHEN** a run's patch touches a path containing the character sequence used
  inside patch headers to separate the two sides of a diff
- **THEN** the file listing reports the whole path, not a truncated suffix

#### Scenario: A patched file is binary

- **WHEN** a run's patch adds a file whose content is binary
- **THEN** the file listing includes that file

#### Scenario: A patched file was renamed

- **WHEN** a run's patch renames a file
- **THEN** the file listing reports the file's new path

### Requirement: A failed reconstruction leaves the repository where it started

When reconstruction cannot apply a run's recorded patch, it SHALL restore the
repository to the commit or branch it stood at before the command ran, remove the
branch it created, and report the failure. This SHALL hold whether the repository
was on a branch or at a detached `HEAD`.

#### Scenario: The patch does not apply and HEAD was detached

- **WHEN** an in-place reconstruction runs in a repository at a detached `HEAD`
  and the run's recorded patch does not apply to the run's commit
- **THEN** the repository is restored to the commit `HEAD` stood at before the
  command, the created branch is gone, and the failure is reported

#### Scenario: The patch does not apply and HEAD was on a branch

- **WHEN** an in-place reconstruction runs in a repository on a branch and the
  run's recorded patch does not apply to the run's commit
- **THEN** the repository is restored to that branch, the created branch is gone,
  and the failure is reported

#### Scenario: The patch does not apply in a worktree

- **WHEN** reconstruction creates a worktree and the run's recorded patch does
  not apply to the run's commit
- **THEN** the worktree and the created branch are removed and the failure is
  reported
