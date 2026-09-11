## Purpose

Reconstructs the code that produced a recorded MLflow run, placing it on a new
branch at the run's base commit with its uncommitted changes reapplied, so the
researcher can read, run, or diff the exact codebase that produced a result.

## ADDED Requirements

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
