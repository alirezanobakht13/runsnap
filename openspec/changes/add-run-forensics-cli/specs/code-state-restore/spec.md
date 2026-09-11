## ADDED Requirements

### Requirement: Two runs can be compared without reconstructing them by hand

`diff` SHALL report, for two named runs, each run's commit and dirty state, the
textual difference between the two recorded code states, and which params were
added, removed, or changed between them. When the textual difference cannot be
produced — a commit missing from the local repository, or a recorded patch that
does not apply — `diff` SHALL report the reason and still report the commit,
dirty-state, and param comparison.

#### Scenario: Two runs from different commits

- **WHEN** `diff` names two runs recorded at different commits
- **THEN** the output names both commits, shows the difference between the two
  code states, and lists the param differences

#### Scenario: Two runs from the same commit with different uncommitted work

- **WHEN** `diff` names two runs recorded at the same commit with different
  patches
- **THEN** the output shows the difference between the two working trees

#### Scenario: Two identical code states

- **WHEN** `diff` names two runs whose commits and patch digests are equal
- **THEN** the output reports the code states as identical and no textual
  difference is shown

#### Scenario: A commit is missing locally

- **WHEN** `diff` names a run whose commit is not in the local repository
- **THEN** the output reports which commit is missing and how to fetch it, and
  still reports the param comparison

#### Scenario: The comparison leaves nothing behind

- **WHEN** `diff` completes, whether it succeeded or reported a reason it could
  not
- **THEN** no worktree, branch, or directory it created remains

### Requirement: A run's recorded invocation can be printed

`rerun` SHALL print the command recorded for a run and the directory to run it
from, in a form that can be copied into a shell. It SHALL NOT execute anything.
A run carrying no recorded invocation SHALL be reported as such.

#### Scenario: A run with a recorded invocation

- **WHEN** `rerun` names a run recorded with an argument vector and working
  directory
- **THEN** the directory and the quoted command are printed, and no process is
  started

#### Scenario: An argument containing whitespace

- **WHEN** a recorded argument contains whitespace or shell metacharacters
- **THEN** the printed command quotes it so that a shell reproduces the original
  argument

#### Scenario: A run recorded before invocations were captured

- **WHEN** `rerun` names a run carrying no recorded invocation
- **THEN** the command reports that the run recorded no invocation and exits
  without printing a command

### Requirement: A run's patch can be exported without a repository

`patch` SHALL write a run's recorded patch bytes to standard output, or to a
file when one is named, without requiring a git repository. `show` SHALL accept
an option that prints the same bytes inline. A run recorded from a clean tree
SHALL be reported as having no patch.

#### Scenario: Exporting to standard output

- **WHEN** `patch` names a run recorded from a dirty tree
- **THEN** the recorded patch bytes are written to standard output unchanged,
  byte for byte

#### Scenario: Exporting to a file

- **WHEN** `patch` names a run and a destination file
- **THEN** the file holds the recorded patch bytes and nothing is written to
  standard output

#### Scenario: The run was recorded from a clean tree

- **WHEN** `patch` names a run recorded from a clean tree
- **THEN** the command reports that the run has no patch and writes no patch
  output

#### Scenario: Showing the patch inline

- **WHEN** `show` is invoked with the patch option on a run recorded from a
  dirty tree
- **THEN** the usual report is printed followed by the patch

### Requirement: Reconstruction leftovers can be found and removed

`clean` SHALL list the worktrees and branches that `checkout` created in a
repository. It SHALL remove nothing unless explicitly asked to, and when asked
it SHALL remove each listed worktree and its branch.

#### Scenario: Listing leftovers

- **WHEN** `clean` runs in a repository holding worktrees created by `checkout`
- **THEN** each worktree's path and branch are listed and nothing is removed

#### Scenario: Removing leftovers

- **WHEN** `clean` runs with the removal option
- **THEN** each listed worktree and its branch are removed and the removals are
  reported

#### Scenario: A branch left without its worktree

- **WHEN** a branch created by `checkout` remains after its worktree directory
  was deleted by hand
- **THEN** `clean` lists it, and with the removal option deletes the branch

#### Scenario: Nothing to clean

- **WHEN** `clean` runs in a repository with no leftovers
- **THEN** it reports that there is nothing to remove and exits successfully

#### Scenario: Unrelated worktrees are untouched

- **WHEN** `clean` runs with the removal option in a repository that also holds
  worktrees not created by `checkout`
- **THEN** those worktrees and their branches remain
