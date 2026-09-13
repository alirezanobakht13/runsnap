## Why

A run's recorded code state currently supports exactly two questions: what did
this run record, and put it back on disk. The questions that actually come up
while working through a set of experiments are different ones — *what changed
between these two runs*, *what command produced this run*, and *give me the
patch* — and each of them today means reconstructing a worktree and comparing by
hand. The data to answer them is already on the tracking server.

`checkout` also leaves worktrees and branches behind and offers nothing to clean
them up, so a week of comparisons accumulates directories beside the repository
that nobody remembers creating.

## What Changes

- `runsnap diff <run-a> <run-b>` reports what changed between two runs: their
  commits, whether each tree was dirty, the textual difference between the two
  recorded code states, and which params were added, removed, or changed.
- Capture records the command that started the run — its argument vector and its
  working directory relative to the repository root — as `runsnap.invocation.*`
  tags.
- Patch capture preserves the copied Git index's timestamps so same-size edits
  with unchanged file timestamps remain visible to Git's content checks.
- `runsnap rerun <run>` prints that recorded invocation, ready to copy. It never
  executes anything.
- `runsnap patch <run>` writes a run's recorded patch to standard output or to a
  file, without needing a repository. `runsnap show --patch` prints it inline.
- `runsnap clean` lists the worktrees and branches `checkout` created and, with
  `--remove`, deletes them. Listing is the default; nothing is removed without
  the flag.

**The recorded argument vector is uploaded to the tracking server, so a secret
passed on the command line is uploaded with it.** This is the same exposure the
patch already carries for uncommitted tracked files, and it is disabled by the
same switches: `capture_code=False` and `RUNSNAP_CAPTURE_CODE=0`.

## Capabilities

### New Capabilities
<!-- None: both capabilities are introduced by earlier changes in this sequence. -->

### Modified Capabilities
- `code-state-capture`: a run additionally records the command and working
  directory that started it.
- `code-state-restore`: four new commands — `diff`, `rerun`, `patch`, `clean` —
  and a `--patch` option on `show`.

## Impact

- `src/runsnap/_cli.py`, `_capture.py`, `_git.py`, `_tags.py`
- New `runsnap.invocation.argv` and `runsnap.invocation.cwd` tags. Runs recorded
  before this change carry neither, and `rerun` reports that rather than failing.
- `diff` reconstructs both runs into throwaway worktrees to produce a textual
  difference, so it needs a repository holding both commits. It removes what it
  creates.
- README gains the new commands and a line extending the existing secret warning
  to the argument vector.
- Depends on `fix-capture-fidelity` (correct patch file listings, which `diff`
  reports) and on `speed-up-run-logging` (the capture path both changes rework).
