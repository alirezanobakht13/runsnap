## Context

See proposal.md — Why. What the existing code already provides shapes most of
this:

- `_git.py` has `add_worktree`, `commit_all`, `remove_worktree`,
  `delete_branch`, and `commit_exists` — between them, everything `diff` needs to
  materialize and compare two recorded states.
- `_cli.py` has `resolve_run`, `code_state`, `download_patch`, `branch_name`,
  and `_reconstruct`, which the new commands reuse rather than reimplement.
- `checkout` already names every branch it creates `runsnap/<slug>-<short id>`,
  so its leftovers are identifiable without recording anything new.
- Capture already runs inside `start_run()` behind the `capture_code` switch and
  never raises, which is exactly where the invocation record belongs.

## Goals / Non-Goals

**Goals:**

- Answer "what changed between these two runs" in one command.
- Make a recorded run's originating command visible and copyable.
- Get a patch out of the tracking server without a repository.
- Make `checkout`'s leftovers findable and removable.

**Non-Goals:**

- Executing anything on the user's behalf. `rerun` prints; it does not run.
- Comparing metrics. The MLflow UI does that well, and a text table would be a
  worse version of it.
- Recording the environment beyond the command line. A uv project's `uv.lock` is
  a tracked file and therefore already inside the commit or the patch.
- A general `clean` for worktrees the tool did not create.

## Decisions

### `diff` compares commits, not working directories

Producing a textual difference between two recorded states means materializing
both. The obvious approach — reconstruct into two worktrees and
`git diff --no-index treeA treeB` — drags each worktree's own `.git` file into
the comparison and loses git's rename detection.

Instead each side is reconstructed into a throwaway worktree, committed with the
existing `commit_all()`, and the two resulting commits are compared with an
ordinary `git diff <sha-a> <sha-b>`. That gives real tree-level diffing, rename
detection, and binary handling for free, and the throwaway worktrees and
branches are removed in a `finally` — the spec requires nothing be left behind
even on the failure path.

Both commits must be present locally. When one is not, or a recorded patch does
not apply, the textual difference is skipped with a stated reason and the
commit, dirty-state, and param comparison are still printed. A comparison is
worth more partially than not at all.

When both runs' commits *and* patch digests are equal, the states are identical
by construction and nothing is reconstructed at all — the common case of
comparing two runs from the same code costs zero worktrees.

**Params** come from `run.data.params` on each run, compared as three sets:
present only in A, present only in B, and present in both with different values.
This reads the flattened params rather than the `hparams/*.json` artifact,
because params are what both runs are guaranteed to have — the artifact exists
only when `log_params()` was used.

### Patch capture preserves the copied index's timestamps

The existing capture requirement includes every uncommitted change. Git uses
the index's modification time to decide when matching file stat data still
needs a content check. Copying the index with `shutil.copyfile()` replaces that
timestamp, which can cause a same-size edit with unchanged file timestamps to
be omitted from the patch. Use `shutil.copy2()` to preserve it while keeping all
index writes confined to the copy. A deterministic regression models matching
timestamps, verifies that the patch reconstructs the edit, and checks that the
original index, working tree, and `HEAD` are unchanged.

### The invocation is two tags, recorded inside `capture()`

`runsnap.invocation.argv` holds `sys.argv` as a JSON array — a list, not a
pre-joined string, so the exact argument boundaries survive and quoting is the
printer's job. `runsnap.invocation.cwd` holds the working directory relative to
the repository root when it is inside the repository, and absolute otherwise;
the relative form is what makes the printed `cd` meaningful in a fresh
reconstruction.

Recording inside `capture()` rather than in `start_run()` is deliberate: it puts
the invocation behind `capture_code=False` and `RUNSNAP_CAPTURE_CODE=0`, which
matters because **argv frequently holds credentials** — `--api-key`, a token in a
URL. That exposure is the same class as the patch's, and the proposal extends the
README's existing warning rather than inventing a second switch nobody would
find.

`rerun` prints with `shlex.join()`, which is the standard library's answer to
"make this argv safe to paste into a shell".

**Non-script invocations.** In a notebook or REPL, `sys.argv[0]` is empty or an
`ipykernel_launcher.py` path, and the recorded vector is not a runnable command.
`rerun` prints what was recorded and says it does not look like a runnable
command rather than pretending otherwise; the tags are still recorded, because
knowing a run came from a notebook is itself the answer to "what produced this".

### `patch` writes bytes, and needs no repository

A patch can contain binary hunks, so output goes to `sys.stdout.buffer` or to
the named file opened in binary mode — never through text encoding. The command
resolves the run, follows `patch_holder()` to whichever run owns the artifact,
and downloads it. Nothing in that path touches a repository, which is the point:
a patch can be pulled on a machine that has never cloned the code.

`show --patch` prints the existing report and then the same bytes, flushing the
text stream first so the two do not interleave.

### `clean` identifies leftovers by the branch prefix it created them with

`git worktree list --porcelain` reports each worktree's path and branch.
A worktree whose branch is under `refs/heads/runsnap/` is one `checkout` created;
everything else is the user's and is not touched. Branches under that prefix with
no worktree — the case where someone deleted the directory by hand — are listed
too, and removed with `--remove`.

Listing is the default and `--remove` is opt-in, because this deletes
directories that may hold uncommitted work: a reconstructed tree's whole purpose
is to carry an uncommitted patch, so `git worktree remove --force` is discarding
exactly the thing the user asked `checkout` to produce. The default output names
what would be removed so the decision is informed.

`--remove` reports each removal as it happens, so an interrupted run leaves the
user knowing how far it got.

## Risks / Trade-offs

- **`diff` writes to the user's repository.** It creates two worktrees, two
  branches, and two commits. → All are removed in a `finally`, all are named
  under the `runsnap/` prefix, and `clean` finds any that survive a hard kill.
- **Recording argv uploads secrets passed on the command line.** → Called out in
  the proposal and the README, and governed by the existing capture switch. It
  is a real exposure; making it silent would be worse than making it documented.
- **`clean --remove` destroys uncommitted work by design.** → Listing is the
  default, the listing names paths and branches, and removal is a separate
  explicit flag.
- **`diff` needs both commits locally**, which a machine that only reads the
  tracking server will not have. → Degrades to the param and metadata comparison
  with a stated reason, rather than failing.
- **Two more tags per run.** → Negligible next to the patch artifact, and absent
  entirely when capture is disabled.

## Migration Plan

Additive. Runs recorded before this change carry no invocation tags; `rerun`
reports that and exits cleanly, and `diff`, `patch`, and `clean` do not depend on
them. No existing command changes behavior except `show`, which gains an
optional flag.

Ordering: apply last, after `fix-capture-fidelity` and `speed-up-run-logging`.
`diff` reports patch file listings, which the first change corrects, and both
earlier changes rework the capture path this one extends.
