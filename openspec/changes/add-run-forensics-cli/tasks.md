## 1. Invocation capture

- [ ] 1.1 Add `TAG_INVOCATION_ARGV` and `TAG_INVOCATION_CWD` to `_tags.py` under
  a `runsnap.invocation.` prefix; verify the names match the proposal
- [ ] 1.2 Record `sys.argv` as a JSON array and the working directory relative to
  the repository root (absolute when outside it) from within `capture()`,
  warning rather than raising on a refused write; verify tests cover a run from
  a subdirectory, a run from outside the repository, and a refusing client
- [ ] 1.3 Verify that `capture_code=False` and `RUNSNAP_CAPTURE_CODE=0` each
  suppress the invocation tags along with the code state

## 2. `runsnap rerun`

- [ ] 2.1 Add the `rerun` command printing the recorded directory and the
  `shlex.join()`ed command; verify a test that an argument containing spaces
  round-trips through a shell-quoted form
- [ ] 2.2 Report a run carrying no invocation tags without printing a command;
  verify a test against a run captured before this change
- [ ] 2.3 Report a recorded vector that is not a runnable command, such as a
  notebook kernel launcher; verify a test asserting the message and that nothing
  is executed

## 3. `runsnap patch` and `show --patch`

- [ ] 3.1 Add the `patch` command writing raw bytes to `sys.stdout.buffer`, or to
  a file named by `--output`; verify a test that a patch containing binary hunks
  is written byte for byte
- [ ] 3.2 Report a run recorded from a clean tree as having no patch, writing no
  patch output; verify a test asserting the exit status and empty output
- [ ] 3.3 Add `--patch` to `show`, flushing the text stream before the bytes;
  verify a test that the report precedes the patch in the captured output
- [ ] 3.4 Verify `patch` works with no git repository in the working directory

## 4. `runsnap diff`

- [ ] 4.1 Add `list_worktrees(root)` to `_git.py` parsing
  `git worktree list --porcelain` into path and branch pairs; verify a test
  against a repository with two worktrees
- [ ] 4.2 Add a diff helper to `_git.py` comparing two commits; verify a test
  that it reports a renamed file as a rename
- [ ] 4.3 Implement `diff` for the identical-state case — equal commits and
  equal patch digests — reporting identity without creating a worktree; verify
  a test asserting no worktree is created
- [ ] 4.4 Implement the reconstruct-commit-compare path, removing both
  throwaway worktrees and branches in a `finally`; verify tests that the diff
  reflects both committed and uncommitted differences and that the repository
  holds no leftover worktree or branch afterwards, including when the comparison
  raises
- [ ] 4.5 Implement the param comparison — added, removed, and changed — from
  each run's params; verify a test covering all three categories
- [ ] 4.6 Degrade to the metadata and param comparison with a stated reason when
  a commit is missing locally or a patch does not apply; verify tests for both
  reasons

## 5. `runsnap clean`

- [ ] 5.1 Implement listing of worktrees whose branch is under `refs/heads/runsnap/`
  plus branches under that prefix with no worktree; verify a test that unrelated
  worktrees and branches are absent from the listing
- [ ] 5.2 Implement `--remove`, deleting each listed worktree and branch and
  reporting each removal; verify a test that unrelated worktrees survive and the
  listed ones are gone
- [ ] 5.3 Report the empty case successfully; verify a test in a repository with
  no leftovers

## 6. Final pass

- [ ] 6.1 Document `diff`, `rerun`, `patch`, `show --patch`, and `clean` in the
  README, and extend the existing secret warning to the recorded argument vector
- [ ] 6.2 Run `uv run pytest`, `uv run ruff check`, `uv run ruff format`, and
  `uv run ty check`; verify all pass
- [ ] 6.3 Run `openspec validate --changes add-run-forensics-cli --strict` and
  verify it reports no errors
