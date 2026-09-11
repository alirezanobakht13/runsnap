## Context

See proposal.md — Why. Three constraints shape the approach:

- Capture runs inside the user's training process and must never raise into it.
  Every correction below stays inside that rule.
- `patch_files()` parses patch bytes downloaded from the tracking server, with
  no repository and possibly no `git` at hand. It has to stay a pure function
  over bytes.
- A patch record for a **binary** file carries `GIT binary patch` and no
  `--- a/… / +++ b/…` pair, and a pure rename carries `rename from/to` and no
  pair either. Only the `diff --git` header is present on every record.

## Goals / Non-Goals

**Goals:**

- Every recorded code state belongs to the run it is recorded on.
- The patch file listing is complete for every path git can name.
- One error path for "this leaf is not a single number", whatever the leaf is.
- No reliance on `tensorboardX` private attributes.

**Non-Goals:**

- Reducing the cost of `start_run()` (see the `speed-up-run-logging` change).
- Turning uncaught `GitError` into a clean CLI message.
- Changing what a patch contains, or the security posture around it.

## Decisions

### Drop the code-state cache outright rather than probe for freshness

`resolve_code_state()` and `resolve_repo_root()` become plain calls. A freshness
probe (compare `HEAD` and working-tree mtimes) was considered and rejected: the
probe is itself a git invocation, so it saves only the `diff`, and it reintroduces
the same class of bug the moment the probe misses a case. Correct by construction
beats fast and subtly wrong for a tool whose output is a provenance claim.

`reset_code_state_cache()` exists only to undo the cache in tests and is removed
with it; `tests/conftest.py` loses that teardown.

### Nested-run patch reuse must key on the patch digest

This one is forced by the decision above. Today a nested run inherits its
parent's uploaded patch whenever the parent id appears in `_patch_holders`,
which is safe *only* because the cache guaranteed the two patches were byte
identical. Without the cache a child's tree can legitimately differ from its
parent's, and blind inheritance would tag the child with a patch that is not its
own — trading a stale-cache bug for a worse one.

`_patch_holders` therefore maps a run id to the pair *(holder run id, patch
digest)*, and a child inherits only when its own digest equals the recorded one.
On a mismatch it uploads its own patch, as an unparented run would.

### Parse the `diff --git` header, not the `---`/`+++` pair

The `+++ b/<path>` line is the tempting target — one path, unambiguous — but it
is absent for binary records and pure renames, so using it alone would drop
exactly the files the current code already drops plus binaries. The header is
the only line present on every record.

Parsing rules, applied per record:

1. If the header's remainder begins with `"`, it holds two C-quoted tokens.
   Scan both, unquote the second, strip its `b/` prefix. Git quotes both sides
   whenever either needs it, so this case is self-identifying.
2. Otherwise the remainder is `a/P b/Q`. Choose the ` b/` split point where the
   text before and after it match (`P == Q`), which resolves paths that contain
   ` b/` themselves.
3. If no split matches, the record is a rename: read its `rename to <path>`
   line and unquote if quoted.

Unquoting implements git's C-escape rules — `\n \t \r \" \\` and `\NNN` octal
bytes — then decodes UTF-8 with `errors="surrogateescape"`, so a path that is
not valid UTF-8 round-trips to the same `str` the filesystem APIs use.

Shelling out to `git apply --numstat -z` and letting git unquote was considered
and rejected: it puts a subprocess and a repository requirement behind a
function that today needs neither, and `runsnap show` would stop working on a
machine without git.

### One error path for multi-element leaves

`_leaf()` currently catches only `ValueError`, which is numpy's error; torch
raises `RuntimeError` and a plain `list` raises nothing at all. The leaf check
becomes:

- A non-string sequence with more than one element, or an object whose `.item()`
  fails with `ValueError`, `RuntimeError`, or `TypeError` → `ValueError` naming
  the key.
- A sequence or array of exactly one element → that element.
- An empty sequence → omitted, like any other leaf carrying no number. There is
  nothing to chart and no ambiguity about the caller's intent.

`str` and `bytes` are excluded from the sequence branch before the length test,
so they stay silently-dropped leaves as they are today.

### Locate event files by the writer's own filename suffix

`_event_path()` walks `writer.file_writer.event_writer._ev_writer._file_name`.
Each writer is already constructed with a suffix unique to its role — `.scalars`
for the light writer, `.media.<n>` for each shard — so the file can be found by
globbing the writer's `get_logdir()` for `events.out.tfevents.*<suffix>`, using
only public API. That glob matches exactly one file by construction.

`pyproject.toml` gains `tensorboardx<3` so a major release cannot change the
file-naming contract underneath the glob without a deliberate bump.

### Assign viewer names in two passes

The current single pass pre-seeds `reserved` with every run name, then
disambiguates only names it saw twice — so a run whose *own* name matches a
generated `<base>-<short id>` collides and `symlink_to` raises. Two passes fix
it: pass one claims every non-colliding name, pass two assigns colliding runs a
`<base>-<short id>`, incrementing to the full run id and then a counter until
the name is unclaimed.

### Replace unrecognized cache entries rather than inspecting them

`fetch_run()` currently writes a link only when nothing is there
(`if not link.is_symlink()`), which raises on a regular file and silently keeps a
symlink pointing at a stale target. Since the link is derived data with a known
correct value, the entry is removed and rewritten whenever it is not already a
symlink to the intended target.

## Risks / Trade-offs

- **Dropping the cache adds a `git add -N` and a `git diff` per run.** → Real but
  small next to the tracking-server round trips `start_run()` already makes. A
  driver starting thousands of runs in one process is the case that would feel
  it; `speed-up-run-logging` bounds the diff cost separately.
- **The digest-keyed nested-run rule can upload the same patch twice** when a
  parent and child genuinely differ. → Correct behavior, and the size ceiling
  still applies to each.
- **The header parser is more code than the line it replaces.** → It is a pure
  function over bytes with no I/O, so it is cheap to test exhaustively; the spec
  scenarios name the cases.
- **The suffix glob depends on `filename_suffix` reaching the filename.** →
  Public, documented `tensorboardX` behavior, now pinned below the next major.
- **Raising on a list of numbers is a behavior change** for callers who relied on
  such fields being dropped. → Their metric was silently missing from every
  chart, so the error surfaces a defect rather than creating one; called out in
  the proposal as part of the metrics contract.

## Migration Plan

No data migration: tags and artifacts already written keep their meaning, and
patches recorded before this change parse the same or better. The only consumer
change is internal — `reset_code_state_cache()` disappears from
`tests/conftest.py`.

Ordering: apply this change before `speed-up-run-logging`, which reworks
`build_patch()` and the params path, and before `add-run-forensics-cli`, which
builds new commands on `patch_files()` and on the capture path.
