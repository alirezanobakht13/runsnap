## Context

See proposal.md — Why. Constraints that shape the approach:

- Logging runs inside the training process. Nothing here may raise into it, and
  nothing may add latency to the training loop itself.
- TensorBoard merges every event file in a directory into one run, which is what
  makes sharding invisible to a viewer — and what lets old single-file runs and
  new sharded runs sit side by side.
- Runs already uploaded keep their existing artifact layout. Anything that reads
  `tb/` has to understand both shapes.
- `fix-capture-fidelity` removes the process-wide code-state cache, so
  `build_patch()` now runs once per run rather than once per process. Its cost
  is on the critical path in a way it was not before.

## Goals / Non-Goals

**Goals:**

- Round trips proportional to batches, not to leaves.
- Upload traffic proportional to bytes logged, not to bytes logged times run
  length.
- Memory during capture bounded by the configured ceiling, whatever the tree.
- Server queries proportional to the invocation, not to the number of arguments.

**Non-Goals:**

- Changing what is logged, only how many calls it takes.
- Making the training loop's `add_scalar` faster; it already writes to a local
  buffer.
- Incremental or delta upload of event files. Sharding gets the same benefit
  without an artifact-store feature that may not exist.

## Decisions

### Batch params at the tracking server's own limit

`log_params()` builds `Param` entities and calls `log_batch()` in chunks of
`MAX_PARAMS_TAGS_PER_BATCH`, imported from `mlflow.utils.validation` rather than
hardcoded, so a server-side change to the limit travels with the dependency.

The per-key length warning stays where it is, emitted while building the
entities, so the message still names the key before anything is sent.

This changes partial-failure behavior: today a refused param leaves every
earlier param logged and the rest absent; with batching a refused chunk leaves
the whole chunk absent. That is a better failure to have — a batch boundary is
at least a documented unit — and `log_params()` already writes its
authoritative artifact separately.

### Shard the scalar stream, but keep uploading the open scalar shard

The media policy already has the right shape: seal at a threshold, upload a
sealed file exactly once, skip the open one. Applying it to scalars unchanged
would break live viewing, because the open shard is exactly where the newest
curves are. So the two streams differ in one respect only:

| | threshold | open file |
|---|---|---|
| media | 8 MiB | skipped until sealed |
| scalars | 1 MiB | uploaded whenever it grew |

A sync pass therefore re-uploads at most one threshold's worth of scalar data
instead of the whole history, which is the quadratic term removed. 1 MiB is
roughly ten thousand scalar events — a run producing that inside one sync
interval is producing data worth the upload.

`_Sync`'s `skip_open_shard` flag narrows to "skip the open *media* shard", which
is what it always meant.

**Naming and backward compatibility.** Scalar files become
`…tfevents.<stamp>.<host>.scalars.<n>` where they were `….scalars`. The fetch
path classifies a file as scalar-or-media by suffix, so its predicate changes
from `endswith(".scalars")` to "the suffix is `.scalars` or `.scalars.<digits>`".
Runs uploaded before this change keep classifying as scalar files, and a mixed
`tb/` directory — possible when a run is resumed by a newer version — works
either way.

### Stream the patch to disk under a hard cap, and drop the digest when capped

`build_patch()` gains a byte ceiling. It reads git's stdout incrementally,
hashing as it goes, and stops the moment the total passes the ceiling — killing
the subprocess rather than draining it. Memory is then bounded by the read
buffer regardless of the tree, which is the actual defect: the ceiling exists to
defend against a huge untracked tree, and today that tree is fully materialized
in memory before the ceiling is consulted.

Consequences that have to be accepted rather than worked around:

- **The digest is lost for an oversize tree.** A patch that was never read to the
  end cannot be hashed. Buffering to a temp file to hash it anyway was
  considered and rejected: an oversize run has no patch artifact, so nothing can
  reconstruct it, and a digest that cannot be paired with a patch answers no
  question the README's "equal commit and digest mean identical code" rule asks.
- **`CodeState.dirty` stops being `bool(self.patch)`.** An abandoned read leaves
  no patch but the tree is still dirty, so `dirty` becomes an explicit field set
  from what git reported, not derived from the patch.

`build_patch()` signals the overrun by raising a `PatchTooLarge` (a `GitError`
subclass) carrying the ceiling, which `capture()` catches alongside the existing
error handling to write the warning and capture-error tag it writes today.

Note that `git add -A -N` records intent-to-add without hashing content — the
expensive work is all in the `diff`, so capping the diff read caps the capture.

### Fetch runs concurrently, and let one failure stand alone

`assemble_logdir()` fetches through a `ThreadPoolExecutor` bounded at
`min(8, len(runs))`. Each task is one `fetch_run()`, which is independent:
separate cache directories, separate downloads, no shared mutable state. The
`_live_logdir()` check stays in the calling thread because it is two tag reads.

A failed fetch currently aborts the whole command. Since the point of the
command is to look at several runs at once, one unreachable run now warns and
drops out of the selection instead. This is a behavior change, stated in the
spec.

Eight workers is a cap on concurrent artifact downloads against one tracking
server, chosen to overlap latency without turning the CLI into a load generator.

### Resolve experiment ids once, by passing them, not by caching them

`_experiment_ids(client, None)` pages the full experiment list, and
`resolve_run()` calls it once per run name — so `runsnap tb a b c` enumerates
everything three times. The fix is to resolve the list once in the command
function and pass it into `resolve_run()`.

A module-level memo was rejected deliberately: `fix-capture-fidelity` exists
largely because a process-wide cache outlived the facts it cached. A parameter
threaded through one invocation cannot go stale.

### Walk the attempt chain once

`attempt_chain()` fetches each run to read its predecessor tag, then
`_with_chains()` fetches every returned id again. An internal
`attempt_runs(client, run_id) -> list[Run]` returns the runs the walk already
fetched; `attempt_chain()` becomes a thin wrapper returning their ids, so the
public signature is untouched.

## Risks / Trade-offs

- **Threads in the CLI fetch path.** → Each task touches its own cache directory
  and its own `MlflowClient` call; no shared state is mutated. The risk is a
  tracking backend whose client object is not thread-safe, which is why the
  worker count is small and the executor is created per invocation.
- **Losing the digest on oversize runs is breaking.** → Flagged in the proposal.
  The affected population is runs that already have no reconstructable patch.
- **1 MiB scalar shards multiply the file count** for very long runs — roughly
  one file per 10k events. → TensorBoard reads a directory of event files
  natively; the media stream already works this way at 8 MiB.
- **Batching changes partial-failure granularity.** → Covered above; the
  artifact remains the authoritative record either way.
- **Dropping a run whose fetch failed could hide a server problem.** → The drop
  warns, naming the run, so it is visible without being fatal.

## Migration Plan

No migration. Existing `tb/` artifacts, params, and capture tags keep their
meaning, and the fetch path is written to read both scalar file shapes. Rollback
is reverting the change; nothing on the server needs undoing.

Ordering: apply after `fix-capture-fidelity` and before `add-run-forensics-cli`.
