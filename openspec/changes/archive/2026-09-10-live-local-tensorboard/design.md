## Context

See proposal.md for motivation.

`runsnap.tensorboard()` writes into a scratch directory from `tempfile.mkdtemp`, mirrors it into the run's `tb/` artifacts from a background thread, and removes it when the block exits. `runsnap tb` calls `fetch_run` once per selected run inside `assemble_logdir`, symlinks each cached run directory into a temporary log directory under its run name, and blocks on the TensorBoard subprocess. TensorBoard's reload only re-reads that log directory; both of its event loaders tail an open file handle by offset, so a file swapped in by rename is never re-read.

The run already carries `runsnap.tb.logdir` = `tb`, and the viewing spec relies on that constant value for tag queries, so it cannot be repurposed to hold a path.

## Goals / Non-Goals

**Goals:**

- A dashboard opened on the training host follows training with no polling, no extra MLflow calls, and no change to the artifact layout.
- Runs that are not live on this host behave exactly as today.

**Non-Goals:**

- Live viewing of a run training on a different host. That needs the viewer to poll artifacts and to append into cached files in place; it is a separate change if the same-host path proves insufficient.
- Picking up runs that start matching a query after the dashboard is opened.
- Cleaning up the scratch directory of a run that was killed without unwinding the writer block.

## Decisions

**Where the local path is recorded.** Two new tags, `runsnap.tb.local_host` and `runsnap.tb.local_dir`, set right after the scratch directory is created, in the same warn-not-raise block that sets `runsnap.tb.logdir`. Two tags rather than one `host:path` value because a path may legally contain a colon. The host is `socket.gethostname()`. The tags are never deleted: the directory's absence already says everything a viewer needs, and a run killed without cleanup keeps a readable directory that is the most complete record of that run anywhere.

*Alternative rejected:* an environment variable or a lookup file on the training host. Tags travel with the run, need no extra state, and already reach the viewer through the run object it holds.

**Where the choice is made.** In `assemble_logdir`, per run, before `fetch_run`: if the run's tags name this host and the directory exists, use the directory as that run's path; otherwise call `fetch_run` as today. Everything downstream, naming, duplicate handling, the check for event files, the symlink into the temporary log directory, already operates on a path and needs no change. `fetch_run` itself is untouched.

*Alternative rejected:* making the check inside `fetch_run`. It takes a run id and a client, not a run object, and its contract is "cache this run's artifacts"; giving it a second meaning would muddy the one function the cache tests pin down.

**Matching on host name plus existence, not run status.** The MLflow status is not consulted. A finished run has no directory, so existence alone rules it out. A killed run stays `RUNNING` forever in MLflow and keeps its directory, and showing that directory is the desired outcome. Checking status would add a round trip and make the killed case worse.

**The whole local directory is linked.** No attempt is made to hide media files from a live run when `--media` is absent. The flag exists to avoid downloads, and a live run downloads nothing. Building a scalars-only view of a live directory would require tracking media shards as they roll, for no saving.

**Flush interval.** `runsnap.tensorboard()` injects `flush_secs=10` into the writer keyword arguments unless the caller supplied one. The value lives beside the other TensorBoard limits in `_tags.py`. Ten seconds keeps the local files and the 30 second upload cycle both fresh; scalar events are tens of bytes, so the extra disk flushes cost nothing measurable. Lowering the writer's `max_queue` was considered and rejected: it changes how often the worker thread wakes, not when data reaches disk.

**Nothing changes on the TensorBoard command line.** Its default reload interval of five seconds is already right for tailing local files.

## Risks / Trade-offs

- [Same host name on two machines with a shared filesystem, or a container reusing a host name] → The path is a random `mkdtemp` name, so a collision requires both the name and the path to coincide. Accepted; the fallback is a stale dashboard, not wrong data.
- [The scratch directory belongs to another user on the same host and is not readable] → `mkdtemp` creates it mode 0700, so TensorBoard would show the run empty. Accepted as out of scope; the check uses existence only, keeping the rule simple. A readability check is a one-line addition if it ever matters.
- [The training run finishes while the dashboard is open] → The symlink dangles when the directory is removed. TensorBoard keeps what it has already read and logs a warning on its next scan. The next `runsnap tb` fetches the finished run from artifacts. Accepted; this matches how a plain `tensorboard --logdir` behaves when a log directory disappears.
- [A caller relying on the previous 120 second flush to reduce disk writes] → They can pass `flush_secs=120` explicitly. Documented in the README.
- [Tests need a host name] → Tests set the tag to `socket.gethostname()` for the live case and to a fixed other name for the other-host case, so they do not depend on the machine's name.
