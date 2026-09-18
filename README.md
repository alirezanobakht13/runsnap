# runsnap

Reproducible MLflow runs: each run records its commit plus a patch of all
uncommitted work, and the CLI puts that code back on disk. Needs `git` on `PATH`.

```python
with runsnap.start_run(run_name="baseline"):  # forwards to mlflow.start_run()
    runsnap.log_params(hparams)  # a pydantic model
hp = runsnap.load_params(run_id, HParams)
```

`log_params` writes one param per scalar leaf (`opt.lr` = `0.001`; sequences
whole) plus `hparams/<name>.json`, the full-fidelity record `load_params` reads.
Tags are `runsnap.git.` + `commit`, `branch`, `dirty`, `repo_url`,
`patch_sha256`, `patch_run_id` (nested runs point at the ancestor holding the
patch), `capture_error`; the patch is artifact `code/state.patch` on dirty runs.
Equal `commit` and `patch_sha256` mean identical code.
Capture also records `sys.argv` as a JSON array in `runsnap.invocation.argv`
and the working directory in `runsnap.invocation.cwd`, relative to the
repository root when inside it and absolute otherwise.

```bash
runsnap show <run-id-or-name>      # commit, branch, dirty, digest, files
runsnap checkout <run-id-or-name>  # branch at that commit, patch reapplied
```

`checkout` lands in a new worktree, patch left uncommitted: `--no-worktree`,
`--path`, `--branch`, `--commit`, `--force`, `--repo`. Capture never fails a run;
disable it with `capture_code=False` or `RUNSNAP_CAPTURE_CODE=0`, and cap the
patch with `RUNSNAP_MAX_PATCH_BYTES` (10 MiB). A patch reaching the cap is
abandoned there, leaving the run marked dirty with a `capture_error` naming the
cap, no patch artifact, and no `patch_sha256`.

**The patch carries uncommitted content of tracked files, and the recorded
argument vector carries command-line arguments. Secrets in either are uploaded
to the tracking server.** Ignored files are excluded from the patch. Both code
and invocation capture are disabled by `capture_code=False` or
`RUNSNAP_CAPTURE_CODE=0`.

## Run forensics

Commands taking runs accept run ids or names. Use `--experiment NAME` to narrow
name resolution and `--tracking-uri URI` to override `MLFLOW_TRACKING_URI`.

```bash
runsnap diff baseline tuned                # code and param changes, baseline -> tuned
runsnap rerun baseline                     # print the recorded directory and argv
runsnap patch baseline                     # raw patch bytes on stdout
runsnap patch baseline --output run.patch  # write the patch to a file instead
runsnap show baseline --patch              # code-state report followed by the patch
runsnap clean                              # list reconstruction worktrees and branches
runsnap clean --remove                     # delete those worktrees and branches
```

`diff` reports both commits and dirty states, added/removed/changed params, and
the difference between the recorded code states, including uncommitted work.
The code comparison needs a local repository holding both commits; pass
`--repo PATH` to select one. Temporary worktrees and branches are removed after
the comparison, including on failure. Identical commits and patch digests skip
reconstruction. If a commit is missing or a patch cannot be applied, the output
explains why the code comparison is unavailable and still reports metadata and
params.

`rerun` prints the recorded directory and shell-quoted argument vector without
executing anything. Relative directories start at the reconstructed repository
root. It reports missing invocation tags on older runs and flags notebook or
REPL argument vectors that do not look like runnable commands.

`patch` exports the recorded bytes unchanged, including binary hunks, and needs
no local repository. A clean run reports that it has no patch and writes no
patch output. `show --patch` appends the same bytes after its usual report.

`clean` lists worktree paths and branches under `runsnap/`, including branches
left after their worktree directories were deleted. Use `--repo PATH` to select
the repository. Listing removes nothing; **`--remove` discards uncommitted work
in the listed worktrees** and reports each removal. Worktrees and branches
outside that prefix are untouched. With no leftovers, it reports that there is
nothing to remove.

## Run lifecycle

Leaving a `start_run` block records how it ended. A `KeyboardInterrupt` ends the
run as `KILLED` rather than `FAILED`, so an interrupted attempt is told apart
from a crash by `attributes.status`. Either way the run carries tag
`runsnap.failure.cause` holding the exception's type and message, and the
exception still reaches the caller.

```python
with runsnap.start_run(continues=previous_run_id):  # resumes an earlier attempt
    ...
runsnap.attempt_chain(run_id)  # [run_id, ..., first attempt]
```

`continues` records tag `runsnap.continues` naming the attempt a run resumes,
independent of MLflow nesting, and `attempt_chain` walks that tag back to the
first attempt. `runsnap show` prints a `continues:` line for a run that has one,
and `runsnap tb --chain` adds each selected run's earlier attempts so a resumed
training draws as one set of curves.

## TensorBoard

Use `runsnap.tensorboard()` inside an active MLflow run to log TensorBoard
events to its `tb/` artifacts. It exposes the `tensorboardX.SummaryWriter`
methods and works independently of your training framework:

```python
import runsnap
import mlflow
import numpy as np

mlflow.set_experiment("ablations")
with runsnap.start_run(run_name="baseline") as run:
    mlflow.log_param("optimizer", "adamw")
    with runsnap.tensorboard() as writer:
        for step in range(10):
            writer.add_scalar("train/loss", 1.0 / (step + 1), step)
            writer.add_record("eval", {"score": step, "solved": step > 4}, step)
        writer.add_text("notes", "Baseline run", 0)
        writer.add_image("sample", np.zeros((3, 32, 32)), 0)
        writer.add_histogram("weights", np.linspace(-1, 1, 100), 0)
    print(run.info.run_id)
```

The writer uploads event files without copying scalars into MLflow metrics.
Histograms default to 30 bins; pass `bins=` to override. Leaving the writer
block flushes and uploads pending events, including on exceptions and Ctrl-C.

`add_record` charts a whole record — a mapping, a dataclass instance, or a
Pydantic model — as one scalar per leaf, so the call above draws `eval/score`
and `eval/solved`, and a nested `actor` field would draw `eval/actor.entropy`.
`runsnap.flatten_metrics(record)` returns that same `{key: number}` mapping for
`mlflow.log_metrics`. Booleans become `0` / `1`, `None` and strings are dropped,
and `NaN` and infinities are kept so a divergence shows as a gap in the curve. A
one-element array, list, or tuple charts as its element and an empty one is
dropped; a field holding more than one number, whether an array or a list,
raises, naming the field.

Every model passed to `runsnap.log_params()` on the run, in the same process,
before entering `runsnap.tensorboard()`, appears in TensorBoard's HParams tab
keyed like its MLflow params (`opt.lr`, prefix included). The session is written
on entry into the run's scalar file, so `runsnap tb` shows it without `--media`
and the first sync pass uploads it. Booleans, numbers, and strings keep their
type; sequences and empty containers show their param's JSON text; `None` is
left out, so a column numeric in some runs stays numeric. Runs with different
keys share one table under the union of their keys, each run's scalar tags
serving as metric columns at their latest values. Params logged inside the
block still reach MLflow but miss the session, and `log_params()` warns; a
second writer block on the run writes a session TensorBoard ignores. Params
logged from another process are not included, a run with no params gets no
session, and a failed write only warns.

Use the same `MLFLOW_TRACKING_URI` for logging and viewing (or pass
`--tracking-uri` to the CLI):

```bash
runsnap tb baseline                        # run name or run id
runsnap tb baseline ablation-nodropout      # compare named runs
runsnap tb --experiment ablations
runsnap tb --experiment ablations --filter "params.optimizer = 'adamw'"
runsnap tb baseline --media                # include images and histograms
runsnap tb baseline --chain                # add the attempts it continues
runsnap tb baseline --bind_all --port 6007 # listen on all network interfaces
runsnap tb baseline --host 127.0.0.1 --port 6008
```

`--bind_all` (also `--bind-all`) and `--host` are mutually exclusive. `--port 0`
asks the operating system to select an unused port; `--port default` searches
for a free port starting at 6006. Omitted options retain TensorBoard's defaults.

For runs logged on the viewing host, the viewer uses the writer's local log
directory while it exists. TensorBoard shows newly flushed events as training
continues, including images, histograms, and other media regardless of `--media`.
Runs logged on another host, or whose local directory is gone, use a snapshot
of their uploaded artifacts: by default only scalars and text, with `--media`
including other media. Rerun the command to fetch newer uploads for those runs.
A run the tracking server will not hand over is reported as a warning and left
out of the dashboard, so the rest of the selection still opens.
Live and cached runs appear together under their MLflow names; unchanged
downloads are reused from a local cache. TensorBoard prints its URL and runs in
the foreground until Ctrl-C. If a live writer block exits while the dashboard
is open, rerun the command to view its final uploaded artifacts.

During logging, writers flush events to local disk every 10 seconds by default;
pass `runsnap.tensorboard(flush_secs=60)` to override the interval. The background
thread waits 30 seconds between upload sync passes.
Media rolls into shards after crossing 8 MiB and scalars after 1 MiB; a sealed
shard uploads once on the next sync. The open scalar shard uploads on every pass
that finds it grown, so a dashboard keeps updating, while the open media shard
waits until it seals. An unannounced kill (`SIGKILL` or unhandled `SIGTERM`)
keeps successfully uploaded data, but the open media shard, pending sealed
shards, and unsynced scalar updates can be lost. Slow or failed uploads extend
this window, so neither 30 seconds nor one shard is a guaranteed loss bound. A
size threshold is checked after writes, so a single large entry can exceed it.
No signal handlers are installed.
