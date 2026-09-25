## Context

`tb` (`src/runsnap/_cli.py`) resolves its runs once with `_tb_runs`, extends
them with `_with_chains` when `--chain` is given, and hands the list to
`assemble_logdir` (`src/runsnap/_tb_fetch.py`). That context links each run
into a temporary directory, from its writer's local directory when
`_live_logdir` finds one on this host, otherwise from the cache via
`_fetch_runs`. Names come from `_run_name`, with runs that share a name
distinguished by `_distinct_names`. When any run was linked live, a
`_LiveWatch` daemon thread polls every `LIVE_POLL_SECONDS` (5 s) and
re-points a live link at the cache once its directory is gone. Nothing adds a
link after startup, which is the gap in proposal.md.

What the writer does on entering `runsnap.tensorboard()`, in order
(`src/runsnap/_tensorboard.py`): it builds `TensorBoardWriter`, which creates
the scalar and media event files immediately (checked by constructing one),
then sets `runsnap.tb.logdir`, `runsnap.tb.local_host`, and
`runsnap.tb.local_dir` with three separate `set_tag` calls.

TensorBoard re-walks its logdir on every reload and adds a subdirectory it has
not seen as a new run. The relink change already relies on this to rediscover
a re-pointed link; a new name is the same case.

## Goals / Non-Goals

**Goals:**

- One watch thread handles both jobs: re-pointing live links and following
  the query.
- Runs added later go through the same linking, fetching, naming, and
  `--chain` code as runs found at startup.
- Runs that will never have TensorBoard data cost no artifact listing on any
  poll.

**Non-Goals:**

- Refreshing runs shown from the cache as more data is uploaded, including
  runs from another host found while the dashboard is open.
- Opening TensorBoard when the startup selection is empty and waiting for runs.
- Narrowing the re-run query, for example to runs started after the dashboard
  opened.
- A flag to turn following off.
- Removing or renaming links that are already shown.

## Decisions

### The existing watch thread also follows the query

`_LiveWatch` gains a second step on each tick: after re-pointing gone live
links, it re-runs the query if one was given. The thread starts when there are
live links or a query to follow, and keeps running while either is true. Runs
it adds live join its own set of watched links, so their switch to the cache
comes for free. Both steps use the same interval, `LIVE_POLL_SECONDS`, whose
docstring changes to cover the query.

Alternative: a second thread for following. It would have the same lifetime,
the same stop and join, and would share the link directory with the first
thread, which then needs a lock. Rejected.

### The query is passed in as a callable; `tb` decides whether to follow

`assemble_logdir` takes `follow: Callable[[], Iterable[Run]] | None`. `tb`
passes `partial(_tb_runs, client, (), experiment, filter)` when `run_refs` is
empty, and `None` otherwise, so a named selection stays fixed. Each call is
one selection pass through the existing `_tb_runs`, which already enumerates
experiments at most once per call. Without `--experiment` it therefore picks up
experiments created after startup.

Alternative: pass `experiment` and `filter` into `_tb_fetch` and query there.
That moves the MLflow selection logic out of the CLI for no gain. Rejected.

### Chain expansion moves into the assembly and runs only for added runs

`_with_chains` moves from `_cli.py` to `_lifecycle.py`, next to `attempt_runs`,
as `with_attempts(client, runs)`. `assemble_logdir` takes `chain: bool`. It
expands the startup runs with it, and on each follow pass it expands only the
runs that pass adds. `tb` no longer expands chains itself. Existing
`--chain` behaviour and its once-per-attempt fetch bound are unchanged.

Alternatives: let the callable from `tb` expand chains. That walks every
selected run's chain with `get_run` on every poll, which is hundreds of calls
every 5 s for a bare `runsnap tb`. Rejected. Or pass a separate `expand`
callable next to `follow`: two parameters that only make sense together.
Rejected.

### A followed run is ready once it carries `runsnap.tb.local_dir`

A pass considers only runs from the query that have the `runsnap.tb.local_dir`
tag. The writer sets that tag last, after its event files exist. So a ready
run written on this host always has its host tag too, and its directory
already holds event files: it is linked live straight away. Checking the tag
is a lookup in the query result. A run that never enters
`runsnap.tensorboard()` is never ready, so nothing is listed or downloaded for
it on any poll.

Alternatives: gate on `runsnap.tb.logdir`, which the writer sets first. A poll
between the first and last `set_tag` would see a run with no local tags, fetch
it before anything was uploaded, and drop it. Rejected. Or have the writer set
all three tags in one `log_batch`. That changes the writer, which this change
leaves alone. Rejected.

Runs logged before the local tags existed never become ready. Those runs
finished long ago, and startup still selects them as before.

### Each run is processed at most once after it is ready

The watch keeps `handled`, a set of run ids. At startup it holds every run
that was shown or that carried `runsnap.tb.local_dir` when it was processed. A
startup run without the tag and without event data stays out of `handled`, so
it is added once its writer starts. That is the "selected before it wrote
events" scenario.

A follow pass takes the ready runs not in `handled`, extends them with
`with_attempts` when `chain` is set, and drops any run now in `handled`. It
then links the rest exactly as startup does: live through `_live_logdir`,
everything else fetched concurrently through `_fetch_runs`, then the
event-data check. Every run the pass processed goes into `handled`, whether it
was shown, empty, or failed to fetch. So a run is fetched at most once, and a
failed fetch is warned once, as at startup.

To share code, the startup body of `assemble_logdir` becomes one function
that takes a batch of runs and returns the paths it could link. Startup and
each pass both call it.

### Naming: one function for startup and later passes

Naming becomes one helper. It takes a batch's base names (`_run_name`), the
base names of every run already shown, and the link names already taken. A
run keeps its plain base name only when no other run in the batch shares it,
no run already shown has that base, and no link already has that name.
Otherwise it gets the first name from `_distinct_names` that is not taken.
Plain names are assigned before distinguished ones, as today. With empty
shown and taken sets this reproduces the current startup naming, including
the collision-with-suffix case. Existing links are never renamed: TensorBoard
would treat a renamed link as a different run and recolour it.

### A failing query is warned once per outage

The pass wraps the `follow()` call. On an exception it warns only if the
previous pass succeeded, keeps every link as it is, and tries again at the
next tick. The first pass that succeeds clears the flag. Nothing that one
pass raises may end the thread. Fetch failures are already warned by
`_fetch_runs`.

### No locking

After startup, only the watch thread creates or replaces links and touches its
bookkeeping. The main thread only waits on TensorBoard and then stops the
thread by joining it, as it does today. A pass in flight delays exit by at
most one query plus the fetches for the runs that pass added.

## Risks / Trade-offs

- [Each pass fetches every run in every selected experiment, with its params,
  metrics, and tags] → Fine for a tracking server of thesis size on the same
  host. If it becomes slow, narrow the re-run query to runs started after
  startup; that is listed as a non-goal, not ruled out.
- [A new run appears up to one poll plus one TensorBoard reload after its
  writer starts, about 10 s] → Acceptable for a dashboard, and the same delay
  the relink switch already has.
- [A run from another host is found when its tags appear, before its first
  upload about 30 s later, so its one fetch is empty and it stays absent] →
  Out of scope. The README keeps "rerun the command" for runs on other hosts.
- [A late run named like a shown run gets a suffix, while two runs that clash
  at startup both get one] → The spec allows this. The alternative, renaming
  shown runs, disrupts what the user is already looking at.
- [TensorBoard's newer data loader, the Rust data server, may discover new
  subdirectories differently from the Python loader the tests exercise] →
  The relink change already depends on rediscovery. The tasks include a
  manual check with a real TensorBoard.
- [Tests depend on timing] → Pass a small `poll_interval` and poll with a
  deadline using `wait_until`, as the relink tests do.
