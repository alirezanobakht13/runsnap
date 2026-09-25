## Why

`runsnap tb` resolves its selection once, at startup. A run that starts after
the dashboard is open is never linked into the assembled directory, so it never
appears: leaving `runsnap tb` open and launching the next training shows
nothing new until the command is restarted, and restarting is needed again for
every run. The dashboard already follows a live run's growing files and its
switch to the cache when the writer exits; it should also pick up runs that
begin after it opened.

## What Changes

- When the selection is a query (no run ids or names given: bare `runsnap tb`,
  `--experiment`, `--filter`), `runsnap tb` re-runs that query while TensorBoard
  is open and adds each newly matching run once the run starts writing
  TensorBoard events, without restarting the command.
- A run added this way is shown exactly as a run found at startup would be:
  from its local directory while it is being written on this host (and switched
  to the cache when its writer exits, as today), otherwise from its cached
  artifacts. `--media` and `--chain` apply to it as they do at startup.
- A run that was already selected at startup but had not yet begun writing
  TensorBoard events is added in the same way once it does.
- Runs already shown keep their names. A newly added run whose name is already
  shown appears under a distinguished name.
- Runs are never removed from the dashboard because they stop matching the
  query.
- A selection of named runs stays fixed, as today.
- The server's experiments are enumerated once per re-run of the query, so runs
  in experiments created after the dashboard opened are followed too.
- The README's TensorBoard section describes the new behaviour.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `tensorboard-viewing`: a query selection keeps adding runs that start writing
  TensorBoard events after the dashboard opened; the naming requirement allows
  a late run to be distinguished without renaming runs already shown; the
  bound on enumerating experiments applies per selection pass rather than per
  invocation.

## Impact

- `src/runsnap/_tb_fetch.py`: `assemble_logdir` accepts the query to re-run and
  whether to extend runs with their earlier attempts; its watch thread re-runs
  the query each poll and links newly ready runs; naming is shared between
  startup and later additions.
- `src/runsnap/_lifecycle.py`: gains the chain expansion that `_cli.py` holds
  today as `_with_chains`, so the assembly can apply it to runs it adds.
- `src/runsnap/_cli.py`: `tb` hands the query and `--chain` to the assembly
  instead of expanding chains itself.
- `tests/test_tb_fetch.py`, `tests/test_tb_cli.py`: tests for late runs,
  runs selected before they started writing, name clashes, `--chain` on a late
  run, named selections staying fixed, and a failing query.
- `README.md`: the TensorBoard section.
- No new dependencies. No change to the writer (`runsnap.tensorboard()`), its
  tags, the artifact layout, or the CLI flags.
- Runs logged on another host remain out of scope: one added while the
  dashboard is open is fetched once when it is found, before it has usually
  uploaded anything, so it is likely to stay absent until the command is
  restarted.
