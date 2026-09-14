## Context

See proposal.md - Why. What shapes the approach in the installed tensorboardX 2.6.5 and TensorBoard 2.21, confirmed against a running TensorBoard's HParams API:

- `SummaryWriter.add_hparams` opens a second writer in `<logdir>/<name>/`. Through runsnap's writer facade it would be routed to the media stream, its file carries no `.scalars` suffix so `runsnap tb` omits it without `--media`, and TensorBoard shows it as a nested run `<run>/<name>`. `tensorboardX.summary.hparams()` builds the same summaries without writing them.
- A session is a `_hparams_/session_start_info` summary carried in summary metadata. TensorBoard keeps the first metadata it reads for a tag in a run, so a run holds exactly one session and a second write is ignored.
- The HParams plugin takes the first `_hparams_/experiment` summary it finds among all runs in a view and uses its columns for every run. Without one it builds the columns from the union of all sessions' keys.
- Every scalar tag in a run holding a session, or in a run nested under it, is offered as a metric column at its latest value.
- Across runs, a hyperparameter's type is the common type of its values; mixed types make it a string column.
- `flatten_model()` encodes every leaf as a string, which the plugin would show as text with no sorting or range filter.
- runsnap's first consumer calls `log_params()` for its configuration and a metadata model derived from it, then enters `runsnap.tensorboard()`, on every training, resume, and evaluation run, all in one process. Several of its fields are optional numbers.

## Goals / Non-Goals

**Goals:**

- A consumer that calls `log_params()` before `runsnap.tensorboard()` gets hparams in TensorBoard with no change.
- A run's session is visible while the run is live and survives an unannounced kill once synced.
- Runs with differing keys and optional numbers stay comparable in one view.

**Non-Goals:**

- Sessions for params logged by another process than the one holding the writer.
- Changing `add_hparams` on the writer facade: its surface stays the underlying writer's.
- Experiment-level metadata such as display names, descriptions, or metric declarations.

## Decisions

### Write the session start only, into the root scalar stream

The writer writes the summary from `tensorboardX.summary.hparams()` directly into the light stream's open shard, which puts it in the run's own directory, in a `.scalars` file, as the first event of the file. The experiment and session-end summaries are discarded: the experiment summary would impose one run's columns on every run in a view, and a session-end status is not needed for the table.

Alternative considered: calling `add_hparams`. Rejected for the nested run and the media routing above.

### Write when the writer opens

The session is written on entry to `runsnap.tensorboard()`, before the writer is yielded. It is then in the first uploaded scalar shard, shown by a live dashboard, and kept by a run killed after its first sync. Params logged later cannot be added, because TensorBoard keeps the first session, so `log_params()` warns.

Alternatives considered: writing on exit, which is always complete but invisible while the run trains and lost entirely on a kill, which runsnap's attempt chains exist to survive; writing just before the first event, which widens the window for late params at the cost of a rule that is harder to predict, and which the consumer's call order does not need.

### An in-process record keyed by run id

`log_params()` records the typed leaves it logs under the run id it resolved; `runsnap.tensorboard()` takes the record for the run it resolved and marks the run as claimed, whether or not the record held anything, so a later `log_params()` on the run knows its values cannot reach this writer. Nothing is read back from the tracking server, and the two functions keep their independent signatures.

Alternatives considered: reading `hparams/*.json` back from the run on entry, which works across processes but costs requests and would need the prefix stored in the artifact; an explicit `writer.add_params(model)`, which asks every consumer for a TensorBoard-specific call.

### One flattening, two encodings

Leaves are flattened once with their JSON types from `model_dump(mode="json")`. MLflow params encode them to strings as before. The session keeps booleans, numbers, and strings, writes other values as their JSON text, and omits `None`. Omitting `None` keeps a column numeric across runs; filtering for "unset" stays available on the MLflow side, where the param reads `null`.

### Module layout

```
src/runsnap/
  _params.py       typed leaves, MLflow encoding, session encoding, per-run record
  _tensorboard.py  session written into the light stream on entry
```

## Risks / Trade-offs

- [Params logged from a different process than the writer are not in the session] -> Documented; a consumer that splits them calls `log_params()` in the writer's process too.
- [A wide HParams table, since every scalar tag is a metric column] -> Accepted; TensorBoard lets columns be hidden, and the tags are the run's own curves.
- [A second writer block on the same run writes a session that TensorBoard ignores] -> It holds the same keys unless params were logged in between, which already warns.
- [The record grows by one entry per run in a long-lived process] -> A few hundred small dicts at most; not pruned.

## Migration Plan

Additive. Runs logged before this change have no session and show no hparams, as today. Reverting removes the session write and the record.
