## Context

See proposal.md — Why. What shapes the approach in the installed environment:

- MLflow's metric validation accepts any `numbers.Number` except `bool`, which it rejects explicitly, and rejects `None`. `NaN` is a float and passes.
- `tensorboardX.SummaryWriter.add_scalar` already converts array-like values through its own `scalar()` helper; `add_scalars` exists but writes each key into its own sub-run directory, which the writing design already notes appears as nested runs in TensorBoard. It is the wrong shape for "one record, one run".
- runsnap's writer facade routes `add_scalar` and `add_text` to the light event file through `_call_light` under a lock and a warn-instead-of-raise guard; every other method goes to media shards. A record is scalars, so it belongs on the light path.
- `_params.flatten_model` flattens Pydantic models to dotted keys but encodes every leaf as a string and keeps sequences whole, because params are for the run table. Metrics need numbers and drop what is not a number, so the two do not share a leaf rule.
- Equinox modules are dataclasses, so `dataclasses.fields` traverses them with no framework import. NumPy, JAX, and PyTorch scalars all expose `.item()`.
- The consumer's current helpers flatten to flat keys (dropping the parent name), replace non-finite floats with `None`, and chart every numeric field except `wall_time` under a `kind` prefix against an `environment_steps` axis, then flush.

## Goals / Non-Goals

**Goals:**

- One leaf rule, defined once, used by both the flattening function and the writer method.
- A result that `mlflow.log_metrics` takes as is.
- No import of any array or ML framework.

**Non-Goals:**

- Changing `log_params` or `flatten_model`. Different contract.
- Expanding sequences or multi-element arrays into indexed keys. A field holding a vector is not a scalar record; raising names the field so the caller reduces it.
- Any per-record policy beyond flattening: which axis to chart against, which fields to exclude, and what prefix to use stay at the call site.

## Decisions

### Dotted keys, consistent with `log_params`

Nested records produce `actor.entropy`, not `entropy`. The consumer's helper flattened flat and the issue claimed that matched `log_params`; it does not. Dotted keys cannot collide across sibling records and match what the run table already shows for params. In TensorBoard the prefix joins with `/`, which is TensorBoard's hierarchy separator, and the dotted key stays inside the leaf name: `train/actor.entropy`.

### Bools become ints, non-numerics are dropped, `NaN` stays

`bool` becomes `0` / `1` because MLflow refuses booleans while the value is a legitimate scalar to chart. `None` and strings are dropped rather than raising: a record commonly carries a label (`kind`) or a sentinel (`successes = None` for tasks without a success signal), and skipping them is what every consumer would write. `NaN` is kept: MLflow accepts it, TensorBoard draws a gap, and the consumer's `None` replacement existed only to pair with its own sentinel convention. Dropping `NaN` would hide the divergence the curve should show.

Alternative considered: a `drop_nonfinite=` flag. Not needed by any current caller; add if one appears.

### `.item()` duck typing for array leaves

A leaf that is not a number is asked for `.item()`; success yields the number, a `ValueError` (the error every array library raises for more than one element) is re-raised naming the key, and a leaf without `.item()` is dropped. This covers NumPy, JAX, and PyTorch zero-dimensional arrays and shape-`(1,)` arrays without naming any of them.

### Traversal covers mappings, dataclasses, and Pydantic models

Mappings and dataclass instances are the two shapes the consumer has. Pydantic models are one more `isinstance` branch through `model_dump()` and the package already depends on Pydantic, so a Pydantic record is not a surprise failure. Anything else is a leaf.

### `add_record` is `flatten_metrics` plus a loop over `add_scalar`

The writer method calls `flatten_metrics(record)` and then the light-path `add_scalar` once per entry, so the leaf rule lives in one place and nested records go straight to the writer. No flush is added: the sync thread and the context exit already flush and upload, and the consumer's trailing `flush()` was doing nothing useful. The name `add_record` avoids shadowing `add_scalars`, whose nested-run behavior the writer promises to leave as the underlying writer defines it.

Alternative considered: extending `flatten_model` to accept dataclasses and array leaves so one function serves both. Rejected: params encode to strings and keep sequences whole; metrics keep numbers and drop non-numerics. One function with two output contracts is harder to describe than two small functions.

### Module layout

```
src/runsnap/
  _metrics.py      flatten_metrics and the leaf rule
  _tensorboard.py  + TensorBoardWriter.add_record
  __init__.py      exports flatten_metrics
```

## Risks / Trade-offs

- **Silently dropped non-numerics can hide a typo in a field name** → Accepted; a string-valued field is never chartable, and a misspelled numeric field is still numeric and still charted under its misspelled name.
- **A record of many fields costs one locked `add_scalar` call per field per step** → The lock is uncontended in the write path and records are tens of fields; no batching is needed at this scale.
- **A one-element array of shape `(1,)` is accepted as a scalar** → Intended; `.item()` defines "scalar" here, and that is how array libraries define it too.
- **Consumers that relied on `None` for non-finite values must change** → The only consumer is the one that raised the issue, and it moves to `NaN` when it adopts these helpers.

## Migration Plan

Additive. No existing behavior changes; runs that never call the new functions are unaffected. Reverting removes the module, the writer method, and the export.
