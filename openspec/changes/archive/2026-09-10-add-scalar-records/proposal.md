## Why

A training iteration produces a small record of numbers: a dataclass of scalar arrays from the update step, or a dictionary describing an evaluation. Before it can be logged or charted it has to be flattened to named Python floats and looped into `add_scalar`, and that loop is identical in every consumer while carrying decisions each project makes differently: the shape of nested keys, what happens to non-numeric fields, and what happens to `NaN`. runsnap's first consumer kept two such helpers; settling the rules once removes both.

## What Changes

- `runsnap.flatten_metrics(obj, prefix="")` turns a mapping, dataclass instance, or Pydantic model into `{dotted_key: number}`: nested records become dotted keys, zero-dimensional arrays become their Python number, booleans become `0` / `1`, non-numeric leaves are dropped, and non-finite values are kept as they are. The result can go straight to `mlflow.log_metrics`.
- The writer from `runsnap.tensorboard()` gains `add_record(prefix, record, global_step=None, walltime=None)`, charting every flattened entry of `record` as `prefix/<key>` in the scalar event file.

`NaN` is deliberately kept rather than replaced: MLflow accepts it as a float and TensorBoard charts it as a gap, and a divergence is something the curves should show.

## Capabilities

### New Capabilities

- `scalar-records`: flattening a record of numbers into named scalars that MLflow and TensorBoard accept directly, with one rule for keys and leaves.

### Modified Capabilities

- `tensorboard-logging`: the writer gains a requirement for charting a whole record in one call.

## Impact

- New module holding `flatten_metrics`; exported from `runsnap`.
- `src/runsnap/_tensorboard.py`: `add_record` on the writer facade.
- README, tests. No new dependencies; array leaves are handled by duck typing on `.item()`, so no framework is imported. The existing `log_params` flattening is untouched: params encode leaves as strings and keep sequences whole, which is a different contract.
