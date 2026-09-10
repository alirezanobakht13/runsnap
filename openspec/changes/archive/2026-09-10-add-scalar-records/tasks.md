## 1. Flattening

- [x] 1.1 Add `_metrics.py` with `flatten_metrics(obj, prefix="")` traversing mappings, dataclass instances, and Pydantic models into dotted keys, applying the leaf rule (numbers pass, bools become ints, `.item()` for array leaves, `ValueError` naming the key for multi-element arrays, non-numerics dropped, non-finite kept); verify with tests for each spec scenario using NumPy arrays as the array library.
- [x] 1.2 Export `flatten_metrics` from `runsnap` and verify `mlflow.log_metrics(runsnap.flatten_metrics(record), step=1)` succeeds inside an active run against a mixed record, as a test in `tests/test_metrics.py`.

## 2. Writer method

- [x] 2.1 Add `TensorBoardWriter.add_record(prefix, record, global_step=None, walltime=None)` that flattens the record and writes each entry through the light path as `prefix/<key>` (bare key when the prefix is empty); verify with tests in `tests/test_tensorboard.py` that entries land in the scalar file with the expected names, step, and wall time, that non-numeric fields are skipped, and that media written alongside stays out of the scalar file.

## 3. Documentation and final pass

- [x] 3.1 Update the README TensorBoard section with `flatten_metrics` and `add_record`, including the `NaN` and boolean rules; verify the example runs as written.
- [x] 3.2 Run `uv run ruff check`, `uv run ruff format`, `uv run ty check`, and `uv run pytest`, and verify all pass.
