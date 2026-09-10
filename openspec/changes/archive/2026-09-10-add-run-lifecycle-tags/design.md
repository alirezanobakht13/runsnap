## Context

See proposal.md — Why. The constraints in the installed MLflow 3.16.0 that shape the approach:

- `mlflow.start_run()` wraps the created run in an `ActiveRun` (`mlflow/tracking/fluent.py:374`), pushes that instance onto a context-local stack, and returns it. `ActiveRun.__exit__` ends the run as `FINISHED` when no exception is set and `FAILED` otherwise, with no special case for `KeyboardInterrupt`, even though `RunStatus.KILLED` exists and counts as a terminal status. The exit checks the stack by run id, not object identity, so another object with the same id can end the run.
- `mlflow.end_run(status)` accepts any terminal status string and pops the stack; nothing else in MLflow decides a run's final status.
- MLflow expresses only nesting between runs (`mlflow.parentRunId`). There is no succession relation, and the run-name tag is free-form, so consumers cannot recover a resume chain without their own tag.
- Tag values cap at 8000 characters and are silently truncated.
- The consumer's current helpers set the failure tags from an `except (Exception, KeyboardInterrupt)` that re-raises, always inside a `with runsnap.start_run()` block, and set the resume tag with `mlflow.set_tag` right after the run starts.

## Goals / Non-Goals

**Goals:**

- Failure and lineage recorded with no code at the call site beyond what `start_run` already requires: a `with` block and, for a resume, one keyword.
- Tag names that live in the `runsnap.` namespace like everything else the package writes.
- Nothing added to the exit path can change which exception the caller sees or leave the run open.

**Non-Goals:**

- Storing a traceback. The cause message plus the recorded code state is what reproducing a failure needs; a traceback artifact can be added later without touching these tags.
- Any other relation between runs, such as "evaluated the checkpoint of". The consumer keeps its `source_run` tag.
- Validating `continues` against the tracking server at run start.
- Resuming a run under its own id. Each attempt is its own run so that it captures its own code state.

## Decisions

### Status carries the kind; one tag carries the cause

`KeyboardInterrupt` ends the run as `KILLED`, everything else as `FAILED`, and one tag `runsnap.failure.cause` holds `f"{type(exc).__qualname__}: {exc}"`. The issue proposed a `runsnap.failure.kind` tag with `failed` / `interrupted`; that duplicates a distinction MLflow's status already makes, shows in the run table without a custom column, and is searchable as `attributes.status = 'KILLED'`. The type name goes into the cause because `str(exc)` alone is often empty or ambiguous (`KeyboardInterrupt` has no message).

Alternative considered: also writing `runsnap.failure.kind` for symmetry with the issue. Rejected as a second spelling of the status.

### An `ActiveRun` subclass owns the exit

`start_run` wraps MLflow's `ActiveRun` in a subclass constructed from it (`Run.__init__` takes `info`, `data`, `inputs`, `outputs`, so the wrap is a copy of references) and returns that. Its `__exit__` records the cause under the package's warn-instead-of-raise guard, then ends the run: `mlflow.end_run("KILLED")` for `KeyboardInterrupt` when the run is still the active one, and the parent class's `__exit__` otherwise. It returns `False` so the exception propagates. MLflow's stack still holds MLflow's own instance, which is why `mlflow.active_run()` keeps working and why ending by id is what makes this safe.

Alternatives considered. A `contextlib` wrapper would lose the non-`with` usage (`run = runsnap.start_run(); ... mlflow.end_run()`) and `run.info`. Catching the exception at the call site, as the consumer does today, is the boilerplate this change removes. Monkeypatching `ActiveRun.__exit__` would also affect bare `mlflow.start_run()` runs, breaking the explicit opt-in contract.

### `continues` is a keyword on `start_run`, written verbatim

`capture_code` already sets the precedent of a runsnap-only keyword on `start_run`. The tag is written with `MlflowClient.set_tag` immediately after the run starts and before capture, so a crash during capture never loses it. The id is not checked against the server: MLflow does not validate `mlflow.parentRunId` either, the id comes from the caller's own checkpoint, and a round trip at every resume buys only an earlier error for a case that `attempt_chain` reports anyway.

### `attempt_chain` walks with the client, newest first

`attempt_chain(run_id)` reads each run with `MlflowClient.get_run` and follows `runsnap.continues` until a run has none, returning ids from the given run back to the first attempt. That order matches how the consumer already uses the chain and how a reader thinks about it ("this run, and what it came from"). A revisited id raises `ValueError` naming it; a missing run lets MLflow's own error through, which already names the id. Chains are a handful of attempts long, so one request per hop is fine.

### CLI: one line in `show`, one flag in `tb`

`show` prints `continues: <id>` between `name:` and `commit:` when the tag is present. Lineage is not code state, so it lives in the `run-lifecycle` spec rather than modifying the `show` requirement in `code-state-restore`. `tb --chain` extends whatever `_tb_runs` selected with `attempt_chain` of each run, deduplicated by id in first-seen order, so it composes with explicit refs, `--experiment`, and `--filter` alike. Attempts usually share a run name, and the viewer already disambiguates duplicate names with a run-id suffix.

### Module layout

```
src/runsnap/
  _lifecycle.py   ActiveRun subclass, cause recording, continues tag, attempt_chain
  _tags.py        + TAG_FAILURE_CAUSE, TAG_CONTINUES
  __init__.py     start_run wraps and tags; exports attempt_chain
  _cli.py         show prints lineage; tb gains --chain
```

## Risks / Trade-offs

- **Some tooling may treat `KILLED` differently from `FAILED`** → Both are terminal statuses in MLflow's own enum and its UI; the difference is the point. Documented in the README.
- **A long exception message is truncated at 8000 characters** → Accepted; MLflow truncates silently, the type name is at the front, and the message is a pointer, not the record.
- **The exit path now makes a network call while an exception is in flight** → Wrapped in the warn-instead-of-raise guard, and `end_run` runs regardless, so the run cannot be left open or the exception replaced.
- **`start_run` no longer returns MLflow's identical object** → The subclass exposes the same interface, and the spec scenario for non-`with` usage guards the one behavior that depends on MLflow's own stack entry.
- **A wrong `continues` id is only detected when the chain is walked** → Accepted; see the decision above.

## Migration Plan

Additive. Existing runs, tags, and artifacts are untouched. The consumer replaces its `except` blocks and its `pomdp.resumed_from` tag with the `continues` keyword and reads the chain through `attempt_chain`. Reverting removes the module, the keyword, and the CLI additions; runs already carrying the new tags keep them as ordinary tags.
