## Why

Two facts every long training job records about its runs have no home in runsnap or MLflow: why a run ended badly, and which earlier attempt it continues. runsnap's first consumer had to invent project-private tags for both (`pomdp.failure_kind`, `pomdp.failure_cause`, `pomdp.resumed_from`), so the records are not comparable across projects and the chain of attempts is only recoverable by whoever knows the private tag name.

## What Changes

- The run returned by `runsnap.start_run()` records how its `with` block ended. A `KeyboardInterrupt` ends the run with MLflow status `KILLED` instead of `FAILED`, and any exception writes tag `runsnap.failure.cause` holding the exception type and message. The exception still propagates.
- `runsnap.start_run(continues=<run_id>)` writes tag `runsnap.continues` naming the attempt this run resumes.
- `runsnap.attempt_chain(run_id)` walks that tag back to the first attempt.
- `runsnap show` prints the recorded predecessor.
- `runsnap tb --chain` adds every predecessor of each selected run, so a resumed training's curves are viewed together.
- The object `start_run()` returns becomes a subclass of MLflow's `ActiveRun` rather than the identical instance. The interface is unchanged; only the wording of the passthrough requirement changes.

No failure-kind tag is added: MLflow's run status already distinguishes `FAILED` from `KILLED` and is searchable as `attributes.status`.

## Capabilities

### New Capabilities

- `run-lifecycle`: how a run ended (status and cause) and which attempt it continues, recorded under `runsnap.` tag names shared by every consumer, walkable from Python and visible from the CLI.

### Modified Capabilities

- `code-state-capture`: the "Run creation passes through to MLflow" requirement no longer promises the identical MLflow object, only the same interface, and names lifecycle recording as the second added behavior.
- `tensorboard-viewing`: run selection gains a flag that extends the selection with each run's attempt chain.

## Impact

- `src/runsnap/__init__.py`: `start_run` returns the subclass, accepts `continues`, and exports `attempt_chain`.
- New module holding the `ActiveRun` subclass, cause recording, and chain walking; `src/runsnap/_tags.py` gains the two tag names.
- `src/runsnap/_cli.py`: `show` prints lineage, `tb` gains `--chain`.
- README, tests. No new dependencies. Existing runs and consumers are unaffected; `pomdp` can drop three of its four private tags.
