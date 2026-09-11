## Why

Four hot paths cost far more than the work they do. Logging a hyperparameter
model opens one HTTP round trip per scalar leaf, so a sixty-field config is
sixty sequential calls to the tracking server. The TensorBoard scalar file is
re-uploaded in full every thirty seconds, which makes a long run's upload traffic
grow with the square of its length. The code-state patch is built entirely in
memory before its size ceiling is consulted, so the one case the ceiling exists
to defend against — a large untracked tree — is also the case that blows up
memory and start-up time. And resolving several run names enumerates every
experiment on the server once per name.

None of this is visible on a laptop against a local file store. All of it is
visible against a shared tracking server, which is where the tool is meant to be
used.

## What Changes

- `log_params()` writes params in batches rather than one call per leaf, so a
  model of any size costs one round trip per hundred leaves.
- The TensorBoard scalar file rolls into sealed shards the way media already
  does, bounding what a sync pass re-uploads. The open scalar shard is still
  re-uploaded each pass so a live dashboard keeps updating.
- The patch ceiling is enforced while the patch is being read, not after. A tree
  whose diff exceeds the ceiling is abandoned at the ceiling instead of being
  materialized in full. The run records the same "patch too large" outcome it
  records today, minus the patch digest: a patch that was never read in full
  cannot be hashed, and an oversize run has no artifact to reconstruct from, so
  the digest had no consumer. **BREAKING** for anyone comparing oversize runs by
  `runsnap.git.patch_sha256`.
- Runs selected for viewing are fetched concurrently instead of one after
  another.
- The list of experiment ids is resolved once per command invocation rather than
  once per run name, and walking a chain of attempts reuses the runs it already
  fetched instead of fetching each one twice.

## Capabilities

### New Capabilities
- `pydantic-params`: how a hyperparameter model is written to and read back from
  a run. Not touched by `fix-capture-fidelity`, so this change introduces it.

### Modified Capabilities
<!-- Introduced by fix-capture-fidelity; extended here. -->
- `code-state-capture`: the patch size ceiling becomes a bound on the work done,
  not only on what is uploaded.
- `tensorboard-logging`: the scalar stream is sharded, and what a sync pass
  uploads is bounded.
- `tensorboard-viewing`: runs are fetched concurrently, and run selection bounds
  its queries to the server.

## Impact

- `src/runsnap/_params.py`, `_capture.py`, `_git.py`, `_tensorboard.py`,
  `_tb_fetch.py`, `_cli.py`, `_tags.py`
- Artifact layout changes: a run's `tb/` directory gains numbered scalar shards
  in place of a single scalar file. TensorBoard merges every event file in a
  directory, so runs logged either way read identically when viewed; no existing
  artifact is rewritten.
- Concurrency is introduced into the CLI fetch path, which previously had none.
- Depends on `fix-capture-fidelity`: this change reworks `build_patch()` and the
  writer's file handling, both of which that change also touches.
