## Why

`runsnap tb` downloads a one-time snapshot of each run's event files before launching TensorBoard, so a dashboard opened during training never shows new data until the command is restarted. The common case, training and viewing on the same machine, does not need the round trip through MLflow at all: the writer already keeps a complete, continuously growing log directory on local disk for as long as the run lasts.

## What Changes

- `runsnap.tensorboard()` records where it writes on local disk: two new run tags carry the host name and the absolute path of the writer's scratch log directory.
- `runsnap tb` links a run straight to that local log directory when the tags name the current host and the directory still exists, instead of fetching from artifacts. TensorBoard then tails the same files the training process is appending to, so its own periodic reload shows new data with no polling and no MLflow traffic.
- Runs whose local directory is absent, or that were logged on another host, keep today's snapshot-from-artifacts behavior. The two kinds mix freely in one invocation, so `--chain` shows finished attempts from the cache beside the live one.
- `runsnap.tensorboard()` flushes the writer every 10 seconds by default instead of the underlying writer's 120, so the local files and the uploads lag by seconds rather than minutes. A caller passing `flush_secs=` still wins.
- README documents the live-on-same-host behavior and the flush default.

The existing `runsnap.tb.logdir` = `tb` tag is unchanged, so tag queries for runs with TensorBoard data keep working.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `tensorboard-logging`: a run records the host and local path of its writer's log directory; the writer's default flush interval is 10 seconds.
- `tensorboard-viewing`: a run still being logged on the viewing host is shown live from its local log directory rather than from a downloaded snapshot; other runs are unaffected.

## Impact

- `src/runsnap/_tags.py`: two new tag names and a flush-interval constant.
- `src/runsnap/_tensorboard.py`: set the tags when the log directory is created; apply the flush default.
- `src/runsnap/_tb_fetch.py`: choose the local directory over the cache when the run's tags point at a readable directory on this host.
- `tests/test_tensorboard.py`, `tests/test_tb_fetch.py`, `tests/test_tb_cli.py`: cover the tags, the flush default, and the local-versus-cached choice.
- `README.md`: TensorBoard section.
- No new dependencies. No change to the artifact layout under `tb/`, so runs logged before this change are viewed exactly as before.
