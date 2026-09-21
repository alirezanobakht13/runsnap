## Why

A run shown live from its local log directory vanishes from an open `runsnap tb`
dashboard the moment its writer block exits: the writer removes its scratch
directory, the assembled link dangles, and TensorBoard drops the run
(`Deleting accumulator '<name>'`). The final curves are already uploaded, but
the only way to see them is to restart the command. A dashboard left open on
a training run should keep showing that run after it finishes.

## What Changes

- While TensorBoard is running, `runsnap tb` watches every run it linked live
  and, when a run's local directory disappears, fetches that run's uploaded
  artifacts into the cache and re-points its link there. TensorBoard
  rediscovers the run on its next reload, so the run reappears with its final
  data without restarting the command.
- A run whose fetch fails after its directory disappears is reported as a
  warning and left as it is, matching how a run that cannot be fetched at
  startup is handled.
- The README's instruction to rerun the command after a live writer exits is
  replaced with a description of the new behaviour.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `tensorboard-viewing`: a run shown live from its local directory stays
  visible after its writer block exits, shown from its cached artifacts as a
  finished run would be, without restarting the command.

## Impact

- `src/runsnap/_tb_fetch.py`: `assemble_logdir` keeps a background thread
  alive for the duration of its context that polls the live links and
  re-points any whose directory has gone.
- `src/runsnap/_cli.py`: no change to `tb` beyond what the context already
  provides; the thread lives as long as the context around `_launch_tensorboard`.
- `tests/test_tb_fetch.py`: new tests for the re-point path, the fetch-failure
  path, and the killed-without-cleanup case staying untouched.
- `README.md`: the TensorBoard section's live-run paragraph.
- No new dependencies. No change to the writer side (`runsnap.tensorboard()`),
  the tags, or the artifact layout.
