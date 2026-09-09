## 1. Dependencies and names

- [x] 1.1 Add the dependencies with `uv add tensorboardx tensorboard` and verify `uv run python -c "import tensorboardX, tensorboard"` succeeds against the existing `protobuf` pin
- [x] 1.2 Add the TensorBoard names to `src/exp_track/_tags.py` — the `exp_track.tb.logdir` tag, the `tb` artifact directory, the light and media filename suffixes, the shard size threshold, and the sync interval — and verify `uv run pytest` still passes

## 2. Writer

- [x] 2.1 Create `src/exp_track/_tensorboard.py` with the two-writer facade: light writer for `add_scalar`/`add_text`, media writer for everything else, resolved through `__getattr__`; verify a test asserting that `add_scalar` and `add_image` land in different event files in the same directory
- [x] 2.2 Add the explicit `add_histogram` override defaulting `bins` to the compact value; verify a test asserting the default produces a substantially smaller entry than `bins="tensorflow"` and that an explicit `bins` argument is passed through unchanged
- [x] 2.3 Add size-based shard rolling in the delegated call under a lock, with the shard index in the filename suffix; verify a test that writes past the threshold produces multiple distinctly named media files and that two shards sealed in the same second do not collide
- [x] 2.4 Verify with a test that a directory holding the light file plus several media shards, written interleaved so step numbers overlap, reads back through TensorBoard's event accumulator as one run with every scalar point present and none purged

## 3. Sync

- [x] 3.1 Add the background sync thread uploading files whose size changed since last upload, skipping the open media shard, to artifact path `tb/`; verify a test against a local MLflow tracking URI that a sealed shard appears in the run's artifacts before the run ends and is not uploaded twice
- [x] 3.2 Set the `exp_track.tb.logdir` tag on entry; verify a test asserting the tag is present on the run
- [x] 3.3 Make the context manager seal the open shard, upload everything outstanding, and close both writers on exit; verify tests covering normal exit, exit by exception, and exit by `KeyboardInterrupt`
- [x] 3.4 Wrap every write, roll, and upload failure so it warns instead of raising; verify tests that an unreachable tracking server and a failing upload at exit both leave user code unaffected
- [x] 3.5 Export `tensorboard` from `src/exp_track/__init__.py` and add it to `__all__`; verify a test that `exp_track.tensorboard()` with no active run raises an error naming the problem

## 4. Fetch and assemble

- [x] 4.1 Create `src/exp_track/_tb_fetch.py` with the run-id-keyed cache: list a run's `tb/` artifacts, skip files already held at their current remote size, download others to a temporary file and rename into place; verify tests that a second fetch downloads nothing and that an interrupted partial file is replaced rather than trusted
- [x] 4.2 Add assembly of a temporary directory of symlinks named by run name, with a short run-id suffix on collision and the run id for an unnamed run; verify a test asserting the resulting directory names for duplicate-named and unnamed runs
- [x] 4.3 Add the scalars-only default and the media opt-in to the fetch path; verify a test that the default fetches only the light file and that requesting media fetches both

## 5. CLI

- [x] 5.1 Add the `tb` command to `src/exp_track/_cli.py` accepting positional run refs, `--experiment`, and `--filter`, reusing the existing run resolution; verify tests covering each selection form and the no-matching-runs message
- [x] 5.2 Launch TensorBoard as a foreground subprocess on the assembled directory, reporting a clear message naming the package when TensorBoard is absent; verify a test that a missing TensorBoard produces the message rather than a traceback

## 6. Documentation

- [x] 6.1 Add the TensorBoard section to `README.md` — the `exp_track.tensorboard()` example, the `exp-track tb` commands, the scalars-by-default and `--media` behavior, and the loss of unsynced data on an unannounced kill (including pending sealed shards, with no fixed loss bound); verify the documented example runs as written
- [x] 6.2 Final pass: run `uv run ruff check`, `uv run ruff format`, and `uv run ty check`, and verify all three pass on the new modules and tests
