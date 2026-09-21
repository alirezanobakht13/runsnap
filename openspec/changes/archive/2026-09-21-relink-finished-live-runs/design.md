## Context

`assemble_logdir` (`src/runsnap/_tb_fetch.py`) yields a temporary directory of
symlinks, one per run, and `tb` keeps that context open around
`_launch_tensorboard`, which runs TensorBoard as a foreground subprocess. A run
tagged as being written on this host is linked straight to its writer's scratch
directory; every other run is fetched into `~/.cache/runsnap/tensorboard/<run_id>/`
and linked there. Nothing touches the links after they are created.

The writer side (`runsnap.tensorboard()`) ends its block by closing the
writer, running a final upload of every event file, and then `rmtree`-ing the
scratch directory. So by the time a live link dangles, the run's complete
artifacts are on the tracking server (or the writer has already warned that
they are not).

TensorBoard's ingester loop, every `--reload_interval` (5 s by default), first
calls `AddRunsFromDirectory` on the logdir and then `Reload()` on each
accumulator. A dangling link makes the run's directory listing fail; the
watcher then raises `DirectoryDeletedError` and the multiplexer drops the
accumulator with the `Deleting accumulator` warning. A link that points at a
directory holding event files again is picked up by the next
`AddRunsFromDirectory` as a fresh run. See proposal.md for why this matters.

## Goals / Non-Goals

**Goals:**

- A run that was linked live is re-pointed at its cached artifacts once its
  directory is gone, while TensorBoard keeps running.
- No change to the writer, the tags, the artifact layout, or the CLI surface.
- No TensorBoard-side workaround: rely on its existing drop-and-rediscover
  behaviour.

**Non-Goals:**

- Refreshing runs that were cached at startup as they upload more data (runs
  training on another host). The README's "rerun the command" advice stays for
  those.
- Retrying a fetch that failed after a directory disappeared.
- Keeping a run's TensorBoard accumulator alive across the switch; a brief
  disappearance between TensorBoard reloads is accepted.

## Decisions

### The watcher lives inside `assemble_logdir`

The context manager already holds everything the watcher needs: the client,
the cache directory, the media flag, which runs were linked live, and the
name each link was given. It starts a daemon thread after creating the links
and stops it before the temporary directory is removed, so the CLI's `tb`
gains the behaviour without changing. Alternative: a separate watcher object
that `tb` starts and stops around `_launch_tensorboard`. That spreads one
concern over two modules and gives `tb` state to manage for no benefit.

The thread is only started when at least one run was linked live; a dashboard
of finished runs behaves exactly as today.

### Poll the directories; do not use inotify

The thread wakes every `LIVE_POLL_SECONDS` (a module constant, 5 s, on the
order of TensorBoard's own reload) and checks `Path.is_dir()` for each live
directory still being watched. That is one stat per live run per poll, adds no
dependency, and mirrors how `_Sync` in `_tensorboard.py` runs its loop on a
`threading.Event`. `assemble_logdir` takes a `poll_interval` keyword so tests
can run the loop fast, the way `tensorboard()` takes `sync_interval`.

### Re-point by unlinking and relinking; atomicity is not needed

When a directory is gone the thread calls `fetch_run` for that run with the
same `cache_dir` and `media` used at assembly, then removes the old link and
creates one with the same name to the returned path. There is a moment where
the name is absent; TensorBoard already tolerates that (the run is simply not
found on that reload and rediscovered on the next), so an atomic
rename-over-symlink would buy nothing and would need a temporary name inside
the directory TensorBoard walks.

Both orderings relative to TensorBoard's reload converge:

```
  writer: sync --> rmtree
                     |
   TB reloads first  |  watcher re-points first
   ------------------+------------------------------------
   listdir fails     |  TB's file reader still holds the old
   -> accumulator    |  inode open and reads it to EOF; the
      deleted        |  new directory lists the same file
   next reload:      |  names, so the watcher finds nothing
   AddRunsFrom-      |  newer. The run never disappears and
   Directory finds   |  its data is complete because the
   the re-pointed    |  writer closed the file before rmtree.
   link -> fresh     |
   accumulator from  |
   the cached file   |
```

### A failed fetch is warned and the run is dropped from watching

`fetch_run` raising after a directory vanished is reported with
`warnings.warn`, the link is left dangling (so the run stays absent, as
today), and the run is removed from the watched set. This is the same policy
`_fetch_runs` applies at startup. Retrying every poll would repeat the warning
indefinitely for a server that is down.

### Stop by joining, as `_Sync` does

On context exit the thread's stop event is set and the thread joined before
`TemporaryDirectory` cleans up. A fetch in flight delays exit by at most that
download, which downloads into the cache rather than the temporary directory,
so nothing is lost by waiting. This mirrors `_Sync.stop`.

## Risks / Trade-offs

- [The run is invisible for up to one poll plus one TensorBoard reload after
  the writer exits] → Both intervals are a few seconds; acceptable for a
  dashboard, and no worse than the run silently vanishing today.
- [The writer's final upload failed, so the cache holds only what was uploaded
  earlier] → The writer already warns in that case, and a restart today would
  show the same partial data. Nothing new is lost.
- [A live directory removed by something other than the writer, before the
  final upload] → Out of scope; the cache shows what has been uploaded, which
  is what a restart shows today.
- [Tests that exercise the switch depend on timing] → Tests pass a small
  `poll_interval` and poll for the expected state with a deadline, as
  `test_assembled_logdir_follows_a_run_still_being_written` already does.
- [Ctrl-C during a fetch waits for the download] → Bounded by one run's
  scalar files; the same bound `_Sync.stop` accepts.
