"""Fetch TensorBoard artifacts into a persistent cache for local viewing."""

import os
import re
import socket
import threading
import warnings
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from itertools import count
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from mlflow.entities import FileInfo, Run
from mlflow.tracking import MlflowClient

from runsnap._lifecycle import with_attempts
from runsnap._tags import (
    TB_ARTIFACT_DIR,
    TB_LIGHT_SUFFIX,
    TB_TAG_LOCAL_DIR,
    TB_TAG_LOCAL_HOST,
)

FETCH_WORKERS = 8
"""Runs fetched at once, a cap on concurrent downloads from one server."""

LIVE_POLL_SECONDS = 5.0
"""How often a live run's directory is checked for having been removed, and a
followed selection is re-run for runs that have begun writing events."""

LIGHT_NAME = re.compile(rf"{re.escape(TB_LIGHT_SUFFIX)}(\.\d+)?$")
"""Matches a scalar event file, sharded or not.

Runs written before the scalar stream was sharded end in `.scalars`; sharded
ones end in `.scalars.<n>`, and a resumed run's directory can hold both.
"""


def fetch_run(
    client: MlflowClient,
    run_id: str,
    *,
    cache_dir: Path | None = None,
    media: bool = False,
) -> Path:
    """Cache light event files, including media only when requested.

    Downloads are shared between both views. The light view links only scalar
    files, so media cached by a previous invocation stays out of default views.
    """
    if cache_dir is None:
        cache_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        cache_dir = cache_dir / "runsnap" / "tensorboard"
    logdir = cache_dir.resolve() / run_id / "events"
    logdir.mkdir(parents=True, exist_ok=True)
    light_dir = logdir.parent / "scalars"
    light_dir.mkdir(exist_ok=True)
    for artifact in _event_artifacts(client, run_id):
        relative = PurePosixPath(artifact.path).relative_to(TB_ARTIFACT_DIR)
        if ".." in relative.parts:
            raise ValueError(f"Invalid TensorBoard artifact path: {artifact.path}")
        is_light = LIGHT_NAME.search(relative.name) is not None
        if not media and not is_light:
            continue
        target = logdir.joinpath(*relative.parts)
        if not target.is_file() or target.stat().st_size != artifact.file_size:
            target.parent.mkdir(parents=True, exist_ok=True)
            # Keep staging on the cache's filesystem so replacement is atomic.
            with TemporaryDirectory(prefix=".download-", dir=logdir.parent) as tmp:
                downloaded = Path(client.download_artifacts(run_id, artifact.path, tmp))
                downloaded.replace(target)
        if is_light:
            link = light_dir.joinpath(*relative.parts)
            link.parent.mkdir(parents=True, exist_ok=True)
            if not (link.is_symlink() and link.readlink() == target):
                link.unlink(missing_ok=True)
                link.symlink_to(target)
    return logdir if media else light_dir


def _event_artifacts(
    client: MlflowClient, run_id: str, path: str = TB_ARTIFACT_DIR
) -> Iterator[FileInfo]:
    for artifact in client.list_artifacts(run_id, path):
        if artifact.is_dir:
            yield from _event_artifacts(client, run_id, artifact.path)
        elif PurePosixPath(artifact.path).name.startswith("events.out.tfevents."):
            yield artifact


@contextmanager
def assemble_logdir(
    client: MlflowClient,
    runs: Iterable[Run],
    *,
    cache_dir: Path | None = None,
    media: bool = False,
    chain: bool = False,
    follow: Callable[[], Iterable[Run]] | None = None,
    poll_interval: float = LIVE_POLL_SECONDS,
) -> Iterator[Path]:
    """Yield a temporary directory of named links to cached runs.

    With `chain`, every earlier attempt the runs continue is linked too.

    A run still being written on this host is linked to the directory it is
    being written to, so TensorBoard tails the growing files themselves; every
    other run is downloaded into the cache first, several runs at a time.

    While the context is open, a thread checks every `poll_interval` seconds
    whether a live run's directory has been removed, which its writer does on
    leaving its block, and then fetches that run into the cache and re-points
    its link there, so TensorBoard rediscovers it on its next reload.

    With `follow`, the same thread also calls it on every check and links each
    run it returns that has since begun writing through `runsnap.tensorboard()`,
    as a run given at startup would be linked. Runs already linked keep their
    names, and a run whose name is already shown is linked under a distinct one.

    Keep this context open while TensorBoard runs. Its links are removed on
    exit, while downloaded event files remain available for later invocations.
    Runs without event data are omitted, as are runs whose fetch failed.
    """
    if chain:
        runs = with_attempts(client, list(runs))
    selected = {run.info.run_id: run for run in runs}
    with TemporaryDirectory(prefix="runsnap-tb-") as tmp:
        logdir = Path(tmp)
        watch = _LiveWatch(
            client,
            logdir,
            follow=follow,
            chain=chain,
            cache_dir=cache_dir,
            media=media,
            interval=poll_interval,
        )
        shown = watch.link(selected)
        # A run neither shown nor tagged with its local directory may not have
        # begun writing yet, so a later pass may still link it.
        tagged = {
            run_id
            for run_id, run in selected.items()
            if TB_TAG_LOCAL_DIR in run.data.tags
        }
        watch.start(handled=shown | tagged)
        try:
            yield logdir
        finally:
            watch.stop()


def _link_targets(
    client: MlflowClient,
    runs: Iterable[Run],
    *,
    cache_dir: Path | None,
    media: bool,
) -> tuple[dict[str, Path], set[str]]:
    """Where each run holding event data is linked from, and which are live.

    A run still being written on this host is linked to the directory it is
    being written to; every other run is downloaded into the cache first.
    Runs without event data are left out, as are runs whose fetch failed.
    """
    live = {run.info.run_id: _live_logdir(run) for run in runs}
    fetched = _fetch_runs(
        client,
        [run_id for run_id, path in live.items() if path is None],
        cache_dir=cache_dir,
        media=media,
    )
    paths = {
        run_id: path if path is not None else fetched[run_id]
        for run_id, path in live.items()
        if path is not None or run_id in fetched
    }
    targets = {
        run_id: path
        for run_id, path in paths.items()
        if any(path.rglob("events.out.tfevents.*"))
    }
    return targets, {run_id for run_id in targets if live[run_id] is not None}


def _fetch_runs(
    client: MlflowClient,
    run_ids: Sequence[str],
    *,
    cache_dir: Path | None,
    media: bool,
) -> dict[str, Path]:
    """Cache several runs at once, keyed in the order they were asked for.

    A run the tracking server will not hand over is reported as a warning and
    left out, so the runs that were fetched can still be viewed together.
    """
    if not run_ids:
        return {}
    with ThreadPoolExecutor(max_workers=min(FETCH_WORKERS, len(run_ids))) as pool:
        futures = {
            run_id: pool.submit(
                fetch_run, client, run_id, cache_dir=cache_dir, media=media
            )
            for run_id in run_ids
        }
    fetched: dict[str, Path] = {}
    for run_id, future in futures.items():
        try:
            fetched[run_id] = future.result()
        except Exception as exc:  # noqa: BLE001 - one unreachable run is not
            # a reason to abandon the runs that were fetched.
            warnings.warn(f"runsnap could not fetch run {run_id}: {exc}", stacklevel=6)
    return fetched


class _LiveWatch:
    """Links runs into `logdir` and keeps those links current.

    A run linked live points at its writer's scratch directory, which the
    writer removes after its final upload. The thread checks each live link
    every `interval` seconds and, when the directory it points at is gone,
    fetches the run and links the same name to the cached copy so TensorBoard
    rediscovers it on its next reload. A run whose fetch fails is reported as
    a warning and left as it is, as `_fetch_runs` does at startup.

    With `follow`, each check then re-runs that selection and links the runs
    that carry their local directory tag, which the writer sets once its event
    files exist, and that were not handled before. A failing pass is reported
    once until a pass succeeds again, and never ends the thread.
    """

    def __init__(
        self,
        client: MlflowClient,
        logdir: Path,
        *,
        follow: Callable[[], Iterable[Run]] | None,
        chain: bool,
        cache_dir: Path | None,
        media: bool,
        interval: float,
    ) -> None:
        self._client = client
        self._logdir = logdir
        self._follow = follow
        self._chain = chain
        self._cache_dir = cache_dir
        self._media = media
        self._interval = interval
        self._links: dict[str, Path] = {}
        self._shown: set[str] = set()
        self._handled: set[str] = set()
        self._failing = False
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, name="runsnap-tb-watch", daemon=True
        )

    def link(self, runs: dict[str, Run]) -> set[str]:
        """Link each of `runs`, keyed by id, that holds event data.

        A run gets its own name unless another run in `runs`, a run linked
        before, or an existing link has it; links made before keep theirs.
        Returns the ids of the runs linked.
        """
        targets, live = _link_targets(
            self._client, runs.values(), cache_dir=self._cache_dir, media=self._media
        )
        bases = {run_id: _run_name(runs[run_id]) for run_id in targets}
        taken = {path.name for path in self._logdir.iterdir()}
        names = _link_names(bases, self._shown, taken)
        for run_id, name in names.items():
            (self._logdir / name).symlink_to(targets[run_id], target_is_directory=True)
        self._shown.update(bases.values())
        self._links.update((run_id, self._logdir / names[run_id]) for run_id in live)
        return set(names)

    def start(self, *, handled: set[str]) -> None:
        """Start the thread when there are live links or a selection to follow.

        A run in `handled` is never linked by a later pass.
        """
        self._handled = handled
        if self._links or self._follow is not None:
            self._thread.start()

    def stop(self) -> None:
        """Stop the thread and wait for a check in flight to finish."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            for run_id in [r for r, link in self._links.items() if not link.is_dir()]:
                self._repoint(run_id)
            if self._follow is not None:
                self._follow_pass(self._follow)
            elif not self._links:
                return

    def _follow_pass(self, follow: Callable[[], Iterable[Run]]) -> None:
        try:
            ready = [
                run
                for run in follow()
                if TB_TAG_LOCAL_DIR in run.data.tags
                and run.info.run_id not in self._handled
            ]
            if self._chain:
                ready = with_attempts(self._client, ready)
            batch = {
                run.info.run_id: run
                for run in ready
                if run.info.run_id not in self._handled
            }
            # Handled before linking, so a pass that fails midway never
            # links the same run twice.
            self._handled.update(batch)
            self.link(batch)
        except Exception as exc:  # noqa: BLE001 - keep following
            if not self._failing:
                warnings.warn(
                    f"runsnap could not follow the selection: {exc}", stacklevel=2
                )
            self._failing = True
        else:
            self._failing = False

    def _repoint(self, run_id: str) -> None:
        link = self._links.pop(run_id)
        try:
            cached = fetch_run(
                self._client, run_id, cache_dir=self._cache_dir, media=self._media
            )
        except Exception as exc:  # noqa: BLE001 - one unreachable run is not
            # a reason to stop watching the others.
            warnings.warn(f"runsnap could not fetch run {run_id}: {exc}", stacklevel=2)
            return
        link.unlink()
        link.symlink_to(cached, target_is_directory=True)


def _link_names(
    bases: dict[str, str], shown: set[str], taken: set[str]
) -> dict[str, str]:
    """Link names for the runs in `bases`, which maps run ids to their own names.

    `shown` holds the own names of runs already linked, and `taken` the names
    of their links. A run keeps its own name when it is unique in `bases` and
    in neither set; every other run gets the first of its `_distinct_names`
    not yet taken.
    """
    counts = Counter(bases.values())
    names = {
        run_id: base
        for run_id, base in bases.items()
        if counts[base] == 1 and base not in shown and base not in taken
    }
    used = taken | set(names.values())
    for run_id, base in bases.items():
        if run_id not in names:
            name = next(n for n in _distinct_names(base, run_id) if n not in used)
            names[run_id] = name
            used.add(name)
    return names


def _distinct_names(base: str, run_id: str) -> Iterator[str]:
    """Names for a run whose own name another selected run shares, shortest first."""
    yield f"{base}-{run_id[:8]}"
    yield f"{base}-{run_id}"
    for counter in count(1):
        yield f"{base}-{run_id}-{counter}"


def _live_logdir(run: Run) -> Path | None:
    """Where `run` is writing its events on this host, if it still is.

    A directory that is absent says the run is over, and one tagged with
    another host says its events only reach here through the artifact store.
    """
    if run.data.tags.get(TB_TAG_LOCAL_HOST) != socket.gethostname():
        return None
    local = run.data.tags.get(TB_TAG_LOCAL_DIR)
    if local is None:
        return None
    path = Path(local)
    return path if path.is_dir() else None


def _run_name(run: Run) -> str:
    name = run.info.run_name or run.data.tags.get("mlflow.runName")
    if not name:
        return run.info.run_id
    name = re.sub(r"[/\\\x00]", "_", name)
    return run.info.run_id if name in {".", ".."} else name
