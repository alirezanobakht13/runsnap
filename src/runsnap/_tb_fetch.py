"""Fetch TensorBoard artifacts into a persistent cache for local viewing."""

import os
import re
import shutil
import socket
import threading
import warnings
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from itertools import batched, count
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from urllib.parse import urlparse

from mlflow.entities import FileInfo, Run, ViewType
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
"""How often shown runs are checked for having been deleted in MLflow, a live
run's directory for having been removed, a followed selection is re-run for
runs that have begun writing events, and running runs shown from the cache are
fetched again, or get links to their new scalar files from an artifact root."""

CHECK_BATCH = 200
"""Run ids checked per search, within SQLite's bound-parameter limits."""

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

    A cached file whose remote copy has grown gains only the new bytes, so a
    TensorBoard already reading it sees them; a shorter remote copy is ignored.
    """
    logdir = _run_cache(cache_dir, run_id) / "events"
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
        cached = target.stat().st_size if target.is_file() else -1
        # A remote copy shorter than the cached one was caught mid-upload.
        if artifact.file_size is None or artifact.file_size > cached:
            target.parent.mkdir(parents=True, exist_ok=True)
            # Keep staging on the cache's filesystem so replacement is atomic.
            with TemporaryDirectory(prefix=".download-", dir=logdir.parent) as tmp:
                downloaded = Path(client.download_artifacts(run_id, artifact.path, tmp))
                _extend_or_replace(target, downloaded)
        if is_light:
            link = light_dir.joinpath(*relative.parts)
            link.parent.mkdir(parents=True, exist_ok=True)
            if not (link.is_symlink() and link.readlink() == target):
                link.unlink(missing_ok=True)
                link.symlink_to(target)
    return logdir if media else light_dir


def _run_cache(cache_dir: Path | None, run_id: str) -> Path:
    """The folder caching `run_id`, under `cache_dir` or the default cache."""
    if cache_dir is None:
        cache_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        cache_dir = cache_dir / "runsnap" / "tensorboard"
    return cache_dir.resolve() / run_id


def _extend_or_replace(target: Path, downloaded: Path) -> None:
    """Append the downloaded tail to `target` if it begins with `target`'s bytes.

    Otherwise the download is moved over `target`. An interrupted append leaves
    a prefix of the remote file, which the next fetch completes.
    """
    if target.is_file():
        cached = target.read_bytes()
        with downloaded.open("rb") as new:
            if new.read(len(cached)) == cached:
                with target.open("ab") as out:
                    out.write(new.read())
                return
    downloaded.replace(target)


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
    artifact_root: Path | None = None,
) -> Iterator[Path]:
    """Yield a temporary directory of named links to cached runs.

    With `chain`, every earlier attempt the runs continue is linked too.

    A run still being written on this host is linked to the directory it is
    being written to, so TensorBoard tails the growing files themselves; every
    other run is downloaded into the cache first, several runs at a time.

    With `artifact_root`, the folder a tracking server on this host stores
    proxied artifacts in, every run is shown from its files there instead:
    its link is a directory holding a link to each of its scalar files, and
    nothing is downloaded, cached, or shown live. A run stored elsewhere is
    reported as a warning and left out.

    While the context is open, a thread checks every `poll_interval` seconds
    whether MLflow still holds the shown runs. A run deleted there, directly or
    with its experiment, loses its link and its cached files, and is linked
    again if it is restored while still selected. The same thread checks
    whether a live run's directory has been removed, which its writer does on
    leaving its block, and then fetches that run into the cache and re-points
    its link there, so TensorBoard rediscovers it on its next reload.

    With `follow`, the same thread also calls it on every check and links each
    run it returns that has since begun writing through `runsnap.tensorboard()`,
    as a run given at startup would be linked. Runs already linked keep their
    names, and a run whose name is already shown is linked under a distinct one.

    A run that has begun writing but has no event data here yet, such as one
    starting on another host before its first upload, is tried again on every
    check while MLflow reports it running, and once more after it ends. A run
    shown from the cache is likewise fetched again on every check while MLflow
    reports it running, and once more after it ends, so its curves keep
    growing; a run shown from `artifact_root` gets links to its new scalar
    files instead.

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
            artifact_root=artifact_root,
        )
        watch.add(selected)
        watch.start()
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
    artifact_root: Path | None,
) -> tuple[dict[str, Path], set[str]]:
    """Where each run holding event data is linked from, and which are live.

    A run still being written on this host is linked to the directory it is
    being written to; every other run is downloaded into the cache first.
    Runs without event data are left out, as are runs whose fetch failed.

    With `artifact_root`, each run is linked from its folder there once that
    holds a scalar file, and none is live.
    """
    if artifact_root is not None:
        folders = {
            run.info.run_id: _artifact_folder(run, artifact_root) for run in runs
        }
        return {
            run_id: folder
            for run_id, folder in folders.items()
            if any(_scalar_files(folder))
        }, set()
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
    quiet: Collection[str] = (),
) -> dict[str, Path]:
    """Cache several runs at once, keyed in the order they were asked for.

    A run the tracking server will not hand over is reported as a warning,
    unless it is in `quiet`, and left out, so the runs that were fetched can
    still be viewed together.
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
            if run_id not in quiet:
                warnings.warn(
                    f"runsnap could not fetch run {run_id}: {exc}", stacklevel=6
                )
    return fetched


def _run_statuses(client: MlflowClient, runs: dict[str, str]) -> dict[str, str]:
    """The status of each of `runs`, which maps run ids to experiment ids.

    A run deleted in MLflow, directly or with its experiment, is left out.
    Runs are searched `CHECK_BATCH` at a time, one search per batch.
    """
    statuses: dict[str, str] = {}
    for batch in batched(runs, CHECK_BATCH):
        experiment_ids = sorted({runs[run_id] for run_id in batch})
        ids = ", ".join(f"'{run_id}'" for run_id in batch)
        token = None
        while True:
            page = client.search_runs(
                experiment_ids,
                filter_string=f"attributes.run_id IN ({ids})",
                run_view_type=ViewType.ACTIVE_ONLY,
                page_token=token,
            )
            statuses.update((run.info.run_id, run.info.status) for run in page)
            token = page.token
            if not token:
                break
    return statuses


class _LiveWatch:
    """Links runs into `logdir` and keeps those links current.

    A run that carries its local directory tag, which the writer sets once its
    event files exist, but has no event data here yet is pending while MLflow
    reports it running, as a run starting on another host is until its first
    upload.

    The thread makes a pass every `interval` seconds. Each pass first checks
    the watched runs with MLflow: the shown and the pending runs. A run the
    check does not find was deleted, directly or with its experiment, so its
    link is removed, freeing its name, and its cached files are deleted.
    Without `follow`, such a run stays pending, so it is linked again once a
    check finds it restored; with `follow`, the selection finds it again. A
    failing check is reported once until a check succeeds again, and removes
    nothing.

    A run linked live points at its writer's scratch directory, which the
    writer removes after its final upload. Each pass then checks each live
    link and, when the directory it points at is gone, fetches the run and
    links the same name to the cached copy so TensorBoard rediscovers it on its
    next reload. A run whose fetch fails is reported as a warning and left as
    it is, as `_fetch_runs` does at startup.

    With `follow`, each pass then re-runs that selection and adds the runs
    that carry their local directory tag and are neither handled nor pending,
    as runs given at startup are added. A failing pass is reported once until a
    pass succeeds again, and never ends the thread.

    Each pass then tries to link the pending runs the check found, and stops
    trying a run once the check reports it ended.

    Last, each pass fetches again the runs shown from the cache that the check
    reports running, so their curves keep growing. A run fetched while it was
    running gets one final fetch once a check reports it ended, and none after
    that; a run that had ended when it was fetched is not fetched again. Runs
    the pass linked or re-pointed are not fetched twice. A failing refresh is
    reported once until that run's refresh succeeds again, keeps the run's
    link and cached files, and is tried again at the next pass.

    A pass whose check failed skips the last two steps.

    With `artifact_root`, a run is linked as a directory of links to the
    scalar files in its folder there, and a refresh links the scalar files
    that appeared since, on the same passes a fetch would run. Nothing is
    linked live, and removing a run leaves its folder to MLflow.
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
        artifact_root: Path | None,
    ) -> None:
        self._client = client
        self._logdir = logdir
        self._follow = follow
        self._chain = chain
        self._cache_dir = cache_dir
        self._media = media
        self._interval = interval
        self._artifact_root = artifact_root
        # Every run passed to `link`, and the link of each one shown.
        self._runs: dict[str, Run] = {}
        self._shown: dict[str, Path] = {}
        self._links: dict[str, Path] = {}
        # Runs a followed selection leaves alone: shown runs, and tagged runs
        # that ended without event data.
        self._handled: set[str] = set()
        self._pending: set[str] = set()
        # Runs shown from the cache that the next pass fetches whatever their
        # status: last fetched while running, or whose last refresh failed.
        self._refreshing: set[str] = set()
        self._refresh_failing: set[str] = set()
        self._check_failing = False
        self._follow_failing = False
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, name="runsnap-tb-watch", daemon=True
        )

    def link(self, runs: dict[str, Run], running: set[str]) -> set[str]:
        """Link each of `runs`, keyed by id, that holds event data.

        A run gets its own name unless another run in `runs`, a shown run, or
        an existing link has it; links made before keep theirs. A run linked
        is handled and no longer pending, even if a later link fails. Returns
        the ids of the runs linked.

        `running` holds the runs MLflow last reported running. Such a run
        linked from the cache may upload more, so later passes fetch it again.
        """
        self._runs.update(runs)
        targets, live = _link_targets(
            self._client,
            runs.values(),
            cache_dir=self._cache_dir,
            media=self._media,
            artifact_root=self._artifact_root,
        )
        bases = {run_id: _run_name(runs[run_id]) for run_id in targets}
        shown = {_run_name(self._runs[run_id]) for run_id in self._shown}
        taken = {path.name for path in self._logdir.iterdir()}
        names = _link_names(bases, shown, taken)
        for run_id, name in names.items():
            link = self._logdir / name
            if self._artifact_root is None:
                link.symlink_to(targets[run_id], target_is_directory=True)
            else:
                link.mkdir()
                _link_scalars(targets[run_id], link)
            self._shown[run_id] = link
            self._handled.add(run_id)
            self._pending.discard(run_id)
        self._links.update((run_id, self._shown[run_id]) for run_id in live)
        self._refreshing |= running & (names.keys() - live)
        return set(names)

    def add(self, runs: dict[str, Run]) -> None:
        """Link each of `runs`, keyed by id, that holds event data.

        A run left out that carries its local directory tag is pending while
        it runs, and handled once it has ended, since this was its last try.
        Untagged runs are left to a followed selection, which adds them once
        they are tagged. With an artifact root, a run stored elsewhere is
        reported as a warning and handled, so it is never tried again.
        """
        if self._artifact_root is not None:
            elsewhere: set[str] = set()
            for run_id, run in runs.items():
                try:
                    _artifact_folder(run, self._artifact_root)
                except ValueError as exc:
                    warnings.warn(
                        f"runsnap leaves out run {run_id}: {exc}", stacklevel=2
                    )
                    elsewhere.add(run_id)
            self._handled |= elsewhere
            runs = {r: run for r, run in runs.items() if r not in elsewhere}
        running = {
            run_id for run_id, run in runs.items() if run.info.status == "RUNNING"
        }
        linked = self.link(runs, running)
        tagged = {
            run_id
            for run_id in runs.keys() - linked
            if TB_TAG_LOCAL_DIR in runs[run_id].data.tags
        }
        self._pending |= tagged & running
        self._handled |= tagged - running

    def start(self) -> None:
        """Start the thread, which makes passes until `stop` is called."""
        self._thread.start()

    def stop(self) -> None:
        """Stop the thread and wait for a pass in flight to finish."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            statuses = self._check_pass()
            # The steps below fetch every run they link, so only the runs shown
            # from the cache before them are refreshed.
            cached = self._shown.keys() - self._links.keys()
            for run_id in [r for r, link in self._links.items() if not link.is_dir()]:
                self._repoint(run_id, statuses)
            if self._follow is not None:
                self._follow_pass(self._follow)
            if statuses is not None:
                self._pending_pass(statuses)
                self._refresh_pass(cached, statuses)

    def _check_pass(self) -> dict[str, str] | None:
        """Remove the watched runs MLflow no longer holds.

        Returns the status of every watched run it still holds, or `None` when
        the check failed.
        """
        watched = self._shown.keys() | self._pending
        try:
            statuses = _run_statuses(
                self._client,
                {run_id: self._runs[run_id].info.experiment_id for run_id in watched},
            )
        except Exception as exc:  # noqa: BLE001 - keep watching
            if not self._check_failing:
                warnings.warn(
                    f"runsnap could not check the shown runs: {exc}", stacklevel=2
                )
            self._check_failing = True
            return None
        self._check_failing = False
        for run_id in watched - statuses.keys():
            self._remove(run_id)
        return statuses

    def _remove(self, run_id: str) -> None:
        """Unlink a run MLflow no longer holds and delete its cached files.

        A run shown from an artifact root loses its directory of links, which
        are removed without being followed, so its folder is left to MLflow.
        """
        link = self._shown.pop(run_id, None)
        if self._artifact_root is not None:
            if link is not None:
                shutil.rmtree(link, ignore_errors=True)
        else:
            if link is not None:
                link.unlink(missing_ok=True)
            shutil.rmtree(_run_cache(self._cache_dir, run_id), ignore_errors=True)
        self._links.pop(run_id, None)
        self._handled.discard(run_id)
        self._refreshing.discard(run_id)
        self._refresh_failing.discard(run_id)
        # Without `follow`, a run stays pending, so the check keeps looking for
        # it and it is linked again once restored, whatever its tags; with
        # `follow`, the selection finds a restored run again by itself.
        if self._follow is None:
            self._pending.add(run_id)
        else:
            self._pending.discard(run_id)

    def _follow_pass(self, follow: Callable[[], Iterable[Run]]) -> None:
        try:
            known = self._handled | self._pending
            ready = [
                run
                for run in follow()
                if TB_TAG_LOCAL_DIR in run.data.tags and run.info.run_id not in known
            ]
            if self._chain:
                ready = with_attempts(self._client, ready)
            self.add(
                {run.info.run_id: run for run in ready if run.info.run_id not in known}
            )
        except Exception as exc:  # noqa: BLE001 - keep following
            if not self._follow_failing:
                warnings.warn(
                    f"runsnap could not follow the selection: {exc}", stacklevel=2
                )
            self._follow_failing = True
        else:
            self._follow_failing = False

    def _pending_pass(self, statuses: dict[str, str]) -> None:
        """Link the pending runs that now hold event data; drop the ended ones.

        Only runs in `statuses` are tried: one the check did not find stays
        removed, and one this pass's selection just added was tried already.
        """
        checked = {r: self._runs[r] for r in self._pending & statuses.keys()}
        ended = {run_id for run_id in checked if statuses[run_id] != "RUNNING"}
        self.link(checked, checked.keys() - ended)
        self._pending -= ended
        self._handled |= ended

    def _refresh_pass(self, cached: set[str], statuses: dict[str, str]) -> None:
        """Fetch again the runs in `cached` that may have uploaded since.

        Those are the runs `statuses` reports running, and the runs last
        fetched while running or whose last refresh failed. A run reported
        ended is not fetched again once this fetch succeeds. With an artifact
        root, those runs get links to their new scalar files instead.
        """
        running = {run_id for run_id in cached if statuses[run_id] == "RUNNING"}
        due = running | (self._refreshing & cached)
        if self._artifact_root is None:
            fetched = set(
                _fetch_runs(
                    self._client,
                    sorted(due),
                    cache_dir=self._cache_dir,
                    media=self._media,
                    quiet=self._refresh_failing,
                )
            )
        else:
            for run_id in due:
                folder = _artifact_folder(self._runs[run_id], self._artifact_root)
                _link_scalars(folder, self._shown[run_id])
            fetched = due
        self._refresh_failing = due - fetched
        # A run fetched after it ended holds its final events.
        self._refreshing |= running
        self._refreshing -= fetched - running

    def _repoint(self, run_id: str, statuses: dict[str, str] | None) -> None:
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
        # Unless this pass's check saw the run end before this fetch, it may
        # upload more.
        if statuses is None or statuses[run_id] == "RUNNING":
            self._refreshing.add(run_id)


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


def _artifact_folder(run: Run, root: Path) -> Path:
    """The folder under `root` holding `run`'s TensorBoard files.

    A tracking server started with `--artifacts-destination root` stores a
    run whose artifact URI is `mlflow-artifacts:/<path>` at `root/<path>`.
    Raises `ValueError` for a run stored any other way, or whose path leads
    out of `root`.
    """
    uri = urlparse(str(run.info.artifact_uri))
    if uri.scheme == "mlflow-artifacts":
        folder = (root / uri.path.lstrip("/") / TB_ARTIFACT_DIR).resolve()
        if folder.is_relative_to(root.resolve()):
            return folder
    raise ValueError(f"its artifacts at {run.info.artifact_uri} are not under {root}")


def _scalar_files(folder: Path) -> Iterator[Path]:
    """The scalar event files under `folder`, in its subfolders too."""
    return (
        path
        for path in folder.rglob("events.out.tfevents.*")
        if LIGHT_NAME.search(path.name) is not None
    )


def _link_scalars(folder: Path, links: Path) -> None:
    """Link each scalar file under `folder` that `links` lacks, mirroring subfolders.

    The links name the files, so a file the tracking server replaces with a
    longer copy is read through the same link.
    """
    for path in _scalar_files(folder):
        link = links / path.relative_to(folder)
        if not link.is_symlink():
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(path)


def _run_name(run: Run) -> str:
    name = run.info.run_name or run.data.tags.get("mlflow.runName")
    if not name:
        return run.info.run_id
    name = re.sub(r"[/\\\x00]", "_", name)
    return run.info.run_id if name in {".", ".."} else name
