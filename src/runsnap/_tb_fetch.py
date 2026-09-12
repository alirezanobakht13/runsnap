"""Fetch TensorBoard artifacts into a persistent cache for local viewing."""

import os
import re
import socket
import warnings
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from itertools import count
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from mlflow.entities import FileInfo, Run
from mlflow.tracking import MlflowClient

from runsnap._tags import (
    TB_ARTIFACT_DIR,
    TB_LIGHT_SUFFIX,
    TB_TAG_LOCAL_DIR,
    TB_TAG_LOCAL_HOST,
)

FETCH_WORKERS = 8
"""Runs fetched at once, a cap on concurrent downloads from one server."""

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
) -> Iterator[Path]:
    """Yield a temporary directory of named links to cached runs.

    A run still being written on this host is linked to the directory it is
    being written to, so TensorBoard tails the growing files themselves; every
    other run is downloaded into the cache first, several runs at a time.

    Keep this context open while TensorBoard runs. Its links are removed on
    exit, while downloaded event files remain available for later invocations.
    Runs without event data are omitted, as are runs whose fetch failed.
    """
    selected = {run.info.run_id: run for run in runs}
    live = {run_id: _live_logdir(run) for run_id, run in selected.items()}
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
    names = {
        run_id: _run_name(selected[run_id])
        for run_id, path in paths.items()
        if any(path.rglob("events.out.tfevents.*"))
    }
    counts = Counter(names.values())
    links = {name: run_id for run_id, name in names.items() if counts[name] == 1}
    for run_id, name in names.items():
        if counts[name] > 1:
            distinct = _distinct_names(name, run_id)
            links[next(n for n in distinct if n not in links)] = run_id
    with TemporaryDirectory(prefix="runsnap-tb-") as tmp:
        logdir = Path(tmp)
        for name, run_id in links.items():
            (logdir / name).symlink_to(paths[run_id], target_is_directory=True)
        yield logdir


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
            warnings.warn(f"runsnap could not fetch run {run_id}: {exc}", stacklevel=4)
    return fetched


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
