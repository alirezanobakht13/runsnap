"""Fetch TensorBoard artifacts into a persistent cache for local viewing."""

import os
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from mlflow.entities import FileInfo, Run
from mlflow.tracking import MlflowClient

from runsnap._tags import TB_ARTIFACT_DIR, TB_LIGHT_SUFFIX


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
        is_light = relative.name.endswith(TB_LIGHT_SUFFIX)
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
            if not link.is_symlink():
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

    Keep this context open while TensorBoard runs. Its links are removed on
    exit, while downloaded event files remain available for later invocations.
    Runs without event data are omitted.
    """
    selected = {run.info.run_id: run for run in runs}
    cached = {
        run_id: fetch_run(client, run_id, cache_dir=cache_dir, media=media)
        for run_id in selected
    }
    names = {
        run_id: _run_name(selected[run_id])
        for run_id, path in cached.items()
        if any(path.rglob("events.out.tfevents.*"))
    }
    counts = Counter(names.values())
    reserved = set(names.values())
    with TemporaryDirectory(prefix="runsnap-tb-") as tmp:
        logdir = Path(tmp)
        for run_id, name in names.items():
            if counts[name] > 1:
                base = name
                name = f"{base}-{run_id[:8]}"
                counter = 0
                while name in reserved:
                    counter += 1
                    name = f"{base}-{run_id}-{counter}"
                reserved.add(name)
            (logdir / name).symlink_to(cached[run_id], target_is_directory=True)
        yield logdir


def _run_name(run: Run) -> str:
    name = run.info.run_name or run.data.tags.get("mlflow.runName")
    if not name:
        return run.info.run_id
    name = re.sub(r"[/\\\x00]", "_", name)
    return run.info.run_id if name in {".", ".."} else name
