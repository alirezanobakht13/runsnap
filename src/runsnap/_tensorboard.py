"""A TensorBoard summary writer that keeps scalars apart from heavy media."""

import shutil
import tempfile
import threading
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient
from tensorboardX import SummaryWriter

from runsnap._tags import (
    TB_ARTIFACT_DIR,
    TB_HISTOGRAM_BINS,
    TB_LIGHT_SUFFIX,
    TB_SHARD_MAX_BYTES,
    TB_SYNC_INTERVAL_SECONDS,
    TB_TAG_LOGDIR,
    tb_media_suffix,
)

EVENT_GLOB = "events.out.tfevents.*"
"""The names the underlying writer gives its event files."""

LIGHT_METHODS = frozenset({"add_scalar", "add_text"})
"""Methods written to the light event file rather than the media shards."""


class TensorBoardWriter:
    """Two summary writers over one log directory, joined by delegation.

    `add_scalar` and `add_text` go to a light event file; every other method of
    the summary-writer surface goes to a media file that rolls into a new shard
    once it passes `shard_max_bytes`. TensorBoard merges every event file in a
    directory into one run, so the split is invisible when viewing and lets a
    run's curves be downloaded without its images.
    """

    def __init__(
        self,
        logdir: str | Path,
        *,
        shard_max_bytes: int = TB_SHARD_MAX_BYTES,
        **kwargs: Any,
    ) -> None:
        self.logdir = str(logdir)
        self._shard_max_bytes = shard_max_bytes
        self._writer_kwargs = kwargs
        self._lock = threading.Lock()
        self._shard = 0
        self._light = SummaryWriter(
            logdir=self.logdir, filename_suffix=TB_LIGHT_SUFFIX, **kwargs
        )
        self._media = self._open_media()

    @property
    def light_path(self) -> Path:
        """The event file holding scalars and text."""
        with self._lock:
            return _event_path(self._light)

    @property
    def media_path(self) -> Path:
        """The media shard currently open for writing."""
        with self._lock:
            return _event_path(self._media)

    def add_histogram(
        self,
        tag: str,
        values: Any,
        global_step: int | None = None,
        bins: Any = TB_HISTOGRAM_BINS,
        **kwargs: Any,
    ) -> Any:
        """Log a histogram, binned compactly rather than into ~775 buckets.

        `bins` takes anything the underlying writer takes, including its own
        `"tensorflow"` default, and is passed through unchanged.
        """
        return self._call_media(
            "add_histogram", tag, values, global_step, bins, **kwargs
        )

    def flush(self) -> None:
        """Push both writers' pending events to disk."""
        with self._lock:
            self._light.flush()
            self._media.flush()

    def close(self) -> None:
        """Close both writers, sealing the open media shard."""
        with self._lock:
            self._light.close()
            self._media.close()

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in LIGHT_METHODS:
            return partial(self._call_light, name)
        attribute = getattr(self._media, name)
        if not callable(attribute):
            return attribute
        return partial(self._call_media, name)

    def _call_light(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Delegate to the light writer, reporting a failure as a warning."""
        with self._lock, _warn_instead_of_raising(f"write {name}"):
            return getattr(self._light, name)(*args, **kwargs)

    def _call_media(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Delegate to the media writer, rolling the shard if the write filled it.

        The lock holds across the write and the roll so that no write lands in a
        file that is being closed.
        """
        with self._lock, _warn_instead_of_raising(f"write {name}"):
            result = getattr(self._media, name)(*args, **kwargs)
            self._roll_if_full()
            return result

    def _roll_if_full(self) -> None:
        with _warn_instead_of_raising("roll the media shard"):
            path = _event_path(self._media)
            if not path.exists() or path.stat().st_size <= self._shard_max_bytes:
                return
            self._media.close()
            self._shard += 1
            self._media = self._open_media()

    def _open_media(self) -> SummaryWriter:
        """A media writer whose filename carries the current shard index.

        The index has to be in the suffix: the underlying writer names files
        after whole seconds and the hostname, so two shards sealed within the
        same second would otherwise share a name.
        """
        return SummaryWriter(
            logdir=self.logdir,
            filename_suffix=tb_media_suffix(self._shard),
            **self._writer_kwargs,
        )


def _event_path(writer: SummaryWriter) -> Path:
    """The file on disk that `writer` is appending events to.

    The underlying writer settles on the name in its own constructor and offers
    no accessor for it, so the name is read off the writer it belongs to.
    """
    event_writer = getattr(writer.file_writer, "event_writer", None)
    if event_writer is None:
        raise RuntimeError("the summary writer has no open event file")
    return Path(event_writer._ev_writer._file_name)


@contextmanager
def _warn_instead_of_raising(action: str) -> Iterator[None]:
    """Report a failure to `action` as a warning rather than letting it out.

    TensorBoard data is a side record of a run; nothing about writing it is
    worth failing the training loop that produced it.
    """
    try:
        yield
    except Exception as exc:  # noqa: BLE001 - logging must never fail the run
        warnings.warn(f"runsnap could not {action}: {exc}", stacklevel=3)


class _Sync:
    """Uploads a log directory's event files to a run's `tb/` artifacts.

    A file is uploaded whenever its size differs from the size last uploaded,
    which gives the whole policy at once: a sealed shard is uploaded exactly
    once because its size never changes again, the light file is re-uploaded
    as it grows, and the open shard is skipped until it seals.
    """

    def __init__(
        self,
        client: MlflowClient,
        run_id: str,
        writer: TensorBoardWriter,
        *,
        interval: float = TB_SYNC_INTERVAL_SECONDS,
    ) -> None:
        self._client = client
        self._run_id = run_id
        self._writer = writer
        self._interval = interval
        self._uploaded: dict[str, int] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, name="runsnap-tb-sync", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Stop the thread and wait for an upload in flight to finish."""
        self._stop.set()
        self._thread.join()

    def sync(self, *, skip_open_shard: bool = True) -> None:
        """Upload every event file that has grown since it was last uploaded."""
        open_shard = self._writer.media_path if skip_open_shard else None
        logdir = Path(self._writer.logdir)
        for path in sorted(logdir.rglob(EVENT_GLOB)):
            if path == open_shard:
                continue
            with _warn_instead_of_raising(f"upload {path.name}"):
                self._upload(logdir, path)

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            self.sync()

    def _upload(self, logdir: Path, path: Path) -> None:
        size = path.stat().st_size
        key = path.relative_to(logdir).as_posix()
        if self._uploaded.get(key) == size:
            return
        self._client.log_artifact(
            self._run_id, str(path), artifact_path=_artifact_path(logdir, path)
        )
        self._uploaded[key] = size


def _artifact_path(logdir: Path, path: Path) -> str:
    """Where under `tb/` an event file belongs.

    Methods like `add_scalars` write into subdirectories of the log directory,
    which TensorBoard reads as nested runs; the subdirectory is preserved so
    they survive the trip through the artifact store.
    """
    parent = path.parent.relative_to(logdir)
    if parent == Path():
        return TB_ARTIFACT_DIR
    return f"{TB_ARTIFACT_DIR}/{parent.as_posix()}"


@contextmanager
def tensorboard(
    run_id: str | None = None,
    *,
    shard_max_bytes: int = TB_SHARD_MAX_BYTES,
    sync_interval: float = TB_SYNC_INTERVAL_SECONDS,
    **kwargs: Any,
) -> Iterator[TensorBoardWriter]:
    """A TensorBoard writer whose event files land in a run's MLflow artifacts.

    Writes go to a scratch directory that a background thread mirrors into the
    run's `tb/` artifacts. Leaving the block seals the open media shard,
    uploads everything outstanding, and closes the writer, whether the block
    ended normally, by an exception, or by `KeyboardInterrupt`.

    `run_id` defaults to the active MLflow run. Extra keyword arguments go to
    the underlying summary writers.
    """
    resolved = run_id or _active_run_id()
    client = MlflowClient()
    logdir = tempfile.mkdtemp(prefix="runsnap-tb-")
    writer = TensorBoardWriter(logdir, shard_max_bytes=shard_max_bytes, **kwargs)
    sync = _Sync(client, resolved, writer, interval=sync_interval)
    with _warn_instead_of_raising(f"tag the run with {TB_TAG_LOGDIR}"):
        client.set_tag(resolved, TB_TAG_LOGDIR, TB_ARTIFACT_DIR)
    sync.start()
    try:
        yield writer
    finally:
        sync.stop()
        with _warn_instead_of_raising("close the TensorBoard writer"):
            writer.close()
        # The writer is closed, so there is no open shard left to hold back.
        sync.sync(skip_open_shard=False)
        shutil.rmtree(logdir, ignore_errors=True)


def _active_run_id() -> str:
    run = mlflow.active_run()
    if run is None:
        raise RuntimeError(
            "runsnap.tensorboard() needs an active MLflow run; "
            "start one with runsnap.start_run() or pass run_id="
        )
    return run.info.run_id
