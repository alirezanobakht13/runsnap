"""A TensorBoard summary writer that keeps scalars apart from heavy media."""

import shutil
import socket
import tempfile
import threading
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient
from tensorboardX import SummaryWriter

from runsnap._metrics import flatten_metrics
from runsnap._tags import (
    TB_ARTIFACT_DIR,
    TB_FLUSH_SECONDS,
    TB_HISTOGRAM_BINS,
    TB_LIGHT_SHARD_MAX_BYTES,
    TB_SHARD_MAX_BYTES,
    TB_SYNC_INTERVAL_SECONDS,
    TB_TAG_LOCAL_DIR,
    TB_TAG_LOCAL_HOST,
    TB_TAG_LOGDIR,
    tb_light_suffix,
    tb_media_suffix,
)

EVENT_GLOB = "events.out.tfevents.*"
"""The names the underlying writer gives its event files."""

LIGHT_METHODS = frozenset({"add_scalar", "add_text"})
"""Methods written to the light stream rather than the media stream."""


class _Stream:
    """One writer's event files, rolling into a new shard at `max_bytes`.

    `suffix` names a shard by its index, and the index has to be in the
    filename suffix: the underlying writer names files after whole seconds and
    the hostname, so two shards sealed within the same second would otherwise
    share a name.
    """

    def __init__(
        self,
        logdir: str,
        name: str,
        suffix: Callable[[int], str],
        max_bytes: int,
        writer_kwargs: dict[str, Any],
    ) -> None:
        self._logdir = logdir
        self._name = name
        self._suffix = suffix
        self._max_bytes = max_bytes
        self._writer_kwargs = writer_kwargs
        self._shard = 0
        self.writer = self._open()

    @property
    def path(self) -> Path:
        """The shard currently open for writing."""
        return _event_path(self.writer, self._suffix(self._shard))

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Write through the open shard, rolling it if the write filled it."""
        result = getattr(self.writer, name)(*args, **kwargs)
        self._roll_if_full()
        return result

    def _roll_if_full(self) -> None:
        with _warn_instead_of_raising(f"roll the {self._name} shard"):
            path = self.path
            if not path.exists() or path.stat().st_size <= self._max_bytes:
                return
            self.writer.close()
            self._shard += 1
            self.writer = self._open()

    def _open(self) -> SummaryWriter:
        return SummaryWriter(
            logdir=self._logdir,
            filename_suffix=self._suffix(self._shard),
            **self._writer_kwargs,
        )


class TensorBoardWriter:
    """Two streams of event files over one log directory, joined by delegation.

    `add_scalar` and `add_text` go to a light stream; every other method of the
    summary-writer surface goes to a media stream. Each stream rolls into a new
    shard once its open file passes that stream's own threshold, media at
    `shard_max_bytes` and scalars at `light_shard_max_bytes`, which bounds what
    a sync pass has to re-upload. TensorBoard merges every event file in a
    directory into one run, so neither the split nor the sharding is visible
    when viewing, and a run's curves can be downloaded without its images.
    """

    def __init__(
        self,
        logdir: str | Path,
        *,
        shard_max_bytes: int = TB_SHARD_MAX_BYTES,
        light_shard_max_bytes: int = TB_LIGHT_SHARD_MAX_BYTES,
        **kwargs: Any,
    ) -> None:
        self.logdir = str(logdir)
        self._lock = threading.Lock()
        self._light = _Stream(
            self.logdir, "scalar", tb_light_suffix, light_shard_max_bytes, kwargs
        )
        self._media = _Stream(
            self.logdir, "media", tb_media_suffix, shard_max_bytes, kwargs
        )

    @property
    def light_path(self) -> Path:
        """The scalar shard currently open for writing."""
        with self._lock:
            return self._light.path

    @property
    def media_path(self) -> Path:
        """The media shard currently open for writing."""
        with self._lock:
            return self._media.path

    def add_record(
        self,
        prefix: str,
        record: Any,
        global_step: int | None = None,
        walltime: float | None = None,
    ) -> None:
        """Chart every number in a record, one scalar per leaf.

        `record` is flattened by `runsnap.flatten_metrics()`, so a mapping, a
        dataclass instance or a model all work and a nested record reads
        `actor.entropy`; non-numeric fields are left out. Each entry is charted
        as `prefix/<key>`, or `<key>` when `prefix` is empty.

        Raises:
            ValueError: A leaf holds more than one element, named by its key.
        """
        for key, value in flatten_metrics(record).items():
            tag = f"{prefix}/{key}" if prefix else key
            self._call_light("add_scalar", tag, value, global_step, walltime)

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
        """Push both streams' pending events to disk."""
        with self._lock:
            self._light.writer.flush()
            self._media.writer.flush()

    def close(self) -> None:
        """Close both streams, sealing the shard each has open."""
        with self._lock:
            self._light.writer.close()
            self._media.writer.close()

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in LIGHT_METHODS:
            return partial(self._call_light, name)
        attribute = getattr(self._media.writer, name)
        if not callable(attribute):
            return attribute
        return partial(self._call_media, name)

    def _call_light(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Delegate to the light stream, reporting a failure as a warning.

        The lock holds across the write and the roll so that no write lands in
        a file that is being closed.
        """
        with self._lock, _warn_instead_of_raising(f"write {name}"):
            return self._light.call(name, *args, **kwargs)

    def _call_media(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Delegate to the media stream, reporting a failure as a warning.

        The lock holds across the write and the roll so that no write lands in
        a file that is being closed.
        """
        with self._lock, _warn_instead_of_raising(f"write {name}"):
            return self._media.call(name, *args, **kwargs)


def _event_path(writer: SummaryWriter, suffix: str) -> Path:
    """The file on disk that `writer` is appending events to.

    The underlying writer settles on the name in its own constructor and offers
    no accessor for it, but the name ends in `suffix`, the `filename_suffix`
    the writer was constructed with, which no other writer over the directory
    shares.
    """
    paths = list(Path(writer.logdir).glob(f"{EVENT_GLOB}{suffix}"))
    if len(paths) != 1:
        raise RuntimeError(
            f"expected one event file ending in {suffix!r}, found {len(paths)}"
        )
    return paths[0]


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
    which gives most of the policy at once: a sealed shard is uploaded exactly
    once because its size never changes again, and an unchanged directory
    uploads nothing. The one file held back is the open media shard, which is
    skipped until it seals; the open scalar shard is uploaded as it grows, so a
    dashboard watching the run keeps seeing the newest curves.
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

    def sync(self, *, skip_open_media: bool = True) -> None:
        """Upload every event file that has grown since it was last uploaded."""
        open_media = self._writer.media_path if skip_open_media else None
        logdir = Path(self._writer.logdir)
        for path in sorted(logdir.rglob(EVENT_GLOB)):
            if path == open_media:
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
    light_shard_max_bytes: int = TB_LIGHT_SHARD_MAX_BYTES,
    sync_interval: float = TB_SYNC_INTERVAL_SECONDS,
    **kwargs: Any,
) -> Iterator[TensorBoardWriter]:
    """A TensorBoard writer whose event files land in a run's MLflow artifacts.

    Writes go to a scratch directory that a background thread mirrors into the
    run's `tb/` artifacts. Leaving the block seals the shard each stream has
    open, uploads everything outstanding, and closes the writer, whether the
    block ended normally, by an exception, or by `KeyboardInterrupt`.

    `run_id` defaults to the active MLflow run. Extra keyword arguments go to
    the underlying summary writers, which flush to disk every
    `TB_FLUSH_SECONDS` unless the caller passes its own `flush_secs`.
    """
    resolved = run_id or _active_run_id()
    client = MlflowClient()
    logdir = tempfile.mkdtemp(prefix="runsnap-tb-")
    kwargs.setdefault("flush_secs", TB_FLUSH_SECONDS)
    writer = TensorBoardWriter(
        logdir,
        shard_max_bytes=shard_max_bytes,
        light_shard_max_bytes=light_shard_max_bytes,
        **kwargs,
    )
    sync = _Sync(client, resolved, writer, interval=sync_interval)
    with _warn_instead_of_raising("tag the run with its TensorBoard location"):
        client.set_tag(resolved, TB_TAG_LOGDIR, TB_ARTIFACT_DIR)
        client.set_tag(resolved, TB_TAG_LOCAL_HOST, socket.gethostname())
        client.set_tag(resolved, TB_TAG_LOCAL_DIR, logdir)
    sync.start()
    try:
        yield writer
    finally:
        sync.stop()
        with _warn_instead_of_raising("close the TensorBoard writer"):
            writer.close()
        # The writer is closed, so there is no open shard left to hold back.
        sync.sync(skip_open_media=False)
        shutil.rmtree(logdir, ignore_errors=True)


def _active_run_id() -> str:
    run = mlflow.active_run()
    if run is None:
        raise RuntimeError(
            "runsnap.tensorboard() needs an active MLflow run; "
            "start one with runsnap.start_run() or pass run_id="
        )
    return run.info.run_id
