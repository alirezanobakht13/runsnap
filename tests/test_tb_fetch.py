"""Artifact cache and named TensorBoard log directory tests."""

import shutil
import socket
import threading
import time
import warnings
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

import mlflow
import pytest
from mlflow.entities import FileInfo, Run
from mlflow.tracking import MlflowClient
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import runsnap
from runsnap._tags import TAG_CONTINUES, TB_TAG_LOCAL_DIR, TB_TAG_LOCAL_HOST
from runsnap._tb_fetch import _run_statuses, assemble_logdir, fetch_run
from runsnap._tensorboard import TensorBoardWriter

LIGHT = "events.out.tfevents.1.host.scalars"
"""A scalar file named the way runs were written before sharding."""

LIGHT_SHARDS = [
    "events.out.tfevents.2.host.scalars.0",
    "events.out.tfevents.3.host.scalars.1",
]
MEDIA = "events.out.tfevents.1.host.media.0"


def logged_run(client: MlflowClient, tmp_path: Path) -> Run:
    with mlflow.start_run() as run:
        source = tmp_path / "source"
        source.mkdir(exist_ok=True)
        (source / LIGHT).write_bytes(b"scalar events")
        (source / MEDIA).write_bytes(b"media events")
        nested = source / "nested"
        nested.mkdir(exist_ok=True)
        (nested / MEDIA).write_bytes(b"nested events")
        (source / "ignored.txt").write_text("not an event file")
        client.log_artifacts(run.info.run_id, str(source), "tb")
        return client.get_run(run.info.run_id)


def test_second_fetch_downloads_nothing(tracking, tmp_path: Path):
    run = logged_run(tracking, tmp_path)
    with patch.object(
        tracking, "download_artifacts", wraps=tracking.download_artifacts
    ) as download:
        logdir = fetch_run(
            tracking, run.info.run_id, cache_dir=tmp_path / "cache", media=True
        )
        assert download.call_count == 3
        download.reset_mock()
        assert (
            fetch_run(
                tracking, run.info.run_id, cache_dir=tmp_path / "cache", media=True
            )
            == logdir
        )
        download.assert_not_called()
    assert run.info.run_id in logdir.parts
    assert (logdir / LIGHT).read_bytes() == b"scalar events"
    assert (logdir / "nested" / MEDIA).read_bytes() == b"nested events"
    assert not (logdir / "ignored.txt").exists()


def test_sharded_scalars_are_all_fetched_without_media(tracking, tmp_path: Path):
    with mlflow.start_run() as active:
        source = tmp_path / "sharded"
        source.mkdir()
        for shard in LIGHT_SHARDS:
            (source / shard).write_bytes(b"scalar events")
        (source / MEDIA).write_bytes(b"media events")
        tracking.log_artifacts(active.info.run_id, str(source), "tb")
        run_id = active.info.run_id

    light = fetch_run(tracking, run_id, cache_dir=tmp_path / "cache")
    assert {p.name for p in light.rglob("events.*")} == set(LIGHT_SHARDS)


def test_both_scalar_shapes_in_one_directory_are_fetched(tracking, tmp_path: Path):
    run = logged_run(tracking, tmp_path)
    source = tmp_path / "source"
    for shard in LIGHT_SHARDS:
        (source / shard).write_bytes(b"more scalar events")
    tracking.log_artifacts(run.info.run_id, str(source), "tb")

    light = fetch_run(tracking, run.info.run_id, cache_dir=tmp_path / "cache")
    assert {p.name for p in light.rglob("events.*")} == {LIGHT, *LIGHT_SHARDS}


def test_only_changed_and_new_files_are_downloaded(tracking, tmp_path: Path):
    run = logged_run(tracking, tmp_path)
    cache = tmp_path / "cache"
    logdir = fetch_run(tracking, run.info.run_id, cache_dir=cache, media=True)
    source = tmp_path / "source"
    (source / LIGHT).write_bytes(b"scalar events with another step")
    new_shard = MEDIA.removesuffix("0") + "1"
    (source / new_shard).write_bytes(b"new shard")
    tracking.log_artifacts(run.info.run_id, str(source), "tb")
    with patch.object(
        tracking, "download_artifacts", wraps=tracking.download_artifacts
    ) as download:
        fetch_run(tracking, run.info.run_id, cache_dir=cache, media=True)
    assert {call.args[1] for call in download.call_args_list} == {
        f"tb/{LIGHT}",
        f"tb/{new_shard}",
    }
    assert (logdir / LIGHT).read_bytes() == (source / LIGHT).read_bytes()


def test_partial_cache_file_is_replaced(tracking, tmp_path: Path):
    run = logged_run(tracking, tmp_path)
    cache = tmp_path / "cache"
    logdir = fetch_run(tracking, run.info.run_id, cache_dir=cache)
    (logdir / LIGHT).write_bytes(b"partial")
    with patch.object(
        tracking, "download_artifacts", wraps=tracking.download_artifacts
    ) as download:
        fetch_run(tracking, run.info.run_id, cache_dir=cache)
    assert [call.args[1] for call in download.call_args_list] == [f"tb/{LIGHT}"]
    assert (logdir / LIGHT).read_bytes() == b"scalar events"


def test_interrupted_download_preserves_previous_file(tracking, tmp_path: Path):
    run = logged_run(tracking, tmp_path)
    cache = tmp_path / "cache"
    logdir = fetch_run(tracking, run.info.run_id, cache_dir=cache)
    source = tmp_path / "source" / LIGHT
    source.write_bytes(b"scalar events with another step")
    tracking.log_artifact(run.info.run_id, str(source), "tb")

    def interrupted(run_id: str, artifact: str, destination: str) -> str:
        (Path(destination) / "partial").write_bytes(b"partial")
        assert (logdir / LIGHT).read_bytes() == b"scalar events"
        raise KeyboardInterrupt

    with (
        patch.object(tracking, "download_artifacts", side_effect=interrupted),
        pytest.raises(KeyboardInterrupt),
    ):
        fetch_run(tracking, run.info.run_id, cache_dir=cache)
    assert (logdir / LIGHT).read_bytes() == b"scalar events"
    assert not list(logdir.parent.glob(".download-*"))
    fetch_run(tracking, run.info.run_id, cache_dir=cache)
    assert (logdir / LIGHT).read_bytes() == source.read_bytes()


def log_light(client: MlflowClient, run: Run, tmp_path: Path, content: bytes) -> None:
    source = tmp_path / "source" / LIGHT
    source.write_bytes(content)
    client.log_artifact(run.info.run_id, str(source), "tb")


def test_grown_remote_file_extends_the_cached_file_in_place(tracking, tmp_path: Path):
    run = logged_run(tracking, tmp_path)
    cache = tmp_path / "cache"
    logdir = fetch_run(tracking, run.info.run_id, cache_dir=cache)
    inode = (logdir / LIGHT).stat().st_ino
    log_light(tracking, run, tmp_path, b"scalar events with another step")
    fetch_run(tracking, run.info.run_id, cache_dir=cache)
    assert (logdir / LIGHT).stat().st_ino == inode
    assert (logdir / LIGHT).read_bytes() == b"scalar events with another step"


def test_shorter_remote_file_leaves_the_cached_file_alone(tracking, tmp_path: Path):
    run = logged_run(tracking, tmp_path)
    cache = tmp_path / "cache"
    logdir = fetch_run(tracking, run.info.run_id, cache_dir=cache)
    log_light(tracking, run, tmp_path, b"scalar")
    with patch.object(
        tracking, "download_artifacts", wraps=tracking.download_artifacts
    ) as download:
        fetch_run(tracking, run.info.run_id, cache_dir=cache)
    download.assert_not_called()
    assert (logdir / LIGHT).read_bytes() == b"scalar events"


def test_remote_file_not_extending_the_cached_one_replaces_it(tracking, tmp_path: Path):
    run = logged_run(tracking, tmp_path)
    cache = tmp_path / "cache"
    logdir = fetch_run(tracking, run.info.run_id, cache_dir=cache)
    log_light(tracking, run, tmp_path, b"rewritten scalar events")
    fetch_run(tracking, run.info.run_id, cache_dir=cache)
    assert (logdir / LIGHT).read_bytes() == b"rewritten scalar events"


def test_cache_file_cut_to_a_prefix_is_completed_by_the_next_fetch(
    tracking, tmp_path: Path
):
    run = logged_run(tracking, tmp_path)
    cache = tmp_path / "cache"
    logdir = fetch_run(tracking, run.info.run_id, cache_dir=cache)
    inode = (logdir / LIGHT).stat().st_ino
    # An interrupted append leaves a prefix of the remote file in place.
    (logdir / LIGHT).write_bytes(b"scalar")
    fetch_run(tracking, run.info.run_id, cache_dir=cache)
    assert (logdir / LIGHT).stat().st_ino == inode
    assert (logdir / LIGHT).read_bytes() == b"scalar events"


def test_stale_link_is_repointed_at_the_cached_file(tracking, tmp_path: Path):
    run = logged_run(tracking, tmp_path)
    cache = tmp_path / "cache"
    link = fetch_run(tracking, run.info.run_id, cache_dir=cache) / LIGHT
    target = link.readlink()
    link.unlink()
    link.symlink_to(tmp_path / "moved" / LIGHT)
    fetch_run(tracking, run.info.run_id, cache_dir=cache)
    assert link.readlink() == target
    assert link.read_bytes() == b"scalar events"


def test_regular_file_in_the_link_path_is_replaced(tracking, tmp_path: Path):
    run = logged_run(tracking, tmp_path)
    cache = tmp_path / "cache"
    link = fetch_run(tracking, run.info.run_id, cache_dir=cache) / LIGHT
    target = link.readlink()
    link.unlink()
    link.write_bytes(b"not a link")
    fetch_run(tracking, run.info.run_id, cache_dir=cache)
    assert link.is_symlink()
    assert link.readlink() == target
    assert link.read_bytes() == b"scalar events"


def test_default_cache_respects_xdg_cache_home(tracking, tmp_path, monkeypatch):
    run = logged_run(tracking, tmp_path)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    logdir = fetch_run(tracking, run.info.run_id)
    assert logdir == (
        tmp_path / "xdg" / "runsnap" / "tensorboard" / run.info.run_id / "scalars"
    )


def named_run(client: MlflowClient, tmp_path: Path, name: str) -> Run:
    run = logged_run(client, tmp_path)
    client.set_tag(run.info.run_id, "mlflow.runName", name)
    return client.get_run(run.info.run_id)


def test_assembly_names_duplicates_and_unnamed_runs(tracking, tmp_path, monkeypatch):
    first = named_run(tracking, tmp_path, "baseline")
    second = named_run(tracking, tmp_path, "baseline")
    unnamed = named_run(tracking, tmp_path, "")
    # This MLflow store generates names even when given an empty string.
    monkeypatch.setattr(unnamed.info, "_run_name", None)
    unnamed.data.tags.pop("mlflow.runName", None)
    unique = named_run(tracking, tmp_path, "baseline-lr0.001")
    runs = [first, second, unnamed, unique, first]
    with assemble_logdir(tracking, runs, cache_dir=tmp_path / "cache") as logdir:
        links = list(logdir.iterdir())
        assert {p.name for p in links} == {
            f"baseline-{first.info.run_id[:8]}",
            f"baseline-{second.info.run_id[:8]}",
            unnamed.info.run_id,
            "baseline-lr0.001",
        }
        assert all(p.is_symlink() for p in links)
        targets = [p.resolve() for p in links]
        assert all((p / LIGHT).read_bytes() == b"scalar events" for p in links)
    assert not logdir.exists()
    assert all(p.is_dir() for p in targets)


def test_assembly_handles_names_that_collide_with_generated_suffixes(
    tracking, tmp_path: Path
):
    first = named_run(tracking, tmp_path, "baseline")
    second = named_run(tracking, tmp_path, "baseline")
    literal_name = f"baseline-{first.info.run_id[:8]}"
    third = named_run(tracking, tmp_path, literal_name)
    with assemble_logdir(
        tracking, [first, second, third], cache_dir=tmp_path / "cache"
    ) as logdir:
        assert {p.name for p in logdir.iterdir()} == {
            f"baseline-{first.info.run_id}",
            f"baseline-{second.info.run_id[:8]}",
            literal_name,
        }
        assert third.info.run_id in (logdir / literal_name).resolve().parts


@pytest.mark.parametrize("name", ["../baseline", "group/baseline", ".."])
def test_run_names_stay_inside_assembled_directory(tracking, tmp_path, name):
    run = named_run(tracking, tmp_path, name)
    with assemble_logdir(tracking, [run], cache_dir=tmp_path / "cache") as logdir:
        links = list(logdir.iterdir())
        assert len(links) == 1
        assert links[0].is_symlink()
        assert (links[0] / LIGHT).exists()


def test_selected_runs_are_fetched_together_whatever_the_finish_order(
    tracking, tmp_path: Path
):
    names = ["first", "second", "third"]
    runs = [named_run(tracking, tmp_path, name) for name in names]
    position = {run.info.run_id: index for index, run in enumerate(runs)}
    download = tracking.download_artifacts
    started = threading.Barrier(len(runs))

    def paced(run_id: str, artifact: str, destination: str) -> str:
        """Hold every download until they are all running, then invert them.

        A sequential fetch never fills the barrier, so a download that returns
        at all is one that overlapped the others. The sleep that follows makes
        the runs finish in the reverse of the order they were selected in.
        """
        started.wait(timeout=30)
        time.sleep(0.05 * (len(runs) - position[run_id]))
        return download(run_id, artifact, destination)

    with (
        patch.object(tracking, "download_artifacts", side_effect=paced),
        assemble_logdir(tracking, runs, cache_dir=tmp_path / "cache") as logdir,
    ):
        assert {p.name for p in logdir.iterdir()} == set(names)
        for name, run in zip(names, runs, strict=True):
            link = logdir / name
            assert run.info.run_id in link.resolve().parts
            assert (link / LIGHT).read_bytes() == b"scalar events"


def test_run_that_cannot_be_fetched_is_dropped_with_a_warning(tracking, tmp_path: Path):
    reachable = named_run(tracking, tmp_path, "reachable")
    unreachable = named_run(tracking, tmp_path, "unreachable")
    other = named_run(tracking, tmp_path, "other")
    download = tracking.download_artifacts

    def refuse_one(run_id: str, artifact: str, destination: str) -> str:
        if run_id == unreachable.info.run_id:
            raise OSError("the artifact store is unreachable")
        return download(run_id, artifact, destination)

    with (
        patch.object(tracking, "download_artifacts", side_effect=refuse_one),
        pytest.warns(UserWarning, match=unreachable.info.run_id),
        assemble_logdir(
            tracking, [reachable, unreachable, other], cache_dir=tmp_path / "cache"
        ) as logdir,
    ):
        events = {p.name: (p / LIGHT).read_bytes() for p in logdir.iterdir()}

    assert events == {"reachable": b"scalar events", "other": b"scalar events"}


def test_runs_without_events_are_omitted(tracking, tmp_path: Path):
    with mlflow.start_run() as run:
        empty = tracking.get_run(run.info.run_id)
    with assemble_logdir(tracking, [empty], cache_dir=tmp_path / "cache") as logdir:
        assert list(logdir.iterdir()) == []


def test_default_fetches_only_light_and_media_opt_in_reuses_it(tracking, tmp_path):
    run = logged_run(tracking, tmp_path)
    cache = tmp_path / "cache"
    with patch.object(
        tracking, "download_artifacts", wraps=tracking.download_artifacts
    ) as download:
        light = fetch_run(tracking, run.info.run_id, cache_dir=cache)
        assert [call.args[1] for call in download.call_args_list] == [f"tb/{LIGHT}"]
        assert {p.name for p in light.rglob("events.*")} == {LIGHT}
        download.reset_mock()
        full = fetch_run(tracking, run.info.run_id, cache_dir=cache, media=True)
        assert {call.args[1] for call in download.call_args_list} == {
            f"tb/{MEDIA}",
            f"tb/nested/{MEDIA}",
        }
        assert (full / LIGHT).read_bytes() == b"scalar events"
        assert (full / MEDIA).read_bytes() == b"media events"
        download.reset_mock()
        assert fetch_run(tracking, run.info.run_id, cache_dir=cache) == light
        download.assert_not_called()
        assert {p.name for p in light.rglob("events.*")} == {LIGHT}


@pytest.mark.parametrize("media", [False, True])
def test_tensorboard_reads_selected_data_through_assembled_links(
    tracking, tmp_path: Path, media: bool
):
    source = tmp_path / "events"
    writer = TensorBoardWriter(source)
    writer.add_scalar("loss", 0.5, 1)
    writer.add_histogram("weights", [1.0, 2.0, 3.0], 1)
    writer.close()
    with mlflow.start_run(run_name="baseline") as active:
        tracking.log_artifacts(active.info.run_id, str(source), "tb")
        run = tracking.get_run(active.info.run_id)
    # Prepopulate media to exercise a later default view of the same cache.
    cache = tmp_path / "cache"
    fetch_run(tracking, run.info.run_id, cache_dir=cache, media=True)
    with assemble_logdir(tracking, [run], cache_dir=cache, media=media) as logdir:
        accumulator = EventAccumulator(str(logdir / "baseline"))
        accumulator.Reload()
        assert [point.value for point in accumulator.Scalars("loss")] == [0.5]
        assert accumulator.Tags()["histograms"] == (["weights"] if media else [])


def live_run(client: MlflowClient, tmp_path: Path, name: str, host: str) -> Run:
    """A run tagged as writing into a local directory of its own on `host`."""
    run = named_run(client, tmp_path, name)
    local = tmp_path / f"local-{name}"
    local.mkdir()
    (local / LIGHT).write_bytes(b"live scalar events")
    client.set_tag(run.info.run_id, TB_TAG_LOCAL_HOST, host)
    client.set_tag(run.info.run_id, TB_TAG_LOCAL_DIR, str(local))
    return client.get_run(run.info.run_id)


def test_run_live_on_this_host_is_linked_without_downloading(tracking, tmp_path: Path):
    run = live_run(tracking, tmp_path, "training", socket.gethostname())
    with (
        patch.object(tracking, "download_artifacts") as download,
        assemble_logdir(tracking, [run], cache_dir=tmp_path / "cache") as logdir,
    ):
        link = logdir / "training"
        assert link.resolve() == (tmp_path / "local-training").resolve()
        assert (link / LIGHT).read_bytes() == b"live scalar events"
    download.assert_not_called()


def test_finished_run_falls_back_to_artifacts(tracking, tmp_path: Path):
    run = live_run(tracking, tmp_path, "training", socket.gethostname())
    shutil.rmtree(tmp_path / "local-training")
    with assemble_logdir(tracking, [run], cache_dir=tmp_path / "cache") as logdir:
        link = logdir / "training"
        assert (tmp_path / "cache") in link.resolve().parents
        assert (link / LIGHT).read_bytes() == b"scalar events"


def test_run_logged_on_another_host_falls_back_to_artifacts(tracking, tmp_path: Path):
    run = live_run(tracking, tmp_path, "training", "elsewhere")
    assert (tmp_path / "local-training").is_dir()
    with assemble_logdir(tracking, [run], cache_dir=tmp_path / "cache") as logdir:
        link = logdir / "training"
        assert (tmp_path / "cache") in link.resolve().parents
        assert (link / LIGHT).read_bytes() == b"scalar events"


def wait_for_steps(logdir: Path, tag: str, wanted: list[int]) -> list[int]:
    """The steps of `tag` under `logdir` once `wanted` has reached disk.

    Reading is retried because the writer hands events to a worker thread, so
    a scalar lands in the file some time after the call that wrote it returns.
    """
    deadline = time.monotonic() + 10.0
    while True:
        accumulator = EventAccumulator(str(logdir))
        accumulator.Reload()
        steps = (
            [point.step for point in accumulator.Scalars(tag)]
            if tag in accumulator.Tags()["scalars"]
            else []
        )
        if steps == wanted or time.monotonic() > deadline:
            return steps
        time.sleep(0.05)


def wait_until(condition: Callable[[], bool], timeout: float = 10.0) -> bool:
    """Whether `condition` came true within `timeout` seconds of polling."""
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


def test_assembled_logdir_follows_a_run_still_being_written(tracking, tmp_path: Path):
    with (
        runsnap.start_run(run_name="training", capture_code=False) as active,
        runsnap.tensorboard(sync_interval=60.0, flush_secs=0.05) as writer,
    ):
        writer.add_scalar("loss", 0.5, 1)
        run = tracking.get_run(active.info.run_id)
        with assemble_logdir(tracking, [run], cache_dir=tmp_path / "cache") as logdir:
            link = logdir / "training"
            assert link.resolve() == Path(writer.logdir).resolve()
            writer.add_scalar("loss", 0.25, 2)
            writer.flush()
            assert wait_for_steps(link, "loss", [1, 2]) == [1, 2]


def test_assembled_logdir_switches_to_the_cache_when_the_writer_exits(
    tracking, tmp_path: Path
):
    cache = tmp_path / "cache"
    with (
        runsnap.start_run(run_name="training", capture_code=False) as active,
        ExitStack() as writing,
    ):
        writer = writing.enter_context(
            runsnap.tensorboard(sync_interval=60.0, flush_secs=0.05)
        )
        writer.add_scalar("loss", 0.5, 1)
        writer.add_scalar("loss", 0.25, 2)
        run = tracking.get_run(active.info.run_id)
        with assemble_logdir(
            tracking, [run], cache_dir=cache, poll_interval=0.05
        ) as logdir:
            link = logdir / "training"
            assert link.resolve() == Path(writer.logdir).resolve()
            writing.close()
            assert not Path(writer.logdir).exists()
            assert wait_until(lambda: cache in link.resolve().parents)
            assert wait_for_steps(link, "loss", [1, 2]) == [1, 2]


def test_live_link_is_repointed_at_the_cache_once_its_directory_goes(
    tracking, tmp_path: Path
):
    run = live_run(tracking, tmp_path, "training", socket.gethostname())
    cache = tmp_path / "cache"
    with assemble_logdir(
        tracking, [run], cache_dir=cache, poll_interval=0.05
    ) as logdir:
        link = logdir / "training"
        assert link.resolve() == (tmp_path / "local-training").resolve()
        shutil.rmtree(tmp_path / "local-training")
        assert wait_until(lambda: cache in link.resolve().parents)
        assert (link / LIGHT).read_bytes() == b"scalar events"


def watch_threads() -> list[threading.Thread]:
    """The live-link watchers currently running, by their thread name."""
    return [t for t in threading.enumerate() if t.name == "runsnap-tb-watch"]


def wait_for_checks(check: Mock, count: int = 2) -> bool:
    """Whether `count` more checks of the shown runs began, so one ran in full."""
    wanted = check.call_count + count
    return wait_until(lambda: check.call_count >= wanted)


def test_context_of_cached_runs_watches_them_until_exit(tracking, tmp_path: Path):
    run = named_run(tracking, tmp_path, "finished")
    with (
        patch("runsnap._tb_fetch._run_statuses", wraps=_run_statuses) as check,
        assemble_logdir(
            tracking, [run], cache_dir=tmp_path / "cache", poll_interval=0.02
        ) as logdir,
    ):
        assert (logdir / "finished").is_symlink()
        (watch,) = watch_threads()
        assert wait_for_checks(check)
    assert not watch.is_alive()
    assert watch_threads() == []
    assert not logdir.exists()


def test_leaving_a_context_with_live_runs_joins_the_watch_thread(
    tracking, tmp_path: Path
):
    run = live_run(tracking, tmp_path, "training", socket.gethostname())
    with assemble_logdir(
        tracking, [run], cache_dir=tmp_path / "cache", poll_interval=60.0
    ) as logdir:
        (watch,) = watch_threads()
        assert watch.is_alive()
    assert not watch.is_alive()
    assert watch_threads() == []
    assert not logdir.exists()


def test_live_run_whose_fetch_fails_is_warned_once_and_left_alone(
    tracking, tmp_path: Path
):
    reachable = live_run(tracking, tmp_path, "reachable", socket.gethostname())
    unreachable = live_run(tracking, tmp_path, "unreachable", socket.gethostname())
    cache = tmp_path / "cache"
    download = tracking.download_artifacts

    def refuse_one(run_id: str, artifact: str, destination: str) -> str:
        if run_id == unreachable.info.run_id:
            raise OSError("the artifact store is unreachable")
        return download(run_id, artifact, destination)

    with (
        patch.object(tracking, "download_artifacts", side_effect=refuse_one),
        warnings.catch_warnings(record=True) as caught,
        assemble_logdir(
            tracking, [reachable, unreachable], cache_dir=cache, poll_interval=0.05
        ) as logdir,
    ):
        warnings.simplefilter("always")
        stale = logdir / "unreachable"
        good = logdir / "reachable"
        shutil.rmtree(tmp_path / "local-unreachable")
        shutil.rmtree(tmp_path / "local-reachable")
        assert wait_until(lambda: cache in good.resolve().parents)
        assert (good / LIGHT).read_bytes() == b"scalar events"
        # Several more polls would each re-warn if the run stayed watched.
        time.sleep(0.5)
        assert stale.is_symlink()
        assert stale.readlink() == tmp_path / "local-unreachable"
        assert not stale.exists()

    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 1
    assert unreachable.info.run_id in str(user_warnings[0].message)


def test_live_run_whose_directory_remains_is_never_fetched(tracking, tmp_path: Path):
    run = live_run(tracking, tmp_path, "training", socket.gethostname())
    with (
        patch.object(tracking, "download_artifacts") as download,
        assemble_logdir(
            tracking, [run], cache_dir=tmp_path / "cache", poll_interval=0.02
        ) as logdir,
    ):
        link = logdir / "training"
        time.sleep(0.3)  # a dozen or so polls
        assert link.resolve() == (tmp_path / "local-training").resolve()
        assert (link / LIGHT).read_bytes() == b"live scalar events"
    download.assert_not_called()


@pytest.mark.parametrize("media", [False, True])
def test_repointed_link_matches_the_media_choice(tracking, tmp_path: Path, media):
    run = live_run(tracking, tmp_path, "training", socket.gethostname())
    cache = tmp_path / "cache"
    with assemble_logdir(
        tracking, [run], cache_dir=cache, media=media, poll_interval=0.05
    ) as logdir:
        link = logdir / "training"
        shutil.rmtree(tmp_path / "local-training")
        assert wait_until(lambda: cache in link.resolve().parents)
        view = "events" if media else "scalars"
        assert link.resolve() == cache.resolve() / run.info.run_id / view
        assert (link / LIGHT).read_bytes() == b"scalar events"
        assert (link / MEDIA).exists() == media


def logging_experiment(client: MlflowClient) -> str:
    """The id of the experiment the `tracking` fixture logs runs into."""
    experiment = client.get_experiment_by_name("runsnap-tests")
    assert experiment is not None
    return experiment.experiment_id


def everything(client: MlflowClient) -> Callable[[], list[Run]]:
    """A selection of every run in the test experiment, re-run on each call."""
    experiment_id = logging_experiment(client)
    return lambda: client.search_runs([experiment_id])


def test_following_a_selection_starts_a_watch_thread_without_live_runs(
    tracking, tmp_path: Path
):
    run = named_run(tracking, tmp_path, "finished")
    with assemble_logdir(
        tracking,
        [run],
        cache_dir=tmp_path / "cache",
        follow=everything(tracking),
        poll_interval=60.0,
    ) as logdir:
        assert (logdir / "finished").is_symlink()
        (watch,) = watch_threads()
        assert watch.is_alive()
    assert not watch.is_alive()
    assert watch_threads() == []


def test_a_run_that_starts_later_is_linked_live_then_from_the_cache(
    tracking, tmp_path: Path
):
    shown = named_run(tracking, tmp_path, "finished")
    cache = tmp_path / "cache"
    with assemble_logdir(
        tracking,
        [shown],
        cache_dir=cache,
        follow=everything(tracking),
        poll_interval=0.05,
    ) as logdir:
        live_run(tracking, tmp_path, "training", socket.gethostname())
        link = logdir / "training"
        assert wait_until(link.exists)
        assert link.resolve() == (tmp_path / "local-training").resolve()
        assert (link / LIGHT).read_bytes() == b"live scalar events"
        shutil.rmtree(tmp_path / "local-training")
        assert wait_until(lambda: cache in link.resolve().parents)
        assert (link / LIGHT).read_bytes() == b"scalar events"
        assert (logdir / "finished" / LIGHT).read_bytes() == b"scalar events"


def test_a_run_is_linked_only_once_it_carries_its_local_directory(
    tracking, tmp_path: Path
):
    with (
        patch.object(
            tracking, "list_artifacts", wraps=tracking.list_artifacts
        ) as listing,
        patch.object(
            tracking, "download_artifacts", wraps=tracking.download_artifacts
        ) as download,
        assemble_logdir(
            tracking,
            [],
            cache_dir=tmp_path / "cache",
            follow=everything(tracking),
            poll_interval=0.02,
        ) as logdir,
    ):
        run = named_run(tracking, tmp_path, "training")
        time.sleep(0.3)  # a dozen or so passes
        assert list(logdir.iterdir()) == []
        listing.assert_not_called()
        download.assert_not_called()
        local = tmp_path / "local"
        local.mkdir()
        (local / LIGHT).write_bytes(b"live scalar events")
        tracking.set_tag(run.info.run_id, TB_TAG_LOCAL_HOST, socket.gethostname())
        tracking.set_tag(run.info.run_id, TB_TAG_LOCAL_DIR, str(local))
        link = logdir / "training"
        assert wait_until(link.exists)
        assert link.resolve() == local.resolve()


def test_a_run_selected_before_it_wrote_events_is_linked_once_it_does(
    tracking, tmp_path: Path
):
    with mlflow.start_run(run_name="training") as active:
        pending = tracking.get_run(active.info.run_id)
    with assemble_logdir(
        tracking,
        [pending],
        cache_dir=tmp_path / "cache",
        follow=everything(tracking),
        poll_interval=0.05,
    ) as logdir:
        assert list(logdir.iterdir()) == []
        local = tmp_path / "local"
        local.mkdir()
        (local / LIGHT).write_bytes(b"live scalar events")
        tracking.set_tag(pending.info.run_id, TB_TAG_LOCAL_HOST, socket.gethostname())
        tracking.set_tag(pending.info.run_id, TB_TAG_LOCAL_DIR, str(local))
        link = logdir / "training"
        assert wait_until(link.exists)
        assert link.resolve() == local.resolve()


def test_a_late_run_whose_writer_already_exited_is_linked_from_the_cache(
    tracking, tmp_path: Path
):
    cache = tmp_path / "cache"
    with assemble_logdir(
        tracking, [], cache_dir=cache, follow=everything(tracking), poll_interval=0.05
    ) as logdir:
        run = named_run(tracking, tmp_path, "training")
        tracking.set_tag(run.info.run_id, TB_TAG_LOCAL_HOST, socket.gethostname())
        tracking.set_tag(run.info.run_id, TB_TAG_LOCAL_DIR, str(tmp_path / "gone"))
        link = logdir / "training"
        assert wait_until(link.exists)
        assert cache in link.resolve().parents
        assert (link / LIGHT).read_bytes() == b"scalar events"


def test_a_late_run_named_like_a_shown_run_is_distinguished(tracking, tmp_path: Path):
    shown = named_run(tracking, tmp_path, "baseline")
    with assemble_logdir(
        tracking,
        [shown],
        cache_dir=tmp_path / "cache",
        follow=everything(tracking),
        poll_interval=0.05,
    ) as logdir:
        target = (logdir / "baseline").resolve()
        late = live_run(tracking, tmp_path, "baseline", socket.gethostname())
        link = logdir / f"baseline-{late.info.run_id[:8]}"
        assert wait_until(link.exists)
        assert link.resolve() == (tmp_path / "local-baseline").resolve()
        assert {p.name for p in logdir.iterdir()} == {"baseline", link.name}
        assert (logdir / "baseline").resolve() == target


def test_a_late_run_named_like_runs_shown_distinguished_is_distinguished_too(
    tracking, tmp_path: Path
):
    first = named_run(tracking, tmp_path, "baseline")
    second = named_run(tracking, tmp_path, "baseline")
    with assemble_logdir(
        tracking,
        [first, second],
        cache_dir=tmp_path / "cache",
        follow=everything(tracking),
        poll_interval=0.05,
    ) as logdir:
        before = {p.name: p.resolve() for p in logdir.iterdir()}
        assert set(before) == {
            f"baseline-{first.info.run_id[:8]}",
            f"baseline-{second.info.run_id[:8]}",
        }
        late = live_run(tracking, tmp_path, "baseline", socket.gethostname())
        link = logdir / f"baseline-{late.info.run_id[:8]}"
        assert wait_until(link.exists)
        after = {p.name: p.resolve() for p in logdir.iterdir()}
        assert after == {**before, link.name: (tmp_path / "local-baseline").resolve()}


def test_a_late_run_brings_its_earlier_attempt_with_chain(tracking, tmp_path: Path):
    earlier = named_run(tracking, tmp_path, "attempt-1")
    late = named_run(tracking, tmp_path, "attempt-2")
    tracking.set_tag(late.info.run_id, TAG_CONTINUES, earlier.info.run_id)
    local = tmp_path / "local"
    local.mkdir()
    (local / LIGHT).write_bytes(b"live scalar events")
    with (
        patch.object(tracking, "get_run", wraps=tracking.get_run) as get_run,
        assemble_logdir(
            tracking,
            [],
            cache_dir=tmp_path / "cache",
            chain=True,
            follow=everything(tracking),
            poll_interval=0.02,
        ) as logdir,
    ):
        tracking.set_tag(late.info.run_id, TB_TAG_LOCAL_HOST, socket.gethostname())
        tracking.set_tag(late.info.run_id, TB_TAG_LOCAL_DIR, str(local))
        assert wait_until(
            lambda: {p.name for p in logdir.iterdir()} == {"attempt-1", "attempt-2"}
        )
        time.sleep(0.3)  # later passes would fetch the chain again
        assert (logdir / "attempt-2").resolve() == local.resolve()
        assert (logdir / "attempt-1" / LIGHT).read_bytes() == b"scalar events"
    assert sorted(call.args[0] for call in get_run.call_args_list) == sorted(
        [earlier.info.run_id, late.info.run_id]
    )


@pytest.mark.parametrize("followed", [False, True])
def test_a_pass_never_fetches_a_run_finished_at_startup(
    tracking, tmp_path: Path, followed: bool
):
    shown = live_run(tracking, tmp_path, "training", "elsewhere")
    assert shown.info.status == "FINISHED"
    listing = Mock(wraps=tracking.list_artifacts)
    check, passes = recording_checks(listing, shown.info.run_id)
    with (
        patch("runsnap._tb_fetch._run_statuses", check),
        patch.object(tracking, "list_artifacts", listing),
        assemble_logdir(
            tracking,
            [shown],
            cache_dir=tmp_path / "cache",
            follow=everything(tracking) if followed else None,
            poll_interval=0.02,
        ) as logdir,
    ):
        assert (logdir / "training" / LIGHT).read_bytes() == b"scalar events"
        assert wait_for_checks(check, 10)
    # Every listing was made at startup, before the first pass.
    assert listings(listing, shown.info.run_id) == passes[0][1]


def test_a_failing_selection_is_warned_once_and_retried(tracking, tmp_path: Path):
    shown = named_run(tracking, tmp_path, "finished")
    query = everything(tracking)
    failures = 5

    def flaky() -> list[Run]:
        nonlocal failures
        if failures:
            failures -= 1
            raise ConnectionError("the tracking server is unreachable")
        return query()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with assemble_logdir(
            tracking,
            [shown],
            cache_dir=tmp_path / "cache",
            follow=flaky,
            poll_interval=0.02,
        ) as logdir:
            target = (logdir / "finished").resolve()
            assert wait_until(lambda: failures == 0)
            live_run(tracking, tmp_path, "training", socket.gethostname())
            assert wait_until((logdir / "training").exists)
            assert (logdir / "finished").resolve() == target

    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 1
    assert "the tracking server is unreachable" in str(user_warnings[0].message)


def test_check_reports_the_status_of_each_run_it_finds(tracking, tmp_path: Path):
    finished = named_run(tracking, tmp_path, "finished").info.run_id
    experiment_id = logging_experiment(tracking)
    running = tracking.create_run(experiment_id).info.run_id
    statuses = _run_statuses(
        tracking, {running: experiment_id, finished: experiment_id}
    )
    assert statuses == {running: "RUNNING", finished: "FINISHED"}


def test_check_leaves_out_deleted_runs_and_runs_of_deleted_experiments(
    tracking, tmp_path: Path
):
    experiment_id = logging_experiment(tracking)
    kept = tracking.create_run(experiment_id).info.run_id
    deleted = tracking.create_run(experiment_id).info.run_id
    tracking.delete_run(deleted)
    doomed = tracking.create_experiment(
        "doomed", artifact_location=(tmp_path / "doomed").as_uri()
    )
    orphan = tracking.create_run(doomed).info.run_id
    tracking.delete_experiment(doomed)
    statuses = _run_statuses(
        tracking, {kept: experiment_id, deleted: experiment_id, orphan: doomed}
    )
    assert statuses == {kept: "RUNNING"}


def test_check_searches_once_per_two_hundred_runs(tracking):
    experiment_id = logging_experiment(tracking)
    found = tracking.create_run(experiment_id).info.run_id
    runs = {f"{n:032x}": experiment_id for n in range(449)} | {found: experiment_id}
    with patch.object(tracking, "search_runs", wraps=tracking.search_runs) as search:
        assert _run_statuses(tracking, runs) == {found: "RUNNING"}
    assert search.call_count == 3


def test_a_deleted_run_leaves_the_dashboard_and_the_cache(tracking, tmp_path: Path):
    kept = named_run(tracking, tmp_path, "kept")
    deleted = named_run(tracking, tmp_path, "deleted")
    cache = tmp_path / "cache"
    with assemble_logdir(
        tracking, [kept, deleted], cache_dir=cache, poll_interval=0.05
    ) as logdir:
        link = logdir / "deleted"
        folder = cache / deleted.info.run_id
        assert link.is_symlink()
        assert folder.is_dir()
        tracking.delete_run(deleted.info.run_id)
        assert wait_until(lambda: not link.is_symlink() and not folder.exists())
        assert [p.name for p in logdir.iterdir()] == ["kept"]
        assert (logdir / "kept" / LIGHT).read_bytes() == b"scalar events"
        assert (cache / kept.info.run_id).is_dir()


def test_deleting_the_experiment_of_shown_runs_removes_them(tracking, tmp_path: Path):
    runs = [named_run(tracking, tmp_path, name) for name in ["first", "second"]]
    with assemble_logdir(
        tracking, runs, cache_dir=tmp_path / "cache", poll_interval=0.05
    ) as logdir:
        assert {p.name for p in logdir.iterdir()} == {"first", "second"}
        tracking.delete_experiment(logging_experiment(tracking))
        assert wait_until(lambda: not any(logdir.iterdir()))


@pytest.mark.parametrize("status", ["FINISHED", "FAILED", "KILLED"])
def test_a_run_that_ends_stays_shown(tracking, tmp_path: Path, status: str):
    run = named_run(tracking, tmp_path, "training")
    tracking.update_run(run.info.run_id, status="RUNNING")
    with (
        patch("runsnap._tb_fetch._run_statuses", wraps=_run_statuses) as check,
        assemble_logdir(
            tracking, [run], cache_dir=tmp_path / "cache", poll_interval=0.02
        ) as logdir,
    ):
        assert wait_for_checks(check)
        tracking.set_terminated(run.info.run_id, status)
        assert wait_for_checks(check)
        assert (logdir / "training" / LIGHT).read_bytes() == b"scalar events"


def test_a_failing_check_is_warned_once_and_removes_nothing(tracking, tmp_path: Path):
    kept = named_run(tracking, tmp_path, "kept")
    deleted = named_run(tracking, tmp_path, "deleted")
    failures = 0
    recovered = threading.Event()

    def flaky(client: MlflowClient, runs: dict[str, str]) -> dict[str, str]:
        nonlocal failures
        if not recovered.is_set():
            failures += 1
            raise ConnectionError("the tracking server is unreachable")
        return _run_statuses(client, runs)

    with (
        patch("runsnap._tb_fetch._run_statuses", side_effect=flaky),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        with assemble_logdir(
            tracking,
            [kept, deleted],
            cache_dir=tmp_path / "cache",
            poll_interval=0.02,
        ) as logdir:
            tracking.delete_run(deleted.info.run_id)
            assert wait_until(lambda: failures >= 5)
            assert {p.name for p in logdir.iterdir()} == {"kept", "deleted"}
            assert (logdir / "deleted" / LIGHT).read_bytes() == b"scalar events"
            recovered.set()
            assert wait_until(lambda: not (logdir / "deleted").is_symlink())
            assert (logdir / "kept" / LIGHT).read_bytes() == b"scalar events"

    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 1
    assert "the tracking server is unreachable" in str(user_warnings[0].message)


def remote_run(client: MlflowClient, tmp_path: Path, name: str) -> Run:
    """A run tagged as being written on another host, so it is shown cached."""
    run = named_run(client, tmp_path, name)
    client.set_tag(run.info.run_id, TB_TAG_LOCAL_HOST, "elsewhere")
    client.set_tag(run.info.run_id, TB_TAG_LOCAL_DIR, f"/elsewhere/{name}")
    return client.get_run(run.info.run_id)


def test_a_restored_run_of_a_followed_query_is_shown_again(tracking, tmp_path: Path):
    run = remote_run(tracking, tmp_path, "training")
    with assemble_logdir(
        tracking,
        [run],
        cache_dir=tmp_path / "cache",
        follow=everything(tracking),
        poll_interval=0.05,
    ) as logdir:
        link = logdir / "training"
        tracking.delete_run(run.info.run_id)
        assert wait_until(lambda: not link.is_symlink())
        tracking.restore_run(run.info.run_id)
        assert wait_until(link.exists)
        assert (link / LIGHT).read_bytes() == b"scalar events"


def test_a_restored_named_run_is_shown_again(tracking, tmp_path: Path):
    run = named_run(tracking, tmp_path, "baseline")
    with assemble_logdir(
        tracking, [run], cache_dir=tmp_path / "cache", poll_interval=0.05
    ) as logdir:
        link = logdir / "baseline"
        tracking.delete_run(run.info.run_id)
        assert wait_until(lambda: not link.is_symlink())
        tracking.restore_run(run.info.run_id)
        assert wait_until(link.exists)
        assert (link / LIGHT).read_bytes() == b"scalar events"


def test_a_restored_run_whose_name_a_late_run_took_is_distinguished(
    tracking, tmp_path: Path
):
    old = remote_run(tracking, tmp_path, "baseline")
    with assemble_logdir(
        tracking,
        [old],
        cache_dir=tmp_path / "cache",
        follow=everything(tracking),
        poll_interval=0.05,
    ) as logdir:
        link = logdir / "baseline"
        tracking.delete_run(old.info.run_id)
        assert wait_until(lambda: not link.is_symlink())
        late = remote_run(tracking, tmp_path, "baseline")
        assert wait_until(link.exists)
        assert late.info.run_id in link.resolve().parts
        tracking.restore_run(old.info.run_id)
        returned = logdir / f"baseline-{old.info.run_id[:8]}"
        assert wait_until(returned.exists)
        assert old.info.run_id in returned.resolve().parts
        assert {p.name for p in logdir.iterdir()} == {"baseline", returned.name}


def starting_run(client: MlflowClient, name: str) -> Run:
    """A running run tagged as writing on another host that uploaded nothing."""
    run_id = client.create_run(logging_experiment(client), run_name=name).info.run_id
    client.set_tag(run_id, TB_TAG_LOCAL_HOST, "elsewhere")
    client.set_tag(run_id, TB_TAG_LOCAL_DIR, f"/elsewhere/{name}")
    return client.get_run(run_id)


def listings(listing: Mock, run_id: str) -> int:
    """How many artifact listings of `run_id` the `listing` spy recorded."""
    return sum(call.args[0] == run_id for call in listing.call_args_list)


def recording_checks(
    listing: Mock, run_id: str
) -> tuple[Mock, list[tuple[str | None, int]]]:
    """A check of the shown runs that records each pass as it begins.

    Each pass notes the status the check reports for `run_id` and how many
    listings of it `listing` recorded before.
    """
    passes: list[tuple[str | None, int]] = []

    def check(client: MlflowClient, runs: dict[str, str]) -> dict[str, str]:
        statuses = _run_statuses(client, runs)
        passes.append((statuses.get(run_id), listings(listing, run_id)))
        return statuses

    return Mock(side_effect=check), passes


def test_a_run_from_another_host_is_linked_once_it_uploads(tracking, tmp_path: Path):
    run = starting_run(tracking, "training")
    with assemble_logdir(
        tracking, [run], cache_dir=tmp_path / "cache", poll_interval=0.05
    ) as logdir:
        assert list(logdir.iterdir()) == []
        (tmp_path / "source").mkdir()
        log_light(tracking, run, tmp_path, b"scalar events")
        link = logdir / "training"
        assert wait_until(link.exists)
        assert (link / LIGHT).read_bytes() == b"scalar events"


def test_a_followed_run_from_another_host_is_linked_once_it_uploads(
    tracking, tmp_path: Path
):
    with (
        patch.object(
            tracking, "list_artifacts", wraps=tracking.list_artifacts
        ) as listing,
        assemble_logdir(
            tracking,
            [],
            cache_dir=tmp_path / "cache",
            follow=everything(tracking),
            poll_interval=0.05,
        ) as logdir,
    ):
        run = starting_run(tracking, "training")
        # Found by the selection, then tried again while pending.
        assert wait_until(lambda: listings(listing, run.info.run_id) >= 2)
        assert list(logdir.iterdir()) == []
        (tmp_path / "source").mkdir()
        log_light(tracking, run, tmp_path, b"scalar events")
        link = logdir / "training"
        assert wait_until(link.exists)
        assert (link / LIGHT).read_bytes() == b"scalar events"


@pytest.mark.parametrize("followed", [False, True])
def test_a_pending_run_that_ends_without_events_is_dropped(
    tracking, tmp_path: Path, followed: bool
):
    run = starting_run(tracking, "training")
    with (
        patch("runsnap._tb_fetch._run_statuses", wraps=_run_statuses) as check,
        patch.object(
            tracking, "list_artifacts", wraps=tracking.list_artifacts
        ) as listing,
        assemble_logdir(
            tracking,
            [run],
            cache_dir=tmp_path / "cache",
            follow=everything(tracking) if followed else None,
            poll_interval=0.02,
        ) as logdir,
    ):
        assert wait_for_checks(check)
        assert listings(listing, run.info.run_id) >= 2
        tracking.set_terminated(run.info.run_id, "FAILED")
        # The pass whose check sees the run ended gives it a last try.
        assert wait_for_checks(check)
        listing.reset_mock()
        assert wait_for_checks(check, 10)
        assert list(logdir.iterdir()) == []
    listing.assert_not_called()


def test_a_pass_never_lists_untagged_or_ended_runs(tracking, tmp_path: Path):
    experiment_id = logging_experiment(tracking)
    ended = starting_run(tracking, "ended").info.run_id
    tracking.set_terminated(ended, "KILLED")
    untagged = tracking.create_run(experiment_id).info.run_id
    selected = [tracking.get_run(ended), tracking.get_run(untagged)]
    with (
        patch("runsnap._tb_fetch._run_statuses", wraps=_run_statuses) as check,
        patch.object(
            tracking, "list_artifacts", wraps=tracking.list_artifacts
        ) as listing,
        assemble_logdir(
            tracking,
            selected,
            cache_dir=tmp_path / "cache",
            follow=everything(tracking),
            poll_interval=0.02,
        ) as logdir,
    ):
        # Startup tried both; an untagged run the selection returns later is
        # never tried at all.
        listing.reset_mock()
        tracking.create_run(experiment_id)
        assert wait_for_checks(check, 10)
        assert list(logdir.iterdir()) == []
    listing.assert_not_called()


def test_a_running_run_from_another_host_gains_its_new_uploads(
    tracking, tmp_path: Path
):
    run = starting_run(tracking, "training")
    source = tmp_path / "events"
    writer = TensorBoardWriter(source, flush_secs=0.05)
    writer.add_scalar("loss", 0.5, 1)
    assert wait_for_steps(source, "loss", [1]) == [1]
    tracking.log_artifact(run.info.run_id, str(writer.light_path), "tb")
    with assemble_logdir(
        tracking, [run], cache_dir=tmp_path / "cache", poll_interval=0.05
    ) as logdir:
        link = logdir / "training"
        assert wait_for_steps(link, "loss", [1]) == [1]
        writer.add_scalar("loss", 0.25, 2)
        writer.close()
        tracking.log_artifact(run.info.run_id, str(writer.light_path), "tb")
        assert wait_for_steps(link, "loss", [1, 2]) == [1, 2]


def test_a_run_that_finishes_is_fetched_once_more_and_never_again(
    tracking, tmp_path: Path
):
    run = starting_run(tracking, "training")
    run_id = run.info.run_id
    (tmp_path / "source").mkdir()
    log_light(tracking, run, tmp_path, b"scalar events")
    listing = Mock(wraps=tracking.list_artifacts)
    check, passes = recording_checks(listing, run_id)
    with (
        patch("runsnap._tb_fetch._run_statuses", check),
        patch.object(tracking, "list_artifacts", listing),
        assemble_logdir(
            tracking, [run], cache_dir=tmp_path / "cache", poll_interval=0.02
        ) as logdir,
    ):
        # Fetched at startup, then again by a pass while running.
        assert wait_for_checks(check)
        assert listings(listing, run_id) >= 2
        log_light(tracking, run, tmp_path, b"scalar events and the last step")
        tracking.set_terminated(run_id)
        assert wait_until(lambda: [s for s, _ in passes].count("FINISHED") >= 10)
        link = logdir / "training"
        assert (link / LIGHT).read_bytes() == b"scalar events and the last step"
    ended = [count for status, count in passes if status == "FINISHED"]
    assert listings(listing, run_id) == ended[0] + 1


def test_a_failing_refresh_is_warned_once_and_keeps_the_run(tracking, tmp_path: Path):
    run = starting_run(tracking, "training")
    (tmp_path / "source").mkdir()
    log_light(tracking, run, tmp_path, b"scalar events")
    list_artifacts = tracking.list_artifacts
    unreachable = threading.Event()

    def flaky(run_id: str, path: str) -> list[FileInfo]:
        if unreachable.is_set():
            raise ConnectionError("the artifact store is unreachable")
        return list_artifacts(run_id, path)

    with (
        patch("runsnap._tb_fetch._run_statuses", wraps=_run_statuses) as check,
        patch.object(tracking, "list_artifacts", side_effect=flaky) as listing,
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        with assemble_logdir(
            tracking, [run], cache_dir=tmp_path / "cache", poll_interval=0.02
        ) as logdir:
            link = logdir / "training"
            unreachable.set()
            tried = listings(listing, run.info.run_id)
            assert wait_for_checks(check, 5)
            # Each pass tried again, and the run kept what it had.
            assert listings(listing, run.info.run_id) >= tried + 3
            assert (link / LIGHT).read_bytes() == b"scalar events"
            unreachable.clear()
            log_light(tracking, run, tmp_path, b"scalar events and more")
            assert wait_until(
                lambda: (link / LIGHT).read_bytes() == b"scalar events and more"
            )

    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 1
    assert run.info.run_id in str(user_warnings[0].message)
    assert "the artifact store is unreachable" in str(user_warnings[0].message)


@pytest.mark.parametrize("uploaded", [True, False])
def test_a_run_that_ends_before_the_first_pass_gets_one_final_fetch(
    tracking, tmp_path: Path, uploaded: bool
):
    """A run ending before the first pass is still fetched one final time.

    Shown at startup, or pending without data, the run uploads its last events
    and ends before the first pass checks it.
    """
    run = starting_run(tracking, "training")
    run_id = run.info.run_id
    (tmp_path / "source").mkdir()
    if uploaded:
        log_light(tracking, run, tmp_path, b"scalar events")
    listing = Mock(wraps=tracking.list_artifacts)
    record, passes = recording_checks(listing, run_id)

    def end_first(client: MlflowClient, runs: dict[str, str]) -> dict[str, str]:
        if not passes:
            log_light(tracking, run, tmp_path, b"scalar events and the last step")
            tracking.set_terminated(run_id)
        return record(client, runs)

    with (
        patch("runsnap._tb_fetch._run_statuses", side_effect=end_first),
        patch.object(tracking, "list_artifacts", listing),
        assemble_logdir(
            tracking, [run], cache_dir=tmp_path / "cache", poll_interval=0.02
        ) as logdir,
    ):
        assert wait_until(lambda: len(passes) >= 10)
        link = logdir / "training"
        assert (link / LIGHT).read_bytes() == b"scalar events and the last step"
    assert passes[0][0] == "FINISHED"
    assert listings(listing, run_id) == passes[0][1] + 1


def stored_run(client: MlflowClient, root: Path, name: str) -> tuple[Run, Path]:
    """A running run tagged as writing on another host, and its TensorBoard folder.

    The run is stored as a tracking server started with
    `--artifacts-destination root` stores it, and its folder under `root` is
    still empty.
    """
    experiment = client.get_experiment_by_name("stored")
    experiment_id = (
        client.create_experiment("stored", artifact_location="mlflow-artifacts:/stored")
        if experiment is None
        else experiment.experiment_id
    )
    run_id = client.create_run(experiment_id, run_name=name).info.run_id
    client.set_tag(run_id, TB_TAG_LOCAL_HOST, "elsewhere")
    client.set_tag(run_id, TB_TAG_LOCAL_DIR, f"/elsewhere/{name}")
    folder = root / "stored" / run_id / "artifacts" / "tb"
    folder.mkdir(parents=True)
    return client.get_run(run_id), folder


def test_a_stored_run_is_linked_from_its_folder_under_the_artifact_root(
    tracking, tmp_path: Path
):
    root = tmp_path / "root"
    run, folder = stored_run(tracking, root, "training")
    (folder / LIGHT).write_bytes(b"stored scalar events")
    with assemble_logdir(
        tracking, [run], cache_dir=tmp_path / "cache", artifact_root=root
    ) as logdir:
        link = logdir / "training"
        assert link.is_dir()
        assert not link.is_symlink()
        assert (link / LIGHT).readlink() == folder.resolve() / LIGHT
        assert (link / LIGHT).read_bytes() == b"stored scalar events"


def test_a_run_stored_elsewhere_is_warned_about_once_and_left_out(
    tracking, tmp_path: Path
):
    root = tmp_path / "root"
    stored, folder = stored_run(tracking, root, "stored")
    (folder / LIGHT).write_bytes(b"stored scalar events")
    # Running in the `file:` experiment, so a followed query keeps finding it.
    elsewhere = starting_run(tracking, "elsewhere")
    (tmp_path / "source").mkdir()
    log_light(tracking, elsewhere, tmp_path, b"scalar events")
    with (
        patch("runsnap._tb_fetch._run_statuses", wraps=_run_statuses) as check,
        patch.object(tracking, "list_artifacts") as listing,
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        with assemble_logdir(
            tracking,
            [stored, elsewhere],
            cache_dir=tmp_path / "cache",
            follow=everything(tracking),
            poll_interval=0.02,
            artifact_root=root,
        ) as logdir:
            assert wait_for_checks(check, 10)
            assert [p.name for p in logdir.iterdir()] == ["stored"]
    listing.assert_not_called()
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 1
    assert elsewhere.info.run_id in str(user_warnings[0].message)


def test_a_run_whose_artifacts_lead_out_of_the_root_is_refused(
    tracking, tmp_path: Path
):
    root = tmp_path / "root"
    root.mkdir()
    experiment_id = tracking.create_experiment(
        "escaping", artifact_location="mlflow-artifacts:/../outside"
    )
    run = tracking.create_run(experiment_id, run_name="escaping")
    folder = tmp_path / "outside" / run.info.run_id / "artifacts" / "tb"
    folder.mkdir(parents=True)
    (folder / LIGHT).write_bytes(b"scalar events")
    with (
        pytest.warns(UserWarning, match=run.info.run_id),
        assemble_logdir(
            tracking, [run], cache_dir=tmp_path / "cache", artifact_root=root
        ) as logdir,
    ):
        assert list(logdir.iterdir()) == []


def test_viewing_from_the_artifact_root_neither_lists_nor_downloads_nor_caches(
    tracking, tmp_path: Path
):
    root = tmp_path / "root"
    shown, folder = stored_run(tracking, root, "training")
    (folder / LIGHT).write_bytes(b"stored scalar events")
    starting, _ = stored_run(tracking, root, "starting")
    cache = tmp_path / "cache"
    with (
        patch("runsnap._tb_fetch._run_statuses", wraps=_run_statuses) as check,
        patch("runsnap._tb_fetch._live_logdir") as live,
        patch.object(tracking, "list_artifacts") as listing,
        patch.object(tracking, "download_artifacts") as download,
        assemble_logdir(
            tracking,
            [shown, starting],
            cache_dir=cache,
            poll_interval=0.02,
            artifact_root=root,
        ) as logdir,
    ):
        # Passes refresh the shown run and retry the pending one, while they
        # run and once more after they end.
        assert wait_for_checks(check, 5)
        tracking.set_terminated(shown.info.run_id)
        tracking.set_terminated(starting.info.run_id)
        assert wait_for_checks(check, 5)
        assert [p.name for p in logdir.iterdir()] == ["training"]
    live.assert_not_called()
    listing.assert_not_called()
    download.assert_not_called()
    assert not cache.exists()


def test_only_scalar_files_are_linked_from_the_artifact_root(tracking, tmp_path: Path):
    root = tmp_path / "root"
    run, folder = stored_run(tracking, root, "training")
    nested = folder / "nested"
    nested.mkdir()
    for directory in (folder, nested):
        (directory / LIGHT_SHARDS[0]).write_bytes(b"scalar events")
        (directory / MEDIA).write_bytes(b"media events")
    with assemble_logdir(
        tracking, [run], cache_dir=tmp_path / "cache", artifact_root=root
    ) as logdir:
        link = logdir / "training"
        entries = {p.relative_to(link).as_posix(): p for p in link.rglob("*")}
        assert set(entries) == {
            LIGHT_SHARDS[0],
            "nested",
            f"nested/{LIGHT_SHARDS[0]}",
        }
        assert not entries["nested"].is_symlink()
        assert entries[f"nested/{LIGHT_SHARDS[0]}"].readlink() == (
            nested.resolve() / LIGHT_SHARDS[0]
        )


def test_a_new_scalar_shard_in_the_artifact_root_is_linked(tracking, tmp_path: Path):
    root = tmp_path / "root"
    run, folder = stored_run(tracking, root, "training")
    with assemble_logdir(
        tracking,
        [run],
        cache_dir=tmp_path / "cache",
        poll_interval=0.05,
        artifact_root=root,
    ) as logdir:
        # Pending until its first shard reaches the folder.
        assert list(logdir.iterdir()) == []
        link = logdir / "training"
        (folder / LIGHT_SHARDS[0]).write_bytes(b"first shard")
        assert wait_until((link / LIGHT_SHARDS[0]).exists)
        (folder / LIGHT_SHARDS[1]).write_bytes(b"second shard")
        assert wait_until((link / LIGHT_SHARDS[1]).exists)
        assert (link / LIGHT_SHARDS[1]).read_bytes() == b"second shard"


def test_a_run_that_ends_gets_its_last_shard_linked_and_no_more(
    tracking, tmp_path: Path
):
    root = tmp_path / "root"
    run, folder = stored_run(tracking, root, "training")
    (folder / LIGHT_SHARDS[0]).write_bytes(b"first shard")
    checks = 0

    def end_first(client: MlflowClient, runs: dict[str, str]) -> dict[str, str]:
        nonlocal checks
        if not checks:
            (folder / LIGHT_SHARDS[1]).write_bytes(b"last shard")
            tracking.set_terminated(run.info.run_id)
        checks += 1
        return _run_statuses(client, runs)

    late = "events.out.tfevents.4.host.scalars.2"
    with (
        patch("runsnap._tb_fetch._run_statuses", side_effect=end_first),
        assemble_logdir(
            tracking,
            [run],
            cache_dir=tmp_path / "cache",
            poll_interval=0.02,
            artifact_root=root,
        ) as logdir,
    ):
        link = logdir / "training"
        # The second check begins once the first pass, which saw the end, is done.
        assert wait_until(lambda: checks >= 2)
        assert (link / LIGHT_SHARDS[1]).read_bytes() == b"last shard"
        (folder / late).write_bytes(b"too late")
        done = checks
        assert wait_until(lambda: checks >= done + 10)
        assert not (link / late).is_symlink()


def test_a_scalar_file_replaced_by_a_longer_copy_shows_its_new_steps(
    tracking, tmp_path: Path
):
    root = tmp_path / "root"
    run, folder = stored_run(tracking, root, "training")
    source = tmp_path / "events"
    writer = TensorBoardWriter(source, flush_secs=0.05)
    writer.add_scalar("loss", 0.5, 1)
    assert wait_for_steps(source, "loss", [1]) == [1]
    stored = folder / writer.light_path.name
    shutil.copy(writer.light_path, stored)
    with assemble_logdir(
        tracking, [run], cache_dir=tmp_path / "cache", artifact_root=root
    ) as logdir:
        accumulator = EventAccumulator(str(logdir / "training"))
        accumulator.Reload()
        assert [point.step for point in accumulator.Scalars("loss")] == [1]
        writer.add_scalar("loss", 0.25, 2)
        writer.close()
        # The tracking server stores each upload by moving a copy over the file.
        upload = tmp_path / "upload"
        shutil.copy(writer.light_path, upload)
        upload.replace(stored)
        accumulator.Reload()
        assert [point.step for point in accumulator.Scalars("loss")] == [1, 2]


def test_a_run_live_on_this_host_is_linked_from_the_artifact_root(
    tracking, tmp_path: Path
):
    root = tmp_path / "root"
    run, folder = stored_run(tracking, root, "training")
    (folder / LIGHT).write_bytes(b"stored scalar events")
    local = tmp_path / "local"
    local.mkdir()
    (local / LIGHT).write_bytes(b"live scalar events")
    tracking.set_tag(run.info.run_id, TB_TAG_LOCAL_HOST, socket.gethostname())
    tracking.set_tag(run.info.run_id, TB_TAG_LOCAL_DIR, str(local))
    run = tracking.get_run(run.info.run_id)
    with assemble_logdir(
        tracking, [run], cache_dir=tmp_path / "cache", artifact_root=root
    ) as logdir:
        link = logdir / "training" / LIGHT
        assert link.readlink() == folder.resolve() / LIGHT
        assert link.read_bytes() == b"stored scalar events"


def test_a_deleted_run_leaves_its_files_in_the_artifact_root(tracking, tmp_path: Path):
    root = tmp_path / "root"
    kept, _ = stored_run(tracking, root, "kept")
    deleted, folder = stored_run(tracking, root, "deleted")
    for directory in (root / "stored").glob("*/artifacts/tb"):
        (directory / "nested").mkdir()
        (directory / LIGHT).write_bytes(b"stored scalar events")
        (directory / "nested" / LIGHT).write_bytes(b"nested scalar events")
    # Left by an earlier invocation that downloaded the run.
    cached = tmp_path / "cache" / deleted.info.run_id / "events" / LIGHT
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"scalar events")
    with assemble_logdir(
        tracking,
        [kept, deleted],
        cache_dir=tmp_path / "cache",
        poll_interval=0.05,
        artifact_root=root,
    ) as logdir:
        link = logdir / "deleted"
        assert (link / "nested" / LIGHT).is_symlink()
        tracking.delete_run(deleted.info.run_id)
        assert wait_until(lambda: not link.exists())
        assert [p.name for p in logdir.iterdir()] == ["kept"]
        assert (folder / LIGHT).read_bytes() == b"stored scalar events"
        assert (folder / "nested" / LIGHT).read_bytes() == b"nested scalar events"
        tracking.restore_run(deleted.info.run_id)
        assert wait_until((link / "nested" / LIGHT).is_symlink)
    assert (folder / LIGHT).read_bytes() == b"stored scalar events"
    assert cached.read_bytes() == b"scalar events"
