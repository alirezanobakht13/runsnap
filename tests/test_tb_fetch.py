"""Artifact cache and named TensorBoard log directory tests."""

import shutil
import socket
import threading
import time
from pathlib import Path
from unittest.mock import patch

import mlflow
import pytest
from mlflow.entities import Run
from mlflow.tracking import MlflowClient
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import runsnap
from runsnap._tags import TB_TAG_LOCAL_DIR, TB_TAG_LOCAL_HOST
from runsnap._tb_fetch import assemble_logdir, fetch_run
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
