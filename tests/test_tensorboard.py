"""Tests for the split, sharded TensorBoard writer."""

import socket
import time
from pathlib import Path

import numpy as np
import pytest
from mlflow.tracking import MlflowClient
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboardX.writer import FileWriter

import runsnap
from runsnap._tags import (
    TB_ARTIFACT_DIR,
    TB_FLUSH_SECONDS,
    TB_LIGHT_SUFFIX,
    TB_MEDIA_SUFFIX,
    TB_TAG_LOCAL_DIR,
    TB_TAG_LOCAL_HOST,
    TB_TAG_LOGDIR,
)
from runsnap._tensorboard import TensorBoardWriter


def event_files(logdir: Path) -> list[Path]:
    return sorted(logdir.glob("events.out.tfevents.*"))


def media_files(logdir: Path) -> list[Path]:
    return sorted(p for p in event_files(logdir) if TB_MEDIA_SUFFIX in p.name)


def image(size: int = 64) -> np.ndarray:
    """An incompressible image, so shard size tracks the number of writes."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(3, size, size), dtype=np.uint8)


def accumulate(logdir: Path) -> EventAccumulator:
    """Read a whole log directory back the way TensorBoard reads it."""
    accumulator = EventAccumulator(str(logdir))
    accumulator.Reload()
    return accumulator


@pytest.fixture
def logdir(tmp_path: Path) -> Path:
    path = tmp_path / "tb"
    path.mkdir()
    return path


def test_scalars_and_media_land_in_different_files(logdir: Path):
    writer = TensorBoardWriter(logdir)
    writer.add_scalar("train/loss", 0.5, 1)
    writer.add_image("train/sample", image(), 1)
    writer.close()

    light = [p for p in event_files(logdir) if p.name.endswith(TB_LIGHT_SUFFIX)]
    media = media_files(logdir)
    assert len(light) == 1
    assert len(media) == 1
    assert light[0].parent == media[0].parent

    assert accumulate_tags(light[0])["scalars"] == ["train/loss"]
    assert accumulate_tags(light[0])["images"] == []
    assert accumulate_tags(media[0])["images"] == ["train/sample"]
    assert accumulate_tags(media[0])["scalars"] == []


def test_text_follows_scalars_and_other_methods_follow_media(logdir: Path):
    writer = TensorBoardWriter(logdir)
    writer.add_text("notes", "hello", 1)
    writer.add_histogram("weights", np.arange(100.0), 1)
    writer.close()

    light = next(p for p in event_files(logdir) if p.name.endswith(TB_LIGHT_SUFFIX))
    media = media_files(logdir)[0]

    assert accumulate_tags(light)["tensors"] == ["notes/text_summary"]
    assert accumulate_tags(media)["histograms"] == ["weights"]


def accumulate_tags(event_file: Path) -> dict[str, list[str]]:
    """The tags of a single event file, read in isolation from its siblings."""
    accumulator = EventAccumulator(str(event_file))
    accumulator.Reload()
    return accumulator.Tags()


def read_light(logdir: Path) -> EventAccumulator:
    """The scalar event file, read in isolation from the media shards."""
    light = next(p for p in event_files(logdir) if p.name.endswith(TB_LIGHT_SUFFIX))
    accumulator = EventAccumulator(str(light))
    accumulator.Reload()
    return accumulator


def test_record_charts_every_numeric_leaf(logdir: Path):
    record = {"loss": 0.5, "kind": "train", "actor": {"entropy": 1.2}}

    writer = TensorBoardWriter(logdir)
    writer.add_record("train", record, 10)
    writer.close()

    light = read_light(logdir)
    assert sorted(light.Tags()["scalars"]) == ["train/actor.entropy", "train/loss"]
    assert [(p.step, p.value) for p in light.Scalars("train/loss")] == [(10, 0.5)]
    assert [(p.step, p.value) for p in light.Scalars("train/actor.entropy")] == [
        (10, pytest.approx(1.2))
    ]


def test_record_forwards_wall_time(logdir: Path):
    writer = TensorBoardWriter(logdir)
    writer.add_record("eval", {"loss": 0.5, "kl": 0.01}, 10, walltime=1700000000.0)
    writer.close()

    light = read_light(logdir)
    points = [p for tag in light.Tags()["scalars"] for p in light.Scalars(tag)]
    assert len(points) == 2
    assert {p.wall_time for p in points} == {1700000000.0}


def test_record_with_an_empty_prefix_charts_bare_keys(logdir: Path):
    writer = TensorBoardWriter(logdir)
    writer.add_record("", {"loss": 0.5}, 1)
    writer.close()

    assert read_light(logdir).Tags()["scalars"] == ["loss"]


def test_records_stay_downloadable_without_media(logdir: Path):
    writer = TensorBoardWriter(logdir)
    for step in range(5):
        writer.add_record("train", {"loss": 1.0 / (step + 1)}, step)
        writer.add_image("train/sample", image(), step)
    writer.close()

    light = read_light(logdir)
    assert [p.step for p in light.Scalars("train/loss")] == list(range(5))
    assert light.Tags()["images"] == []
    assert accumulate_tags(media_files(logdir)[0])["scalars"] == []


def test_scalar_file_is_a_small_fraction_of_the_total(logdir: Path):
    writer = TensorBoardWriter(logdir)
    for step in range(20):
        writer.add_scalar("train/loss", 1.0 / (step + 1), step)
        writer.add_image("train/sample", image(), step)
    writer.close()

    light = next(p for p in event_files(logdir) if p.name.endswith(TB_LIGHT_SUFFIX))
    total = sum(p.stat().st_size for p in event_files(logdir))
    assert light.stat().st_size * 10 < total


def test_histogram_default_bins_are_compact(logdir: Path):
    values = np.random.default_rng(0).normal(size=10_000)

    compact = TensorBoardWriter(logdir / "compact")
    compact.add_histogram("weights", values, 1)
    compact.close()

    wide = TensorBoardWriter(logdir / "wide")
    wide.add_histogram("weights", values, 1, bins="tensorflow")
    wide.close()

    compact_size = media_files(logdir / "compact")[0].stat().st_size
    wide_size = media_files(logdir / "wide")[0].stat().st_size
    assert compact_size * 4 < wide_size
    assert accumulate_tags(media_files(logdir / "compact")[0])["histograms"] == [
        "weights"
    ]


def test_histogram_bins_argument_is_passed_through(logdir: Path):
    values = np.arange(1000.0)

    writer = TensorBoardWriter(logdir)
    writer.add_histogram("explicit", values, 1, bins=7)
    writer.close()

    accumulator = accumulate(logdir)
    buckets = accumulator.Histograms("explicit")[0].histogram_value.bucket
    # The writer prepends one empty bucket below the requested seven.
    assert len(buckets) == 8
    assert buckets[0] == 0
    assert sum(buckets) == len(values)


def test_media_rolls_into_distinctly_named_shards(logdir: Path):
    writer = TensorBoardWriter(logdir, shard_max_bytes=16 * 1024)
    for step in range(40):
        writer.add_image("train/sample", image(), step)
    writer.close()

    shards = media_files(logdir)
    assert len(shards) > 1
    assert len({p.name for p in shards}) == len(shards)


def test_shards_sealed_in_the_same_second_do_not_collide(logdir: Path):
    writer = TensorBoardWriter(logdir, shard_max_bytes=1)
    for step in range(6):
        writer.flush()
        writer.add_image("train/sample", image(16), step)
    writer.close()

    shards = media_files(logdir)
    seconds = {p.name.split(".")[3] for p in shards}
    assert len(shards) > len(seconds)
    assert len({p.name for p in shards}) == len(shards)


def test_paths_name_the_light_file_and_the_open_shard(logdir: Path):
    writer = TensorBoardWriter(logdir, shard_max_bytes=1)
    writer.add_scalar("train/loss", 0.5, 1)
    writer.add_image("train/sample", image(16), 1)
    writer.flush()
    sealed = writer.media_path
    # The flushed shard is over the limit, so this write rolls into a new one.
    writer.add_image("train/sample", image(16), 2)
    open_shard = writer.media_path
    light = writer.light_path
    writer.close()

    assert light.parent == logdir
    assert light.name.endswith(TB_LIGHT_SUFFIX)
    assert open_shard != sealed
    assert {sealed, open_shard} <= set(media_files(logdir))


def test_unlocatable_shard_warns_and_writing_continues(logdir: Path):
    writer = TensorBoardWriter(logdir)
    decoy = logdir / "events.out.tfevents.0.decoy.media.0"
    decoy.touch()
    with pytest.warns(UserWarning, match="roll the media shard"):
        writer.add_image("train/sample", image(16), 1)
    writer.add_scalar("train/loss", 0.5, 1)
    writer.close()

    assert read_light(logdir).Tags()["scalars"] == ["train/loss"]
    shards = [p for p in media_files(logdir) if p != decoy]
    assert len(shards) == 1
    assert accumulate_tags(shards[0])["images"] == ["train/sample"]


def test_interleaved_writing_reads_back_as_one_complete_run(logdir: Path):
    writer = TensorBoardWriter(logdir, shard_max_bytes=16 * 1024)
    for step in range(60):
        writer.add_scalar("train/loss", 1.0 / (step + 1), step)
        # Media steps run backwards against the scalars, so the two files'
        # step numbers overlap and restart relative to each other.
        writer.add_image("train/sample", image(), 60 - step)
    writer.close()

    assert len(media_files(logdir)) > 2

    accumulator = accumulate(logdir)
    assert accumulator.Tags()["scalars"] == ["train/loss"]
    assert accumulator.Tags()["images"] == ["train/sample"]
    points = accumulator.Scalars("train/loss")
    assert [point.step for point in points] == list(range(60))


def artifact_names(client, run_id: str) -> list[str]:
    """The event files uploaded under a run's `tb/` artifact directory."""
    infos = client.list_artifacts(run_id, TB_ARTIFACT_DIR)
    return sorted(info.path.split("/")[-1] for info in infos)


def uploaded_shards(client, run_id: str) -> list[str]:
    """The media shards uploaded so far, sealed ones only."""
    return [name for name in artifact_names(client, run_id) if TB_MEDIA_SUFFIX in name]


def wait_for_shards(client, run_id: str, count: int, timeout: float = 20.0):
    """Poll until `count` shards have been uploaded, so no sync tick is raced."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        shards = uploaded_shards(client, run_id)
        if len(shards) >= count:
            return shards
        time.sleep(0.02)
    raise AssertionError(f"only {uploaded_shards(client, run_id)} were uploaded")


def test_sealed_shards_upload_before_the_run_ends(tracking):
    with runsnap.start_run(capture_code=False) as run:
        run_id = run.info.run_id
        with runsnap.tensorboard(
            shard_max_bytes=16 * 1024, sync_interval=0.05
        ) as writer:
            for step in range(40):
                writer.add_image("train/sample", image(), step)
            sealed = wait_for_shards(tracking, run_id, 2)
            assert writer.media_path.name not in sealed


def test_sealed_shard_is_not_uploaded_twice(tracking, monkeypatch):
    uploaded: list[str] = []
    original = MlflowClient.log_artifact

    def spy(self, run_id, local_path, artifact_path=None):
        uploaded.append(Path(local_path).name)
        return original(self, run_id, local_path, artifact_path=artifact_path)

    monkeypatch.setattr(MlflowClient, "log_artifact", spy)

    with (
        runsnap.start_run(capture_code=False) as run,
        runsnap.tensorboard(shard_max_bytes=16 * 1024, sync_interval=0.05) as writer,
    ):
        for step in range(40):
            writer.add_image("train/sample", image(), step)
        sealed = wait_for_shards(tracking, run.info.run_id, 2)
        # Several more ticks pass with the sealed shards unchanged.
        time.sleep(0.3)

    for name in sealed:
        assert uploaded.count(name) == 1


def test_logdir_tag_is_set_on_entry(tracking):
    with (
        runsnap.start_run(capture_code=False) as run,
        runsnap.tensorboard(sync_interval=60.0),
    ):
        tags = tracking.get_run(run.info.run_id).data.tags
        assert tags[TB_TAG_LOGDIR] == TB_ARTIFACT_DIR


def test_local_tags_name_the_live_log_directory(tracking):
    with (
        runsnap.start_run(capture_code=False) as run,
        runsnap.tensorboard(sync_interval=60.0),
    ):
        tags = tracking.get_run(run.info.run_id).data.tags
        assert tags[TB_TAG_LOCAL_HOST] == socket.gethostname()
        local = Path(tags[TB_TAG_LOCAL_DIR])
        assert local.is_absolute()
        assert local.is_dir()
        assert event_files(local)


def test_local_tags_outlive_the_directory(tracking):
    with runsnap.start_run(capture_code=False) as run:
        with runsnap.tensorboard(sync_interval=60.0):
            pass
        tags = tracking.get_run(run.info.run_id).data.tags
        assert tags[TB_TAG_LOCAL_HOST] == socket.gethostname()
        assert not Path(tags[TB_TAG_LOCAL_DIR]).exists()


def wait_for_scalar(logdir: Path, tag: str, timeout: float) -> list[int]:
    """The steps of `tag` once the writer's own flushing has put them on disk.

    Reading is retried because the writer hands events to a worker thread, so
    a scalar reaches the file some time after the call that wrote it returns.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            return [point.step for point in read_light(logdir).Scalars(tag)]
        except KeyError:
            if time.monotonic() > deadline:
                raise AssertionError(f"{tag} never reached {logdir}") from None
            time.sleep(0.05)


def test_refused_tags_warn_and_still_yield_a_writer(tracking, monkeypatch):
    def refuse(*args, **kwargs):
        raise ConnectionError("tags refused")

    def logging_with_refused_tags():
        with runsnap.tensorboard(sync_interval=60.0, flush_secs=0.05) as writer:
            writer.add_scalar("train/loss", 0.5, 1)
            assert wait_for_scalar(Path(writer.logdir), "train/loss", 10.0) == [1]

    monkeypatch.setattr(MlflowClient, "set_tag", refuse)
    with (
        runsnap.start_run(capture_code=False),
        pytest.warns(UserWarning, match="tags refused"),
    ):
        logging_with_refused_tags()


def flush_intervals(writer: TensorBoardWriter) -> set[float]:
    """The flush interval each of the two underlying writers is running with."""
    intervals = set()
    for underlying in (writer._light, writer._media):
        file_writer = underlying.file_writer
        assert isinstance(file_writer, FileWriter)
        intervals.add(file_writer.event_writer._flush_secs)
    return intervals


def test_scalars_reach_disk_without_the_user_flushing(tracking):
    with (
        runsnap.start_run(capture_code=False),
        runsnap.tensorboard(sync_interval=60.0) as writer,
    ):
        assert flush_intervals(writer) == {TB_FLUSH_SECONDS}
        writer.add_scalar("train/loss", 0.5, 1)
        steps = wait_for_scalar(Path(writer.logdir), "train/loss", 3 * TB_FLUSH_SECONDS)
        assert steps == [1]


def test_caller_flush_secs_reaches_the_writers(tracking):
    with (
        runsnap.start_run(capture_code=False),
        runsnap.tensorboard(sync_interval=60.0, flush_secs=60) as writer,
    ):
        assert flush_intervals(writer) == {60}


def uploaded_scalars(client, run_id: str, tmp_path: Path) -> list[int]:
    """The scalar steps TensorBoard reads back from a run's uploaded artifacts."""
    local = tmp_path / run_id
    local.mkdir(parents=True, exist_ok=True)
    downloaded = client.download_artifacts(run_id, TB_ARTIFACT_DIR, str(local))
    accumulator = accumulate(Path(downloaded))
    return [point.step for point in accumulator.Scalars("train/loss")]


def write_some(writer) -> None:
    for step in range(10):
        writer.add_scalar("train/loss", 1.0 / (step + 1), step)
        writer.add_image("train/sample", image(), step)


def test_normal_exit_uploads_everything(tracking, tmp_path):
    with runsnap.start_run(capture_code=False) as run:
        with runsnap.tensorboard(sync_interval=60.0) as writer:
            write_some(writer)
        assert uploaded_scalars(tracking, run.info.run_id, tmp_path) == list(range(10))


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_exit_by_raising_uploads_everything(tracking, tmp_path, failure):
    def interrupted_logging():
        with runsnap.tensorboard(sync_interval=60.0) as writer:
            write_some(writer)
            raise failure("stop")

    with runsnap.start_run(capture_code=False) as run:
        with pytest.raises(failure):
            interrupted_logging()
        assert uploaded_scalars(tracking, run.info.run_id, tmp_path) == list(range(10))


def test_unreachable_tracking_server_does_not_reach_user_code(tracking, monkeypatch):
    def refuse(*args, **kwargs):
        raise ConnectionError("tracking server unreachable")

    def logging_while_unreachable():
        with runsnap.tensorboard(sync_interval=0.05) as writer:
            write_some(writer)
            time.sleep(0.2)
        return True

    monkeypatch.setattr(MlflowClient, "log_artifact", refuse)
    monkeypatch.setattr(MlflowClient, "set_tag", refuse)
    with (
        runsnap.start_run(capture_code=False),
        pytest.warns(UserWarning, match="tracking server unreachable"),
    ):
        reached_the_end = logging_while_unreachable()

    assert reached_the_end


def test_failing_upload_at_exit_does_not_raise(tracking, monkeypatch):
    def logging_with_failed_exit_upload():
        with runsnap.tensorboard(sync_interval=60.0) as writer:
            write_some(writer)
            monkeypatch.setattr(
                MlflowClient,
                "log_artifact",
                lambda *args, **kwargs: (_ for _ in ()).throw(OSError("upload failed")),
            )
        return True

    with (
        runsnap.start_run(capture_code=False),
        pytest.warns(UserWarning, match="upload failed"),
    ):
        reached_the_end = logging_with_failed_exit_upload()

    assert reached_the_end


def test_write_failure_does_not_reach_user_code(logdir):
    writer = TensorBoardWriter(logdir)
    with pytest.warns(UserWarning, match="add_scalar"):
        writer.add_scalar("train/loss", object(), 1)
    with pytest.warns(UserWarning, match="add_image"):
        writer.add_image("train/sample", object(), 1)
    writer.close()


def test_no_active_run_raises_a_named_error(tracking):
    with (
        pytest.raises(RuntimeError, match="active MLflow run"),
        runsnap.tensorboard(),
    ):
        pass
