"""Run selection and foreground TensorBoard launch through the CLI."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import mlflow
import pytest
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

from exp_track import _cli

LIGHT = "events.out.tfevents.1.host.scalars"
MEDIA = "events.out.tfevents.1.host.media.0"


def logged_run(
    client: MlflowClient,
    tmp_path: Path,
    name: str,
    *,
    experiment_id: str | None = None,
    optimizer: str = "adamw",
) -> Run:
    with mlflow.start_run(run_name=name, experiment_id=experiment_id) as active:
        source = tmp_path / "source"
        source.mkdir(exist_ok=True)
        (source / LIGHT).write_bytes(b"scalar events")
        (source / MEDIA).write_bytes(b"media events")
        mlflow.log_param("optimizer", optimizer)
        client.log_artifacts(active.info.run_id, str(source), "tb")
        return client.get_run(active.info.run_id)


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(path))
    return path


@pytest.fixture
def viewer(monkeypatch: pytest.MonkeyPatch) -> dict[str, set[str]]:
    viewed = {}

    def launch(logdir: Path) -> None:
        viewed.update(
            (run.name, {path.name for path in run.iterdir()})
            for run in logdir.iterdir()
        )

    monkeypatch.setattr(_cli, "_launch_tensorboard", launch)
    return viewed


def invoke(monkeypatch: pytest.MonkeyPatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["exp-track", "tb", *args])
    try:
        _cli.main()
    except SystemExit as exc:
        if exc.code != 0:
            raise


def test_explicit_ids_and_names(tracking, tmp_path, cache, viewer, monkeypatch):
    first = logged_run(tracking, tmp_path, "baseline")
    logged_run(tracking, tmp_path, "ablation")
    logged_run(tracking, tmp_path, "unselected")

    invoke(monkeypatch, first.info.run_id, "ablation")

    assert viewer == {"baseline": {LIGHT}, "ablation": {LIGHT}}


@pytest.mark.parametrize("filtered", [False, True])
def test_experiment_selection(tracking, tmp_path, cache, viewer, monkeypatch, filtered):
    logged_run(tracking, tmp_path, "adamw")
    logged_run(tracking, tmp_path, "sgd", optimizer="sgd")
    other = tracking.create_experiment(
        "elsewhere", artifact_location=(tmp_path / "elsewhere").as_uri()
    )
    logged_run(tracking, tmp_path, "elsewhere", experiment_id=other)
    with mlflow.start_run(run_name="no-events"):
        pass
    args = ["--experiment", "exp-track-tests"]
    if filtered:
        args.extend(["--filter", "params.optimizer = 'adamw'"])

    invoke(monkeypatch, *args)

    assert set(viewer) == ({"adamw"} if filtered else {"adamw", "sgd"})


def test_filter_without_experiment(tracking, tmp_path, cache, viewer, monkeypatch):
    logged_run(tracking, tmp_path, "first")
    other = tracking.create_experiment(
        "elsewhere", artifact_location=(tmp_path / "elsewhere").as_uri()
    )
    logged_run(tracking, tmp_path, "second", experiment_id=other)
    logged_run(tracking, tmp_path, "sgd", optimizer="sgd")

    invoke(monkeypatch, "--filter", "params.optimizer = 'adamw'")

    assert set(viewer) == {"first", "second"}


def test_filter_narrows_explicit_runs(tracking, tmp_path, cache, viewer, monkeypatch):
    logged_run(tracking, tmp_path, "adamw")
    logged_run(tracking, tmp_path, "sgd", optimizer="sgd")
    logged_run(tracking, tmp_path, "unselected")

    invoke(monkeypatch, "adamw", "sgd", "--filter", "params.optimizer = 'adamw'")

    assert set(viewer) == {"adamw"}


def test_experiment_resolves_duplicate_names(
    tracking, tmp_path, cache, viewer, monkeypatch
):
    wanted = logged_run(tracking, tmp_path, "twin")
    other = tracking.create_experiment(
        "elsewhere", artifact_location=(tmp_path / "elsewhere").as_uri()
    )
    logged_run(tracking, tmp_path, "twin", experiment_id=other)

    invoke(monkeypatch, "twin", "--experiment", "exp-track-tests")

    assert set(viewer) == {"twin"}
    assert {path.name for path in (cache / "exp-track" / "tensorboard").iterdir()} == {
        wanted.info.run_id
    }


@pytest.mark.parametrize("selection", ["empty", "no-events", "filter"])
def test_no_matching_data_does_not_launch(
    tracking, tmp_path, cache, monkeypatch, capsys, selection
):
    args = ["--experiment", "exp-track-tests"]
    if selection == "no-events":
        with mlflow.start_run(run_name="no-events"):
            pass
        args = ["no-events"]
    elif selection == "filter":
        logged_run(tracking, tmp_path, "baseline")
        args.extend(["--filter", "params.optimizer = 'missing'"])
    launch = Mock()
    monkeypatch.setattr(_cli, "_launch_tensorboard", launch)

    invoke(monkeypatch, *args)

    launch.assert_not_called()
    assert "No runs found with TensorBoard data" in capsys.readouterr().out


def test_media_opt_in(tracking, tmp_path, cache, viewer, monkeypatch):
    logged_run(tracking, tmp_path, "baseline")

    invoke(monkeypatch, "baseline", "--media")

    assert viewer == {"baseline": {LIGHT, MEDIA}}


def test_search_follows_all_pages(tracking, tmp_path, cache, viewer, monkeypatch):
    logged_run(tracking, tmp_path, "first")
    logged_run(tracking, tmp_path, "second")
    other = tracking.create_experiment(
        "elsewhere", artifact_location=(tmp_path / "elsewhere").as_uri()
    )
    logged_run(tracking, tmp_path, "third", experiment_id=other)
    search_runs = tracking.search_runs
    search_experiments = tracking.search_experiments

    def runs(*args, **kwargs):
        return search_runs(*args, **kwargs, max_results=1)

    def experiments(**kwargs):
        return search_experiments(**kwargs, max_results=1)

    monkeypatch.setattr(tracking, "search_runs", runs)
    monkeypatch.setattr(tracking, "search_experiments", experiments)
    monkeypatch.setattr(_cli, "make_client", lambda uri: tracking)

    invoke(monkeypatch, "--filter", "params.optimizer = 'adamw'")

    assert set(viewer) == {"first", "second", "third"}


def test_tracking_uri_override(tracking, tmp_path, cache, viewer, monkeypatch):
    logged_run(tracking, tmp_path, "baseline")
    uri = tracking.tracking_uri
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'other.db'}")

    invoke(monkeypatch, "baseline", "--tracking-uri", uri)

    assert set(viewer) == {"baseline"}


def test_launch_keeps_logdir_until_subprocess_exits(
    tracking, tmp_path, cache, monkeypatch, capsys
):
    logged_run(tracking, tmp_path, "baseline")
    directories = []

    def run(args, *, check):
        assert args[:4] == [sys.executable, "-m", "tensorboard.main", "--logdir"]
        assert check is True
        logdir = Path(args[4])
        directories.append(logdir)
        assert (logdir / "baseline" / LIGHT).read_bytes() == b"scalar events"
        print("TensorBoard at http://localhost:6006/")

    monkeypatch.setattr(_cli.subprocess, "run", run)

    invoke(monkeypatch, "baseline")

    assert len(directories) == 1
    assert not directories[0].exists()
    assert "http://localhost:6006/" in capsys.readouterr().out
    assert list(cache.rglob(LIGHT))


def test_missing_tensorboard_is_actionable(
    tracking, tmp_path, cache, monkeypatch, capsys
):
    logged_run(tracking, tmp_path, "baseline")
    monkeypatch.setattr(_cli, "find_spec", lambda name: None)
    launch = Mock()
    monkeypatch.setattr(_cli.subprocess, "run", launch)

    with pytest.raises(SystemExit) as raised:
        invoke(monkeypatch, "baseline")

    assert raised.value.code == 1
    message = capsys.readouterr().err
    assert "TensorBoard is required" in message
    assert "uv add tensorboard" in message
    assert "Traceback" not in message
    launch.assert_not_called()


@pytest.mark.parametrize("interrupted", [False, True])
def test_viewer_failure_or_interrupt_cleans_up(
    tracking, tmp_path, cache, monkeypatch, capsys, interrupted
):
    logged_run(tracking, tmp_path, "baseline")
    directories = []

    def run(args, *, check):
        directories.append(Path(args[4]))
        if interrupted:
            raise KeyboardInterrupt
        raise subprocess.CalledProcessError(2, args)

    monkeypatch.setattr(_cli.subprocess, "run", run)

    if interrupted:
        invoke(monkeypatch, "baseline")
    else:
        with pytest.raises(SystemExit) as raised:
            invoke(monkeypatch, "baseline")
        assert raised.value.code == 1
        assert "TensorBoard could not run" in capsys.readouterr().err

    assert len(directories) == 1
    assert not directories[0].exists()
