"""Run selection and foreground TensorBoard launch through the CLI."""

import socket
import subprocess
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from unittest.mock import Mock

import mlflow
import pytest
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

from runsnap import _cli
from runsnap._tags import TAG_CONTINUES, TB_TAG_LOCAL_DIR, TB_TAG_LOCAL_HOST

LIGHT = "events.out.tfevents.1.host.scalars"
MEDIA = "events.out.tfevents.1.host.media.0"


def logged_run(
    client: MlflowClient,
    tmp_path: Path,
    name: str,
    *,
    experiment_id: str | None = None,
    optimizer: str = "adamw",
    continues: str | None = None,
) -> Run:
    with mlflow.start_run(run_name=name, experiment_id=experiment_id) as active:
        source = tmp_path / "source"
        source.mkdir(exist_ok=True)
        (source / LIGHT).write_bytes(b"scalar events")
        (source / MEDIA).write_bytes(b"media events")
        mlflow.log_param("optimizer", optimizer)
        if continues is not None:
            mlflow.set_tag(TAG_CONTINUES, continues)
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

    def launch(logdir: Path, **_options) -> None:
        viewed.update(
            (run.name, {path.name for path in run.iterdir()})
            for run in logdir.iterdir()
        )

    monkeypatch.setattr(_cli, "_launch_tensorboard", launch)
    return viewed


def invoke(monkeypatch: pytest.MonkeyPatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["runsnap", "tb", *args])
    try:
        _cli.main()
    except SystemExit as exc:
        if exc.code != 0:
            raise


def record_calls(
    monkeypatch: pytest.MonkeyPatch, client: MlflowClient, method: str
) -> list[tuple]:
    """The positional arguments of each `method` call on `client`, in call order.

    `client` also becomes the client the CLI queries, so the record covers
    everything one invocation asks the tracking server for.
    """
    original = getattr(client, method)
    calls: list[tuple] = []

    def spy(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(client, method, spy)
    monkeypatch.setattr(_cli, "make_client", lambda uri: client)
    return calls


@pytest.fixture
def fast_polls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the assembled directory re-run a followed selection every 20 ms."""
    monkeypatch.setattr(
        _cli, "assemble_logdir", partial(_cli.assemble_logdir, poll_interval=0.02)
    )


def launch_while(
    monkeypatch: pytest.MonkeyPatch, during: Callable[[Path], None]
) -> dict[str, Path]:
    """Where each link points once `during` returns, called with the open logdir.

    The dict is filled when the launch returns.
    """
    linked: dict[str, Path] = {}

    def launch(logdir: Path, **_options) -> None:
        during(logdir)
        linked.update((link.name, link.resolve()) for link in logdir.iterdir())

    monkeypatch.setattr(_cli, "_launch_tensorboard", launch)
    return linked


def tag_live(client: MlflowClient, run: Run, local: Path) -> None:
    """Tag `run` as writing into `local` on this host, as its writer would."""
    local.mkdir()
    (local / LIGHT).write_bytes(b"live events")
    client.set_tag(run.info.run_id, TB_TAG_LOCAL_HOST, socket.gethostname())
    client.set_tag(run.info.run_id, TB_TAG_LOCAL_DIR, str(local))


def wait_until(condition: Callable[[], bool], timeout: float = 10.0) -> bool:
    """Whether `condition` came true within `timeout` seconds of polling."""
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


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
    args = ["--experiment", "runsnap-tests"]
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

    invoke(monkeypatch, "twin", "--experiment", "runsnap-tests")

    assert set(viewer) == {"twin"}
    assert {path.name for path in (cache / "runsnap" / "tensorboard").iterdir()} == {
        wanted.info.run_id
    }


def test_several_names_enumerate_experiments_once(
    tracking, tmp_path, cache, viewer, monkeypatch
):
    logged_run(tracking, tmp_path, "first")
    logged_run(tracking, tmp_path, "second")
    logged_run(tracking, tmp_path, "third")
    enumerations = record_calls(monkeypatch, tracking, "search_experiments")

    invoke(monkeypatch, "first", "second", "third")

    assert set(viewer) == {"first", "second", "third"}
    assert len(enumerations) == 1


@pytest.mark.parametrize(
    "selection", [("baseline",), ("--filter", "params.optimizer = 'adamw'")]
)
def test_a_named_experiment_is_resolved_without_enumerating_them(
    tracking, tmp_path, cache, viewer, monkeypatch, selection
):
    logged_run(tracking, tmp_path, "baseline")
    enumerations = record_calls(monkeypatch, tracking, "search_experiments")

    invoke(monkeypatch, *selection, "--experiment", "runsnap-tests")

    assert set(viewer) == {"baseline"}
    assert enumerations == []


def test_each_followed_pass_enumerates_experiments_once(
    tracking, tmp_path, cache, fast_polls, monkeypatch
):
    logged_run(tracking, tmp_path, "finished")
    enumerations = record_calls(monkeypatch, tracking, "search_experiments")
    searches = record_calls(monkeypatch, tracking, "search_runs")

    def start_training(logdir: Path) -> None:
        training = logged_run(tracking, tmp_path, "training")
        tag_live(tracking, training, tmp_path / "local")
        assert wait_until((logdir / "training").exists)

    launch_while(monkeypatch, start_training)

    invoke(monkeypatch)

    # One page of runs per pass, so each search is one pass: startup's or a
    # followed one.
    assert len(searches) >= 2
    assert len(enumerations) == len(searches)


def test_chain_pulls_in_earlier_attempts(
    tracking, tmp_path, cache, viewer, monkeypatch
):
    first = logged_run(tracking, tmp_path, "attempt-1")
    second = logged_run(tracking, tmp_path, "attempt-2", continues=first.info.run_id)
    logged_run(tracking, tmp_path, "attempt-3", continues=second.info.run_id)
    logged_run(tracking, tmp_path, "unrelated")

    invoke(monkeypatch, "attempt-3", "--chain")

    assert set(viewer) == {"attempt-1", "attempt-2", "attempt-3"}


def test_chain_fetches_each_attempt_once(
    tracking, tmp_path, cache, viewer, monkeypatch
):
    first = logged_run(tracking, tmp_path, "attempt-1")
    second = logged_run(tracking, tmp_path, "attempt-2", continues=first.info.run_id)
    third = logged_run(tracking, tmp_path, "attempt-3", continues=second.info.run_id)
    fetched = record_calls(monkeypatch, tracking, "get_run")

    invoke(monkeypatch, "attempt-3", "--chain")

    assert set(viewer) == {"attempt-1", "attempt-2", "attempt-3"}
    assert sorted(run_id for (run_id,) in fetched) == sorted(
        run.info.run_id for run in (first, second, third)
    )


def test_chain_shows_a_live_run_beside_a_cached_attempt(
    tracking, tmp_path, cache, viewer, monkeypatch
):
    first = logged_run(tracking, tmp_path, "attempt-1")
    second = logged_run(tracking, tmp_path, "attempt-2", continues=first.info.run_id)
    local = tmp_path / "live"
    local.mkdir()
    (local / "events.out.tfevents.2.host.media.0").write_bytes(b"live events")
    tracking.set_tag(second.info.run_id, TB_TAG_LOCAL_HOST, socket.gethostname())
    tracking.set_tag(second.info.run_id, TB_TAG_LOCAL_DIR, str(local))

    invoke(monkeypatch, "attempt-2", "--chain")

    assert viewer == {
        "attempt-1": {LIGHT},
        "attempt-2": {"events.out.tfevents.2.host.media.0"},
    }


def test_chain_extends_a_filtered_selection(
    tracking, tmp_path, cache, viewer, monkeypatch
):
    first = logged_run(tracking, tmp_path, "attempt-1")
    tracking.set_terminated(first.info.run_id, "KILLED")
    logged_run(tracking, tmp_path, "attempt-2", continues=first.info.run_id)

    invoke(
        monkeypatch,
        "--experiment",
        "runsnap-tests",
        "--filter",
        "attributes.status = 'FINISHED'",
        "--chain",
    )

    assert set(viewer) == {"attempt-1", "attempt-2"}


def test_without_the_flag_only_the_named_run_is_shown(
    tracking, tmp_path, cache, viewer, monkeypatch
):
    first = logged_run(tracking, tmp_path, "attempt-1")
    logged_run(tracking, tmp_path, "attempt-2", continues=first.info.run_id)

    invoke(monkeypatch, "attempt-2")

    assert set(viewer) == {"attempt-2"}


@pytest.mark.parametrize("new_experiment", [False, True])
def test_a_query_adds_a_run_that_starts_after_launch(
    tracking, tmp_path, cache, fast_polls, monkeypatch, new_experiment
):
    logged_run(tracking, tmp_path, "finished")
    local = tmp_path / "local"

    def start_training(logdir: Path) -> None:
        experiment_id = (
            tracking.create_experiment(
                "later", artifact_location=(tmp_path / "later").as_uri()
            )
            if new_experiment
            else None
        )
        run = logged_run(tracking, tmp_path, "training", experiment_id=experiment_id)
        tag_live(tracking, run, local)
        assert wait_until((logdir / "training").exists)

    linked = launch_while(monkeypatch, start_training)

    invoke(monkeypatch)

    assert set(linked) == {"finished", "training"}
    assert linked["training"] == local.resolve()


def test_a_followed_experiment_ignores_a_new_run_elsewhere(
    tracking, tmp_path, cache, fast_polls, monkeypatch
):
    logged_run(tracking, tmp_path, "finished")
    other = tracking.create_experiment(
        "elsewhere", artifact_location=(tmp_path / "elsewhere").as_uri()
    )

    def start_both(logdir: Path) -> None:
        outside = logged_run(tracking, tmp_path, "outside", experiment_id=other)
        tag_live(tracking, outside, tmp_path / "outside")
        # Tagged after the outside run, so a pass that links it saw both.
        tag_live(tracking, logged_run(tracking, tmp_path, "inside"), tmp_path / "in")
        assert wait_until((logdir / "inside").exists)

    linked = launch_while(monkeypatch, start_both)

    invoke(monkeypatch, "--experiment", "runsnap-tests")

    assert set(linked) == {"finished", "inside"}


def test_a_named_selection_adds_no_new_run(
    tracking, tmp_path, cache, fast_polls, monkeypatch
):
    logged_run(tracking, tmp_path, "baseline")

    def start_training(logdir: Path) -> None:
        training = logged_run(tracking, tmp_path, "training")
        tag_live(tracking, training, tmp_path / "local")
        time.sleep(0.3)  # a dozen or so polls

    linked = launch_while(monkeypatch, start_training)

    invoke(monkeypatch, "baseline")

    assert set(linked) == {"baseline"}


@pytest.mark.parametrize("selection", ["empty", "no-events", "filter"])
def test_no_matching_data_does_not_launch(
    tracking, tmp_path, cache, monkeypatch, capsys, selection
):
    args = ["--experiment", "runsnap-tests"]
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


@pytest.mark.parametrize(
    ("options", "forwarded"),
    [
        ([], []),
        (["--bind_all"], ["--bind_all"]),
        (["--bind-all"], ["--bind_all"]),
        (["--host", "127.0.0.1"], ["--host", "127.0.0.1"]),
        (["--port", "6007"], ["--port", "6007"]),
        (["--port", "0"], ["--port", "0"]),
        (["--port", "default"], ["--port", "default"]),
        (["--bind_all", "--port", "6007"], ["--bind_all", "--port", "6007"]),
        (
            ["--host=127.0.0.1", "--port=6008"],
            ["--port", "6008", "--host", "127.0.0.1"],
        ),
    ],
)
def test_launch_forwards_options_and_keeps_logdir_until_subprocess_exits(
    tracking, tmp_path, cache, monkeypatch, capsys, options, forwarded
):
    logged_run(tracking, tmp_path, "baseline")
    directories = []

    def run(args, *, check):
        assert args[:4] == [sys.executable, "-m", "tensorboard.main", "--logdir"]
        assert args[5:] == forwarded
        assert check is True
        logdir = Path(args[4])
        directories.append(logdir)
        assert (logdir / "baseline" / LIGHT).read_bytes() == b"scalar events"
        print("TensorBoard at http://localhost:6006/")

    monkeypatch.setattr(_cli.subprocess, "run", run)

    invoke(monkeypatch, "baseline", *options)

    assert len(directories) == 1
    assert not directories[0].exists()
    assert "http://localhost:6006/" in capsys.readouterr().out
    assert list(cache.rglob(LIGHT))


@pytest.mark.parametrize("bind_flag", ["--bind_all", "--bind-all"])
def test_host_and_bind_all_conflict_before_querying(monkeypatch, capsys, bind_flag):
    client = Mock()
    monkeypatch.setattr(_cli, "make_client", client)

    with pytest.raises(SystemExit) as raised:
        invoke(monkeypatch, "baseline", bind_flag, "--host", "127.0.0.1")

    assert raised.value.code == 1
    assert "Cannot combine --bind_all with --host" in capsys.readouterr().err
    client.assert_not_called()


def test_invalid_port_is_rejected_before_querying(monkeypatch, capsys):
    client = Mock()
    monkeypatch.setattr(_cli, "make_client", client)

    with pytest.raises(SystemExit) as raised:
        invoke(monkeypatch, "baseline", "--port", "invalid")

    assert raised.value.code != 0
    assert "port" in capsys.readouterr().err.lower()
    client.assert_not_called()


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
