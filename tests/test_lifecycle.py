"""Tests for how a run records the way its block ended."""

import mlflow
import pytest

import runsnap
from runsnap import _lifecycle
from runsnap._capture import MLFLOW_PARENT_RUN_ID
from runsnap._lifecycle import LifecycleRun
from runsnap._tags import TAG_CONTINUES, TAG_FAILURE_CAUSE


def status_and_tags(client: mlflow.MlflowClient, run_id: str):
    run = client.get_run(run_id)
    return run.info.status, run.data.tags


def test_normal_exit_finishes_without_a_cause(tracking):
    with LifecycleRun(mlflow.start_run()) as run:
        run_id = run.info.run_id

    status, recorded = status_and_tags(tracking, run_id)
    assert status == "FINISHED"
    assert TAG_FAILURE_CAUSE not in recorded


def test_exception_fails_the_run_and_names_the_cause(tracking):
    run = LifecycleRun(mlflow.start_run())
    run_id = run.info.run_id

    with pytest.raises(FloatingPointError), run:
        raise FloatingPointError("nonfinite update 7")

    status, recorded = status_and_tags(tracking, run_id)
    assert status == "FAILED"
    assert recorded[TAG_FAILURE_CAUSE] == "FloatingPointError: nonfinite update 7"


def test_keyboard_interrupt_kills_the_run(tracking):
    run = LifecycleRun(mlflow.start_run())
    run_id = run.info.run_id

    with pytest.raises(KeyboardInterrupt), run:
        raise KeyboardInterrupt

    status, recorded = status_and_tags(tracking, run_id)
    assert status == "KILLED"
    assert recorded[TAG_FAILURE_CAUSE].startswith("KeyboardInterrupt")


def test_refused_cause_tag_warns_and_still_ends_the_run(tracking, monkeypatch):
    class Refusing:
        def set_tag(self, *args, **kwargs):
            raise RuntimeError("tag refused")

    monkeypatch.setattr(_lifecycle, "MlflowClient", Refusing)
    run = LifecycleRun(mlflow.start_run())
    run_id = run.info.run_id

    with (
        pytest.raises(FloatingPointError),
        pytest.warns(UserWarning, match="tag refused"),
        run,
    ):
        raise FloatingPointError("boom")

    status, recorded = status_and_tags(tracking, run_id)
    assert status == "FAILED"
    assert TAG_FAILURE_CAUSE not in recorded


def test_continues_records_the_earlier_attempt(tracking):
    with runsnap.start_run(capture_code=False) as first:
        first_id = first.info.run_id

    with runsnap.start_run(capture_code=False, continues=first_id) as second:
        second_id = second.info.run_id

    assert tracking.get_run(second_id).data.tags[TAG_CONTINUES] == first_id


def test_refused_continues_tag_warns_and_still_starts_the_run(tracking, monkeypatch):
    class Refusing:
        def set_tag(self, *args, **kwargs):
            raise RuntimeError("tag refused")

    monkeypatch.setattr(_lifecycle, "MlflowClient", Refusing)

    with (
        pytest.warns(UserWarning, match="tag refused"),
        runsnap.start_run(capture_code=False, continues="0" * 32) as run,
    ):
        run_id = run.info.run_id

    status, recorded = status_and_tags(tracking, run_id)
    assert status == "FINISHED"
    assert TAG_CONTINUES not in recorded


def test_fresh_attempt_records_no_predecessor(tracking):
    with runsnap.start_run(capture_code=False) as run:
        run_id = run.info.run_id

    assert TAG_CONTINUES not in tracking.get_run(run_id).data.tags


def test_a_nested_run_can_also_continue_an_attempt(tracking):
    with runsnap.start_run(capture_code=False) as first:
        first_id = first.info.run_id

    with runsnap.start_run(capture_code=False) as parent:
        parent_id = parent.info.run_id
        with runsnap.start_run(
            capture_code=False, nested=True, continues=first_id
        ) as child:
            child_id = child.info.run_id

    recorded = tracking.get_run(child_id).data.tags
    assert recorded[MLFLOW_PARENT_RUN_ID] == parent_id
    assert recorded[TAG_CONTINUES] == first_id


def test_the_wrapped_run_is_the_active_one(tracking):
    with runsnap.start_run(capture_code=False) as run:
        active = mlflow.active_run()
        assert active is not None
        assert active.info.run_id == run.info.run_id

    assert mlflow.active_run() is None


def test_a_run_without_a_block_ends_through_mlflow(tracking):
    run = runsnap.start_run(capture_code=False)
    run_id = run.info.run_id
    mlflow.end_run()

    status, recorded = status_and_tags(tracking, run_id)
    assert status == "FINISHED"
    assert TAG_FAILURE_CAUSE not in recorded


def attempt(continues: str | None = None) -> str:
    """A finished run continuing `continues`, as its id."""
    with runsnap.start_run(capture_code=False, continues=continues) as run:
        return run.info.run_id


def test_chain_walks_back_to_the_first_attempt(tracking):
    first = attempt()
    second = attempt(continues=first)
    third = attempt(continues=second)

    assert runsnap.attempt_chain(third) == [third, second, first]


def test_chain_of_a_first_attempt_is_the_run_itself(tracking):
    first = attempt()

    assert runsnap.attempt_chain(first) == [first]


def test_chain_that_revisits_a_run_is_refused(tracking):
    first = attempt()
    second = attempt(continues=first)
    tracking.set_tag(first, TAG_CONTINUES, second)

    with pytest.raises(ValueError, match=second):
        runsnap.attempt_chain(second)


def test_chain_naming_an_unknown_run_reports_it(tracking):
    missing = "0" * 32
    run_id = attempt(continues=missing)

    with pytest.raises(mlflow.MlflowException, match=missing):
        runsnap.attempt_chain(run_id)


def test_a_caught_child_failure_leaves_the_parent_finished(tracking):
    with runsnap.start_run(capture_code=False) as parent:
        parent_id = parent.info.run_id
        try:
            with runsnap.start_run(capture_code=False, nested=True) as child:
                child_id = child.info.run_id
                raise FloatingPointError("nonfinite update 7")
        except FloatingPointError:
            pass

    child_status, child_tags = status_and_tags(tracking, child_id)
    assert child_status == "FAILED"
    assert child_tags[TAG_FAILURE_CAUSE] == "FloatingPointError: nonfinite update 7"

    parent_status, parent_tags = status_and_tags(tracking, parent_id)
    assert parent_status == "FINISHED"
    assert TAG_FAILURE_CAUSE not in parent_tags
