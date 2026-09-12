"""How a run ended and which attempt it continues."""

import warnings
from types import TracebackType
from typing import Literal

import mlflow
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

from runsnap._tags import TAG_CONTINUES, TAG_FAILURE_CAUSE

KILLED = "KILLED"


class LifecycleRun(mlflow.ActiveRun):
    """An MLflow run that records how its `with` block ended.

    A `KeyboardInterrupt` ends the run as `KILLED`; every other exception
    leaves it `FAILED` as MLflow itself would. Either way the cause is
    recorded as a tag and the exception propagates untouched.
    """

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        if exc_val is not None:
            record_cause(self.info.run_id, exc_val)
        if isinstance(exc_val, KeyboardInterrupt) and _is_active(self.info.run_id):
            mlflow.end_run(KILLED)
        else:
            super().__exit__(exc_type, exc_val, exc_tb)
        return False


def record_cause(run_id: str, exc: BaseException) -> None:
    """Tag `run_id` with the exception that ended it.

    A tracking store that refuses the tag is reported as a warning: the run
    still has to end with the right status and the exception still has to
    reach the caller.
    """
    try:
        MlflowClient().set_tag(
            run_id, TAG_FAILURE_CAUSE, f"{type(exc).__qualname__}: {exc}"
        )
    except Exception as tag_error:  # noqa: BLE001 - never replace the exception
        warnings.warn(
            f"runsnap could not record the failure cause: {tag_error}", stacklevel=3
        )


def record_continues(run_id: str, continues: str) -> None:
    """Tag `run_id` with the id of the attempt it continues.

    A tracking store that refuses the tag is reported as a warning: the run
    has already started and the caller's training loop must go on.
    """
    try:
        MlflowClient().set_tag(run_id, TAG_CONTINUES, continues)
    except Exception as tag_error:  # noqa: BLE001 - never raise into the run
        warnings.warn(
            f"runsnap could not record the continued attempt: {tag_error}",
            stacklevel=3,
        )


def attempt_runs(client: MlflowClient, run_id: str) -> list[Run]:
    """The run `run_id` and every attempt it continues, newest attempt first.

    Each run in the chain is fetched once, so a caller that needs the runs
    themselves does not have to fetch them again.

    Raises `ValueError` when the chain revisits a run, and lets MLflow's own
    error through when a recorded predecessor is not on the tracking server.
    """
    chain: list[Run] = []
    seen: set[str] = set()
    current: str | None = run_id
    while current is not None:
        if current in seen:
            raise ValueError(f"the chain of attempts revisits run {current}")
        seen.add(current)
        run = client.get_run(current)
        chain.append(run)
        current = run.data.tags.get(TAG_CONTINUES)
    return chain


def attempt_chain(run_id: str, client: MlflowClient | None = None) -> list[str]:
    """`run_id` and every attempt it continues, newest attempt first.

    `client` names the tracking server to walk, defaulting to the one the
    MLflow environment points at.

    Raises `ValueError` when the chain revisits a run, and lets MLflow's own
    error through when a recorded predecessor is not on the tracking server.
    """
    return [run.info.run_id for run in attempt_runs(client or MlflowClient(), run_id)]


def _is_active(run_id: str) -> bool:
    """Whether `run_id` is the run MLflow would end next."""
    active = mlflow.active_run()
    return active is not None and active.info.run_id == run_id
