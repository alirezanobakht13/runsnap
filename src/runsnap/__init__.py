"""Reproducible MLflow runs: record the code that produced them."""

from typing import Any

import mlflow

from runsnap._capture import capture, capture_enabled
from runsnap._lifecycle import LifecycleRun, attempt_chain, record_continues
from runsnap._metrics import flatten_metrics
from runsnap._params import load_params, log_params
from runsnap._tensorboard import tensorboard

__all__ = [
    "attempt_chain",
    "flatten_metrics",
    "load_params",
    "log_params",
    "start_run",
    "tensorboard",
]


def start_run(
    *args: Any,
    capture_code: bool = True,
    continues: str | None = None,
    **kwargs: Any,
) -> LifecycleRun:
    """Start an MLflow run and record the code state that produced it.

    Arguments are forwarded unchanged to `mlflow.start_run()` and the run it
    creates comes back wrapped in an object exposing the same `ActiveRun`
    interface, so everything downstream is stock MLflow. The wrapper records
    how the `with` block ended: `KILLED` on a `KeyboardInterrupt`, `FAILED`
    with a `runsnap.failure.cause` tag on any other exception.

    `continues` names the id of the attempt this run resumes and is recorded
    as the `runsnap.continues` tag.

    Capture is skipped when `capture_code` is false or the environment sets
    `RUNSNAP_CAPTURE_CODE=0`, and never raises into the caller.
    """
    run = LifecycleRun(mlflow.start_run(*args, **kwargs))
    if continues is not None:
        record_continues(run.info.run_id, continues)
    if capture_code and capture_enabled():
        capture(run)
    return run
