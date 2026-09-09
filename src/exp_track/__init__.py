"""Reproducible MLflow runs: record the code that produced them."""

from typing import Any

import mlflow

from exp_track._capture import capture, capture_enabled
from exp_track._params import load_params, log_params

__all__ = ["load_params", "log_params", "start_run"]


def start_run(*args: Any, capture_code: bool = True, **kwargs: Any) -> Any:
    """Start an MLflow run and record the code state that produced it.

    Arguments are forwarded unchanged to `mlflow.start_run()` and its return
    value — an `ActiveRun`, usable as a context manager — is returned as is, so
    everything downstream is stock MLflow.

    Capture is skipped when `capture_code` is false or the environment sets
    `EXP_TRACK_CAPTURE_CODE=0`, and never raises into the caller.
    """
    run = mlflow.start_run(*args, **kwargs)
    if capture_code and capture_enabled():
        capture(run)
    return run
