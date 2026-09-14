"""Pydantic hyperparameter flattening, artifact write and read."""

import json
import tempfile
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mlflow
from mlflow.entities import Param
from mlflow.tracking import MlflowClient
from mlflow.utils.validation import (
    MAX_PARAM_VAL_LENGTH,
    MAX_PARAMS_TAGS_PER_BATCH,
)
from pydantic import BaseModel

from runsnap._tags import (
    HPARAMS_CLASS_KEY,
    HPARAMS_DATA_KEY,
    HPARAMS_DEFAULT_NAME,
    hparams_artifact_path,
)


def flatten_model(model: BaseModel, prefix: str = "") -> dict[str, str]:
    """Flatten a model into one MLflow param per scalar leaf.

    Nested models and mappings become dotted keys. Strings are logged verbatim
    and every other leaf as its JSON representation, so `lr` reads `0.001`,
    `use_amp` reads `true` and `limit` reads `null`. Sequences are single leaves
    rather than indexed keys, which keeps runs holding different-length
    sequences comparable in MLflow's run table, and empty containers are leaves
    too so the field does not silently vanish. A model with no fields has no
    leaves and yields no params, with or without a prefix.
    """
    return {key: _encode(value) for key, value in _leaves(model, prefix).items()}


def _leaves(model: BaseModel, prefix: str) -> dict[str, Any]:
    """Every leaf of a model under its dotted key, holding its JSON value."""
    return _flatten_fields(model.model_dump(mode="json"), prefix)


def _flatten_fields(fields: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in fields.items():
        flat.update(_flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    return flat


def _flatten(value: Any, prefix: str) -> dict[str, Any]:
    if isinstance(value, Mapping) and value:
        return _flatten_fields(value, prefix)
    return {prefix: value}


def _encode(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value)


type SessionValue = bool | int | float | str

_sessions: dict[str, dict[str, SessionValue]] = {}
"""Each run's TensorBoard session values, as logged in this process."""

_claimed: set[str] = set()
"""Runs whose TensorBoard session has been taken by a writer in this process."""


def claim_session(run_id: str) -> dict[str, SessionValue]:
    """The session values logged to a run, marking the run as claimed.

    The run counts as claimed whether or not anything was logged to it, so a
    later `log_params()` on the run warns that its values miss the session.
    """
    _claimed.add(run_id)
    return dict(_sessions.get(run_id, {}))


def _session_values(leaves: Mapping[str, Any]) -> dict[str, SessionValue]:
    """A run's TensorBoard session values from its leaves.

    Booleans, numbers and strings are kept as they are, `None` is left out, and
    the rest become the JSON text of their MLflow param.
    """
    return {
        key: value if isinstance(value, bool | int | float | str) else _encode(value)
        for key, value in leaves.items()
        if value is not None
    }


def log_params(
    model: BaseModel,
    *,
    name: str = HPARAMS_DEFAULT_NAME,
    prefix: str = "",
    run_id: str | None = None,
) -> None:
    """Log a Pydantic hyperparameter model to a run.

    Fields are flattened into one param per scalar leaf under `prefix`, for the
    MLflow UI and for search, and the model's own serialization is written to
    `hparams/<name>.json` alongside its class's fully qualified name. That
    artifact is the authoritative record: it keeps the fidelity the params
    encoding loses, and `load_params()` reads it back into the model class.

    Params go to the tracking server in batches, so a model of any size costs
    one request per hundred leaves rather than one per leaf.

    The leaves are also recorded in this process as the run's TensorBoard
    session values; params logged after `runsnap.tensorboard()` was entered
    for the run miss its session and warn.

    Logs to the active run unless `run_id` names another one.
    """
    if not isinstance(model, BaseModel):
        raise TypeError(
            f"log_params() expects a pydantic BaseModel instance, "
            f"got {type(model).__name__}"
        )
    client = MlflowClient()
    target = _resolve_run_id(run_id)
    leaves = _leaves(model, prefix)
    params: list[Param] = []
    for key, leaf in leaves.items():
        value = _encode(leaf)
        if len(value) > MAX_PARAM_VAL_LENGTH:
            warnings.warn(
                f"param {key!r} is {len(value)} characters and MLflow truncates "
                f"past {MAX_PARAM_VAL_LENGTH}; the complete value is in "
                f"{hparams_artifact_path(name)}",
                stacklevel=2,
            )
        params.append(Param(key, value))
    for start in range(0, len(params), MAX_PARAMS_TAGS_PER_BATCH):
        batch = params[start : start + MAX_PARAMS_TAGS_PER_BATCH]
        client.log_batch(target, params=batch)
    client.log_dict(
        target,
        {
            HPARAMS_CLASS_KEY: _qualified_name(type(model)),
            HPARAMS_DATA_KEY: model.model_dump(mode="json"),
        },
        hparams_artifact_path(name),
    )
    _sessions.setdefault(target, {}).update(_session_values(leaves))
    if target in _claimed:
        warnings.warn(
            f"{hparams_artifact_path(name)} was logged to run {target} after "
            f"runsnap.tensorboard() was entered, so its values are missing from "
            f"the run's TensorBoard hyperparameters; log params before entering "
            f"runsnap.tensorboard()",
            stacklevel=2,
        )


def load_params[M: BaseModel](
    run_id: str, model_class: type[M], *, name: str = HPARAMS_DEFAULT_NAME
) -> M:
    """Reconstruct a logged hyperparameter model from a run.

    The artifact records the class it was logged from; a different class here
    warns and validation is attempted anyway, so a class that moved or was
    renamed still loads.
    """
    payload = _read_artifact(run_id, name)
    recorded = payload.get(HPARAMS_CLASS_KEY)
    expected = _qualified_name(model_class)
    if recorded != expected:
        warnings.warn(
            f"{hparams_artifact_path(name)} on run {run_id} was logged from "
            f"{recorded}, loading as {expected}",
            stacklevel=2,
        )
    return model_class.model_validate(payload[HPARAMS_DATA_KEY])


def _read_artifact(run_id: str, name: str) -> dict[str, Any]:
    path = hparams_artifact_path(name)
    client = MlflowClient()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            local = client.download_artifacts(run_id, path, tmp)
        except Exception as exc:
            raise FileNotFoundError(f"run {run_id} has no artifact {path}") from exc
        return json.loads(Path(local).read_text())


def _resolve_run_id(run_id: str | None) -> str:
    if run_id is not None:
        return run_id
    active = mlflow.active_run()
    if active is None:
        raise RuntimeError(
            "log_params() found no active MLflow run; start one or pass run_id"
        )
    return active.info.run_id


def _qualified_name(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"
