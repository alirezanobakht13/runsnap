"""Pydantic hyperparameter flattening, artifact write and read."""

import json
import tempfile
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.utils.validation import MAX_PARAM_VAL_LENGTH
from pydantic import BaseModel

from exp_track._tags import (
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
    too so the field does not silently vanish.
    """
    data = model.model_dump(mode="json")
    if not prefix and not data:
        return {}
    return _flatten(data, prefix)


def _flatten(value: Any, prefix: str) -> dict[str, str]:
    if isinstance(value, Mapping) and value:
        flat: dict[str, str] = {}
        for key, item in value.items():
            flat.update(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return flat
    return {prefix: _encode(value)}


def _encode(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value)


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

    Logs to the active run unless `run_id` names another one.
    """
    if not isinstance(model, BaseModel):
        raise TypeError(
            f"log_params() expects a pydantic BaseModel instance, "
            f"got {type(model).__name__}"
        )
    client = MlflowClient()
    target = _resolve_run_id(run_id)
    for key, value in flatten_model(model, prefix).items():
        if len(value) > MAX_PARAM_VAL_LENGTH:
            warnings.warn(
                f"param {key!r} is {len(value)} characters and MLflow truncates "
                f"past {MAX_PARAM_VAL_LENGTH}; the complete value is in "
                f"{hparams_artifact_path(name)}",
                stacklevel=2,
            )
        client.log_param(target, key, value)
    client.log_dict(
        target,
        {
            HPARAMS_CLASS_KEY: _qualified_name(type(model)),
            HPARAMS_DATA_KEY: model.model_dump(mode="json"),
        },
        hparams_artifact_path(name),
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
