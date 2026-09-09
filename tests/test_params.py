"""Tests for logging Pydantic hyperparameter models onto runs."""

import json
from pathlib import Path

import mlflow
import pytest
from pydantic import BaseModel, ValidationError

import runsnap
from runsnap._params import flatten_model


def artifact_paths(client: mlflow.MlflowClient, run_id: str) -> set[str]:
    """Every artifact file on a run, as slash-separated paths."""
    found: set[str] = set()
    pending = [""]
    while pending:
        for item in client.list_artifacts(run_id, pending.pop()):
            if item.is_dir:
                pending.append(item.path)
            else:
                found.add(item.path)
    return found


def read_artifact(client: mlflow.MlflowClient, run_id: str, path: str) -> dict:
    return json.loads(Path(client.download_artifacts(run_id, path)).read_text())


class Scheduler(BaseModel):
    warmup_steps: int = 500


class Trainer(BaseModel):
    scheduler: Scheduler = Scheduler()


class Optimizer(BaseModel):
    lr: float = 0.001


class HParams(BaseModel):
    seed: int = 42
    opt: Optimizer = Optimizer()


def test_flattens_nested_model():
    assert flatten_model(HParams()) == {"seed": "42", "opt.lr": "0.001"}


def test_flattens_three_levels():
    class Deep(BaseModel):
        trainer: Trainer = Trainer()

    assert flatten_model(Deep()) == {"trainer.scheduler.warmup_steps": "500"}


def test_encodes_each_scalar_type():
    class Scalars(BaseModel):
        name: str = "resnet"
        lr: float = 0.001
        use_amp: bool = True
        limit: int | None = None

    assert flatten_model(Scalars()) == {
        "name": "resnet",
        "lr": "0.001",
        "use_amp": "true",
        "limit": "null",
    }


def test_logs_sequences_whole():
    class WithList(BaseModel):
        layers: list[int] = [64, 128, 256]

    assert flatten_model(WithList()) == {"layers": "[64, 128, 256]"}


def test_sequences_of_differing_length_share_one_key():
    class WithList(BaseModel):
        layers: list[int] = []

    short = flatten_model(WithList(layers=[64, 128]))
    long = flatten_model(WithList(layers=[64, 128, 256]))
    assert short.keys() == long.keys() == {"layers"}
    assert short["layers"] != long["layers"]


def test_keeps_empty_containers():
    class WithEmpty(BaseModel):
        overrides: dict[str, str] = {}
        stages: list[str] = []

    assert flatten_model(WithEmpty()) == {"overrides": "{}", "stages": "[]"}


def test_prefix_namespaces_every_key():
    assert flatten_model(HParams(), prefix="model") == {
        "model.seed": "42",
        "model.opt.lr": "0.001",
    }


def test_logs_params_and_artifact_to_the_active_run(tracking):
    hp = HParams(seed=7)
    with mlflow.start_run() as run:
        runsnap.log_params(hp)
    run_id = run.info.run_id

    params = tracking.get_run(run_id).data.params
    assert params["seed"] == "7"
    assert params["opt.lr"] == "0.001"

    payload = read_artifact(tracking, run_id, "hparams/params.json")
    assert payload["class"] == f"{HParams.__module__}.HParams"
    assert payload["data"] == hp.model_dump(mode="json")


def test_logs_to_a_run_named_explicitly(tracking):
    with mlflow.start_run() as run:
        pass
    runsnap.log_params(HParams(seed=3), run_id=run.info.run_id)
    assert tracking.get_run(run.info.run_id).data.params["seed"] == "3"


def test_rejects_non_pydantic_input(tracking):
    with mlflow.start_run(), pytest.raises(TypeError, match="BaseModel"):
        runsnap.log_params({"seed": 42})  # ty: ignore[invalid-argument-type]


def test_named_models_get_distinct_artifacts(tracking):
    with mlflow.start_run() as run:
        runsnap.log_params(HParams(), name="model", prefix="model")
        runsnap.log_params(Trainer(), name="data", prefix="data")
    run_id = run.info.run_id

    assert artifact_paths(tracking, run_id) == {
        "hparams/model.json",
        "hparams/data.json",
    }
    assert set(tracking.get_run(run_id).data.params) == {
        "model.seed",
        "model.opt.lr",
        "data.scheduler.warmup_steps",
    }


def test_prefixes_keep_shared_field_names_apart(tracking):
    with mlflow.start_run() as run:
        runsnap.log_params(Optimizer(lr=0.1), name="a", prefix="a")
        runsnap.log_params(Optimizer(lr=0.2), name="b", prefix="b")
    params = tracking.get_run(run.info.run_id).data.params
    assert params == {"a.lr": "0.1", "b.lr": "0.2"}


def test_warns_when_a_value_exceeds_mlflows_param_limit(tracking):
    class Wide(BaseModel):
        notes: list[str] = []

    hp = Wide(notes=["x" * 100] * 100)
    with mlflow.start_run() as run, pytest.warns(UserWarning, match="notes"):
        runsnap.log_params(hp)

    payload = read_artifact(tracking, run.info.run_id, "hparams/params.json")
    assert payload["data"] == hp.model_dump(mode="json")


def test_round_trips_through_load_params(tracking):
    hp = HParams(seed=11, opt=Optimizer(lr=0.5))
    with mlflow.start_run() as run:
        runsnap.log_params(hp)
    assert runsnap.load_params(run.info.run_id, HParams) == hp


def test_load_params_warns_on_class_mismatch(tracking):
    class Renamed(BaseModel):
        seed: int = 42
        opt: Optimizer = Optimizer()

    with mlflow.start_run() as run:
        runsnap.log_params(HParams(seed=5))
    with pytest.warns(UserWarning, match="HParams"):
        loaded = runsnap.load_params(run.info.run_id, Renamed)
    assert loaded.seed == 5


def test_load_params_raises_pydantics_validation_error(tracking):
    class Incompatible(BaseModel):
        seed: dict[str, int]

    with mlflow.start_run() as run:
        runsnap.log_params(HParams())
    with pytest.warns(UserWarning), pytest.raises(ValidationError):
        runsnap.load_params(run.info.run_id, Incompatible)


def test_load_params_reports_a_missing_artifact(tracking):
    with mlflow.start_run() as run:
        pass
    run_id = run.info.run_id
    with pytest.raises(FileNotFoundError, match=f"{run_id}.*hparams/absent.json"):
        runsnap.load_params(run_id, HParams, name="absent")
