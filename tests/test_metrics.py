"""Tests for flattening a record of numbers into named scalars."""

import math
from dataclasses import dataclass

import mlflow
import numpy as np
import pytest
from pydantic import BaseModel

import runsnap
from runsnap._metrics import flatten_metrics


@dataclass
class Actor:
    entropy: float


@dataclass
class Diagnostics:
    loss: float
    actor: Actor


def test_flattens_nested_dataclass():
    flat = flatten_metrics(Diagnostics(loss=0.5, actor=Actor(entropy=1.2)))
    assert flat == {"loss": 0.5, "actor.entropy": 1.2}


def test_flattens_nested_mapping():
    assert flatten_metrics({"train": {"loss": 0.5}, "kl": 0.01}) == {
        "train.loss": 0.5,
        "kl": 0.01,
    }


def test_flattens_nested_model():
    class Critic(BaseModel):
        value_loss: float = 1.2

    class Report(BaseModel):
        loss: float = 0.5
        critic: Critic = Critic()

    assert flatten_metrics(Report()) == {"loss": 0.5, "critic.value_loss": 1.2}


def test_prefix_namespaces_every_key():
    assert flatten_metrics({"loss": 0.5}, prefix="train") == {"train.loss": 0.5}


def test_passes_numbers_through_unchanged():
    assert flatten_metrics({"steps": 7, "loss": 0.5}) == {"steps": 7, "loss": 0.5}


def test_zero_dimensional_array_becomes_a_python_number():
    flat = flatten_metrics({"reward": np.array(0.25)})
    assert flat == {"reward": 0.25}
    assert type(flat["reward"]) is float


def test_one_element_array_becomes_its_element():
    assert flatten_metrics({"reward": np.array([0.25])}) == {"reward": 0.25}


def test_one_element_sequence_becomes_its_element():
    assert flatten_metrics({"reward": [0.25], "steps": (7,)}) == {
        "reward": 0.25,
        "steps": 7,
    }


def test_empty_sequence_is_omitted():
    assert flatten_metrics({"rewards": [], "loss": 0.5}) == {"loss": 0.5}


def test_boolean_becomes_an_int():
    flat = flatten_metrics({"converged": True})
    assert flat == {"converged": 1}
    assert not isinstance(flat["converged"], bool)


def test_boolean_array_becomes_an_int():
    flat = flatten_metrics({"converged": np.array(True)})
    assert flat == {"converged": 1}
    assert not isinstance(flat["converged"], bool)


def test_multi_element_array_raises_naming_the_key():
    with pytest.raises(ValueError, match="grads"):
        flatten_metrics({"grads": np.array([1.0, 2.0, 3.0])})


def test_multi_element_tensor_raises_naming_the_key():
    class Tensor:
        def item(self):
            raise RuntimeError("a Tensor with 3 elements cannot be converted to Scalar")

    with pytest.raises(ValueError, match="grads"):
        flatten_metrics({"grads": Tensor()})


def test_list_of_numbers_raises_naming_the_key():
    with pytest.raises(ValueError, match="grads"):
        flatten_metrics({"grads": [1.0, 2.0, 3.0]})


def test_drops_non_numeric_leaves():
    record = {"kind": "train", "raw": b"train", "successes": None, "loss": 0.5}
    assert flatten_metrics(record) == {"loss": 0.5}


def test_keeps_non_finite_values():
    flat = flatten_metrics({"loss": float("nan"), "bound": float("inf")})
    assert math.isnan(flat["loss"])
    assert flat["bound"] == math.inf


def test_flattened_record_logs_straight_to_mlflow(tracking):
    record = {
        "loss": 0.5,
        "reward": np.array(2.5),
        "converged": True,
        "successes": None,
        "kind": "train",
    }
    with mlflow.start_run() as run:
        mlflow.log_metrics(runsnap.flatten_metrics(record), step=1)

    metrics = tracking.get_run(run.info.run_id).data.metrics
    assert metrics == {"loss": 0.5, "reward": 2.5, "converged": 1.0}
