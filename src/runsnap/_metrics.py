"""Flattening a record of numbers into named scalars."""

import dataclasses
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel


def flatten_metrics(obj: Any, prefix: str = "") -> dict[str, float]:
    """Flatten a record of numbers into one named scalar per leaf.

    Mappings, dataclass instances and Pydantic models are traversed into dotted
    keys, so a record whose `actor` field is itself a record reads
    `actor.entropy`, and `prefix` namespaces every key the same way. Leaves
    become Python numbers: booleans become `0` or `1`, a zero-dimensional array
    from any array library becomes the element behind it, and a leaf that is
    neither a number nor a single-element array — a `None` sentinel, a `kind`
    label — is left out. Non-finite values are kept as they are, so a diverged
    loss stays visible as the `NaN` it is.

    The result is what `mlflow.log_metrics()` takes.

    Raises:
        ValueError: A leaf holds more than one element, named by its key.
    """
    flat: dict[str, float] = {}
    _flatten(obj, prefix, flat)
    return flat


def _flatten(value: Any, key: str, flat: dict[str, float]) -> None:
    fields = _fields(value)
    if fields is None:
        number = _leaf(value, key)
        if number is not None:
            flat[key] = number
        return
    for name, item in fields.items():
        _flatten(item, f"{key}.{name}" if key else str(name), flat)


def _fields(value: Any) -> Mapping[Any, Any] | None:
    """The fields of `value` when it is a record, otherwise None."""
    if isinstance(value, Mapping):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}
    return None


def _leaf(value: Any, key: str) -> float | None:
    """The number `value` stands for, or None when it stands for none."""
    number = _number(value)
    if number is not None:
        return number
    if not callable(getattr(value, "item", None)):
        return None
    try:
        element = value.item()
    except ValueError as exc:
        raise ValueError(
            f"metric {key!r} holds more than one element; reduce it to a scalar"
        ) from exc
    return _number(element)


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return value
    return None
