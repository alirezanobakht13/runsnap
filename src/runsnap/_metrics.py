"""Flattening a record of numbers into named scalars."""

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel


def flatten_metrics(obj: Any, prefix: str = "") -> dict[str, float]:
    """Flatten a record of numbers into one named scalar per leaf.

    Mappings, dataclass instances and Pydantic models are traversed into dotted
    keys, so a record whose `actor` field is itself a record reads
    `actor.entropy`, and `prefix` namespaces every key the same way. Leaves
    become Python numbers: booleans become `0` or `1`, and an array from any
    array library or a sequence such as a list or tuple holding exactly one
    element becomes that element. A leaf carrying no number — a `None`
    sentinel, a `kind` label, an empty sequence — is left out. Non-finite
    values are kept as they are, so a diverged loss stays visible as the `NaN`
    it is.

    The result is what `mlflow.log_metrics()` takes.

    Raises:
        ValueError: A leaf holds more than one element, whether an array or a
            sequence of numbers, named by its key.
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
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        if len(value) > 1:
            raise _too_many(key)
        return _leaf(value[0], key) if value else None
    if not callable(getattr(value, "item", None)):
        return None
    try:
        element = value.item()
    except (ValueError, RuntimeError, TypeError) as exc:
        raise _too_many(key) from exc
    return _number(element)


def _too_many(key: str) -> ValueError:
    return ValueError(
        f"metric {key!r} holds more than one element; reduce it to a scalar"
    )


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return value
    return None
