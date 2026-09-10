## Purpose

Turns a record of numbers, whether a mapping, a dataclass instance, or a model, into flat named scalars that MLflow metrics and TensorBoard accept directly, with one rule for keys and leaves shared by every consumer.

## Requirements

### Requirement: Nested records flatten to dotted keys

`runsnap.flatten_metrics()` SHALL accept a mapping, a dataclass instance, or a Pydantic model and return a mapping from key to number. Nested records SHALL flatten into dotted keys, one entry per leaf. An optional prefix SHALL be prepended to every key with a dot.

#### Scenario: Nested dataclass

- **WHEN** a dataclass `diagnostics` has field `loss` and a nested dataclass field `actor` with field `entropy`
- **THEN** the result has keys `loss` and `actor.entropy`

#### Scenario: Nested mapping

- **WHEN** the input is `{"train": {"loss": 0.5}, "kl": 0.01}`
- **THEN** the result is `{"train.loss": 0.5, "kl": 0.01}`

#### Scenario: Prefix

- **WHEN** a user calls `flatten_metrics({"loss": 0.5}, prefix="train")`
- **THEN** the result is `{"train.loss": 0.5}`

### Requirement: Leaves become Python numbers

Integer and float leaves SHALL pass through unchanged. Boolean leaves SHALL become `0` or `1`. A leaf exposing a single-element `.item()`, such as a zero-dimensional array from any array library, SHALL be replaced by that element. A leaf whose `.item()` fails because it holds more than one element SHALL raise an error naming the key. No array library SHALL be imported to do this.

#### Scenario: Zero-dimensional array

- **WHEN** a field holds a zero-dimensional array with value `0.25`
- **THEN** the result holds the Python float `0.25`

#### Scenario: Boolean flag

- **WHEN** a field holds `True`
- **THEN** the result holds `1`, which MLflow accepts as a metric value

#### Scenario: Multi-element array

- **WHEN** a field `grads` holds an array of three values
- **THEN** an error is raised that names `grads`

### Requirement: Non-numeric leaves are dropped

Leaves that are neither numbers nor single-element arrays, such as `None` and strings, SHALL be omitted from the result.

#### Scenario: Sentinel and label fields

- **WHEN** the input is `{"kind": "train", "successes": None, "loss": 0.5}`
- **THEN** the result is `{"loss": 0.5}`

### Requirement: Non-finite values are preserved

`NaN` and infinities SHALL appear in the result unchanged.

#### Scenario: Diverged loss

- **WHEN** a field holds `float("nan")`
- **THEN** the result holds `NaN` under that key

### Requirement: The result is directly loggable

The returned mapping SHALL be accepted by `mlflow.log_metrics()` without further conversion.

#### Scenario: Straight to MLflow

- **WHEN** a user calls `mlflow.log_metrics(runsnap.flatten_metrics(diagnostics), step=step)` inside an active run, where `diagnostics` mixes floats, zero-dimensional arrays, a boolean, and a `None`
- **THEN** every numeric leaf is logged as a metric and the call raises nothing
