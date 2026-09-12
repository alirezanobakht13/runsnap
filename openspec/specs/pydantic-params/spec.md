## Purpose

Lets hyperparameters defined as Pydantic models be logged to an MLflow run without hand-conversion, producing flat searchable params for the MLflow UI alongside a full-fidelity artifact that reloads back into the original model class.

## Requirements

### Requirement: Pydantic models can be logged as params

`runsnap.log_params()` SHALL accept a Pydantic model instance and log its fields as MLflow params on the active run, or on a run named explicitly. Passing a value that is not a Pydantic model SHALL raise `TypeError`.

#### Scenario: Model logged to the active run

- **WHEN** a user calls `runsnap.log_params(hp)` inside an active run, where `hp` is a Pydantic model with field `seed = 42`
- **THEN** the run has param `seed` with value `42`

#### Scenario: Non-model rejected

- **WHEN** a user calls `runsnap.log_params({"seed": 42})`
- **THEN** a `TypeError` is raised naming the expected type

### Requirement: Nested models flatten to dotted param names

Nested models and nested mappings SHALL flatten into dotted param keys, one param per scalar leaf, so that individual hyperparameters are sortable and filterable in the MLflow UI.

#### Scenario: Nested model

- **WHEN** a model contains a field `opt` holding a nested model with `lr = 0.001`
- **THEN** the run has param `opt.lr` with value `0.001`, and no param whose value is a serialized representation of the whole `opt` object

#### Scenario: Deeply nested model

- **WHEN** a model nests three levels, ending in `trainer.scheduler.warmup_steps = 500`
- **THEN** the run has param `trainer.scheduler.warmup_steps` with value `500`

### Requirement: Param values use a single deterministic encoding

Leaf values SHALL be encoded by one rule: strings are logged verbatim, and every other value is logged as its JSON representation. Sequences SHALL be logged whole as a single JSON value rather than expanded into indexed keys, so that runs with different-length sequences remain comparable in the MLflow run table.

#### Scenario: Scalar encodings

- **WHEN** a model has `name = "resnet"`, `lr = 0.001`, `use_amp = True`, and `limit = None`
- **THEN** the params are `name` = `resnet`, `lr` = `0.001`, `use_amp` = `true`, and `limit` = `null`

#### Scenario: Sequence logged whole

- **WHEN** a model has `layers = [64, 128, 256]`
- **THEN** the run has a single param `layers` with value `[64, 128, 256]`, and no params `layers.0`, `layers.1`, or `layers.2`

#### Scenario: Sequences of differing length stay comparable

- **WHEN** one run logs `layers = [64, 128]` and another logs `layers = [64, 128, 256]`
- **THEN** both runs have exactly one `layers` param, so comparing them in MLflow shows a single differing column

#### Scenario: Empty container preserved

- **WHEN** a model has `overrides = {}`
- **THEN** the run has param `overrides` with value `{}` rather than the field being omitted

#### Scenario: Value exceeding MLflow's param limit

- **WHEN** a leaf's encoded value is longer than MLflow's maximum param value length
- **THEN** a warning is emitted identifying the param key, and the complete value remains available in the artifact

### Requirement: The full model is stored as a round-trippable artifact

`runsnap.log_params()` SHALL additionally write the model's complete JSON serialization as a run artifact, together with the fully qualified name of the model class. The artifact SHALL be the authoritative record, preserving values that the flattened params encode lossily or truncate.

#### Scenario: Artifact written

- **WHEN** a user calls `runsnap.log_params(hp)`
- **THEN** the run has an artifact under `hparams/` containing the model's serialized data and the fully qualified name of its class

#### Scenario: Multiple models on one run

- **WHEN** a user calls `runsnap.log_params(model_hp, name="model")` and `runsnap.log_params(data_hp, name="data")` on the same run
- **THEN** the run has two distinct artifacts, `hparams/model.json` and `hparams/data.json`

#### Scenario: Param key collision avoided

- **WHEN** two models sharing a field name are logged to one run with distinct prefixes
- **THEN** their params are namespaced under those prefixes and do not collide

### Requirement: Logged models can be reloaded into their class

`runsnap.load_params()` SHALL reconstruct a logged model from a run by downloading its artifact and validating it against a caller-supplied model class, returning a fully typed instance.

#### Scenario: Round trip

- **WHEN** a user logs a model to a run and later calls `runsnap.load_params(run_id, HParams)`
- **THEN** an `HParams` instance equal to the original is returned

#### Scenario: Class mismatch

- **WHEN** the supplied class's fully qualified name differs from the one recorded in the artifact
- **THEN** a warning naming both is emitted and validation is still attempted, so a moved or renamed class can still be loaded

#### Scenario: Validation failure

- **WHEN** the recorded data does not validate against the supplied class
- **THEN** Pydantic's validation error is raised

#### Scenario: No such artifact

- **WHEN** the named model was never logged to that run
- **THEN** an error is raised naming the run and the requested artifact

### Requirement: Logging a hyperparameter model costs a bounded number of round trips

Logging a model SHALL write its flattened params in batches, so the number of
requests to the tracking server grows with the number of batches rather than
with the number of leaves. A model with no more leaves than one batch holds
SHALL cost a single params request.

#### Scenario: A model with many leaves

- **WHEN** a model flattening to several hundred params is logged
- **THEN** every param is recorded on the run, and the params are written in as
  many requests as there are full batches plus any remainder

#### Scenario: A model smaller than one batch

- **WHEN** a model flattening to fewer params than a batch holds is logged
- **THEN** every param is recorded in a single params request

#### Scenario: A leaf exceeding the param value limit

- **WHEN** a model holds a leaf whose encoded value is longer than the tracking
  server's param value limit
- **THEN** a warning names that key and points at the artifact holding the
  complete value, and logging proceeds
