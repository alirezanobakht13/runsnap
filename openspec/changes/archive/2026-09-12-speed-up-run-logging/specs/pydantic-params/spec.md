## Purpose

Lets hyperparameters defined as Pydantic models be logged to an MLflow run
without hand-conversion, producing flat searchable params for the MLflow UI
alongside a full-fidelity artifact that reloads back into the original model
class.

## ADDED Requirements

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
