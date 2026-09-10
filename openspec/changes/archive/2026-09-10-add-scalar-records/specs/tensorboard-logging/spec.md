## ADDED Requirements

### Requirement: A record of scalars is charted in one call

The writer returned by `runsnap.tensorboard()` SHALL provide `add_record(prefix, record, global_step=None, walltime=None)`, which flattens `record` by the same rule as `runsnap.flatten_metrics()` and charts each entry as a scalar named `prefix/<key>`, or `<key>` when the prefix is empty, at the given step and wall time. Entries SHALL land in the scalar event file, and non-numeric fields SHALL be skipped.

#### Scenario: Training record

- **WHEN** a user calls `writer.add_record("train", {"loss": 0.5, "kind": "train", "actor": {"entropy": 1.2}}, 10)`
- **THEN** the run's scalar event file holds `train/loss` = `0.5` and `train/actor.entropy` = `1.2` at step `10`, and nothing for `kind`

#### Scenario: Wall time forwarded

- **WHEN** a user calls `add_record("eval", record, 10, walltime=1700000000.0)`
- **THEN** every charted event carries wall time `1700000000.0`

#### Scenario: Scalars stay downloadable without media

- **WHEN** a run logs records through `add_record` and images through `add_image`
- **THEN** downloading only the scalar event file shows every record's curves

#### Scenario: Empty prefix

- **WHEN** a user calls `add_record("", {"loss": 0.5}, 1)`
- **THEN** the scalar is named `loss`
