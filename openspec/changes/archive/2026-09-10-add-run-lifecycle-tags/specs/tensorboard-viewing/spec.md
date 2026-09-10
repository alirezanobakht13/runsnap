## ADDED Requirements

### Requirement: Attempt chains can be viewed together

`runsnap tb` SHALL accept a flag that extends the selected runs with every attempt each selected run transitively continues, so a training that was interrupted and resumed is viewed as one set of curves. A predecessor already in the selection SHALL appear once.

#### Scenario: Resumed training

- **WHEN** run `third` continues `second`, which continues `first`, and a user runs `runsnap tb third --chain`
- **THEN** TensorBoard opens showing `third`, `second`, and `first`

#### Scenario: Chain over a query

- **WHEN** a user runs `runsnap tb --experiment ablations --filter "attributes.status = 'FINISHED'" --chain` and a matching run continues an earlier `KILLED` attempt
- **THEN** the earlier attempt is shown alongside the matching run even though it does not match the filter

#### Scenario: Without the flag

- **WHEN** a user runs `runsnap tb third` without the flag
- **THEN** only `third` is shown
