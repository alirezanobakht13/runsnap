## ADDED Requirements

### Requirement: Scalar events are sharded so a sync pass uploads a bounded amount

The scalar event stream SHALL roll into a new shard once the open shard passes a
size threshold. A sealed scalar shard SHALL be uploaded once and not re-uploaded;
the open scalar shard SHALL be uploaded on every sync pass that finds it changed,
so a dashboard watching the run keeps updating.

#### Scenario: A run logs scalars past the shard threshold

- **WHEN** a run logs enough scalars for the scalar stream to pass the shard
  threshold several times
- **THEN** each sealed shard is uploaded exactly once, and the data available to
  a viewer is the same as if the stream had been one file

#### Scenario: A dashboard watches a run still being written

- **WHEN** a sync pass runs while the open scalar shard has grown since the last
  pass
- **THEN** the open shard is uploaded, and a viewer fetching the run afterwards
  sees the newly logged scalars

#### Scenario: Nothing has changed since the last pass

- **WHEN** a sync pass runs and no event file has grown since the last pass
- **THEN** nothing is uploaded
