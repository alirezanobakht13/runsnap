## Purpose

Lets a run log metrics, images, and histograms as TensorBoard event files that
live in the run's own MLflow artifacts, so that MLflow stays the system of record
while TensorBoard does the viewing. Logging works from any framework, and a job
that is killed mid-training keeps everything already synced.

## ADDED Requirements

### Requirement: Writing TensorBoard data never fails the run that produced it

A failure anywhere in TensorBoard logging — writing an event, locating the file
an event was written to, rolling a shard, or uploading — SHALL be reported as a
warning and SHALL NOT raise into the training loop.

#### Scenario: The underlying writer's event file cannot be located

- **WHEN** the summary writer does not expose the file it is appending events to
  in the expected form
- **THEN** a warning is issued, the training loop continues, and events already
  written remain uploadable
