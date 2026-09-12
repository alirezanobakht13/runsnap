## Purpose

Turns a record of numbers, whether a mapping, a dataclass instance, or a model,
into flat named scalars that MLflow metrics and TensorBoard accept directly, with
one rule for keys and leaves shared by every consumer.

## ADDED Requirements

### Requirement: A leaf holding more than one number is an error naming the field

Flattening a record SHALL raise `ValueError` naming the offending key when a
leaf holds more than one number, whichever array library the leaf comes from and
whether the leaf is an array or an ordinary sequence. The error message SHALL
identify the key so the caller can locate the field.

#### Scenario: A leaf is a multi-element tensor

- **WHEN** a record holds a field whose value is an array or tensor with more
  than one element
- **THEN** flattening raises `ValueError` whose message names that field's key

#### Scenario: A leaf is a list of numbers

- **WHEN** a record holds a field whose value is a sequence of more than one
  number
- **THEN** flattening raises `ValueError` whose message names that field's key,
  rather than dropping the field

#### Scenario: A leaf is a single-element array

- **WHEN** a record holds a field whose value is an array with exactly one
  element
- **THEN** flattening yields that element as the field's number

#### Scenario: A leaf carries no number

- **WHEN** a record holds a field whose value is `None` or a string
- **THEN** flattening omits that field without raising
