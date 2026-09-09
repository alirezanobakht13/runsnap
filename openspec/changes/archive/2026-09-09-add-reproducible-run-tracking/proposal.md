## Why

Research runs are only reproducible if the exact code that produced them can be recovered. MLflow records the current commit but not uncommitted work, so any run started from a dirty working tree — the normal state during research — cannot be reconstructed. MLflow also rejects Pydantic hyperparameter models, and its one existing diff feature (`mlflow.genai.enable_git_model_versioning`) is unusable for this purpose: it is scoped to GenAI logged models, stores the diff in an 8000-character tag that silently truncates, ignores untracked files, and concatenates `git diff` with `git diff --cached` — two diffs against different bases, producing a patch that does not apply.

## What Changes

- Add `exp_track.start_run()`, a thin pass-through wrapper over `mlflow.start_run()` that records the code state of the working tree at run start. Every other MLflow call stays stock MLflow.
- Capture code state as the recorded commit SHA plus a single `git diff HEAD --binary` patch taken against a throwaway index copy, so staged changes, unstaged changes, untracked files, deletions, and file-mode changes are all recovered while the user's real staging area is untouched.
- Store the patch as a run artifact (not a tag), with a `sha256` tag so runs sharing byte-identical code state can be found by search.
- Add `exp_track.log_params()` / `exp_track.load_params()` for Pydantic hyperparameter models: flattened dotted params for the MLflow UI and search, plus a full-fidelity JSON artifact for exact round-trip back into the model class.
- Add an `exp-track` CLI (built on `cyclopts`) with `checkout`, which reconstructs a run's code state onto a new branch — in a separate git worktree by default, or in the current working tree on request — and `show`, which prints a run's recorded code state.
- Degrade quietly: a missing git repository, a detached HEAD, or an oversized patch produces a warning and a partial record, never a failed experiment.

## Capabilities

### New Capabilities

- `code-state-capture`: Recording the exact code state of the working tree onto an MLflow run at run start, via `exp_track.start_run()`.
- `pydantic-params`: Logging and reloading Pydantic hyperparameter models as MLflow params and a round-trippable artifact.
- `code-state-restore`: The `exp-track` CLI that reconstructs a recorded run's code state into a working tree.

### Modified Capabilities

None. This is the project's first capability set.

## Impact

- **New source**: `src/exp_track/` gains the public API (`start_run`, `log_params`, `load_params`), git capture and apply primitives, the Pydantic flattener, and the CLI.
- **Dependencies**: adds `pydantic>=2` and `cyclopts>=4`. `mlflow>=3.16` is already present. Git operations shell out to the `git` binary rather than GitPython, since the capture depends on a custom `GIT_INDEX_FILE` and maps one-to-one onto git's command line.
- **Entry point**: `pyproject.toml`'s existing `exp-track` script is repointed from the placeholder `exp_track:main` to the CLI app.
- **Run records**: runs gain `exp_track.git.*` tags, a `code/state.patch` artifact when dirty, and `hparams/*.json` artifacts. MLflow's standard `mlflow.source.git.*` tags are mirrored when absent so the MLflow UI's source column works from notebooks, where MLflow's own `GitRunContext` resolves nothing.
- **Runtime requirement**: `git` must be on `PATH` for capture and restore; its absence is a warning, not an error.
