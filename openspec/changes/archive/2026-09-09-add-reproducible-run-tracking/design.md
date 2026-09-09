## Context

See proposal.md — Why. The relevant constraints in the installed environment:

- MLflow 3.16.0 registers a `GitRunContext` that stamps `mlflow.source.git.commit`/`.branch`/`.repoURL` on every run, but resolves the repository from `sys.argv[0]`, so it yields nothing under a Jupyter kernel or REPL.
- MLflow's plugin system exposes an entry point group `mlflow.run_context_provider`, but providers can only contribute **tags**. Tag values cap at `MAX_TAG_VAL_LENGTH = 8000` and are silently truncated (`truncate=True`), and there is no post-run-creation hook for logging artifacts. The plugin path therefore cannot carry a patch.
- `mlflow.log_params` requires a mapping and stringifies each value with `str()`, so a nested Pydantic model becomes one unusable param.
- Param values cap at `MAX_PARAM_VAL_LENGTH = 6000` and param keys at `MAX_ENTITY_KEY_LENGTH = 250`, both truncated rather than rejected.
- GitPython 3.1.62 is installed but MLflow imports it lazily behind `ImportError` guards, so it is an optional transitive dependency, not a guaranteed one.

## Goals / Non-Goals

**Goals:**

- One added symbol at the call site (`exp_track.start_run`), with everything downstream remaining stock MLflow.
- A patch that is provably applicable, not merely informative.
- Capture that is invisible to the user's git working state and incapable of failing an experiment.

**Non-Goals:**

- Reproducing the Python environment beyond what the repository already tracks. `uv.lock` is a tracked file, so dependency state travels inside the patch; no separate environment capture is added.
- Submodules, git LFS pointers resolved to content, and multi-repository workspaces. Single repository only.
- Capturing runs created by bare `mlflow.start_run()`. Opting in is the explicit contract.
- Pushing reconstructed branches, or any network git operation.

## Decisions

### Wrapper over plugin or monkeypatch

`exp_track.start_run()` delegates to `mlflow.start_run()` and then performs capture against the returned run.

Alternatives considered. A `mlflow.run_context_provider` entry point would make capture fully transparent with zero code change, but providers return only tags, and an 8000-character truncating tag cannot hold a patch — this is precisely the defect in MLflow's own `mlflow.genai.git_versioning`. Monkeypatching `mlflow.start_run` at import time would reach the same transparency, but it depends on MLflow's private call graph and would fail obscurely mid-experiment on a minor version bump. The wrapper costs one symbol and stays entirely on MLflow's public API.

### Patch construction: one base, throwaway index

Capture copies `.git/index` to a temporary file and runs, with `GIT_INDEX_FILE` pointing at the copy:

```
git add -A -N .              # intent-to-add: untracked files enter the diff
git diff HEAD --binary       # single base, applicable output
```

The intent-to-add marks land only in the copy, so the user's staging area is untouched — satisfying the "does not disturb git state" requirement. `git diff HEAD` yields one diff against one base, which is what makes the patch apply; MLflow's `git diff` + `git diff --cached` concatenation produces overlapping hunks against different bases that `git apply` rejects outright. `--binary` carries binary file content. `.gitignore` is honoured by `git add` for free.

This was verified against a repository combining staged-then-re-edited files, untracked additions, deletions, mode changes, binary blobs, and ignored files: the patch applied cleanly to a checkout of the base commit and reproduced every case, with `git status` unchanged in the source repository.

Alternative considered: `git stash create`, which produces a commit object capturing the working tree. It is a single well-formed object, but it writes into the object database, does not include untracked files without `-u`, and yields a commit rather than a portable patch — harder to store as an artifact and harder to inspect.

### Shell out to git rather than use GitPython

Capture is a handful of git invocations, one of which requires a per-invocation `GIT_INDEX_FILE`. `subprocess` expresses that directly; GitPython's `custom_environment` adds an abstraction over the same commands. Shelling out also avoids depending on a package MLflow treats as optional. The `git` binary is already a hard requirement for the feature to mean anything.

### Recorded schema

Tags under an `exp_track.git.` prefix, so nothing collides with MLflow's own namespace:

| Tag | Value |
| --- | --- |
| `exp_track.git.commit` | full base commit SHA |
| `exp_track.git.branch` | branch name; absent when detached |
| `exp_track.git.dirty` | `true` / `false` |
| `exp_track.git.repo_url` | remote URL, credentials stripped |
| `exp_track.git.patch_sha256` | digest of patch bytes; absent when clean |
| `exp_track.git.patch_run_id` | run holding the artifact; set on nested runs only |
| `exp_track.git.capture_error` | short reason; set only on partial capture |

Artifacts: `code/state.patch` (dirty runs only), `hparams/<name>.json`.

The patch digest is the payload that makes "which runs used identical code" a single `search_runs` filter rather than a manual diffing exercise — cheap to record and the main affordance the capture buys beyond bare reproduction.

MLflow's `mlflow.source.git.*` tags are mirrored only when unset, so notebook runs gain a working source display without overriding what MLflow itself resolved. `mlflow.source.git.diff` is deliberately never written — that is the truncating field.

### Nested runs store the patch once

A sweep with one parent and fifty children would otherwise upload the same patch fifty-one times. Children receive the full tag set plus `exp_track.git.patch_run_id` pointing at the ancestor holding the artifact; the CLI follows that pointer. Git state is resolved once per process and cached, so children cost only tag writes.

Alternative considered: capture on every run unconditionally. Simpler, but multiplies artifact storage by the sweep width, which is the common case in this project.

### Flattening: one encoding rule

`model_dump(mode="json")`, then recursive descent through mappings producing dotted keys. At each leaf: strings are logged verbatim, everything else is `json.dumps`-encoded. That single rule yields `lr` = `0.001`, `use_amp` = `true`, `limit` = `null`, `name` = `resnet`.

Sequences are logged whole as one JSON value rather than expanded into `layers.0`, `layers.1`. Expansion would make two runs with different-length lists produce different param *sets*, and MLflow's comparison table renders that as a spray of half-empty columns. One column that differs as a whole is the readable outcome. Empty mappings and sequences are treated as leaves so the field does not silently vanish.

Fidelity lost to this encoding — truncation past 6000 characters, tuples rendering as JSON arrays — is recovered from the artifact, which holds the model's own serialization and is the authoritative record. Params exist for the UI and for search.

### Reload requires an explicit class

`load_params(run_id, HParams)` takes the target class from the caller. The artifact records the original class's fully qualified name for verification, and a mismatch warns rather than fails, so a class that moved between packages still loads. Importing the recorded name dynamically was rejected: it turns an artifact fetched from a tracking server into an import instruction.

### CLI on cyclopts

Two commands: `checkout <run-ref>` and `show <run-ref>`. Reconstruction defaults to `git worktree add`, because the researcher's normal reason to reach for this is comparing old code against work currently in progress — switching the current checkout would interrupt exactly the work that prompted the question. `--no-worktree` gives the in-place branch checkout, guarded against clobbering uncommitted changes unless forced.

The applied patch is left uncommitted by default: that state is what the run actually had, and `git diff` in the reconstructed tree then shows precisely what was uncommitted at run time. `--commit` snapshots it into one commit when a durable, shareable reference is wanted.

Run references accept an MLflow run id (a 32-character hex string) or a run name, resolved through `search_runs` on `tags."mlflow.runName"`, optionally narrowed by `--experiment`. Ambiguity lists the candidates rather than guessing.

### Module layout

```
src/exp_track/
  __init__.py    public API: start_run, log_params, load_params
  _tags.py       tag names, artifact paths, size defaults
  _git.py        git primitives: read state, build patch, worktree, apply
  _capture.py    orchestration: what gets tagged and logged on a run
  _params.py     pydantic flatten, artifact write and read
  _cli.py        cyclopts app: checkout, show
```

`_git.py` holds every `subprocess` call, which keeps the git behaviour testable against real temporary repositories — the only way to have confidence in a patch's applicability. Tests build repositories exercising each fidelity case and assert round-trip through `git apply`.

## Risks / Trade-offs

- **A large tracked file edited in the working tree produces a huge patch** → A configurable size ceiling (`EXP_TRACK_MAX_PATCH_BYTES`) skips the artifact and records `capture_error` rather than uploading it.
- **A patch may contain uncommitted secrets** → Ignored files are excluded, which covers the usual `.env` case, but a secret typed into a tracked file will be captured and stored on the tracking server. Documented; not defended against in code.
- **Capture adds latency to every run start** → Git state is resolved once per process and cached, so cost is one `git diff` per process, not per run.
- **Silent degradation can hide a misconfiguration** → Every skipped capture warns and, where a run exists to carry it, records `capture_error`, so a run that looks captured but is not can be told apart from one that was never captured.
- **`git worktree` requires git 2.5+ and leaves registrations behind** → The created path is reported so the user can `git worktree remove` it; no automatic cleanup is attempted, since the whole point is that the tree outlives the command.
- **Reconstruction cannot recover a commit that was never pushed and no longer exists locally** → Detected as a precondition failure naming the missing commit; unrecoverable by design, since the code state genuinely no longer exists.
