"""Orchestration of what a run's code state records."""

import hashlib
import os
import tempfile
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from mlflow.entities import Run
from mlflow.tracking import MlflowClient

from runsnap._git import (
    GitError,
    GitState,
    build_patch,
    current_branch,
    find_repo_root,
    head_commit,
    remote_url,
)
from runsnap._tags import (
    DEFAULT_MAX_PATCH_BYTES,
    ENV_CAPTURE_CODE,
    ENV_MAX_PATCH_BYTES,
    MLFLOW_TAG_BRANCH,
    MLFLOW_TAG_COMMIT,
    MLFLOW_TAG_DIRTY,
    MLFLOW_TAG_REPO_URL,
    PATCH_ARTIFACT_DIR,
    PATCH_ARTIFACT_NAME,
    TAG_BRANCH,
    TAG_CAPTURE_ERROR,
    TAG_COMMIT,
    TAG_DIRTY,
    TAG_PATCH_RUN_ID,
    TAG_PATCH_SHA256,
    TAG_REPO_URL,
)

MLFLOW_PARENT_RUN_ID = "mlflow.parentRunId"


@dataclass(frozen=True)
class CodeState:
    """A repository's state plus the patch describing its uncommitted work."""

    git: GitState
    patch: bytes

    @property
    def dirty(self) -> bool:
        return bool(self.patch)

    @property
    def patch_sha256(self) -> str:
        return hashlib.sha256(self.patch).hexdigest()


@lru_cache(maxsize=1)
def resolve_repo_root() -> Path:
    """Root of the repository the process runs in, resolved once per process.

    Raises `GitError` outside a repository and when `git` is unavailable.
    """
    return find_repo_root()


@lru_cache(maxsize=1)
def resolve_code_state() -> CodeState:
    """The current code state, resolved once per process and reused after that.

    Raises `GitError` when the repository has no commit at `HEAD`.
    """
    root = resolve_repo_root()
    git = GitState(
        root=root,
        commit=head_commit(root),
        branch=current_branch(root),
        repo_url=remote_url(root),
    )
    return CodeState(git=git, patch=build_patch(root))


def reset_code_state_cache() -> None:
    """Forget the resolved code state so the next capture reads git again."""
    resolve_repo_root.cache_clear()
    resolve_code_state.cache_clear()
    _patch_holders.clear()


def capture_enabled() -> bool:
    """Whether the environment leaves code-state capture switched on."""
    return os.environ.get(ENV_CAPTURE_CODE, "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }


def max_patch_bytes() -> int:
    """The configured ceiling on the size of an uploaded patch."""
    raw = os.environ.get(ENV_MAX_PATCH_BYTES)
    if raw is None:
        return DEFAULT_MAX_PATCH_BYTES
    try:
        return int(raw)
    except ValueError:
        warnings.warn(
            f"{ENV_MAX_PATCH_BYTES}={raw!r} is not an integer; "
            f"using the default of {DEFAULT_MAX_PATCH_BYTES} bytes",
            stacklevel=2,
        )
        return DEFAULT_MAX_PATCH_BYTES


# Run id -> id of the run holding that run's patch artifact. A nested run points
# at its ancestor's artifact rather than uploading the same patch again.
_patch_holders: dict[str, str] = {}


def capture(run: Run) -> None:
    """Record the code state of the working tree onto `run`.

    Every failure is reported as a warning and, where a run exists to carry it,
    as an `runsnap.git.capture_error` tag. Nothing raises into user code.
    """
    client = MlflowClient()
    run_id = run.info.run_id
    try:
        resolve_repo_root()
    except GitError as exc:
        warnings.warn(f"runsnap captured no code state: {exc}", stacklevel=3)
        return
    try:
        state = resolve_code_state()
    except Exception as exc:  # noqa: BLE001 - capture must never fail the run
        warnings.warn(f"runsnap could not capture code state: {exc}", stacklevel=3)
        _record_error(client, run_id, str(exc))
        return
    try:
        _record_state(client, run, state)
    except Exception as exc:  # noqa: BLE001 - capture must never fail the run
        warnings.warn(f"runsnap could not capture code state: {exc}", stacklevel=3)
        _record_error(client, run_id, str(exc))


def _record_error(client: MlflowClient, run_id: str, message: str) -> None:
    try:
        client.set_tag(run_id, TAG_CAPTURE_ERROR, message)
    except Exception:  # noqa: BLE001,S110 - the caller already warned; a
        # tracking store that cannot take the tag must not fail the run either.
        pass


def _record_state(client: MlflowClient, run: Run, state: CodeState) -> None:
    run_id = run.info.run_id
    client.set_tag(run_id, TAG_COMMIT, state.git.commit)
    client.set_tag(run_id, TAG_DIRTY, "true" if state.dirty else "false")
    if state.git.branch is not None:
        client.set_tag(run_id, TAG_BRANCH, state.git.branch)
    if state.git.repo_url is not None:
        client.set_tag(run_id, TAG_REPO_URL, state.git.repo_url)
    if state.dirty:
        client.set_tag(run_id, TAG_PATCH_SHA256, state.patch_sha256)
        _record_patch(client, run, state)
    _mirror_mlflow_tags(client, run, state)


def _record_patch(client: MlflowClient, run: Run, state: CodeState) -> None:
    run_id = run.info.run_id
    holder = _inherited_patch_holder(run)
    if holder is not None:
        client.set_tag(run_id, TAG_PATCH_RUN_ID, holder)
        _patch_holders[run_id] = holder
        return
    ceiling = max_patch_bytes()
    if len(state.patch) > ceiling:
        warnings.warn(
            f"runsnap skipped the code state patch: "
            f"{len(state.patch)} bytes exceeds the {ceiling} byte limit",
            stacklevel=4,
        )
        client.set_tag(
            run_id,
            TAG_CAPTURE_ERROR,
            f"patch too large: {len(state.patch)} bytes exceeds {ceiling}",
        )
        return
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / PATCH_ARTIFACT_NAME
        local.write_bytes(state.patch)
        client.log_artifact(run_id, str(local), artifact_path=PATCH_ARTIFACT_DIR)
    _patch_holders[run_id] = run_id


def _inherited_patch_holder(run: Run) -> str | None:
    """The ancestor run already holding this run's patch, when there is one."""
    parent_id = run.data.tags.get(MLFLOW_PARENT_RUN_ID)
    if parent_id is None:
        return None
    return _patch_holders.get(parent_id)


def _mirror_mlflow_tags(client: MlflowClient, run: Run, state: CodeState) -> None:
    """Fill MLflow's own git tags where MLflow itself resolved nothing.

    `mlflow.source.git.diff` is never written: it is a tag, and tag values are
    silently truncated, so a patch cannot survive there.
    """
    mirrored = {
        MLFLOW_TAG_COMMIT: state.git.commit,
        MLFLOW_TAG_BRANCH: state.git.branch,
        MLFLOW_TAG_REPO_URL: state.git.repo_url,
        MLFLOW_TAG_DIRTY: "true" if state.dirty else "false",
    }
    for key, value in mirrored.items():
        if value is not None and key not in run.data.tags:
            client.set_tag(run.info.run_id, key, value)
