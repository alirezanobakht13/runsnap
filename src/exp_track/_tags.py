"""Names and limits for what a captured run records."""

TAG_PREFIX = "exp_track.git."

TAG_COMMIT = f"{TAG_PREFIX}commit"
TAG_BRANCH = f"{TAG_PREFIX}branch"
TAG_DIRTY = f"{TAG_PREFIX}dirty"
TAG_REPO_URL = f"{TAG_PREFIX}repo_url"
TAG_PATCH_SHA256 = f"{TAG_PREFIX}patch_sha256"
TAG_PATCH_RUN_ID = f"{TAG_PREFIX}patch_run_id"
TAG_CAPTURE_ERROR = f"{TAG_PREFIX}capture_error"

MLFLOW_TAG_COMMIT = "mlflow.source.git.commit"
MLFLOW_TAG_BRANCH = "mlflow.source.git.branch"
MLFLOW_TAG_DIRTY = "mlflow.source.git.dirty"
MLFLOW_TAG_REPO_URL = "mlflow.source.git.repoURL"

PATCH_ARTIFACT_DIR = "code"
PATCH_ARTIFACT_NAME = "state.patch"
PATCH_ARTIFACT_PATH = f"{PATCH_ARTIFACT_DIR}/{PATCH_ARTIFACT_NAME}"

HPARAMS_ARTIFACT_DIR = "hparams"
HPARAMS_DEFAULT_NAME = "params"
HPARAMS_CLASS_KEY = "class"
HPARAMS_DATA_KEY = "data"


def hparams_artifact_path(name: str) -> str:
    """Artifact path holding the serialized hyperparameter model called `name`."""
    return f"{HPARAMS_ARTIFACT_DIR}/{name}.json"


DEFAULT_MAX_PATCH_BYTES = 10 * 1024 * 1024

ENV_CAPTURE_CODE = "EXP_TRACK_CAPTURE_CODE"
ENV_MAX_PATCH_BYTES = "EXP_TRACK_MAX_PATCH_BYTES"
