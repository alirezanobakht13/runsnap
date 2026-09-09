"""Names and limits for what a captured run records."""

TAG_PREFIX = "runsnap.git."

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

ENV_CAPTURE_CODE = "RUNSNAP_CAPTURE_CODE"
ENV_MAX_PATCH_BYTES = "RUNSNAP_MAX_PATCH_BYTES"

TB_TAG_LOGDIR = "runsnap.tb.logdir"

TB_ARTIFACT_DIR = "tb"
TB_LIGHT_SUFFIX = ".scalars"
TB_MEDIA_SUFFIX = ".media"


def tb_media_suffix(shard: int) -> str:
    """Filename suffix for media shard number `shard`."""
    return f"{TB_MEDIA_SUFFIX}.{shard}"


TB_HISTOGRAM_BINS = 30
TB_SHARD_MAX_BYTES = 8 * 1024 * 1024
TB_SYNC_INTERVAL_SECONDS = 30.0
