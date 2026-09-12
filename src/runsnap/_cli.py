"""The `runsnap` command line application."""

import re
import subprocess
import sys
import tempfile
from importlib.util import find_spec
from pathlib import Path

from cyclopts import App
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

from runsnap._git import (
    GitError,
    add_worktree,
    apply_patch,
    branch_exists,
    checkout_existing,
    checkout_new_branch,
    commit_all,
    commit_exists,
    current_branch,
    delete_branch,
    find_repo_root,
    head_commit,
    is_dirty,
    patch_files,
    remove_worktree,
)
from runsnap._lifecycle import attempt_runs
from runsnap._tags import (
    PATCH_ARTIFACT_PATH,
    TAG_BRANCH,
    TAG_COMMIT,
    TAG_CONTINUES,
    TAG_DIRTY,
    TAG_PATCH_RUN_ID,
    TAG_PATCH_SHA256,
)
from runsnap._tb_fetch import assemble_logdir

app = App(
    name="runsnap",
    help="Inspect MLflow runs, reconstruct their code, and view TensorBoard data.",
)

RUN_ID = re.compile(r"\A[0-9a-f]{32}\Z")

MLFLOW_RUN_NAME_TAG = "mlflow.runName"


class CliError(Exception):
    """A command cannot proceed, for a reason the user can act on."""


def make_client(tracking_uri: str | None = None) -> MlflowClient:
    """A client for the tracking server, `tracking_uri` overriding the environment."""
    return MlflowClient(tracking_uri=tracking_uri)


def resolve_run(
    client: MlflowClient,
    run_ref: str,
    experiment: str | None = None,
    experiment_ids: list[str] | None = None,
) -> Run:
    """The run `run_ref` names, as either a run id or a run name.

    A value in MLflow's run id form is looked up directly; anything else is
    matched against run names, across every experiment unless `experiment`
    narrows the search. `experiment_ids` are the ids to search, already
    resolved, so a caller resolving several names asks the server for the
    experiment list once.
    """
    if RUN_ID.match(run_ref):
        try:
            return client.get_run(run_ref)
        except Exception as exc:
            raise CliError(f"no run {run_ref} on {client.tracking_uri}: {exc}") from exc
    if experiment_ids is None:
        experiment_ids = _experiment_ids(client, experiment)
    matches = client.search_runs(
        experiment_ids,
        filter_string=f"tags.\"{MLFLOW_RUN_NAME_TAG}\" = '{_quote(run_ref)}'",
    )
    if not matches:
        raise CliError(
            f"no run named {run_ref!r} on {client.tracking_uri}"
            + (f" in experiment {experiment!r}" if experiment else "")
        )
    if len(matches) > 1:
        listed = "\n".join(
            f"  {run.info.run_id}  {client.get_experiment(run.info.experiment_id).name}"
            for run in matches
        )
        raise CliError(
            f"{len(matches)} runs are named {run_ref!r}; "
            f"pass one of these run ids instead:\n{listed}"
        )
    return matches[0]


def _experiment_ids(client: MlflowClient, experiment: str | None) -> list[str]:
    if experiment is None:
        ids = []
        token = None
        while True:
            page = client.search_experiments(page_token=token)
            ids.extend(found.experiment_id for found in page)
            token = page.token
            if not token:
                return ids
    found = client.get_experiment_by_name(experiment)
    if found is None:
        raise CliError(f"no experiment named {experiment!r} on {client.tracking_uri}")
    return [found.experiment_id]


def _quote(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _tb_runs(
    client: MlflowClient,
    run_refs: tuple[str, ...],
    experiment: str | None,
    filter_string: str,
) -> list[Run]:
    """The selected runs, enumerating the server's experiments at most once."""
    if run_refs and not filter_string:
        ids = (
            _experiment_ids(client, experiment)
            if any(not RUN_ID.match(ref) for ref in run_refs)
            else None
        )
        return [resolve_run(client, ref, experiment, ids) for ref in run_refs]
    experiment_ids = _experiment_ids(client, experiment)
    selected_ids = {
        resolve_run(client, ref, experiment, experiment_ids).info.run_id
        for ref in run_refs
    }
    if not experiment_ids:
        return []
    matches = []
    token = None
    while True:
        page = client.search_runs(
            experiment_ids, filter_string=filter_string, page_token=token
        )
        matches.extend(
            run for run in page if not run_refs or run.info.run_id in selected_ids
        )
        token = page.token
        if not token:
            return matches


def _with_chains(client: MlflowClient, runs: list[Run]) -> list[Run]:
    """`runs` followed by every attempt they continue, each run appearing once."""
    extended = {run.info.run_id: run for run in runs}
    for run in runs:
        for attempt in attempt_runs(client, run.info.run_id):
            extended.setdefault(attempt.info.run_id, attempt)
    return list(extended.values())


@app.command
def tb(
    *run_refs: str,
    experiment: str | None = None,
    filter: str = "",
    chain: bool = False,
    media: bool = False,
    tracking_uri: str | None = None,
) -> None:
    """Open TensorBoard on selected MLflow runs.

    Parameters
    ----------
    run_refs
        Run ids or names. Omit to search all runs in the selected experiments.
    experiment
        Experiment name, defaulting to all experiments.
    filter
        MLflow filter expression to narrow the selected runs.
    chain
        Also show every earlier attempt each selected run continues.
    media
        Include images, histograms, and other media alongside scalar data.
    tracking_uri
        Tracking server to query, overriding the MLflow environment.
    """
    client = make_client(tracking_uri)
    runs = _tb_runs(client, run_refs, experiment, filter)
    if chain:
        runs = _with_chains(client, runs)
    with assemble_logdir(client, runs, media=media) as logdir:
        if not any(logdir.iterdir()):
            print("No runs found with TensorBoard data matching the selection.")
            return
        _launch_tensorboard(logdir)


def _launch_tensorboard(logdir: Path) -> None:
    if find_spec("tensorboard") is None:
        raise CliError("TensorBoard is required; install it with `uv add tensorboard`.")
    try:
        subprocess.run(
            [sys.executable, "-m", "tensorboard.main", "--logdir", str(logdir)],
            check=True,
        )
    except KeyboardInterrupt:
        return
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CliError(f"TensorBoard could not run: {exc}") from exc


def code_state(run: Run) -> dict[str, str]:
    """The `runsnap.git.*` tags on `run`, or nothing when it was never captured."""
    tags = run.data.tags
    if TAG_COMMIT not in tags:
        raise CliError(
            f"run {run.info.run_id} carries no code state; it was not started "
            f"with runsnap.start_run()"
        )
    return tags


def patch_holder(run: Run) -> str:
    """The run holding this run's patch artifact: an ancestor's, or its own."""
    return run.data.tags.get(TAG_PATCH_RUN_ID, run.info.run_id)


def download_patch(client: MlflowClient, run: Run) -> bytes:
    """The recorded patch bytes for `run`, followed to the run that holds them."""
    holder = patch_holder(run)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            local = client.download_artifacts(holder, PATCH_ARTIFACT_PATH, tmp)
        except Exception as exc:
            raise CliError(
                f"run {run.info.run_id} records a dirty tree but run {holder} "
                f"has no {PATCH_ARTIFACT_PATH} artifact: {exc}"
            ) from exc
        return Path(local).read_bytes()


@app.command
def show(
    run_ref: str,
    *,
    experiment: str | None = None,
    tracking_uri: str | None = None,
) -> None:
    """Print a run's recorded code state, without touching any repository.

    Parameters
    ----------
    run_ref
        An MLflow run id or a run name.
    experiment
        Name of the experiment to resolve a run name in.
    tracking_uri
        Tracking server to query, overriding the MLflow environment.
    """
    client = make_client(tracking_uri)
    run = resolve_run(client, run_ref, experiment)
    tags = code_state(run)
    dirty = tags.get(TAG_DIRTY) == "true"
    print(f"run:     {run.info.run_id}")
    print(f"name:    {run.info.run_name}")
    if TAG_CONTINUES in tags:
        print(f"continues: {tags[TAG_CONTINUES]}")
    print(f"commit:  {tags[TAG_COMMIT]}")
    print(f"branch:  {tags.get(TAG_BRANCH, '(detached)')}")
    print(f"dirty:   {'true' if dirty else 'false'}")
    if not dirty:
        print("patch:   (none)")
        return
    print(f"patch:   sha256:{tags.get(TAG_PATCH_SHA256, '(unrecorded)')}")
    print(f"held by: {patch_holder(run)}")
    print("files:")
    for path in patch_files(download_patch(client, run)):
        print(f"  {path}")


def resolve_repo(repo: Path | None) -> Path:
    """Root of the repository to reconstruct into."""
    try:
        return find_repo_root(repo)
    except GitError as exc:
        where = f"{repo}" if repo is not None else "the working directory"
        raise CliError(
            f"{where} is not inside a git repository; run runsnap from the "
            f"repository holding the run's code, or pass --repo PATH"
        ) from exc


def branch_name(run: Run) -> str:
    """A branch name carrying the run's own name and a short form of its id."""
    short = run.info.run_id[:8]
    name = _slug(run.info.run_name or "")
    return f"runsnap/{name}-{short}" if name else f"runsnap/{short}"


def _slug(name: str) -> str:
    return re.sub(r"-{2,}", "-", re.sub(r"[^A-Za-z0-9._]", "-", name)).strip("-.")


def worktree_path(root: Path, branch: str) -> Path:
    """Where a worktree for `branch` goes when the user names no path."""
    return root.parent / f"{root.name}-{branch.replace('/', '-')}"


@app.command
def checkout(
    run_ref: str,
    *,
    experiment: str | None = None,
    tracking_uri: str | None = None,
    repo: Path | None = None,
    path: Path | None = None,
    branch: str | None = None,
    worktree: bool = True,
    commit: bool = False,
    force: bool = False,
) -> None:
    """Reconstruct a run's code state on a new branch and report where it landed.

    A branch is created at the run's recorded commit and the run's recorded
    patch is applied on top, left uncommitted so the reconstructed tree carries
    exactly the uncommitted work the run was started with.

    Parameters
    ----------
    run_ref
        An MLflow run id or a run name.
    experiment
        Name of the experiment to resolve a run name in.
    tracking_uri
        Tracking server to query, overriding the MLflow environment.
    repo
        Repository to reconstruct into, defaulting to the working directory's.
    path
        Where to create the worktree, defaulting to a directory beside the
        repository named after the branch.
    branch
        Name for the created branch, defaulting to one derived from the run.
    worktree
        Create a separate worktree. `--no-worktree` switches the current
        working tree to the new branch instead.
    commit
        Commit the applied patch, leaving the reconstructed tree clean.
    force
        Allow `--no-worktree` to proceed with uncommitted changes present,
        discarding modifications to tracked files.
    """
    client = make_client(tracking_uri)
    run = resolve_run(client, run_ref, experiment)
    tags = code_state(run)
    base = tags[TAG_COMMIT]
    dirty = tags.get(TAG_DIRTY) == "true"

    root = resolve_repo(repo)
    if not commit_exists(root, base):
        raise CliError(
            f"commit {base} is not present in {root}; fetch it first, for "
            f"example with `git fetch --all`"
        )
    target_branch = branch or branch_name(run)
    if branch_exists(root, target_branch):
        raise CliError(
            f"branch {target_branch!r} already exists in {root}; "
            f"pass --branch NAME to use a different name"
        )
    target_path = Path(path) if path is not None else worktree_path(root, target_branch)
    if worktree and target_path.exists():
        raise CliError(
            f"{target_path} already exists; pass --path PATH to place the "
            f"worktree elsewhere"
        )
    if not worktree and not force and is_dirty(root):
        raise CliError(
            f"{root} has uncommitted changes that an in-place checkout would "
            f"carry onto the new branch; commit or stash them, drop "
            f"--no-worktree to reconstruct in a separate worktree, or pass "
            f"--force to discard them"
        )
    patch = download_patch(client, run) if dirty else b""

    tree = _reconstruct(root, target_path, target_branch, base, patch, worktree, force)
    if commit:
        commit_all(tree, f"runsnap: code state of run {run.info.run_id}")

    print(f"branch: {target_branch}")
    print(f"commit: {base}")
    print(f"path:   {tree}")
    if not dirty:
        print("note:   the run's working tree was clean; no patch applied")


def _reconstruct(
    root: Path,
    target_path: Path,
    target_branch: str,
    base: str,
    patch: bytes,
    worktree: bool,
    force: bool,
) -> Path:
    """Create the branch, apply the patch, and undo both if the patch fails."""
    previous = current_branch(root) or head_commit(root)
    if worktree:
        tree = add_worktree(root, target_path, target_branch, base)
    else:
        checkout_new_branch(root, target_branch, base, force=force)
        tree = root
    try:
        apply_patch(tree, patch)
    except GitError as exc:
        if worktree:
            remove_worktree(root, tree)
        else:
            checkout_existing(root, previous)
        delete_branch(root, target_branch)
        raise CliError(f"the recorded patch does not apply to {base}: {exc}") from exc
    return tree


def main() -> None:
    """Entry point of the `runsnap` command."""
    try:
        app(sys.argv[1:])
    except CliError as exc:
        print(f"runsnap: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
