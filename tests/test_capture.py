"""Tests for code state capture onto real MLflow runs."""

import json
import subprocess
import sys
from pathlib import Path

import mlflow
import pytest
from conftest import GitRepo

import runsnap
from runsnap import _capture, _git
from runsnap._tags import (
    MLFLOW_TAG_BRANCH,
    MLFLOW_TAG_COMMIT,
    MLFLOW_TAG_DIRTY,
    MLFLOW_TAG_REPO_URL,
    PATCH_ARTIFACT_PATH,
    TAG_BRANCH,
    TAG_CAPTURE_ERROR,
    TAG_COMMIT,
    TAG_DIRTY,
    TAG_INVOCATION_ARGV,
    TAG_INVOCATION_CWD,
    TAG_PATCH_RUN_ID,
    TAG_PATCH_SHA256,
    TAG_REPO_URL,
)


def tags(client: mlflow.MlflowClient, run_id: str) -> dict[str, str]:
    return client.get_run(run_id).data.tags


def artifact_paths(client: mlflow.MlflowClient, run_id: str) -> set[str]:
    """Every artifact file on a run, as slash-separated paths."""
    found: set[str] = set()
    pending = [""]
    while pending:
        for item in client.list_artifacts(run_id, pending.pop()):
            if item.is_dir:
                pending.append(item.path)
            else:
                found.add(item.path)
    return found


def test_dirty_run_records_tags_and_patch(in_repo: GitRepo, tracking):
    in_repo.git("remote", "add", "origin", "https://user:token@example.com/o/r.git")
    base = in_repo.git("rev-parse", "HEAD").strip()
    in_repo.write("main.py", "print('changed')\n")
    in_repo.write("added.txt", "new\n")

    with runsnap.start_run() as run:
        run_id = run.info.run_id

    recorded = tags(tracking, run_id)
    assert recorded[TAG_COMMIT] == base
    assert recorded[TAG_BRANCH] == "main"
    assert recorded[TAG_DIRTY] == "true"
    assert recorded[TAG_REPO_URL] == "https://example.com/o/r.git"
    assert TAG_PATCH_SHA256 in recorded
    assert artifact_paths(tracking, run_id) == {PATCH_ARTIFACT_PATH}

    local = tracking.download_artifacts(run_id, PATCH_ARTIFACT_PATH)
    patch = Path(local).read_bytes()
    assert b"added.txt" in patch
    assert b"print('changed')" in patch


def test_clean_tree_records_no_patch(in_repo: GitRepo, tracking):
    with runsnap.start_run() as run:
        run_id = run.info.run_id

    recorded = tags(tracking, run_id)
    assert recorded[TAG_DIRTY] == "false"
    assert TAG_PATCH_SHA256 not in recorded
    assert artifact_paths(tracking, run_id) == set()


@pytest.mark.parametrize("directory", [".", "experiments/nested"])
def test_invocation_records_arguments_and_relative_directory(
    in_repo: GitRepo, tracking, monkeypatch: pytest.MonkeyPatch, directory: str
):
    cwd = in_repo.path / directory
    cwd.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(cwd)
    argv = ["train.py", "--name", "two words", "$(echo value);", "", "café"]
    monkeypatch.setattr(sys, "argv", argv)

    with runsnap.start_run() as run:
        run_id = run.info.run_id

    recorded = tags(tracking, run_id)
    assert json.loads(recorded[TAG_INVOCATION_ARGV]) == argv
    assert recorded[TAG_INVOCATION_CWD] == directory
    assert recorded[TAG_COMMIT] == in_repo.git("rev-parse", "HEAD").strip()


def test_invocation_records_absolute_directory_outside_repository(
    in_repo: GitRepo, tracking, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    outside = tmp_path / "repo-other"
    outside.mkdir()
    monkeypatch.setenv("GIT_DIR", str(in_repo.path / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(in_repo.path))
    monkeypatch.chdir(outside)
    argv = [str(in_repo.path / "main.py"), "--steps", "10"]
    monkeypatch.setattr(sys, "argv", argv)

    with runsnap.start_run() as run:
        run_id = run.info.run_id

    recorded = tags(tracking, run_id)
    assert json.loads(recorded[TAG_INVOCATION_ARGV]) == argv
    assert recorded[TAG_INVOCATION_CWD] == str(outside)
    assert recorded[TAG_COMMIT] == in_repo.git("rev-parse", "HEAD").strip()


@pytest.mark.parametrize("refused_tag", [TAG_INVOCATION_ARGV, TAG_INVOCATION_CWD])
def test_refused_invocation_warns_and_leaves_run_usable(
    in_repo: GitRepo, tracking, monkeypatch: pytest.MonkeyPatch, refused_tag: str
):
    set_tag = tracking.set_tag

    def refusing_set_tag(run_id: str, key: str, value):
        if key == refused_tag:
            raise RuntimeError("invocation rejected")
        return set_tag(run_id, key, value)

    monkeypatch.setattr(tracking, "set_tag", refusing_set_tag)
    monkeypatch.setattr(_capture, "MlflowClient", lambda: tracking)

    with pytest.warns(UserWarning, match="invocation rejected"):
        run = runsnap.start_run()
    with run:
        run_id = run.info.run_id
        mlflow.log_metric("score", 1.0)

    finished = tracking.get_run(run_id)
    assert finished.info.status == "FINISHED"
    assert finished.data.metrics["score"] == 1.0
    assert finished.data.tags[TAG_CAPTURE_ERROR] == "invocation rejected"
    assert refused_tag not in finished.data.tags
    assert TAG_COMMIT in finished.data.tags


def test_identical_trees_share_a_digest(in_repo: GitRepo, tracking):
    in_repo.write("main.py", "print('changed')\n")
    with runsnap.start_run() as first:
        first_id = first.info.run_id
    with runsnap.start_run() as second:
        second_id = second.info.run_id
    assert (
        tags(tracking, first_id)[TAG_PATCH_SHA256]
        == (tags(tracking, second_id)[TAG_PATCH_SHA256])
    )


def test_edit_between_runs_records_a_new_patch(in_repo: GitRepo, tracking):
    in_repo.write("main.py", "print('changed')\n")
    with runsnap.start_run() as first:
        first_id = first.info.run_id
    in_repo.write("main.py", "print('changed again')\n")
    with runsnap.start_run() as second:
        second_id = second.info.run_id

    assert (
        tags(tracking, first_id)[TAG_PATCH_SHA256]
        != (tags(tracking, second_id)[TAG_PATCH_SHA256])
    )
    local = tracking.download_artifacts(second_id, PATCH_ARTIFACT_PATH)
    assert b"print('changed again')" in Path(local).read_bytes()


def test_commit_between_runs_records_the_new_commit(in_repo: GitRepo, tracking):
    in_repo.write("main.py", "print('changed')\n")
    with runsnap.start_run() as first:
        first_id = first.info.run_id
    committed = in_repo.commit("between runs")
    with runsnap.start_run() as second:
        second_id = second.info.run_id

    assert tags(tracking, first_id)[TAG_COMMIT] != committed
    recorded = tags(tracking, second_id)
    assert recorded[TAG_COMMIT] == committed
    assert recorded[TAG_DIRTY] == "false"
    assert TAG_PATCH_SHA256 not in recorded
    assert artifact_paths(tracking, second_id) == set()


def test_nested_runs_point_at_the_parents_patch(in_repo: GitRepo, tracking):
    in_repo.write("main.py", "print('changed')\n")
    child_ids = []
    with runsnap.start_run() as parent:
        parent_id = parent.info.run_id
        for _ in range(2):
            with runsnap.start_run(nested=True) as child:
                child_ids.append(child.info.run_id)

    assert artifact_paths(tracking, parent_id) == {PATCH_ARTIFACT_PATH}
    assert TAG_PATCH_RUN_ID not in tags(tracking, parent_id)
    for child_id in child_ids:
        recorded = tags(tracking, child_id)
        assert recorded[TAG_COMMIT] == tags(tracking, parent_id)[TAG_COMMIT]
        assert recorded[TAG_PATCH_SHA256] == tags(tracking, parent_id)[TAG_PATCH_SHA256]
        assert recorded[TAG_PATCH_RUN_ID] == parent_id
        assert artifact_paths(tracking, child_id) == set()


def test_nested_run_with_a_changed_tree_uploads_its_own_patch(
    in_repo: GitRepo, tracking
):
    in_repo.write("main.py", "print('changed')\n")
    with runsnap.start_run() as parent:
        parent_id = parent.info.run_id
        in_repo.write("main.py", "print('changed again')\n")
        with runsnap.start_run(nested=True) as child:
            child_id = child.info.run_id

    recorded = tags(tracking, child_id)
    assert recorded[TAG_PATCH_SHA256] != tags(tracking, parent_id)[TAG_PATCH_SHA256]
    assert TAG_PATCH_RUN_ID not in recorded
    assert artifact_paths(tracking, child_id) == {PATCH_ARTIFACT_PATH}
    local = tracking.download_artifacts(child_id, PATCH_ARTIFACT_PATH)
    assert b"print('changed again')" in Path(local).read_bytes()


def test_oversized_patch_is_skipped(in_repo: GitRepo, tracking, monkeypatch):
    monkeypatch.setenv("RUNSNAP_MAX_PATCH_BYTES", "10")
    in_repo.write("main.py", "print('a much longer line than ten bytes')\n")

    with pytest.warns(UserWarning, match="exceeds"), runsnap.start_run() as run:
        run_id = run.info.run_id

    recorded = tags(tracking, run_id)
    assert recorded[TAG_DIRTY] == "true"
    assert "too large" in recorded[TAG_CAPTURE_ERROR]
    assert "10 byte" in recorded[TAG_CAPTURE_ERROR]
    assert TAG_PATCH_SHA256 not in recorded
    assert artifact_paths(tracking, run_id) == set()


def test_oversized_patch_leaves_the_run_usable(in_repo: GitRepo, tracking, monkeypatch):
    monkeypatch.setenv("RUNSNAP_MAX_PATCH_BYTES", "10")
    base = in_repo.git("rev-parse", "HEAD").strip()
    in_repo.write("main.py", "print('a much longer line than ten bytes')\n")

    with pytest.warns(UserWarning, match="exceeds"), runsnap.start_run() as run:
        run_id = run.info.run_id
    mlflow.log_metric("loss", 0.5, run_id=run_id)

    recorded = tags(tracking, run_id)
    assert recorded[TAG_COMMIT] == base
    assert recorded[TAG_BRANCH] == "main"
    assert recorded[MLFLOW_TAG_DIRTY] == "true"
    assert tracking.get_run(run_id).data.metrics["loss"] == 0.5


def test_mlflow_git_tags_are_filled_when_absent(in_repo: GitRepo, tracking):
    in_repo.git("remote", "add", "origin", "https://example.com/o/r.git")
    in_repo.write("main.py", "print('changed')\n")

    with runsnap.start_run() as run:
        run_id = run.info.run_id

    recorded = tags(tracking, run_id)
    assert recorded[MLFLOW_TAG_COMMIT] == recorded[TAG_COMMIT]
    assert recorded[MLFLOW_TAG_BRANCH] == "main"
    assert recorded[MLFLOW_TAG_REPO_URL] == "https://example.com/o/r.git"
    assert recorded[MLFLOW_TAG_DIRTY] == "true"
    assert "mlflow.source.git.diff" not in recorded


def test_mlflow_git_tags_already_set_are_left_alone(in_repo: GitRepo, tracking):
    with runsnap.start_run(tags={MLFLOW_TAG_COMMIT: "resolved-by-mlflow"}) as run:
        run_id = run.info.run_id

    recorded = tags(tracking, run_id)
    assert recorded[MLFLOW_TAG_COMMIT] == "resolved-by-mlflow"
    assert recorded[TAG_COMMIT] != "resolved-by-mlflow"


def test_outside_a_repository_records_nothing(tmp_path: Path, tracking, monkeypatch):
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.chdir(outside)

    with pytest.warns(UserWarning, match="no code state"), runsnap.start_run() as run:
        run_id = run.info.run_id

    recorded = tags(tracking, run_id)
    assert not [key for key in recorded if key.startswith("runsnap.")]


def test_missing_git_binary_warns_only(in_repo: GitRepo, tracking, monkeypatch):
    def no_git(*args, **kwargs):
        raise OSError("No such file or directory: 'git'")

    monkeypatch.setattr(subprocess, "run", no_git)

    with (
        pytest.warns(UserWarning, match="could not run git"),
        runsnap.start_run() as run,
    ):
        run_id = run.info.run_id

    monkeypatch.undo()
    assert not [key for key in tags(tracking, run_id) if key.startswith("runsnap.")]


def test_repository_with_no_commits_records_the_reason(
    repo: GitRepo, tracking, monkeypatch
):
    monkeypatch.chdir(repo.path)

    with pytest.warns(UserWarning, match="no commit"), runsnap.start_run() as run:
        run_id = run.info.run_id

    recorded = tags(tracking, run_id)
    assert "no commit" in recorded[TAG_CAPTURE_ERROR]
    assert TAG_COMMIT not in recorded
    assert artifact_paths(tracking, run_id) == set()


def test_unexpected_git_failure_does_not_escape(
    in_repo: GitRepo, tracking, monkeypatch
):
    def broken(root, max_bytes):
        raise _git.GitError("git diff failed: catastrophe")

    monkeypatch.setattr(_capture, "build_patch", broken)

    with pytest.warns(UserWarning, match="catastrophe"), runsnap.start_run() as run:
        run_id = run.info.run_id

    assert "catastrophe" in tags(tracking, run_id)[TAG_CAPTURE_ERROR]


def test_start_run_forwards_arguments_and_wraps_mlflows_run(in_repo: GitRepo, tracking):
    with runsnap.start_run(run_name="named", tags={"custom": "value"}) as run:
        assert isinstance(run, mlflow.ActiveRun)
        active = mlflow.active_run()
        assert active is not None and active.info.run_id == run.info.run_id
        run_id = run.info.run_id
        mlflow.log_metric("score", 1.0)

    assert mlflow.active_run() is None
    finished = tracking.get_run(run_id)
    assert finished.data.tags["mlflow.runName"] == "named"
    assert finished.data.tags["custom"] == "value"
    assert finished.data.metrics["score"] == 1.0


def test_capture_code_false_suppresses_capture(in_repo: GitRepo, tracking):
    in_repo.write("main.py", "print('changed')\n")
    with runsnap.start_run(capture_code=False) as run:
        run_id = run.info.run_id

    assert not [key for key in tags(tracking, run_id) if key.startswith("runsnap.")]
    assert artifact_paths(tracking, run_id) == set()


def test_environment_opt_out_suppresses_capture(
    in_repo: GitRepo, tracking, monkeypatch
):
    monkeypatch.setenv("RUNSNAP_CAPTURE_CODE", "0")
    in_repo.write("main.py", "print('changed')\n")
    with runsnap.start_run() as run:
        run_id = run.info.run_id

    assert not [key for key in tags(tracking, run_id) if key.startswith("runsnap.")]
    assert artifact_paths(tracking, run_id) == set()
