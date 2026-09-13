"""Run comparisons against real repositories and a local tracking store."""

import sys
from pathlib import Path
from unittest.mock import Mock

import mlflow
import pytest
from conftest import GitRepo

import runsnap
from runsnap import _cli, _git
from runsnap._tags import PATCH_ARTIFACT_PATH, TAG_COMMIT, TAG_DIRTY


@pytest.fixture
def temporary_trees(in_repo: GitRepo, monkeypatch):
    """Track comparison worktrees and verify that all resources are removed."""
    worktrees = in_repo.git("worktree", "list", "--porcelain")
    branches = in_repo.git("show-ref", "--heads")
    created: list[Path] = []
    add_worktree = _cli.add_worktree

    def record(root, path, branch, commit):
        created.append(path)
        return add_worktree(root, path, branch, commit)

    monkeypatch.setattr(_cli, "add_worktree", record)
    yield created
    assert in_repo.git("worktree", "list", "--porcelain") == worktrees
    assert in_repo.git("show-ref", "--heads") == branches
    assert all(not tree.parent.exists() for tree in created)


@pytest.mark.parametrize("dirty", [False, True])
def test_identical_states_need_no_repository_or_worktree(
    dirty, in_repo: GitRepo, tracking, tmp_path, capsys, monkeypatch
) -> None:
    if dirty:
        in_repo.write("main.py", "print('same patch')\n")
    for name in ("first", "second"):
        with runsnap.start_run(run_name=name):
            pass
    base = _git.head_commit(in_repo.path)
    create = Mock(side_effect=AssertionError("identical states need no worktrees"))
    download = Mock(side_effect=AssertionError("identical states need no patches"))
    monkeypatch.setattr(_cli, "add_worktree", create)
    monkeypatch.setattr(_cli, "download_patch", download)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runsnap",
            "diff",
            "first",
            "second",
            "--experiment",
            "runsnap-tests",
            "--tracking-uri",
            mlflow.get_tracking_uri(),
        ],
    )

    with pytest.raises(SystemExit) as raised:
        _cli.main()

    assert raised.value.code == 0
    printed = capsys.readouterr().out
    assert "Code states are identical." in printed
    assert printed.count(f"commit:  {base}") == 2
    assert printed.count(f"dirty:   {str(dirty).lower()}") == 2
    assert "diff --git" not in printed
    create.assert_not_called()
    download.assert_not_called()


@pytest.mark.parametrize("differences", ["commits", "patches", "both"])
def test_diff_compares_recorded_states_and_preserves_the_users_tree(
    differences, in_repo: GitRepo, tracking, capsys, monkeypatch, tmp_path
) -> None:
    in_repo.write("main.py", "print('A')\n")
    if differences == "commits":
        in_repo.commit("first state")
    with runsnap.start_run() as first:
        first_id = first.info.run_id
    if differences == "both":
        in_repo.write("committed.txt", "new commit\n")
        in_repo.commit("committed difference")
    in_repo.write("main.py", "print('B')\n")
    in_repo.write("added.txt", "added on B\n")
    in_repo.write("weights.bin", b"\x00\xff\x01")
    if differences == "commits":
        in_repo.commit("second state")
    with runsnap.start_run() as second:
        second_id = second.info.run_id
    before = in_repo.status()
    branches = in_repo.git("show-ref", "--heads")
    worktrees = _git.list_worktrees(in_repo.path)
    created = []
    add_worktree = _cli.add_worktree

    def record(root, path, branch, commit):
        created.append(path)
        return add_worktree(root, path, branch, commit)

    monkeypatch.setattr(_cli, "add_worktree", record)
    monkeypatch.chdir(tmp_path)

    _cli.diff(first_id, second_id, repo=in_repo.path)

    printed = capsys.readouterr().out
    for run_id in (first_id, second_id):
        assert run_id in printed
        assert tracking.get_run(run_id).data.tags[TAG_COMMIT] in printed
    assert "-print('A')\n+print('B')" in printed
    assert "+added on B" in printed
    assert "Binary files /dev/null and b/weights.bin differ" in printed
    if differences == "both":
        assert "+new commit" in printed
    assert "diff --git a/.git" not in printed
    assert in_repo.status() == before
    assert in_repo.git("show-ref", "--heads") == branches
    assert _git.list_worktrees(in_repo.path) == worktrees
    assert len(created) == 2
    assert all(not tree.parent.exists() for tree in created)


@pytest.mark.parametrize("failure", ["comparison", "commit"])
def test_diff_cleans_up_even_when_comparison_or_commit_raises(
    failure, in_repo: GitRepo, tracking, temporary_trees, monkeypatch
) -> None:
    with runsnap.start_run() as first:
        first_id = first.info.run_id
    in_repo.write("main.py", "print('changed')\n")
    with runsnap.start_run() as second:
        second_id = second.info.run_id
    fail = Mock(side_effect=RuntimeError("injected failure"))
    monkeypatch.setattr(
        _cli, "diff_commits" if failure == "comparison" else "commit_all", fail
    )

    with pytest.raises(RuntimeError, match="injected failure"):
        _cli.diff(first_id, second_id)

    assert len(temporary_trees) == 2
    fail.assert_called_once()


def test_diff_handles_different_records_with_the_same_final_tree(
    in_repo: GitRepo, tracking, capsys
) -> None:
    in_repo.write("main.py", "print('changed')\n")
    with runsnap.start_run() as first:
        first_id = first.info.run_id
    in_repo.commit("commit the recorded patch")
    with runsnap.start_run() as second:
        second_id = second.info.run_id

    _cli.diff(first_id, second_id)

    assert "No code differences." in capsys.readouterr().out


@pytest.mark.parametrize("failing_side", ["a", "b"])
def test_diff_cleans_up_when_worktree_creation_fails_after_creating_resources(
    failing_side, in_repo: GitRepo, tracking, capsys, temporary_trees, monkeypatch
) -> None:
    with runsnap.start_run() as first:
        first_id = first.info.run_id
    in_repo.write("main.py", "print('changed')\n")
    with runsnap.start_run() as second:
        second_id = second.info.run_id
    add_worktree = _cli.add_worktree

    def fail_after_creation(root, path, branch, commit):
        tree = add_worktree(root, path, branch, commit)
        if path.name == failing_side:
            raise _git.GitError("injected worktree creation failure")
        return tree

    monkeypatch.setattr(_cli, "add_worktree", fail_after_creation)

    _cli.diff(first_id, second_id)

    assert "injected worktree creation failure" in capsys.readouterr().out
    assert len(temporary_trees) == (1 if failing_side == "a" else 2)


def test_diff_compares_flattened_params_without_hparams_artifacts(
    in_repo: GitRepo, tracking, capsys, monkeypatch
) -> None:
    with runsnap.start_run() as first:
        first_id = first.info.run_id
        mlflow.log_params({"model.lr": "0.1", "removed": "old", "same": "value"})
    with runsnap.start_run() as second:
        second_id = second.info.run_id
        mlflow.log_params({"model.lr": "0.2", "added": "new", "same": "value"})
    download = Mock(side_effect=AssertionError("params need no artifacts"))
    monkeypatch.setattr(_cli.MlflowClient, "download_artifacts", download)

    _cli.diff(first_id, second_id)

    printed = capsys.readouterr().out
    assert "  added: added = 'new'\n" in printed
    assert "  removed: removed = 'old'\n" in printed
    assert "  changed: model.lr: '0.1' -> '0.2'\n" in printed
    assert all(
        f"{category}: same" not in printed
        for category in ("added", "removed", "changed")
    )
    download.assert_not_called()


@pytest.mark.parametrize("missing_side", [0, 1])
def test_missing_commit_still_reports_metadata_and_params(
    missing_side, in_repo: GitRepo, tracking, capsys, monkeypatch, temporary_trees
) -> None:
    ids = []
    for value in ("before", "after"):
        with runsnap.start_run() as active:
            ids.append(active.info.run_id)
            mlflow.log_param("value", value)
    missing = "0" * 40
    tracking.set_tag(ids[missing_side], TAG_COMMIT, missing)
    monkeypatch.setattr(sys, "argv", ["runsnap", "diff", *ids])

    with pytest.raises(SystemExit) as raised:
        _cli.main()

    assert raised.value.code == 0
    printed = capsys.readouterr().out
    assert f"commit {missing} is not present" in printed
    assert "git fetch --all" in printed
    assert "Code difference unavailable:" in printed
    assert f"commit:  {missing}" in printed
    assert f"commit:  {_git.head_commit(in_repo.path)}" in printed
    assert printed.count("dirty:   false") == 2
    assert "changed: value: 'before' -> 'after'" in printed
    assert temporary_trees == []


@pytest.mark.parametrize("failing_side", [0, 1])
def test_unapplicable_patch_still_reports_metadata_and_params_and_cleans_up(
    failing_side, in_repo: GitRepo, tracking, capsys, request
) -> None:
    ids = []
    for value in ("before", "after"):
        in_repo.write("main.py", f"print('{value}')\n")
        with runsnap.start_run() as active:
            ids.append(active.info.run_id)
            mlflow.log_param("value", value)
    in_repo.write("main.py", "print('incompatible base')\n")
    incompatible = in_repo.commit("different patch context")
    tracking.set_tag(ids[failing_side], TAG_COMMIT, incompatible)
    temporary_trees = request.getfixturevalue("temporary_trees")

    _cli.diff(*ids)

    printed = capsys.readouterr().out
    assert "Code difference unavailable:" in printed
    assert f"the recorded patch does not apply to {incompatible}" in printed
    assert f"commit:  {incompatible}" in printed
    assert printed.count("dirty:   true") == 2
    assert "changed: value: 'before' -> 'after'" in printed
    assert len(temporary_trees) == failing_side + 1


def test_diff_follows_an_inherited_patch(
    in_repo: GitRepo, tracking, capsys, temporary_trees
) -> None:
    with runsnap.start_run() as first:
        first_id = first.info.run_id
    in_repo.write("main.py", "print('inherited')\n")
    with runsnap.start_run(), runsnap.start_run(nested=True) as child:
        child_id = child.info.run_id

    _cli.diff(first_id, child_id)

    assert "+print('inherited')" in capsys.readouterr().out
    assert len(temporary_trees) == 2


def test_diff_does_not_claim_identity_for_missing_dirty_patches(
    in_repo: GitRepo, tracking, capsys, temporary_trees
) -> None:
    ids = []
    for value in ("before", "after"):
        with mlflow.start_run(
            tags={TAG_COMMIT: _git.head_commit(in_repo.path), TAG_DIRTY: "true"}
        ) as active:
            ids.append(active.info.run_id)
            mlflow.log_param("value", value)

    _cli.diff(*ids)

    printed = capsys.readouterr().out
    assert "identical" not in printed
    assert "Code difference unavailable:" in printed
    assert f"has no {PATCH_ARTIFACT_PATH} artifact" in printed
    assert "changed: value: 'before' -> 'after'" in printed
    assert temporary_trees == []


def test_diff_without_a_repository_still_compares_params(
    in_repo: GitRepo, tracking, tmp_path, capsys, monkeypatch
) -> None:
    with runsnap.start_run() as first:
        first_id = first.info.run_id
        mlflow.log_param("value", "before")
    in_repo.write("main.py", "print('changed')\n")
    with runsnap.start_run() as second:
        second_id = second.info.run_id
        mlflow.log_param("value", "after")
    monkeypatch.chdir(tmp_path)

    _cli.diff(first_id, second_id)

    printed = capsys.readouterr().out
    assert "Code difference unavailable:" in printed
    assert "not inside a git repository" in printed
    assert "--repo PATH" in printed
    assert "changed: value: 'before' -> 'after'" in printed
