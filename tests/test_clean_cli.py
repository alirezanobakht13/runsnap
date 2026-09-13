"""Reconstruction cleanup against real repositories and worktrees."""

import shutil
import sys
from pathlib import Path

import pytest
from conftest import GitRepo

from runsnap import _cli, _git


@pytest.fixture
def leftovers(in_repo: GitRepo, tmp_path: Path) -> dict[str, Path]:
    trees = {
        "runsnap/restored": tmp_path / "restored tree",
        "runsnap/missing": tmp_path / "missing tree",
        "runsnap-other/keep": tmp_path / "unrelated tree",
    }
    for branch, path in trees.items():
        _git.add_worktree(in_repo.path, path, branch, "HEAD")
        (path / "main.py").write_text("print('uncommitted')\n")
        (path / "untracked.txt").write_text("untracked\n")
    shutil.rmtree(trees["runsnap/missing"])
    in_repo.git("branch", "runsnap/orphan")
    in_repo.git("branch", "unrelated-orphan")
    trees["detached"] = tmp_path / "detached tree"
    in_repo.git("worktree", "add", "--detach", str(trees["detached"]), "HEAD")
    in_repo.write("main.py", "print('user work')\n")
    return trees


@pytest.mark.parametrize("prune", [False, True])
def test_clean_lists_only_runsnap_leftovers_without_changing_them(
    prune, in_repo: GitRepo, leftovers, capsys, monkeypatch
) -> None:
    if prune:
        in_repo.git("worktree", "prune")
    worktrees = in_repo.git("worktree", "list", "--porcelain")
    branches = in_repo.git("show-ref", "--heads")
    status = in_repo.status()
    monkeypatch.setattr(sys, "argv", ["runsnap", "clean"])

    with pytest.raises(SystemExit) as raised:
        _cli.main()

    assert raised.value.code == 0
    captured = capsys.readouterr()
    missing = "(no worktree)" if prune else str(leftovers["runsnap/missing"])
    assert captured.out.splitlines() == [
        f"runsnap/missing  {missing}",
        "runsnap/orphan  (no worktree)",
        f"runsnap/restored  {leftovers['runsnap/restored']}",
    ]
    assert captured.err == ""
    assert in_repo.git("worktree", "list", "--porcelain") == worktrees
    assert in_repo.git("show-ref", "--heads") == branches
    assert in_repo.status() == status
    for branch in ("runsnap/restored", "runsnap-other/keep"):
        assert (leftovers[branch] / "main.py").read_text() == "print('uncommitted')\n"
        assert (leftovers[branch] / "untracked.txt").read_text() == "untracked\n"


@pytest.mark.parametrize("prune", [False, True])
def test_clean_removes_only_runsnap_leftovers_and_reports_each_removal(
    prune, in_repo: GitRepo, leftovers, tmp_path, capsys, monkeypatch
) -> None:
    if prune:
        in_repo.git("worktree", "prune")
    status = in_repo.status()
    worktrees = [
        (path, branch)
        for path, branch in _git.list_worktrees(in_repo.path)
        if branch is None or not branch.startswith("refs/heads/runsnap/")
    ]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["runsnap", "clean", "--remove", "--repo", str(in_repo.path)]
    )

    with pytest.raises(SystemExit) as raised:
        _cli.main()

    assert raised.value.code == 0
    captured = capsys.readouterr()
    expected = [] if prune else [f"Removed worktree: {leftovers['runsnap/missing']}"]
    expected.extend(
        [
            "Removed branch: runsnap/missing",
            "Removed branch: runsnap/orphan",
            f"Removed worktree: {leftovers['runsnap/restored']}",
            "Removed branch: runsnap/restored",
        ]
    )
    assert captured.out.splitlines() == expected
    assert captured.err == ""
    assert not leftovers["runsnap/restored"].exists()
    assert not leftovers["runsnap/missing"].exists()
    assert _git.list_worktrees(in_repo.path) == worktrees
    assert in_repo.git(
        "for-each-ref", "--format=%(refname)", "refs/heads/"
    ).splitlines() == [
        "refs/heads/main",
        "refs/heads/runsnap-other/keep",
        "refs/heads/unrelated-orphan",
    ]
    assert in_repo.status() == status
    assert (
        leftovers["runsnap-other/keep"] / "main.py"
    ).read_text() == "print('uncommitted')\n"
    assert (
        leftovers["runsnap-other/keep"] / "untracked.txt"
    ).read_text() == "untracked\n"
    assert (leftovers["detached"] / "main.py").read_text() == "print('hello')\n"


def test_clean_reports_completed_removals_when_a_later_worktree_is_locked(
    in_repo: GitRepo, leftovers, capsys, monkeypatch
) -> None:
    tree = leftovers["runsnap/restored"]
    in_repo.git("worktree", "lock", str(tree))
    monkeypatch.setattr(sys, "argv", ["runsnap", "clean", "--remove"])

    with pytest.raises(SystemExit) as raised:
        _cli.main()

    assert raised.value.code == 1
    captured = capsys.readouterr()
    assert captured.out.splitlines() == [
        f"Removed worktree: {leftovers['runsnap/missing']}",
        "Removed branch: runsnap/missing",
        "Removed branch: runsnap/orphan",
    ]
    assert "locked" in captured.err
    assert str(tree) in captured.err
    assert tree.exists()
    assert _git.branch_exists(in_repo.path, "runsnap/restored")


@pytest.mark.parametrize("remove", [False, True])
def test_clean_reports_nothing_to_remove_successfully(
    remove, in_repo: GitRepo, capsys, monkeypatch
) -> None:
    worktrees = in_repo.git("worktree", "list", "--porcelain")
    branches = in_repo.git("show-ref", "--heads")
    monkeypatch.setattr(
        sys, "argv", ["runsnap", "clean", *(["--remove"] if remove else [])]
    )

    with pytest.raises(SystemExit) as raised:
        _cli.main()

    assert raised.value.code == 0
    captured = capsys.readouterr()
    assert captured.out == "No runsnap worktrees or branches to remove.\n"
    assert captured.err == ""
    assert in_repo.git("worktree", "list", "--porcelain") == worktrees
    assert in_repo.git("show-ref", "--heads") == branches
