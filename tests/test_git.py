"""Tests for the git primitives, run against real temporary repositories."""

import os
from pathlib import Path

import pytest
from conftest import GitRepo

from exp_track._git import (
    GitError,
    add_worktree,
    apply_patch,
    build_patch,
    checkout_new_branch,
    current_branch,
    find_repo_root,
    head_commit,
    read_state,
    remote_url,
)


def make_dirty(repo: GitRepo) -> str:
    """A working tree exercising every fidelity case; returns the base commit."""
    repo.write(".gitignore", "ignored.txt\n")
    repo.write("tracked.txt", "one\n")
    repo.write("deleted.txt", "gone soon\n")
    repo.write("exec.sh", "#!/bin/sh\n")
    repo.write("bin.dat", b"\x00\x01\x02original")
    base = repo.commit("base")

    repo.write("tracked.txt", "one\ntwo\n")
    repo.git("add", "tracked.txt")
    repo.write("tracked.txt", "one\ntwo\nthree\n")
    repo.write("untracked.txt", "new file\n")
    (repo.path / "deleted.txt").unlink()
    (repo.path / "exec.sh").chmod(0o755)
    repo.write("bin.dat", b"\x00\xff\x02changed")
    repo.write("ignored.txt", "secret\n")
    return base


def snapshot(root: Path) -> dict[str, tuple[bytes, bool]]:
    """Every file under `root` outside `.git`, as content and executability."""
    return {
        str(path.relative_to(root)): (path.read_bytes(), os.access(path, os.X_OK))
        for path in sorted(root.rglob("*"))
        if path.is_file() and ".git" not in path.relative_to(root).parts
    }


def test_fixture_builds_a_working_repository(repo: GitRepo) -> None:
    repo.write("hello.txt", "hi\n")
    commit = repo.commit("first")

    assert len(commit) == 40
    assert repo.status() == ""


def test_read_state_on_a_branch(repo: GitRepo) -> None:
    repo.write("a.txt", "one\n")
    commit = repo.commit("base")

    state = read_state(repo.path)

    assert state.root == repo.path
    assert state.commit == commit
    assert state.branch == "main"
    assert state.repo_url is None


def test_read_state_with_detached_head(repo: GitRepo) -> None:
    repo.write("a.txt", "one\n")
    first = repo.commit("first")
    repo.write("a.txt", "two\n")
    repo.commit("second")
    repo.git("checkout", "--quiet", first)

    state = read_state(repo.path)

    assert state.commit == first
    assert state.branch is None


def test_read_state_without_commits(repo: GitRepo) -> None:
    assert find_repo_root(repo.path) == repo.path
    assert current_branch(repo.path) == "main"

    with pytest.raises(GitError, match="no commit at HEAD"):
        head_commit(repo.path)


def test_remote_url_strips_credentials(repo: GitRepo) -> None:
    repo.git("remote", "add", "origin", "https://user:token@host/org/repo.git")

    assert remote_url(repo.path) == "https://host/org/repo.git"


def test_build_patch_leaves_git_status_untouched(repo: GitRepo) -> None:
    make_dirty(repo)
    before = repo.status()

    build_patch(repo.path)

    assert repo.status() == before


def test_patch_round_trip_reproduces_the_working_tree(
    repo: GitRepo, tmp_path: Path
) -> None:
    base = make_dirty(repo)
    patch = build_patch(repo.path)

    clone = repo.clone_at(base, tmp_path / "clone")
    apply_patch(clone, patch)

    assert (clone / "tracked.txt").read_text() == "one\ntwo\nthree\n"
    assert (clone / "untracked.txt").read_text() == "new file\n"
    assert not (clone / "deleted.txt").exists()
    assert os.access(clone / "exec.sh", os.X_OK)
    assert (clone / "bin.dat").read_bytes() == b"\x00\xff\x02changed"


def test_patch_excludes_ignored_files(repo: GitRepo, tmp_path: Path) -> None:
    base = make_dirty(repo)
    patch = build_patch(repo.path)

    assert b"ignored.txt" not in patch

    clone = repo.clone_at(base, tmp_path / "clone")
    apply_patch(clone, patch)

    assert not (clone / "ignored.txt").exists()


def test_clean_tree_produces_an_empty_patch(repo: GitRepo) -> None:
    repo.write("a.txt", "one\n")
    repo.commit("base")

    assert build_patch(repo.path) == b""


def test_worktree_reconstructs_the_recorded_state(
    repo: GitRepo, tmp_path: Path
) -> None:
    repo.write("a.txt", "one\n")
    base = repo.commit("base")
    repo.write("a.txt", "one\ntwo\n")
    repo.write("b.txt", "new\n")
    patch = build_patch(repo.path)
    expected = snapshot(repo.path)

    tree = add_worktree(repo.path, tmp_path / "worktree", "restored", base)
    apply_patch(tree, patch)

    assert snapshot(tree) == expected


def test_in_place_checkout_reconstructs_the_recorded_state(
    repo: GitRepo, tmp_path: Path
) -> None:
    repo.write("a.txt", "one\n")
    base = repo.commit("base")
    repo.write("a.txt", "one\ntwo\n")
    repo.write("b.txt", "new\n")
    patch = build_patch(repo.path)
    expected = snapshot(repo.path)

    clone = repo.clone_at(base, tmp_path / "clone")
    checkout_new_branch(clone, "restored", base)
    apply_patch(clone, patch)

    assert current_branch(clone) == "restored"
    assert snapshot(clone) == expected


def test_missing_git_binary_raises(repo: GitRepo, monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))

    with pytest.raises(GitError, match="could not run git"):
        read_state(repo.path)


def test_outside_a_repository_raises(tmp_path: Path) -> None:
    with pytest.raises(GitError, match="not a git repository"):
        find_repo_root(tmp_path)


def test_failing_apply_raises_with_git_stderr(repo: GitRepo) -> None:
    repo.write("a.txt", "one\n")
    repo.commit("base")
    patch = b"""diff --git a/missing.txt b/missing.txt
--- a/missing.txt
+++ b/missing.txt
@@ -1 +1 @@
-old
+new
"""

    with pytest.raises(GitError, match="missing.txt"):
        apply_patch(repo.path, patch)
