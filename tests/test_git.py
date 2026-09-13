"""Tests for the git primitives, run against real temporary repositories."""

import os
import tracemalloc
from pathlib import Path

import pytest
from conftest import GitRepo

from runsnap._git import (
    GitError,
    PatchTooLarge,
    add_worktree,
    apply_patch,
    build_patch,
    checkout_new_branch,
    current_branch,
    diff_commits,
    find_repo_root,
    head_commit,
    list_worktrees,
    patch_files,
    read_state,
    remote_url,
    unquote_path,
)

ROOMY = 10 * 1024 * 1024
"""A ceiling no test tree comes near, so the cap stays out of the way."""


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

    build_patch(repo.path, ROOMY)

    assert repo.status() == before


def test_patch_captures_same_size_edit_with_unchanged_timestamp(
    repo: GitRepo, tmp_path: Path
) -> None:
    # Model a filesystem where an edit shares the cached file's stat data.
    repo.git("config", "core.trustctime", "false")
    repo.git("config", "core.checkstat", "minimal")
    source = repo.write("main.py", "print('A')\n")
    timestamp = 1_700_000_000
    os.utime(source, (timestamp, timestamp))
    base = repo.commit("base")
    index = repo.path / ".git" / "index"
    # Git must treat the file as potentially changed when its timestamp
    # matches the index's, even though its stat data looks unchanged.
    os.utime(index, (timestamp, timestamp))
    repo.write("main.py", "print('B')\n")
    os.utime(source, (timestamp, timestamp))
    original_index = index.read_bytes()
    original_mtime = index.stat().st_mtime_ns

    patch = build_patch(repo.path, ROOMY)

    assert b"-print('A')\n+print('B')" in patch
    assert index.read_bytes() == original_index
    assert index.stat().st_mtime_ns == original_mtime
    assert source.read_text() == "print('B')\n"
    assert head_commit(repo.path) == base
    clone = repo.clone_at(base, tmp_path / "clone")
    apply_patch(clone, patch)
    assert (clone / "main.py").read_text() == "print('B')\n"


def test_patch_round_trip_reproduces_the_working_tree(
    repo: GitRepo, tmp_path: Path
) -> None:
    base = make_dirty(repo)
    patch = build_patch(repo.path, ROOMY)

    clone = repo.clone_at(base, tmp_path / "clone")
    apply_patch(clone, patch)

    assert (clone / "tracked.txt").read_text() == "one\ntwo\nthree\n"
    assert (clone / "untracked.txt").read_text() == "new file\n"
    assert not (clone / "deleted.txt").exists()
    assert os.access(clone / "exec.sh", os.X_OK)
    assert (clone / "bin.dat").read_bytes() == b"\x00\xff\x02changed"


def test_patch_excludes_ignored_files(repo: GitRepo, tmp_path: Path) -> None:
    base = make_dirty(repo)
    patch = build_patch(repo.path, ROOMY)

    assert b"ignored.txt" not in patch

    clone = repo.clone_at(base, tmp_path / "clone")
    apply_patch(clone, patch)

    assert not (clone / "ignored.txt").exists()


def test_clean_tree_produces_an_empty_patch(repo: GitRepo) -> None:
    repo.write("a.txt", "one\n")
    repo.commit("base")

    assert build_patch(repo.path, ROOMY) == b""


def test_patch_over_the_ceiling_is_abandoned_unread(repo: GitRepo) -> None:
    repo.write("a.txt", "one\n")
    repo.commit("base")
    repo.write("big.bin", os.urandom(4 * 1024 * 1024))

    tracemalloc.start()
    try:
        with pytest.raises(PatchTooLarge, match="1024 byte limit"):
            build_patch(repo.path, 1024)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < 1024 * 1024


def test_patch_just_under_the_ceiling_is_returned_whole(repo: GitRepo) -> None:
    repo.write("a.txt", "one\n")
    repo.commit("base")
    repo.write("b.txt", "two\n")

    patch = build_patch(repo.path, ROOMY)

    assert b"b.txt" in patch
    assert len(patch) < ROOMY


@pytest.mark.parametrize(
    ("token", "path"),
    [
        pytest.param(b'"caf\\303\\251.txt"', "caf\u00e9.txt", id="non-ascii"),
        pytest.param(b'"say \\"hi\\".txt"', 'say "hi".txt', id="double-quote"),
        pytest.param(b'"back\\\\slash\\tt.txt"', "back\\slash\tt.txt", id="escapes"),
        pytest.param(b'"raw\\377.txt"', "raw\udcff.txt", id="invalid-utf8"),
        pytest.param(b"plain.txt", "plain.txt", id="bare"),
    ],
)
def test_unquote_path_reads_git_quoting(token: bytes, path: str) -> None:
    assert unquote_path(token) == path


@pytest.mark.parametrize(
    ("record", "path"),
    [
        pytest.param(
            rb"""diff --git "a/caf\303\251.txt" "b/caf\303\251.txt"
index 5626abf..814f4a4 100644
--- "a/caf\303\251.txt"
+++ "b/caf\303\251.txt"
@@ -1 +1,2 @@
 one
+two
""",
            "café.txt",
            id="non-ascii",
        ),
        pytest.param(
            rb"""diff --git "a/say \"hi\".txt" "b/say \"hi\".txt"
new file mode 100644
index 0000000..5626abf
--- /dev/null
+++ "b/say \"hi\".txt"
@@ -0,0 +1 @@
+q
""",
            'say "hi".txt',
            id="double-quote",
        ),
        pytest.param(
            rb"""diff --git a/dir b/x.txt b/dir b/x.txt
new file mode 100644
index 0000000..5626abf
--- /dev/null
+++ b/dir b/x.txt
@@ -0,0 +1 @@
+sep
""",
            "dir b/x.txt",
            id="separator-in-path",
        ),
        pytest.param(
            rb"""diff --git a/old.txt b/new.txt
similarity index 100%
rename from old.txt
rename to new.txt
""",
            "new.txt",
            id="rename",
        ),
        pytest.param(
            rb"""diff --git a/plain.txt "b/caf\303\251.txt"
similarity index 100%
rename from plain.txt
rename to "caf\303\251.txt"
""",
            "café.txt",
            id="rename-to-quoted",
        ),
        pytest.param(
            rb"""diff --git "a/caf\303\251.txt" b/plain b/name.txt
similarity index 100%
rename from "caf\303\251.txt"
rename to plain b/name.txt
""",
            "plain b/name.txt",
            id="rename-from-quoted",
        ),
        pytest.param(
            rb"""diff --git a/gone.txt b/gone.txt
deleted file mode 100644
index 5626abf..0000000
--- a/gone.txt
+++ /dev/null
@@ -1 +0,0 @@
-gone soon
""",
            "gone.txt",
            id="deletion",
        ),
        pytest.param(
            rb"""diff --git a/bin.dat b/bin.dat
new file mode 100644
index 0000000..d98e7c5
GIT binary patch
literal 4
LcmZQzWMT#Y01f~L

literal 0
HcmV?d00001

""",
            "bin.dat",
            id="binary-add",
        ),
    ],
)
def test_patch_files_reads_each_kind_of_header(record: bytes, path: str) -> None:
    assert patch_files(record) == [path]


def test_patch_files_keeps_patch_order_across_records() -> None:
    patch = b"".join(
        [
            b"diff --git a/z.txt b/z.txt\n--- a/z.txt\n+++ b/z.txt\n@@ -1 +1 @@\n-1\n+2\n",
            b"diff --git a/old.txt b/new.txt\nrename from old.txt\nrename to new.txt\n",
            b"diff --git a/a.txt b/a.txt\n--- /dev/null\n+++ b/a.txt\n@@ -0,0 +1 @@\n+x\n",
        ]
    )

    assert patch_files(patch) == ["z.txt", "new.txt", "a.txt"]


def test_patch_files_lists_every_path_git_names(repo: GitRepo) -> None:
    repo.write("café.txt", "one\n")
    repo.write("old.txt", "kept\n")
    repo.write("gone.txt", "gone soon\n")
    repo.commit("base")
    repo.write("café.txt", "one\ntwo\n")
    repo.write('say "hi".txt', "q\n")
    repo.write("dir b/x.txt", "sep\n")
    (repo.path / "old.txt").rename(repo.path / "new.txt")
    (repo.path / "gone.txt").unlink()
    repo.write("bin.dat", b"\x00\x01\x02\xff")

    files = patch_files(build_patch(repo.path, ROOMY))

    assert sorted(files) == sorted(
        ["café.txt", 'say "hi".txt', "dir b/x.txt", "new.txt", "gone.txt", "bin.dat"]
    )


def test_worktree_reconstructs_the_recorded_state(
    repo: GitRepo, tmp_path: Path
) -> None:
    repo.write("a.txt", "one\n")
    base = repo.commit("base")
    repo.write("a.txt", "one\ntwo\n")
    repo.write("b.txt", "new\n")
    patch = build_patch(repo.path, ROOMY)
    expected = snapshot(repo.path)

    tree = add_worktree(repo.path, tmp_path / "worktree", "restored", base)
    apply_patch(tree, patch)

    assert snapshot(tree) == expected


@pytest.mark.parametrize("detached", [False, True])
def test_list_worktrees_reports_paths_and_full_branch_refs(
    in_repo: GitRepo, tmp_path: Path, detached: bool
) -> None:
    tree = tmp_path / 'worktree with spaces and "quotes" café'
    if detached:
        in_repo.git("worktree", "add", "--detach", str(tree), "HEAD")
    else:
        add_worktree(in_repo.path, tree, "runsnap/restored", head_commit(in_repo.path))

    assert list_worktrees(in_repo.path) == [
        (in_repo.path, "refs/heads/main"),
        (tree, None if detached else "refs/heads/runsnap/restored"),
    ]


def test_diff_commits_detects_a_rename(in_repo: GitRepo) -> None:
    before = head_commit(in_repo.path)
    (in_repo.path / "main.py").rename(in_repo.path / "renamed.py")
    after = in_repo.commit("rename script")
    in_repo.git("config", "diff.renames", "false")

    compared = diff_commits(in_repo.path, before, after)

    assert (
        "similarity index 100%\nrename from main.py\nrename to renamed.py\n" in compared
    )
    assert "deleted file mode" not in compared
    assert "new file mode" not in compared


def test_in_place_checkout_reconstructs_the_recorded_state(
    repo: GitRepo, tmp_path: Path
) -> None:
    repo.write("a.txt", "one\n")
    base = repo.commit("base")
    repo.write("a.txt", "one\ntwo\n")
    repo.write("b.txt", "new\n")
    patch = build_patch(repo.path, ROOMY)
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
