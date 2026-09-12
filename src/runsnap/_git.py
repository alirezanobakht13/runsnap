"""Git primitives: reading repository state, building and applying patches."""

import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


class GitError(RuntimeError):
    """A git command failed, or git itself could not be run."""


class PatchTooLarge(GitError):
    """A patch passed the byte ceiling it was read under and was abandoned."""

    def __init__(self, limit: int) -> None:
        super().__init__(f"patch too large: exceeds the {limit} byte limit")
        self.limit = limit


def _git(
    args: Sequence[str],
    cwd: Path | str,
    *,
    env: dict[str, str] | None = None,
    stdin: bytes | None = None,
) -> bytes:
    """Raw stdout of `git <args>`, raising `GitError` with git's stderr on failure."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            env=env,
            input=stdin,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise GitError(f"could not run git: {exc}") from exc
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip()
        raise GitError(f"git {' '.join(args)} failed: {stderr}")
    return result.stdout


def _git_text(
    args: Sequence[str], cwd: Path | str, *, env: dict[str, str] | None = None
) -> str:
    return _git(args, cwd, env=env).decode(errors="replace").strip()


@dataclass(frozen=True)
class GitState:
    """Where a repository stands: its root, base commit, branch, and remote."""

    root: Path
    commit: str
    branch: str | None
    repo_url: str | None


def find_repo_root(path: Path | str | None = None) -> Path:
    """Root of the repository containing `path` (the working directory by default)."""
    cwd = Path(path) if path is not None else Path.cwd()
    return Path(_git_text(["rev-parse", "--show-toplevel"], cwd))


def head_commit(root: Path | str) -> str:
    """Full SHA of the commit at `HEAD`."""
    try:
        return _git_text(["rev-parse", "--verify", "HEAD"], root)
    except GitError as exc:
        raise GitError(f"repository has no commit at HEAD: {exc}") from exc


def current_branch(root: Path | str) -> str | None:
    """Name of the checked-out branch, or `None` when `HEAD` is detached."""
    try:
        return _git_text(["symbolic-ref", "--short", "HEAD"], root)
    except GitError:
        return None


def remote_url(root: Path | str) -> str | None:
    """URL of the `origin` remote with credentials stripped, or `None` if unset."""
    try:
        url = _git_text(["remote", "get-url", "origin"], root)
    except GitError:
        return None
    return _strip_credentials(url)


def _strip_credentials(url: str) -> str:
    parts = urlsplit(url)
    if "@" not in parts.netloc:
        return url
    host = parts.netloc.rsplit("@", 1)[1]
    return urlunsplit(parts._replace(netloc=host))


def commit_exists(root: Path | str, commit: str) -> bool:
    """Whether `commit` is present in the repository at `root`."""
    try:
        _git(["cat-file", "-e", f"{commit}^{{commit}}"], root)
    except GitError:
        return False
    return True


def branch_exists(root: Path | str, branch: str) -> bool:
    """Whether a branch called `branch` is already defined at `root`."""
    try:
        _git(["show-ref", "--verify", "--quiet", f"refs/heads/{branch}"], root)
    except GitError:
        return False
    return True


def is_dirty(root: Path | str) -> bool:
    """Whether the working tree at `root` holds uncommitted or untracked changes."""
    return bool(_git_text(["status", "--porcelain", "--untracked-files=all"], root))


def read_state(path: Path | str | None = None) -> GitState:
    """Repository state at `path`, raising `GitError` outside a repository."""
    root = find_repo_root(path)
    return GitState(
        root=root,
        commit=head_commit(root),
        branch=current_branch(root),
        repo_url=remote_url(root),
    )


_READ_CHUNK = 64 * 1024


def _git_capped(
    args: Sequence[str],
    cwd: Path | str,
    *,
    env: dict[str, str] | None = None,
    limit: int,
) -> bytes:
    """Raw stdout of `git <args>`, read in chunks and abandoned past `limit` bytes.

    Raises `PatchTooLarge` the moment the output passes `limit`, killing git
    rather than draining it, so nothing larger than one chunk past the ceiling
    is ever held.
    """
    try:
        process = subprocess.Popen(
            ["git", *args],
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise GitError(f"could not run git: {exc}") from exc
    chunks: list[bytes] = []
    total = 0
    with process:
        stdout, stderr = process.stdout, process.stderr
        if stdout is None or stderr is None:
            process.kill()
            raise GitError(f"git {' '.join(args)} gave no output to read")
        for chunk in iter(lambda: stdout.read(_READ_CHUNK), b""):
            total += len(chunk)
            if total > limit:
                process.kill()
                raise PatchTooLarge(limit)
            chunks.append(chunk)
        failure = stderr.read().decode(errors="replace").strip()
    if process.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed: {failure}")
    return b"".join(chunks)


def build_patch(root: Path | str, max_bytes: int) -> bytes:
    """Every working-tree change against `HEAD` as one applicable patch.

    Staged changes, unstaged changes, untracked files, deletions, mode changes,
    and binary content are all included; ignored files are not. The repository's
    own index is copied first and only the copy is written to, so the caller's
    staging area, working tree, and `HEAD` are left untouched.

    Raises `PatchTooLarge` once the diff passes `max_bytes`, at which point it is
    abandoned unread; the work and the memory are both bounded by the ceiling.
    """
    index = Path(_git_text(["rev-parse", "--git-path", "index"], root))
    if not index.is_absolute():
        index = Path(root) / index
    with tempfile.TemporaryDirectory() as tmp:
        index_copy = Path(tmp) / "index"
        if index.exists():
            shutil.copyfile(index, index_copy)
        env = {**os.environ, "GIT_INDEX_FILE": str(index_copy)}
        _git(["add", "-A", "-N", "--", "."], root, env=env)
        return _git_capped(
            ["diff", "--no-ext-diff", "--binary", "HEAD"],
            root,
            env=env,
            limit=max_bytes,
        )


_ESCAPES = {b"n": b"\n", b"t": b"\t", b"r": b"\r", b'"': b'"', b"\\": b"\\"}
_ESCAPE = re.compile(rb'\\([0-3][0-7]{2}|[ntr"\\])')


def _unescape(match: re.Match[bytes]) -> bytes:
    escape = match.group(1)
    return bytes([int(escape, 8)]) if len(escape) == 3 else _ESCAPES[escape]


def unquote_path(token: bytes) -> str:
    r"""A path as git prints it, C-quoted or bare, as the `str` the filesystem uses.

    Git quotes a path holding non-ASCII bytes, double quotes, backslashes, or
    control characters, escaping them as `\n \t \r \" \\` and three-digit
    octal `\NNN`. Bytes that are not valid UTF-8 survive as surrogate escapes.
    """
    if token.startswith(b'"'):
        token = _ESCAPE.sub(_unescape, token[1:-1])
    return token.decode(errors="surrogateescape")


_QUOTED_A_SIDE = re.compile(rb'"(?:[^"\\]|\\.)*" (.*)')


def _header_path(header: bytes) -> str | None:
    """Path Q of a `diff --git a/P b/Q` header, or `None` when only `rename to` has it.

    Git quotes each side on its own: a quoted `a/` side delimits itself, and
    the `b/` side is whatever follows. Two bare sides are told apart at the
    ` b/` whose halves match, which finds Q even when it holds ` b/` itself.
    Without such a split P and Q differ, and the record's `rename to` line
    names Q.
    """
    if quoted := _QUOTED_A_SIDE.match(header):
        return unquote_path(quoted.group(1)).removeprefix("b/")
    for split in re.finditer(rb" b/", header):
        if header[2 : split.start()] == header[split.end() :]:
            return unquote_path(header[split.end() :])
    return None


def patch_files(patch: bytes) -> list[str]:
    """Paths a patch touches, in the order the patch names them."""
    files: list[str] = []
    renamed = False
    for line in patch.split(b"\n"):
        if line.startswith(b"diff --git "):
            path = _header_path(line.removeprefix(b"diff --git "))
            renamed = path is None
            if path is not None:
                files.append(path)
        elif renamed and line.startswith(b"rename to "):
            files.append(unquote_path(line.removeprefix(b"rename to ")))
            renamed = False
    return files


def apply_patch(tree: Path | str, patch: bytes) -> None:
    """Apply `patch` to the working tree at `tree`, leaving the result uncommitted."""
    if not patch:
        return
    _git(["apply", "--binary", "--whitespace=nowarn", "-"], tree, stdin=patch)


def add_worktree(root: Path | str, path: Path | str, branch: str, commit: str) -> Path:
    """Create a worktree at `path` on a new `branch` starting at `commit`."""
    _git(["worktree", "add", "-b", branch, str(path), commit], root)
    return Path(path)


def checkout_new_branch(
    root: Path | str, branch: str, commit: str, *, force: bool = False
) -> None:
    """Switch the working tree at `root` to a new `branch` starting at `commit`.

    `force` discards uncommitted changes to tracked files; untracked files are
    left where they are either way.
    """
    _git(["checkout", *(["--force"] if force else []), "-b", branch, commit], root)


def checkout_existing(root: Path | str, ref: str) -> None:
    """Switch the working tree at `root` back to `ref`, discarding tracked changes."""
    _git(["checkout", "--force", ref], root)


def remove_worktree(root: Path | str, path: Path | str) -> None:
    """Delete the worktree at `path` and its registration in the repository."""
    _git(["worktree", "remove", "--force", str(path)], root)


def delete_branch(root: Path | str, branch: str) -> None:
    """Delete `branch`, whether or not its commits are merged."""
    _git(["branch", "--delete", "--force", branch], root)


def commit_all(tree: Path | str, message: str) -> str:
    """Commit everything in the working tree at `tree`, returning the new SHA."""
    _git(["add", "-A", "--", "."], tree)
    _git(["commit", "--quiet", "--message", message], tree)
    return _git_text(["rev-parse", "HEAD"], tree)
