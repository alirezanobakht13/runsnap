"""Throwaway git repositories for exercising the git primitives."""

import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest


@dataclass
class GitRepo:
    """A throwaway repository whose working tree a test can shape freely."""

    path: Path

    def git(self, *args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=self.path,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout

    def write(
        self, name: str, content: str | bytes = "", *, executable: bool = False
    ) -> Path:
        """Create or overwrite a file in the working tree."""
        target = self.path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content)
        if executable:
            target.chmod(0o755)
        return target

    def commit(self, message: str = "change") -> str:
        """Commit everything in the working tree and return the new commit SHA."""
        self.git("add", "-A")
        self.git("commit", "--quiet", "-m", message)
        return self.git("rev-parse", "HEAD").strip()

    def status(self) -> str:
        return self.git("status", "--porcelain=v1", "--untracked-files=all")

    def clone_at(self, commit: str, dest: Path) -> Path:
        """A clean checkout of `commit` in a separate directory."""
        subprocess.run(
            ["git", "clone", "--quiet", str(self.path), str(dest)],
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "checkout", "--quiet", commit],
            cwd=dest,
            capture_output=True,
            check=True,
        )
        return dest


@pytest.fixture
def repo(tmp_path: Path) -> GitRepo:
    """An initialized repository with no commits and an empty working tree."""
    path = tmp_path / "repo"
    path.mkdir()
    created = GitRepo(path)
    created.git("init", "--quiet", "-b", "main")
    created.git("config", "user.email", "test@example.com")
    created.git("config", "user.name", "Test")
    created.git("config", "commit.gpgsign", "false")
    return created


@pytest.fixture
def tracking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A local sqlite-backed MLflow tracking store, isolated per test."""
    import mlflow

    from runsnap._capture import reset_code_state_cache

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    client = mlflow.MlflowClient()
    experiment_id = client.create_experiment(
        "runsnap-tests", artifact_location=artifacts.as_uri()
    )
    mlflow.set_experiment(experiment_id=experiment_id)
    reset_code_state_cache()
    yield client
    while mlflow.active_run() is not None:
        mlflow.end_run()
    reset_code_state_cache()


@pytest.fixture
def in_repo(repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> GitRepo:
    """A repository with one commit, made the process working directory."""
    repo.write("main.py", "print('hello')\n")
    repo.commit("base")
    monkeypatch.chdir(repo.path)
    return repo
