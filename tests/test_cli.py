"""Tests for the `runsnap` command line application."""

import sys
from pathlib import Path

import mlflow
import pytest
from conftest import GitRepo

import runsnap
from runsnap import _cli, _git
from runsnap._cli import (
    CliError,
    checkout,
    main,
    make_client,
    resolve_run,
    show,
)
from runsnap._tags import TAG_COMMIT, TAG_PATCH_RUN_ID, TAG_PATCH_SHA256


def test_run_id_reference_is_looked_up_directly(tracking) -> None:
    with mlflow.start_run() as active:
        run_id = active.info.run_id

    assert resolve_run(make_client(), run_id).info.run_id == run_id


def test_unique_run_name_resolves(tracking) -> None:
    with mlflow.start_run(run_name="baseline-lr3") as active:
        run_id = active.info.run_id

    assert resolve_run(make_client(), "baseline-lr3").info.run_id == run_id


def test_ambiguous_run_name_lists_candidates(tracking) -> None:
    ids = []
    for _ in range(2):
        with mlflow.start_run(run_name="twin") as active:
            ids.append(active.info.run_id)

    with pytest.raises(CliError) as raised:
        resolve_run(make_client(), "twin")

    message = str(raised.value)
    assert "2 runs are named 'twin'" in message
    assert all(run_id in message for run_id in ids)
    assert "runsnap-tests" in message


def test_unknown_reference_names_the_reference_and_server(tracking) -> None:
    with pytest.raises(CliError) as raised:
        resolve_run(make_client(), "nowhere")

    message = str(raised.value)
    assert "'nowhere'" in message
    assert mlflow.get_tracking_uri() in message


def test_missing_run_id_names_the_reference_and_server(tracking) -> None:
    absent = "0" * 32

    with pytest.raises(CliError) as raised:
        resolve_run(make_client(), absent)

    message = str(raised.value)
    assert absent in message
    assert mlflow.get_tracking_uri() in message


def test_experiment_narrows_the_name_search(tracking) -> None:
    with mlflow.start_run(run_name="twin") as active:
        wanted = active.info.run_id
    other = tracking.create_experiment("elsewhere")
    with mlflow.start_run(run_name="twin", experiment_id=other):
        pass

    resolved = resolve_run(make_client(), "twin", experiment="runsnap-tests")

    assert resolved.info.run_id == wanted


def test_unknown_experiment_is_reported(tracking) -> None:
    with pytest.raises(CliError) as raised:
        resolve_run(make_client(), "anything", experiment="absent")

    assert "'absent'" in str(raised.value)


def test_tracking_uri_override_wins(tracking, tmp_path, monkeypatch) -> None:
    override = f"sqlite:///{tmp_path / 'other.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'env.db'}")

    assert make_client(override).tracking_uri == override


def test_tracking_uri_defaults_to_the_environment(tracking) -> None:
    assert make_client().tracking_uri == mlflow.get_tracking_uri()


def dirty_run(repo: GitRepo) -> str:
    """A run captured from a dirty tree; returns its run id."""
    repo.write("main.py", "print('changed')\n")
    repo.write("added.txt", "new\n")
    with runsnap.start_run() as active:
        return active.info.run_id


def test_show_prints_the_recorded_code_state(
    in_repo: GitRepo, tracking, capsys
) -> None:
    run_id = dirty_run(in_repo)
    before = in_repo.status()

    show(run_id)

    printed = capsys.readouterr().out
    tags = tracking.get_run(run_id).data.tags
    assert f"run:     {run_id}" in printed
    assert f"commit:  {tags[TAG_COMMIT]}" in printed
    assert "branch:  main" in printed
    assert "dirty:   true" in printed
    assert f"patch:   sha256:{tags[TAG_PATCH_SHA256]}" in printed
    assert "  main.py" in printed
    assert "  added.txt" in printed
    assert in_repo.status() == before


def test_show_lists_a_binary_file_added_to_the_tree(
    in_repo: GitRepo, tracking, capsys
) -> None:
    in_repo.write("weights.bin", b"\x00\x01\x02\xff")
    with runsnap.start_run() as active:
        run_id = active.info.run_id

    show(run_id)

    assert "  weights.bin" in capsys.readouterr().out


def test_show_on_a_clean_run_reports_no_patch(
    in_repo: GitRepo, tracking, capsys
) -> None:
    with runsnap.start_run() as active:
        run_id = active.info.run_id

    show(run_id)

    printed = capsys.readouterr().out
    assert "dirty:   false" in printed
    assert "patch:   (none)" in printed
    assert "files:" not in printed


def test_show_prints_the_attempt_a_run_continues(
    in_repo: GitRepo, tracking, capsys
) -> None:
    with runsnap.start_run() as first:
        first_id = first.info.run_id
    with runsnap.start_run(continues=first_id) as second:
        second_id = second.info.run_id

    show(second_id)

    assert f"continues: {first_id}" in capsys.readouterr().out


def test_show_on_a_first_attempt_prints_no_lineage(
    in_repo: GitRepo, tracking, capsys
) -> None:
    with runsnap.start_run() as active:
        run_id = active.info.run_id

    show(run_id)

    assert "continues:" not in capsys.readouterr().out


def test_show_rejects_a_run_without_code_state(tracking) -> None:
    with mlflow.start_run() as active:
        run_id = active.info.run_id

    with pytest.raises(CliError, match="carries no code state"):
        show(run_id)


def printed_path(capsys) -> Path:
    """The tree path a checkout reported."""
    printed = capsys.readouterr().out
    line = next(l for l in printed.splitlines() if l.startswith("path:"))
    return Path(line.removeprefix("path:").strip())


def tree_state(path: Path) -> dict[str, bytes]:
    """Every non-git file under `path`, keyed by its relative path."""
    return {
        str(item.relative_to(path)): item.read_bytes()
        for item in sorted(path.rglob("*"))
        if item.is_file() and ".git" not in item.relative_to(path).parts
    }


def test_checkout_reconstructs_the_working_tree(
    in_repo: GitRepo, tracking, capsys
) -> None:
    in_repo.write("main.py", "print('changed')\n")
    in_repo.write("added.txt", "new\n")
    with runsnap.start_run(run_name="baseline-lr3") as active:
        run_id = active.info.run_id
    original = tree_state(in_repo.path)

    checkout(run_id)

    printed = capsys.readouterr().out
    line = next(l for l in printed.splitlines() if l.startswith("path:"))
    created = Path(line.removeprefix("path:").strip())
    assert created.is_dir()
    assert tree_state(created) == original
    assert f"runsnap/baseline-lr3-{run_id[:8]}" in printed


def test_checkout_leaves_the_patch_uncommitted(in_repo: GitRepo, tracking) -> None:
    in_repo.write("main.py", "print('changed')\n")
    with runsnap.start_run() as active:
        run_id = active.info.run_id

    checkout(run_id, path=in_repo.path.parent / "rebuilt")

    rebuilt = GitRepo(in_repo.path.parent / "rebuilt")
    assert rebuilt.status().strip()


def test_checkout_leaves_the_users_tree_untouched(in_repo: GitRepo, tracking) -> None:
    in_repo.write("main.py", "print('changed')\n")
    with runsnap.start_run() as active:
        run_id = active.info.run_id
    before = in_repo.status()

    checkout(run_id)

    assert in_repo.status() == before
    assert in_repo.git("rev-parse", "--abbrev-ref", "HEAD").strip() == "main"


def test_checkout_of_a_clean_run_reports_the_clean_tree(
    in_repo: GitRepo, tracking, capsys
) -> None:
    with runsnap.start_run() as active:
        run_id = active.info.run_id

    checkout(run_id)

    printed = capsys.readouterr().out
    line = next(l for l in printed.splitlines() if l.startswith("path:"))
    assert "clean" in printed
    assert GitRepo(Path(line.removeprefix("path:").strip())).status() == ""


def test_nested_run_follows_its_parents_patch(
    in_repo: GitRepo, tracking, capsys
) -> None:
    in_repo.write("main.py", "print('changed')\n")
    with runsnap.start_run(), runsnap.start_run(nested=True) as child:
        child_id = child.info.run_id
    assert TAG_PATCH_RUN_ID in tracking.get_run(child_id).data.tags
    original = tree_state(in_repo.path)

    checkout(child_id)

    assert tree_state(printed_path(capsys)) == original


def test_explicit_branch_name_is_used(in_repo: GitRepo, tracking) -> None:
    with runsnap.start_run() as active:
        run_id = active.info.run_id

    checkout(run_id, branch="chosen")

    assert "chosen" in in_repo.git("branch", "--list", "chosen")


def test_in_place_checkout_switches_the_current_tree(
    in_repo: GitRepo, tracking, capsys
) -> None:
    in_repo.write("main.py", "print('changed')\n")
    with runsnap.start_run() as active:
        run_id = active.info.run_id
    original = tree_state(in_repo.path)
    in_repo.commit("keep the working tree clean")

    checkout(run_id, worktree=False, branch="rebuilt")

    assert printed_path(capsys) == in_repo.path
    assert in_repo.git("rev-parse", "--abbrev-ref", "HEAD").strip() == "rebuilt"
    assert tree_state(in_repo.path) == original


def test_in_place_refuses_a_dirty_tree(in_repo: GitRepo, tracking) -> None:
    with runsnap.start_run() as active:
        run_id = active.info.run_id
    in_repo.write("scratch.py", "work in progress\n")

    with pytest.raises(CliError, match="uncommitted changes"):
        checkout(run_id, worktree=False)

    assert in_repo.git("rev-parse", "--abbrev-ref", "HEAD").strip() == "main"


def test_force_overrides_the_dirty_tree_refusal(in_repo: GitRepo, tracking) -> None:
    with runsnap.start_run() as active:
        run_id = active.info.run_id
    in_repo.write("main.py", "print('work in progress')\n")

    checkout(run_id, worktree=False, branch="rebuilt", force=True)

    assert in_repo.git("rev-parse", "--abbrev-ref", "HEAD").strip() == "rebuilt"
    assert (in_repo.path / "main.py").read_text() == "print('hello')\n"


def test_commit_snapshots_the_patch(in_repo: GitRepo, tracking, capsys) -> None:
    in_repo.write("main.py", "print('changed')\n")
    with runsnap.start_run() as active:
        run_id = active.info.run_id
    base = tracking.get_run(run_id).data.tags[TAG_COMMIT]

    checkout(run_id, commit=True)

    rebuilt = GitRepo(printed_path(capsys))
    assert rebuilt.status() == ""
    assert rebuilt.git("rev-list", "--count", f"{base}..HEAD").strip() == "1"


def test_missing_base_commit_is_reported(in_repo: GitRepo, tracking, tmp_path) -> None:
    with runsnap.start_run() as active:
        run_id = active.info.run_id
    elsewhere = GitRepo(tmp_path / "elsewhere")
    elsewhere.path.mkdir()
    elsewhere.git("init", "--quiet", "-b", "main")
    elsewhere.git("config", "user.email", "test@example.com")
    elsewhere.git("config", "user.name", "Test")
    elsewhere.write("other.py", "")
    elsewhere.commit("unrelated")
    base = tracking.get_run(run_id).data.tags[TAG_COMMIT]

    with pytest.raises(CliError, match=f"commit {base} is not present"):
        checkout(run_id, repo=elsewhere.path)


def test_existing_branch_is_reported(in_repo: GitRepo, tracking) -> None:
    with runsnap.start_run() as active:
        run_id = active.info.run_id
    in_repo.git("branch", "taken")

    with pytest.raises(CliError, match="already exists"):
        checkout(run_id, branch="taken")


def test_run_without_code_state_is_reported(in_repo: GitRepo, tracking) -> None:
    with mlflow.start_run() as active:
        run_id = active.info.run_id

    with pytest.raises(CliError, match="carries no code state"):
        checkout(run_id)


def test_outside_a_repository_is_reported(
    in_repo: GitRepo, tracking, tmp_path, monkeypatch
) -> None:
    with runsnap.start_run() as active:
        run_id = active.info.run_id
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.chdir(outside)

    with pytest.raises(CliError, match="not inside a git repository"):
        checkout(run_id, repo=outside)


def test_failing_patch_leaves_nothing_behind(
    in_repo: GitRepo, tracking, monkeypatch
) -> None:
    in_repo.write("main.py", "print('changed')\n")
    with runsnap.start_run() as active:
        run_id = active.info.run_id

    def refuse(tree, patch):
        raise _git.GitError("git apply failed: patch does not apply")

    monkeypatch.setattr(_cli, "apply_patch", refuse)

    with pytest.raises(CliError, match="does not apply"):
        checkout(run_id)

    assert "runsnap" not in in_repo.git("branch", "--list")
    assert not (in_repo.path.parent / f"repo-runsnap-{run_id[:8]}").exists()


def test_in_place_failure_restores_the_previous_branch(
    in_repo: GitRepo, tracking, monkeypatch
) -> None:
    in_repo.write("main.py", "print('changed')\n")
    with runsnap.start_run() as active:
        run_id = active.info.run_id
    in_repo.commit("keep the working tree clean")

    def refuse(tree, patch):
        raise _git.GitError("git apply failed: patch does not apply")

    monkeypatch.setattr(_cli, "apply_patch", refuse)

    with pytest.raises(CliError, match="does not apply"):
        checkout(run_id, worktree=False, branch="rebuilt")

    assert in_repo.git("rev-parse", "--abbrev-ref", "HEAD").strip() == "main"
    assert in_repo.git("branch", "--list", "rebuilt").strip() == ""


def test_in_place_failure_restores_a_detached_head(
    in_repo: GitRepo, tracking, monkeypatch
) -> None:
    in_repo.write("main.py", "print('changed')\n")
    with runsnap.start_run() as active:
        run_id = active.info.run_id
    detached_at = in_repo.commit("keep the working tree clean")
    in_repo.git("checkout", "--quiet", "--detach")

    def refuse(tree, patch):
        raise _git.GitError("git apply failed: patch does not apply")

    monkeypatch.setattr(_cli, "apply_patch", refuse)

    with pytest.raises(CliError, match="does not apply"):
        checkout(run_id, worktree=False, branch="rebuilt")

    assert in_repo.git("rev-parse", "HEAD").strip() == detached_at
    assert in_repo.git("rev-parse", "--abbrev-ref", "HEAD").strip() == "HEAD"
    assert in_repo.git("branch", "--list", "rebuilt").strip() == ""


def test_main_reports_a_failure_without_a_traceback(
    in_repo: GitRepo, tracking, capsys, monkeypatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["runsnap", "show", "nowhere"])

    with pytest.raises(SystemExit) as raised:
        main()

    assert raised.value.code == 1
    assert "runsnap: no run named 'nowhere'" in capsys.readouterr().err
