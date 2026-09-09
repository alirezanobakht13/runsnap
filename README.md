# exp-track

Reproducible MLflow runs: each run records its commit plus a patch of all
uncommitted work, and the CLI puts that code back on disk. Needs `git` on `PATH`.

```python
with exp_track.start_run(run_name="baseline"):  # forwards to mlflow.start_run()
    exp_track.log_params(hparams)  # a pydantic model
hp = exp_track.load_params(run_id, HParams)
```

`log_params` writes one param per scalar leaf (`opt.lr` = `0.001`; sequences
whole) plus `hparams/<name>.json`, the full-fidelity record `load_params` reads.
Tags are `exp_track.git.` + `commit`, `branch`, `dirty`, `repo_url`,
`patch_sha256`, `patch_run_id` (nested runs point at the ancestor holding the
patch), `capture_error`; the patch is artifact `code/state.patch` on dirty runs.
Equal `commit` and `patch_sha256` mean identical code.

```bash
exp-track show <run-id-or-name>      # commit, branch, dirty, digest, files
exp-track checkout <run-id-or-name>  # branch at that commit, patch reapplied
```

`checkout` lands in a new worktree, patch left uncommitted: `--no-worktree`,
`--path`, `--branch`, `--commit`, `--force`, `--repo`. Capture never fails a run;
disable it with `capture_code=False` or `EXP_TRACK_CAPTURE_CODE=0`, and cap the
patch with `EXP_TRACK_MAX_PATCH_BYTES` (10 MiB).

**The patch carries uncommitted content of tracked files, so a secret in one is
uploaded to the tracking server.** Ignored files are excluded.
