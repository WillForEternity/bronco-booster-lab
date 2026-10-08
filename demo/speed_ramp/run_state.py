"""Run bookkeeping that needs no Isaac Lab: file hashes, code provenance, and the inputs for resuming a run.

Runs on: GPU host (imported by train_v3.py, final_test.py, record_stage.py); laptop for the unit tests.
"""

from __future__ import annotations

import glob
import hashlib
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_RE = re.compile(r"model_(\d+)\.pt$")
STAGE_RE = re.compile(r"stage_(\d+)_vmax(\d+\.\d+)_it(\d+)\.pt$")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_provenance() -> dict:
    """SHA-256 of every .py file next to this script, plus the Git commit and dirty files if it is a checkout."""
    files = {os.path.basename(p): sha256_file(p) for p in sorted(glob.glob(os.path.join(HERE, "*.py")))}
    git = None
    try:
        commit = subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "-C", HERE, "status", "--porcelain", "."], capture_output=True, text=True, check=True)
        git = {"commit": commit.stdout.strip(), "dirty_files": dirty.stdout.splitlines()}
    except (OSError, subprocess.CalledProcessError):
        pass
    return {"dir": HERE, "sha256": files, "git": git}


def latest_checkpoint(run_dir: str) -> tuple[str, int]:
    """The run's latest resumable checkpoint: model_<it>.pt or stages/stage_NN_vmax<v>_it<it>.pt (NN >= 1).

    Raises FileNotFoundError if there is none.
    """
    found = []
    for path in glob.glob(os.path.join(run_dir, "model_*.pt")):
        found.append((int(MODEL_RE.search(path).group(1)), path))
    for path in glob.glob(os.path.join(run_dir, "stages", "stage_*.pt")):
        m = STAGE_RE.search(path)
        if m and int(m.group(1)) >= 1:
            found.append((int(m.group(3)), path))
    if not found:
        raise FileNotFoundError(f"no model_*.pt or promoted stage checkpoint in {run_dir}")
    it, path = max(found)
    return path, it


def merge_resume_args(saved: dict, current: dict, defaults: dict, run_keys: tuple[str, ...]) -> dict:
    """Arguments for resuming a run: the run's own values for every key in run_keys, the current ones otherwise.

    A run-defining flag passed with a value other than its default and the run's is an error, so a resumed run
    cannot silently become a different experiment. Raises ValueError, naming the flags.
    """
    missing = [k for k in run_keys if k not in saved]
    if missing:
        raise ValueError(f"the run's curriculum_args.json has no {missing}: it was started by an older train_v3.py. "
                         "Resume it with the code that started it (Git commit in params/code_*.json).")
    conflicts = {k: (saved[k], current[k]) for k in run_keys if current[k] != defaults[k] and current[k] != saved[k]}
    if conflicts:
        detail = ", ".join(f"--{k} {new} (run: {old})" for k, (old, new) in conflicts.items())
        raise ValueError(f"these flags would change the run being resumed: {detail}. Start a new run instead.")
    return {**current, **{k: saved[k] for k in run_keys}}
