"""Run bookkeeping for train_v3.py that needs no Isaac Lab: code provenance and resuming a run.

Runs on: GPU host (imported by train_v3.py); also imported by the unit tests on a laptop.
"""

from __future__ import annotations

import glob
import hashlib
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_RE = re.compile(r"model_(\d+)\.pt$")
STAGE_RE = re.compile(r"stage_(\d+)_vmax\d+\.\d+_it(\d+)\.pt$")


def code_provenance() -> dict:
    """SHA-256 of every .py file next to this script, plus the Git commit and dirty state if it is a checkout."""
    files = {}
    for path in sorted(glob.glob(os.path.join(HERE, "*.py"))):
        with open(path, "rb") as f:
            files[os.path.basename(path)] = hashlib.sha256(f.read()).hexdigest()
    git = None
    try:
        commit = subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "-C", HERE, "status", "--porcelain", "."], capture_output=True, text=True)
        git = {"commit": commit.stdout.strip(), "dirty_files": dirty.stdout.splitlines()}
    except (OSError, subprocess.CalledProcessError):
        pass
    return {"dir": HERE, "sha256": files, "git": git}


def latest_checkpoint(run_dir: str) -> tuple[str, int]:
    """The run's latest resumable checkpoint: model_<it>.pt or stages/stage_NN_vmax<v>_it<it>.pt (NN >= 1)."""
    found = []
    for path in glob.glob(os.path.join(run_dir, "model_*.pt")):
        found.append((int(MODEL_RE.search(path).group(1)), path))
    for path in glob.glob(os.path.join(run_dir, "stages", "stage_*.pt")):
        m = STAGE_RE.search(path)
        if m and int(m.group(1)) >= 1:
            found.append((int(m.group(2)), path))
    if not found:
        sys.exit(f"--resume: no model_*.pt or promoted stage checkpoint in {run_dir}")
    it, path = max(found)
    return path, it


def curriculum_state(events: list[dict], a, up_to: int) -> dict:
    """Rebuild the curriculum from ramp_events.jsonl, ignoring anything after iteration `up_to`."""
    evs = [e for e in events if "iteration" in e and e["iteration"] <= up_to]
    hits = [e for e in evs if e["event"] == "target_hit"]
    level_start = hits[-1]["iteration"] if hits else a.warmup
    tries = [e["iteration"] for e in evs if e["event"] in ("target_hit", "validation_failed", "validation_error")]
    return {
        "stage": hits[-1]["stage"] if hits else 0,
        "v_max": round(hits[-1]["target"] + a.step, 4) if hits else a.start_speed,
        "level_start": level_start,
        "last_mj_try": max(tries) if tries else None,
        "stall_logged": any(e["event"] == "stalled" and e["iteration"] > level_start for e in evs),
    }
