"""Run bookkeeping: hashes, provenance, and resuming a run (demo/speed_ramp/run_state.py). Runs on: laptop."""

import hashlib

import pytest
from run_state import code_provenance, latest_checkpoint, merge_resume_args, sha256_file

KEYS = ("seed", "step", "patience")
DEFAULTS = {"seed": 1, "step": 0.25, "patience": 1000, "max_iterations": 20000, "resume": None}
SAVED = {"seed": 2, "step": 0.5, "patience": 800, "max_iterations": 20000, "resume": None}


def test_latest_checkpoint_prefers_the_later_iteration(tmp_path):
    (tmp_path / "stages").mkdir()
    for name in ("model_1750.pt", "model_1500.pt", "stages/stage_00_k1_walk_baseline.pt",
                 "stages/stage_03_vmax1.50_it1900.pt"):
        (tmp_path / name).write_bytes(b"")
    path, it = latest_checkpoint(str(tmp_path))
    assert (it, path.endswith("stage_03_vmax1.50_it1900.pt")) == (1900, True)
    (tmp_path / "model_2000.pt").write_bytes(b"")
    assert latest_checkpoint(str(tmp_path))[1] == 2000


def test_latest_checkpoint_ignores_the_baseline(tmp_path):
    (tmp_path / "stages").mkdir()
    (tmp_path / "stages" / "stage_00_k1_walk_baseline.pt").write_bytes(b"")
    with pytest.raises(FileNotFoundError):
        latest_checkpoint(str(tmp_path))


def test_resume_uses_the_runs_own_arguments():
    current = DEFAULTS | {"resume": "/runs/x", "max_iterations": 30000}
    merged = merge_resume_args(SAVED, current, DEFAULTS, KEYS)
    assert merged == {"seed": 2, "step": 0.5, "patience": 800, "max_iterations": 30000, "resume": "/runs/x"}


def test_resume_accepts_a_flag_equal_to_the_runs_value():
    assert merge_resume_args(SAVED, DEFAULTS | {"seed": 2}, DEFAULTS, KEYS)["seed"] == 2


def test_resume_refuses_a_flag_that_changes_the_run():
    with pytest.raises(ValueError, match=r"--step 0.75 \(run: 0.5\)"):
        merge_resume_args(SAVED, DEFAULTS | {"step": 0.75}, DEFAULTS, KEYS)


def test_resume_refuses_a_run_from_older_code():
    with pytest.raises(ValueError, match="older train_v3.py"):
        merge_resume_args({"seed": 2, "step": 0.5}, DEFAULTS, DEFAULTS, KEYS)


def test_sha256_file(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"k1" * 1_000_000)
    assert sha256_file(str(p)) == hashlib.sha256(b"k1" * 1_000_000).hexdigest()


def test_code_provenance_hashes_the_scripts():
    p = code_provenance()
    for name in ("train_v3.py", "criteria.py", "curriculum.py", "k1_conventions.py"):
        assert len(p["sha256"][name]) == 64
