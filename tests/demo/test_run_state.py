"""Resuming a version-3 run (demo/speed_ramp/run_state.py). Runs on: laptop."""

from types import SimpleNamespace

from run_state import code_provenance, curriculum_state, latest_checkpoint

ARGS = SimpleNamespace(start_speed=1.0, step=0.25, warmup=100)

# Shape of a real run's ramp_events.jsonl (version 3, seed 1, abridged).
EVENTS = [
    {"event": "start", "version": 3},
    {"event": "actor_unfrozen", "iteration": 99},
    {"event": "validation_failed", "target": 1.0, "iteration": 1000},
    {"event": "target_hit", "stage": 1, "target": 1.0, "iteration": 1300},
    {"event": "target_hit", "stage": 2, "target": 1.25, "iteration": 1600},
    {"event": "validation_failed", "target": 1.5, "iteration": 1800},
    {"event": "target_hit", "stage": 3, "target": 1.5, "iteration": 1900},
    {"event": "stalled", "target": 1.75, "iteration": 4900},
]


def test_state_after_last_promotion():
    s = curriculum_state(EVENTS, ARGS, up_to=1950)
    assert s == {"stage": 3, "v_max": 1.75, "level_start": 1900, "last_mj_try": 1900, "stall_logged": False}


def test_events_after_the_checkpoint_are_ignored():
    # Resuming from iteration 1750: the 1800 failure and the 1900 promotion are redone.
    s = curriculum_state(EVENTS, ARGS, up_to=1750)
    assert s == {"stage": 2, "v_max": 1.5, "level_start": 1600, "last_mj_try": 1600, "stall_logged": False}


def test_fresh_run_state():
    s = curriculum_state(EVENTS[:2], ARGS, up_to=150)
    assert s == {"stage": 0, "v_max": 1.0, "level_start": 100, "last_mj_try": None, "stall_logged": False}


def test_stall_on_current_level_is_remembered():
    assert curriculum_state(EVENTS, ARGS, up_to=5000)["stall_logged"]


def test_latest_checkpoint_prefers_the_later_iteration(tmp_path):
    (tmp_path / "stages").mkdir()
    for name in ("model_1750.pt", "model_1500.pt", "stages/stage_00_k1_walk_baseline.pt",
                 "stages/stage_03_vmax1.50_it1900.pt"):
        (tmp_path / name).write_bytes(b"")
    path, it = latest_checkpoint(str(tmp_path))
    assert (it, path.endswith("stage_03_vmax1.50_it1900.pt")) == (1900, True)
    (tmp_path / "model_2000.pt").write_bytes(b"")
    assert latest_checkpoint(str(tmp_path))[1] == 2000


def test_code_provenance_hashes_the_scripts():
    p = code_provenance()
    assert "train_v3.py" in p["sha256"] and len(p["sha256"]["train_v3.py"]) == 64
