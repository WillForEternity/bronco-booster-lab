"""Pass/fail rules shared by the training gate and the final test (RECIPE.md, notes 3-7), and the final test's
bookkeeping. Runs on: laptop."""

import json
import math
from types import SimpleNamespace

import pytest
from criteria import GAIT_KEYS, recipe_result, run_result, stage_criteria, survivor_gait_means
from final_test import find_stages, write_summary
from run_state import STAGE_RE
from speed_metrics import epte_sp, heading_speed


def trial(survived=True, fwd=1.0, steady=None, epte=0.05, asym=0.05, flight=0.1):
    gait = {"flight_fraction": flight, "duty_factor": 0.5, "stance_asymmetry": asym, "cost_of_transport": 0.8,
            "leg_torque_saturation": 0.0, "trunk_pitch_rms_deg": 2.0, "trunk_roll_rms_deg": 1.0}
    return {"survived": survived, "forward_speed_mps": fwd if survived else None,
            "steady_speed_mps": steady if steady is not None else fwd, "epte_sp": epte, "gait": gait if survived else None}


def test_success_needs_survival_and_own_speed():
    # Note 3: 18 fast survivors + 2 that stand still pass only if 18 >= min_success.
    c = stage_criteria([trial(fwd=1.52)] * 18 + [trial(fwd=0.02)] * 2, 1.5)
    assert (c["successes"], c["passes"]) == (18, True)
    c = stage_criteria([trial(fwd=1.52)] * 17 + [trial(fwd=0.02)] * 3, 1.5)
    assert (c["successes"], c["passes"]) == (17, False)


def test_survivor_mean_cannot_hide_slow_trials():
    # The case that motivated the shared rule: every trial survives and the survivors' mean speed clears 90%,
    # but most trials are individually slow. A mean-speed gate would pass this; the per-trial rule does not.
    trials = [trial(fwd=1.30)] * 8 + [trial(fwd=0.80)] * 12
    mean = sum(t["forward_speed_mps"] for t in trials) / len(trials)
    assert mean >= 0.9 * 1.0
    c = stage_criteria(trials, 1.0)
    assert c["survived"] == 20 and c["successes"] == 8 and not c["passes"]


def test_fallen_trials_never_succeed():
    c = stage_criteria([trial(survived=False)] * 3 + [trial(fwd=2.0)] * 17, 2.0)
    assert c["successes"] == 17 and c["survived"] == 17 and not c["passes"]


def test_speed_ratio_boundary():
    assert stage_criteria([trial(fwd=0.9)] * 20, 1.0)["successes"] == 20
    assert stage_criteria([trial(fwd=0.8999)] * 20, 1.0)["successes"] == 0


def test_speed_metric_choice():
    # Forward speed misses the bar while straight-line displacement speed (drift included) clears it.
    trials = [trial(fwd=0.85, steady=0.95)] * 20
    assert not stage_criteria(trials, 1.0, metric="forward")["passes"]
    assert stage_criteria(trials, 1.0, metric="steady")["passes"]
    with pytest.raises(ValueError):
        stage_criteria(trials, 1.0, metric="sideways")


def test_command_must_be_positive():
    with pytest.raises(ValueError):
        stage_criteria([trial()], 0.0)


def test_overshoot_flag_uses_median_error():
    assert stage_criteria([trial(fwd=1.12)] * 20, 1.0)["overshoots"]
    assert not stage_criteria([trial(fwd=1.08)] * 20, 1.0)["overshoots"]


def test_symmetry_flag():
    assert stage_criteria([trial(asym=0.10)] * 20, 1.0)["symmetric"]
    assert not stage_criteria([trial(asym=0.11)] * 20, 1.0)["symmetric"]


def test_gait_means_use_survivors_only():
    means = survivor_gait_means([trial(flight=0.2), trial(flight=0.4), trial(survived=False)])
    assert set(means) == set(GAIT_KEYS)
    assert means["flight_fraction"] == pytest.approx(0.3)
    assert survivor_gait_means([trial(survived=False)])["flight_fraction"] is None


def stage(n, v, passes):
    return {"stage": n, "v_max": v, "checkpoint": f"s{n}.pt", "criteria": {"passes": passes, "symmetric": True}}


def test_run_result_is_highest_passing_stage_even_after_a_failure():
    assert run_result([stage(1, 1.0, True), stage(2, 1.25, False), stage(3, 1.5, True)])["v_max"] == 1.5
    assert run_result([stage(1, 1.0, False)]) is None


def test_recipe_result_is_the_lowest_seed():
    r = recipe_result({"seed1": {"v_max": 2.0}, "seed2": {"v_max": 1.75}})
    assert (r["v_max"], r["limited_by"]) == (1.75, "seed2")
    assert recipe_result({"seed1": {"v_max": 2.0}, "seed2": None})["v_max"] is None


def test_stage_names():
    m = STAGE_RE.search("runs/x/stages/stage_05_vmax2.00_it2500.pt")
    assert m.groups() == ("05", "2.00", "2500")
    assert STAGE_RE.search("stage_00_k1_walk_baseline.pt") is None


def test_find_stages_skips_the_baseline(tmp_path):
    stages = tmp_path / "2026-10-08_01-50-19_seed1" / "stages"
    stages.mkdir(parents=True)
    for name in ("stage_00_k1_walk_baseline.pt", "stage_01_vmax1.00_it1300.pt", "stage_02_vmax1.25_it1600.pt"):
        (stages / name).write_bytes(b"")
    found = find_stages(str(tmp_path))
    assert [(s["run"], s["stage"], s["v_max"], s["iteration"]) for s in found] == [
        ("2026-10-08_01-50-19_seed1", 1, 1.0, 1300), ("2026-10-08_01-50-19_seed1", 2, 1.25, 1600)]


def test_write_summary(tmp_path):
    args = {"trials": 20, "seed0": 1000, "damping_profile": "v3", "speed_metric": "forward", "seconds": 10.0,
            "speed_ratio": 0.9, "min_success": 18}
    results = []
    for run, v, fwd in (("seed1", 2.0, 2.05), ("seed2", 2.0, 2.02), ("seed2", 2.25, 1.5)):
        results.append({"run": run, "stage": int(v * 4 - 3), "v_max": v, "iteration": 100, "checkpoint": f"{run}_{v}.pt",
                        "criteria": stage_criteria([trial(fwd=fwd)] * 20, v)})
    summary = write_summary(str(tmp_path), results, args)
    assert summary["recipe"]["v_max"] == 2.0
    assert json.loads((tmp_path / "summary.json").read_text())["per_run"]["seed2"]["v_max"] == 2.0
    md = (tmp_path / "summary.md").read_text()
    assert "| seed2 | 6 | 2.25 | 0/20 | no |" in md
    assert "- Recipe (lowest of the runs): 2.00 m/s" in md


def test_epte_sp():
    assert epte_sp([1.0] * 10, 1.0, 10) == 0.0
    assert epte_sp([0.9] * 10, 1.0, 10) == pytest.approx(0.1)
    # Fell after 4 of 10 steps with 10% error: (4 x 0.1 + 6 x 1.0) / 10.
    assert epte_sp([1.1] * 4, 1.0, 10) == pytest.approx(0.64)


def test_heading_speed_ignores_sideways_motion():
    yaw = math.radians(30)
    quat = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]  # wxyz, yaw only
    fwd = [math.cos(yaw), math.sin(yaw)]
    side = [-math.sin(yaw), math.cos(yaw)]
    d = SimpleNamespace(qpos=[0, 0, 0.5, *quat], qvel=[2.0 * fwd[0] + 0.5 * side[0], 2.0 * fwd[1] + 0.5 * side[1], 0])
    assert heading_speed(d) == pytest.approx(2.0)
