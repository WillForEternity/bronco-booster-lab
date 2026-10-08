"""Final-test criteria (RECIPE.md, notes 3-7) and speed measures. Runs on: laptop."""

import math
from types import SimpleNamespace

import pytest
from final_test import STAGE_RE, recipe_result, run_result, stage_criteria
from speed_metrics import epte_sp, heading_speed


def trial(survived=True, fwd=1.0, steady=None, epte=0.05, asym=0.05, flight=0.1):
    gait = {"flight_fraction": flight, "duty_factor": 0.5, "stance_asymmetry": asym, "cost_of_transport": 0.8,
            "leg_torque_saturation": 0.0, "trunk_pitch_rms_deg": 2.0, "trunk_roll_rms_deg": 1.0}
    return {"survived": survived, "forward_speed_mps": fwd if survived else None,
            "steady_speed_mps": steady if steady is not None else fwd, "epte_sp": epte, "gait": gait if survived else None}


def test_success_needs_survival_and_own_speed():
    # Amendment 3: 18 fast survivors + 2 that stand still pass only if 18 >= min_success.
    trials = [trial(fwd=1.52)] * 18 + [trial(fwd=0.02)] * 2
    c = stage_criteria(trials, 1.5)
    assert (c["successes"], c["passes"]) == (18, True)
    c = stage_criteria([trial(fwd=1.52)] * 17 + [trial(fwd=0.02)] * 3, 1.5)
    assert (c["successes"], c["passes"]) == (17, False)


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


def test_overshoot_flag_uses_median_error():
    assert stage_criteria([trial(fwd=1.12)] * 20, 1.0)["overshoots"]
    assert not stage_criteria([trial(fwd=1.08)] * 20, 1.0)["overshoots"]


def test_symmetry_flag():
    assert stage_criteria([trial(asym=0.10)] * 20, 1.0)["symmetric"]
    assert not stage_criteria([trial(asym=0.11)] * 20, 1.0)["symmetric"]


def stage(n, v, passes):
    return {"stage": n, "v_max": v, "checkpoint": f"s{n}.pt", "criteria": {"passes": passes, "symmetric": True}}


def test_run_result_is_highest_passing_stage_even_after_a_failure():
    assert run_result([stage(1, 1.0, True), stage(2, 1.25, False), stage(3, 1.5, True)])["v_max"] == 1.5
    assert run_result([stage(1, 1.0, False)]) is None


def test_recipe_result_is_the_lower_seed():
    r = recipe_result({"seed1": {"v_max": 2.0}, "seed2": {"v_max": 1.75}})
    assert (r["v_max"], r["limited_by"]) == (1.75, "seed2")
    assert recipe_result({"seed1": {"v_max": 2.0}, "seed2": None})["v_max"] is None


def test_stage_names():
    m = STAGE_RE.search("runs/x/stages/stage_05_vmax2.00_it2500.pt")
    assert m.groups() == ("05", "2.00", "2500")
    assert STAGE_RE.search("stage_00_k1_walk_baseline.pt") is None


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
