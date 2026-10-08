"""Pass/fail rules for MuJoCo trials, shared by the training gate (train_v3.py) and the final test (final_test.py).

Runs on: GPU host or laptop. Pure Python, so the unit tests can load it.

One rule decides both gates (RECIPE.md, note 3): a trial succeeds if it survives AND its own forward speed is at
least SPEED_RATIO x the command; a set of trials passes with at least MIN_SUCCESSES successes. The gate and the
final test differ only in their trial seeds (0-19 and 1000-1019).
"""

from __future__ import annotations

import statistics

SPEED_RATIO = 0.9
MIN_SUCCESSES = 18
OVERSHOOT = 0.10
MAX_ASYMMETRY = 0.10
GAIT_KEYS = ("flight_fraction", "duty_factor", "stance_asymmetry", "cost_of_transport", "leg_torque_saturation",
             "trunk_pitch_rms_deg", "trunk_roll_rms_deg")


def trial_speed(t: dict, metric: str) -> float | None:
    """A trial's speed: "forward" (along the trunk's heading, recipe note 7) or "steady" (straight-line displacement)."""
    if metric not in ("forward", "steady"):
        raise ValueError(f"unknown speed metric {metric!r}")
    return t["forward_speed_mps"] if metric == "forward" else t["steady_speed_mps"]


def survivor_gait_means(per_trial: list[dict]) -> dict:
    """Mean of each gait measure over the surviving trials that have it (None if none do)."""
    survivors = [t for t in per_trial if t["survived"] and t.get("gait")]
    means = {}
    for k in GAIT_KEYS:
        vals = [t["gait"][k] for t in survivors if t["gait"].get(k) is not None]
        means[k] = round(statistics.fmean(vals), 4) if vals else None
    return means


def stage_criteria(per_trial: list[dict], command: float, *, metric: str = "forward", speed_ratio: float = SPEED_RATIO,
                   min_success: int = MIN_SUCCESSES, overshoot: float = OVERSHOOT,
                   max_asymmetry: float = MAX_ASYMMETRY) -> dict:
    """Recipe notes 3-5 and 7 applied to one policy's trials at one commanded speed."""
    if command <= 0:
        raise ValueError("command must be positive")
    success, errors, epte = [], [], []
    for t in per_trial:
        v = trial_speed(t, metric)
        success.append(bool(t["survived"] and v is not None and v >= speed_ratio * command))
        if t["survived"] and v is not None:
            errors.append((v - command) / command)
        if t.get("epte_sp") is not None:
            epte.append(t["epte_sp"])
    survivors = [t for t in per_trial if t["survived"]]
    gait = survivor_gait_means(per_trial)
    median_err = statistics.median(errors) if errors else None
    asym = gait["stance_asymmetry"]
    speeds = [v for v in (trial_speed(t, metric) for t in survivors) if v is not None]
    return {
        "command_mps": command,
        "speed_metric": metric,
        "trials": len(per_trial),
        "survived": len(survivors),
        "successes": sum(success),
        "passes": sum(success) >= min_success,
        "median_speed_survivors_mps": round(statistics.median(speeds), 3) if speeds else None,
        "median_speed_error": round(median_err, 4) if median_err is not None else None,
        "overshoots": median_err is not None and median_err > overshoot,
        "epte_sp_median": round(statistics.median(epte), 4) if epte else None,
        "epte_sp_worst": round(max(epte), 4) if epte else None,
        "symmetric": asym is not None and asym <= max_asymmetry,
        "gait_survivors_mean": gait,
        "per_trial_success": success,
    }


def run_result(stages: list[dict]) -> dict | None:
    """Highest promoted stage that passes (stages: dicts with 'stage', 'v_max', 'checkpoint', 'criteria')."""
    passing = [s for s in stages if s["criteria"]["passes"]]
    if not passing:
        return None
    best = max(passing, key=lambda s: s["stage"])
    return {"stage": best["stage"], "v_max": best["v_max"], "symmetric": best["criteria"]["symmetric"],
            "checkpoint": best["checkpoint"]}


def recipe_result(results: dict[str, dict | None]) -> dict:
    """Recipe note 6: the recipe's result is the lowest of the runs' results."""
    speeds = {run: (r["v_max"] if r else None) for run, r in results.items()}
    if not speeds or any(v is None for v in speeds.values()):
        return {"v_max": None, "per_run": speeds, "note": "at least one run has no passing stage"}
    low = min(speeds, key=speeds.get)
    return {"v_max": speeds[low], "limited_by": low, "per_run": speeds}
