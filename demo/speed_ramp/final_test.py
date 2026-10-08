"""Version-3 final test (RECIPE.md, notes 3-7). Never used for training decisions.

Runs on: GPU host (recorder environment, /workspace/venv_rec) or laptop (booster_deploy's environment),
from booster_deploy's repository root, after training has stopped:
    python <repo>/demo/speed_ramp/final_test.py --runs /workspace/runs/k1_run_v3 --out /workspace/final_test_v3

For every promoted stage (stages/stage_NN_vmax<v>_it<i>.pt, NN >= 1) of every run under --runs:
- export it to booster_deploy's format and play 20 perturbed trials (eval_mujoco.py) at the stage's
  v_max, with trial seeds 1000-1019 (the validation gate uses 0-19) and damping profile v3;
- per trial (recipe note 3): success = survives 10 s AND its own forward speed >= 0.9 x command;
  speed error = (forward speed - command) / command; EPTE-SP (note 5);
- per stage: passes with >= 18/20 successes. Also reported: median speed error, flagged "overshoots"
  above +10% (note 4); EPTE-SP median and worst; survivor means of the gait measures; and
  "symmetric" (mean stance asymmetry <= 10%, version 3 success definition).
Result per run: the highest promoted stage that passes. Recipe result: the lower of the runs' results
(note 6); the higher one is reported, not claimed.

Outputs: <out>/<run>/<stage>.json (all trials and provenance), <out>/summary.json, <out>/summary.md.
A stage whose JSON already exists is not re-run unless --force is given.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import platform
import re
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE_RE = re.compile(r"stage_(\d+)_vmax(\d+\.\d+)_it(\d+)\.pt$")
GAIT_KEYS = ("flight_fraction", "duty_factor", "stance_asymmetry", "cost_of_transport", "leg_torque_saturation",
             "trunk_pitch_rms_deg", "trunk_roll_rms_deg")


# --- Criteria (pure functions; tested in tests/demo/test_final_test.py) --------------------------------------


def trial_speed(t: dict, metric: str) -> float | None:
    return t["forward_speed_mps"] if metric == "forward" else t["steady_speed_mps"]


def stage_criteria(per_trial: list[dict], command: float, *, metric: str = "forward", speed_ratio: float = 0.9,
                   min_success: int = 18, overshoot: float = 0.10, max_asymmetry: float = 0.10) -> dict:
    """Amendments 3-5 applied to one stage's trials at one commanded speed."""
    success, errors, epte = [], [], []
    for t in per_trial:
        v = trial_speed(t, metric)
        ok = bool(t["survived"] and v is not None and v >= speed_ratio * command)
        success.append(ok)
        if t["survived"] and v is not None:
            errors.append((v - command) / command)
        if t.get("epte_sp") is not None:
            epte.append(t["epte_sp"])
    survivors = [t for t in per_trial if t["survived"]]
    gait = {}
    for k in GAIT_KEYS:
        vals = [t["gait"][k] for t in survivors if t.get("gait") and t["gait"].get(k) is not None]
        gait[k] = round(statistics.fmean(vals), 4) if vals else None
    median_err = statistics.median(errors) if errors else None
    asym = gait["stance_asymmetry"]
    speeds = [trial_speed(t, metric) for t in survivors if trial_speed(t, metric) is not None]
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
    """Highest promoted stage that passes (stages: dicts with 'stage', 'v_max', 'criteria')."""
    passing = [s for s in stages if s["criteria"]["passes"]]
    if not passing:
        return None
    best = max(passing, key=lambda s: s["stage"])
    return {"stage": best["stage"], "v_max": best["v_max"], "symmetric": best["criteria"]["symmetric"],
            "checkpoint": best["checkpoint"]}


def recipe_result(results: dict[str, dict | None]) -> dict:
    """Amendment 6: the recipe's result is the lower of the runs' results."""
    speeds = {run: (r["v_max"] if r else None) for run, r in results.items()}
    if not speeds or any(v is None for v in speeds.values()):
        return {"v_max": None, "per_run": speeds, "note": "at least one run has no passing stage"}
    low = min(speeds, key=speeds.get)
    return {"v_max": speeds[low], "limited_by": low, "per_run": speeds}


# --- Running ---------------------------------------------------------------------------------------------------


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_stages(runs_dir: str) -> list[dict]:
    out = []
    for run_dir in sorted(glob.glob(os.path.join(runs_dir, "*"))):
        for path in sorted(glob.glob(os.path.join(run_dir, "stages", "stage_*.pt"))):
            m = STAGE_RE.search(path)
            if m and int(m.group(1)) >= 1:
                out.append({"run": os.path.basename(run_dir), "stage": int(m.group(1)), "v_max": float(m.group(2)),
                            "iteration": int(m.group(3)), "checkpoint": path})
    return out


def test_stage(job: dict) -> dict:
    """Export, evaluate and score one stage. Runs in a worker process (cwd = booster_deploy's root)."""
    sys.path.insert(0, HERE)
    import eval_mujoco  # noqa: PLC0415  (imports booster_deploy; needs booster_deploy's root as cwd)
    import export_stage  # noqa: PLC0415
    import mujoco  # noqa: PLC0415
    import numpy  # noqa: PLC0415
    import torch  # noqa: PLC0415

    a = job["args"]
    policy = export_stage.export(job["checkpoint"], os.path.join(a["out"], job["run"], "policies"))
    t0 = time.time()
    r = eval_mujoco.evaluate(policy, job["v_max"], a["trials"], a["seconds"], a["seed0"], a["damping_profile"])
    crit = stage_criteria(r["per_trial"], job["v_max"], metric=a["speed_metric"], speed_ratio=a["speed_ratio"],
                          min_success=a["min_success"])
    return {
        **{k: job[k] for k in ("run", "stage", "v_max", "iteration", "checkpoint")},
        "criteria": crit,
        "evaluation": r,
        "provenance": {
            "checkpoint_sha256": sha256(job["checkpoint"]),
            "policy": policy,
            "policy_sha256": sha256(policy),
            "final_test_sha256": sha256(os.path.join(HERE, "final_test.py")),
            "eval_mujoco_sha256": sha256(os.path.join(HERE, "eval_mujoco.py")),
            "versions": {"mujoco": mujoco.__version__, "torch": torch.__version__, "numpy": numpy.__version__,
                         "python": platform.python_version(), "platform": platform.platform()},
            "wall_s": round(time.time() - t0, 1),
        },
    }


def write_summary(out: str, results: list[dict], args: dict) -> dict:
    by_run: dict[str, list[dict]] = {}
    for r in sorted(results, key=lambda r: (r["run"], r["stage"])):
        by_run.setdefault(r["run"], []).append(r)
    per_run = {run: run_result(stages) for run, stages in by_run.items()}
    summary = {"args": args, "per_run": per_run, "recipe": recipe_result(per_run),
               "stages": [{k: r[k] for k in ("run", "stage", "v_max", "iteration", "checkpoint", "criteria")}
                          for rs in by_run.values() for r in rs]}
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    def fmt(v, spec=".2f"):
        return "-" if v is None else format(v, spec)

    lines = [
        "# Version-3 final test",
        "",
        f"MuJoCo (booster_deploy), {args['trials']} perturbed trials per stage, seeds {args['seed0']}+, "
        f"damping profile {args['damping_profile']}, speed metric: {args['speed_metric']}. "
        f"A trial succeeds if it survives {args['seconds']:.0f} s at >= {args['speed_ratio']:.0%} of the command; "
        f"a stage passes with >= {args['min_success']}/{args['trials']}.",
        "",
        "| Run | Stage | v_max | Successes | Passes | Median speed | Speed error | Overshoots | EPTE-SP median / worst "
        "| Stance asym. | Flight frac. | CoT |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summary["stages"]:
        c, g = s["criteria"], s["criteria"]["gait_survivors_mean"]
        lines.append(
            f"| {s['run']} | {s['stage']} | {s['v_max']:.2f} | {c['successes']}/{c['trials']} | "
            f"{'yes' if c['passes'] else 'no'} | {fmt(c['median_speed_survivors_mps'])} | "
            f"{fmt(c['median_speed_error'], '+.1%')} | {'yes' if c['overshoots'] else 'no'} | "
            f"{fmt(c['epte_sp_median'], '.1%')} / {fmt(c['epte_sp_worst'], '.1%')} | "
            f"{fmt(g['stance_asymmetry'], '.1%')} | {fmt(g['flight_fraction'], '.1%')} | {fmt(g['cost_of_transport'])} |")
    lines += ["", "## Result", ""]
    for run, r in per_run.items():
        lines.append(f"- {run}: " + (f"stage {r['stage']} ({r['v_max']:.2f} m/s), symmetric: "
                                     f"{'yes' if r['symmetric'] else 'no'}" if r else "no stage passes"))
    rec = summary["recipe"]
    lines.append(f"- Recipe (lower of the runs): {fmt(rec['v_max'])} m/s" + (f", limited by {rec['limited_by']}"
                                                                           if rec.get("limited_by") else ""))
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", required=True, help="folder holding the run folders (each with stages/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--damping_profile", default="v3")
    ap.add_argument("--speed_metric", choices=["forward", "steady"], default="forward",
                    help="trial speed for the success rule (recipe note 7: forward)")
    ap.add_argument("--speed_ratio", type=float, default=0.9)
    ap.add_argument("--min_success", type=int, default=18)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--force", action="store_true", help="re-run stages that already have results")
    a = ap.parse_args()
    if a.seed0 < 1000:
        sys.exit("Final-test seeds start at 1000; seeds 0-19 belong to the training gate.")
    if not os.path.isdir("tasks"):
        sys.exit("Run from booster_deploy's repository root (eval_mujoco imports its tasks package).")
    args = {k: v for k, v in vars(a).items() if k not in ("jobs", "force")}
    args["out"] = os.path.abspath(a.out)

    stages = find_stages(a.runs)
    if not stages:
        sys.exit(f"No promoted stages under {a.runs}/*/stages/")
    results, todo = [], []
    for s in stages:
        path = os.path.join(args["out"], s["run"], f"stage_{s['stage']:02d}.json")
        if os.path.exists(path) and not a.force:
            with open(path) as f:
                results.append(json.load(f))
        else:
            todo.append({**s, "args": args, "result_path": path})
    print(f"{len(stages)} promoted stages; {len(todo)} to test, {len(results)} already done; {a.jobs} workers", flush=True)

    failed = []
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        futures = {pool.submit(test_stage, job): job for job in todo}
        for fut in as_completed(futures):
            job = futures[fut]
            try:
                r = fut.result()
            except Exception as e:  # one broken stage must not lose the others' results
                failed.append(job["checkpoint"])
                print(f"ERROR {job['run']} stage {job['stage']:02d}: {type(e).__name__}: {e}", flush=True)
                continue
            os.makedirs(os.path.dirname(job["result_path"]), exist_ok=True)
            with open(job["result_path"], "w") as f:
                json.dump(r, f, indent=2)
            c = r["criteria"]
            print(f"{r['run']} stage {r['stage']:02d} @ {r['v_max']:.2f} m/s: {c['successes']}/{c['trials']} "
                  f"{'PASS' if c['passes'] else 'fail'}", flush=True)
            results.append(r)

    write_summary(args["out"], results, args)
    with open(os.path.join(args["out"], "summary.md")) as f:
        print(f.read())
    if failed:
        sys.exit(f"{len(failed)} stage(s) failed and are missing from the summary: {failed}")


if __name__ == "__main__":
    main()
