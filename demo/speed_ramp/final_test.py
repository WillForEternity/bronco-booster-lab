"""The final test of the version-3 recipe (RECIPE.md, notes 3-7). Never used for training decisions.

Runs on: GPU host (recorder environment, /workspace/venv_rec) or laptop (booster_deploy's environment),
from booster_deploy's repository root, after training has stopped:
    python <repo>/demo/speed_ramp/final_test.py --runs /workspace/runs/k1_run_v3 --out /workspace/final_test_v3

For every promoted stage (stages/stage_NN_vmax<v>_it<i>.pt, NN >= 1) of every run under --runs:
- export it to booster_deploy's format and play 20 perturbed trials (eval_mujoco.py) at the stage's
  v_max, with trial seeds 1000-1019 (the training gate uses 0-19) and damping profile v3;
- score it with the rule the training gate also uses (criteria.py; note 3): a trial succeeds if it survives
  10 s AND its own forward speed is >= 0.9 x command; a stage passes with >= 18/20 successes.
  Also reported: median speed error and an "overshoots" flag above +10% (note 4); EPTE-SP median and worst
  (note 5); survivor means of the gait measures; and "symmetric" (mean stance asymmetry <= 10%).
Result per run: the highest promoted stage that passes. Recipe result: the lowest of the runs' results
(note 6); a higher one is reported, not claimed.

Outputs: <out>/<run>/stage_NN.json (all trials and provenance), <out>/summary.json, <out>/summary.md.
A stage whose JSON already exists is not re-run unless --force is given.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from criteria import MIN_SUCCESSES, SPEED_RATIO, recipe_result, run_result, stage_criteria
from run_state import STAGE_RE, sha256_file

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL_SEED0 = 1000  # seeds 0-19 belong to the training gate


def find_stages(runs_dir: str) -> list[dict]:
    out = []
    for run_dir in sorted(glob.glob(os.path.join(runs_dir, "*"))):
        for path in sorted(glob.glob(os.path.join(run_dir, "stages", "stage_*.pt"))):
            m = STAGE_RE.search(path)
            if m and int(m.group(1)) >= 1:
                out.append({"run": os.path.basename(run_dir), "stage": int(m.group(1)), "v_max": float(m.group(2)),
                            "iteration": int(m.group(3)), "checkpoint": path})
    return out


def score_stage(job: dict) -> dict:
    """Export, evaluate and score one stage. Runs in a worker process (cwd = booster_deploy's root)."""
    # Imported here, not at the top: they load booster_deploy and MuJoCo, which only the workers need.
    import eval_mujoco
    import export_stage
    import mujoco
    import numpy
    import torch

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
            "checkpoint_sha256": sha256_file(job["checkpoint"]),
            "policy": policy,
            "policy_sha256": sha256_file(policy),
            "code_sha256": {name: sha256_file(os.path.join(HERE, name))
                            for name in ("final_test.py", "eval_mujoco.py", "criteria.py", "booster_player.py",
                                         "export_stage.py", "k1_conventions.py", "speed_metrics.py")},
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
    lines.append(f"- Recipe (lowest of the runs): {fmt(rec['v_max'])} m/s" + (f", limited by {rec['limited_by']}"
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
    ap.add_argument("--seed0", type=int, default=FINAL_SEED0)
    ap.add_argument("--damping_profile", default="v3", help="v3 for policies from train_v3.py")
    ap.add_argument("--speed_metric", choices=["forward", "steady"], default="forward",
                    help="trial speed for the success rule (recipe note 7: forward)")
    ap.add_argument("--speed_ratio", type=float, default=SPEED_RATIO)
    ap.add_argument("--min_success", type=int, default=MIN_SUCCESSES)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--force", action="store_true", help="re-run stages that already have results")
    a = ap.parse_args()
    if a.min_success > a.trials:
        ap.error(f"--min_success {a.min_success} > --trials {a.trials}: no stage could pass")
    if a.seed0 < FINAL_SEED0:
        sys.exit(f"Final-test seeds start at {FINAL_SEED0}; seeds 0-19 belong to the training gate.")
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
        futures = {pool.submit(score_stage, job): job for job in todo}
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
