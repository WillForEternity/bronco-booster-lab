"""Watch the speed-ramp runs and record every new checkpoint in MuJoCo at a grid of speeds.

Runs on: GPU host, in the recorder environment (/workspace/venv_rec), alongside training:
    python record_watch.py [--runs /workspace/runs/k1_speed_ramp] [--videos /workspace/videos]

For each run (<date>_<reward>) it records:
- every stage checkpoint (stages/stage_NN_*.pt, including stage 00 = Booster's k1_walk unchanged);
- every periodic checkpoint whose iteration is a multiple of --every (model_<iter>.pt).

Each checkpoint is exported to booster_deploy's format and played at SPEED_GRID plus its own
target speed. Outputs, per checkpoint: <videos>/<reward>/<checkpoint>/policy.pt, v<speed>.mp4,
v<speed>.json, done.json. Every video is also appended to <videos>/index.jsonl, and
<videos>/summary.csv is rewritten on each pass. Nothing is ever deleted.
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
import time
import traceback

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")  # must be set before mujoco is first imported

HERE = os.path.dirname(os.path.abspath(__file__))
DEPLOY = "/workspace/upstream/booster_deploy"
SPEED_GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="/workspace/runs/k1_speed_ramp")
ap.add_argument("--videos", default="/workspace/videos")
ap.add_argument("--every", type=int, default=1000, help="also record model_<iter>.pt when iter is a multiple of this")
ap.add_argument("--seconds", type=float, default=10.0)
ap.add_argument("--poll", type=float, default=60.0)
ap.add_argument("--once", action="store_true", help="one pass, then exit")
ap.add_argument("--trials", type=int, default=0, help="also run eval_mujoco.py with this many perturbed trials per video")
ap.add_argument("--run_glob", default="*_*", help="which run folders under --runs to watch")
ap.add_argument("--damping_profile", default=None, help="MuJoCo joint damping profile (version 3: v3)")
args = ap.parse_args()

os.chdir(DEPLOY)  # booster_deploy imports its task registry from the working directory
sys.path.insert(0, HERE)
import eval_mujoco  # noqa: E402
import export_stage  # noqa: E402
import record_stage  # noqa: E402

STAGE_RE = re.compile(r"stage_(\d+)_(?:(?:vmax)?(\d+\.\d+)(?:mps)?_it(\d+)|k1_walk_baseline)\.pt$")
MODEL_RE = re.compile(r"model_(\d+)\.pt$")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ramp_info(run_dir: str):
    """(start_speed, step, [hit iterations]) from the run's ramp_events.jsonl."""
    start, step, hits = 1.0, 0.25, []
    path = os.path.join(run_dir, "ramp_events.jsonl")
    if os.path.exists(path):
        for line in open(path):
            e = json.loads(line)
            if e["event"] == "start":
                start, step = e["start_speed"], e["step"]
            elif e["event"] == "target_hit":
                hits.append(e["iteration"])
    return start, step, hits


def describe(run_dir: str, reward: str, ckpt: str):
    """(stage, iteration, target, label) for a checkpoint file, or None to skip it."""
    name = os.path.basename(ckpt)
    m = STAGE_RE.search(name)
    if m:
        stage = int(m.group(1))
        if stage == 0:
            return 0, 0, None, f"Booster's k1_walk, unchanged (stage 0 of the {reward} run)"
        target, it = float(m.group(2)), int(m.group(3))
        what = f"max target {target:.2f} m/s" if "vmax" in name else f"target {target:.2f} m/s"
        who = f"version 3, {reward}" if reward.startswith("seed") else f"reward: {reward}"
        return stage, it, target, (f"Booster k1_walk fine-tuned by Bronco Robotics | {who} | "
                                   f"stage {stage} ({what}, iter {it})")
    m = MODEL_RE.search(name)
    if m:
        it = int(m.group(1))
        if it == 0 or it % args.every:
            return None
        start, step, hits = ramp_info(run_dir)
        target = start + step * sum(1 for h in hits if h < it)
        return None, it, target, (f"Booster k1_walk fine-tuned by Bronco Robotics | reward: {reward} | "
                                  f"iter {it} (training target then {target:.2f} m/s)")
    return None


def process(run_dir: str, reward: str, ckpt: str, index_path: str) -> None:
    info = describe(run_dir, reward, ckpt)
    if info is None:
        return
    stage, it, target, label = info
    out_dir = os.path.join(args.videos, reward, os.path.splitext(os.path.basename(ckpt))[0])
    if os.path.exists(os.path.join(out_dir, "done.json")):
        if args.trials:
            add_robust(out_dir)
        return
    if time.time() - os.path.getmtime(ckpt) < 30:  # may still be being written
        return
    os.makedirs(out_dir, exist_ok=True)
    exported = export_stage.export(ckpt, out_dir)
    policy = os.path.join(out_dir, "policy.pt")
    os.replace(exported, policy)
    speeds = sorted(set(SPEED_GRID + ([round(target, 2)] if target else [])))
    results = []
    for speed in speeds:
        out = os.path.join(out_dir, f"v{speed:.2f}.mp4")
        m = record_stage.record(policy, speed, out, label, seconds=args.seconds, damping_profile=args.damping_profile)
        if args.trials:
            m["robust"] = robust_summary(eval_mujoco.evaluate(policy, speed, args.trials, args.seconds, damping_profile=args.damping_profile))
        m.update({"run_dir": run_dir, "reward": reward, "training_checkpoint": ckpt,
                  "training_checkpoint_sha256": record_stage.sha256(ckpt), "stage": stage, "iteration": it,
                  "stage_target_mps": target})
        with open(os.path.splitext(out)[0] + ".json", "w") as f:
            json.dump(m, f, indent=2)
        with open(index_path, "a") as f:
            f.write(json.dumps(m) + "\n")
        results.append(m)
        stop = "no stop" if m["safety_stop_s"] is None else f"STOP at {m['safety_stop_s']} s"
        rob = f", survived {m['robust']['survived']}/{m['robust']['trials']}" if "robust" in m else ""
        log(f"{reward} {os.path.basename(ckpt)} @ {speed:.2f} m/s -> {m['steady_speed_mps']:.2f} m/s, {stop}{rob}")
    with open(os.path.join(out_dir, "done.json"), "w") as f:
        json.dump({"checkpoint": ckpt, "speeds": speeds, "label": label, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)


def robust_summary(r: dict) -> dict:
    keys = ("trials", "survived", "survival_rate", "mean_speed_survivors_mps", "perturbation", "gait_survivors_mean", "damping_profile")
    return {k: r.get(k) for k in keys} | {"per_trial": r["per_trial"]}


def add_robust(out_dir: str) -> None:
    """Back-fill the perturbed-trial evaluation into videos recorded without it."""
    policy = os.path.join(out_dir, "policy.pt")
    for path in sorted(glob.glob(os.path.join(out_dir, "v*.json"))):
        m = json.load(open(path))
        if "robust" in m:
            continue
        m["robust"] = robust_summary(eval_mujoco.evaluate(policy, m["commanded_speed_mps"], args.trials, args.seconds,
                                                           damping_profile=args.damping_profile))
        with open(path, "w") as f:
            json.dump(m, f, indent=2)
        log(f"robust {os.path.relpath(path, args.videos)}: survived {m['robust']['survived']}/{m['robust']['trials']}")


def write_summary(index_path: str) -> None:
    rows = []
    for path in glob.glob(os.path.join(args.videos, "**", "v*.json"), recursive=True):
        m = json.load(open(path))
        if "reward" not in m:
            continue
        r = m.get("robust") or {}
        m["survived_of_trials"] = f"{r['survived']}/{r['trials']}" if r else ""
        m["robust_mean_speed_mps"] = r.get("mean_speed_survivors_mps", "")
        rows.append(m)
    if not rows:
        return
    cols = ["run_dir", "reward", "stage", "iteration", "stage_target_mps", "commanded_speed_mps", "steady_speed_mps", "safety_stop_s",
            "survived_of_trials", "robust_mean_speed_mps", "seconds_run", "distance_m", "min_trunk_height_m",
            "min_uprightness", "video", "checkpoint_sha256"]
    with open(os.path.join(args.videos, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["reward"], r["iteration"] or 0, r["commanded_speed_mps"])):
            w.writerow(r)


def one_pass() -> None:
    index_path = os.path.join(args.videos, "index.jsonl")
    for run_dir in sorted(glob.glob(os.path.join(args.runs, args.run_glob))):
        reward = os.path.basename(run_dir).split("_", 2)[-1]  # <date>_<time>_<run name>: speed_only, shaped, seed1, ...
        ckpts = sorted(glob.glob(os.path.join(run_dir, "stages", "stage_*.pt")))
        ckpts += sorted(glob.glob(os.path.join(run_dir, "model_*.pt")), key=lambda p: int(MODEL_RE.search(p).group(1)))
        for ckpt in ckpts:
            try:
                process(run_dir, reward, ckpt, index_path)
            except Exception:
                log(f"ERROR on {ckpt}:\n{traceback.format_exc()}")
    write_summary(index_path)


os.makedirs(args.videos, exist_ok=True)
log(f"watching {args.runs}; videos in {args.videos}; speeds {SPEED_GRID} + each stage's own target")
while True:
    one_pass()
    if args.once:
        break
    time.sleep(args.poll)
