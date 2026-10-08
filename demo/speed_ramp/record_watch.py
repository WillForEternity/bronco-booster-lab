"""Watch training runs and record every new checkpoint in booster_deploy's MuJoCo player at a grid of speeds.

Runs on: GPU host, in the recorder environment (/workspace/venv_rec), alongside training (launch_v3.sh starts it):
    python record_watch.py [--runs /workspace/runs/k1_run_v3] [--videos /workspace/videos/v3] [--trials 20]

For each run folder (<date>_<time>_<run name>) it records:
- every stage checkpoint (stages/stage_NN_*.pt, including stage 00 = Booster's k1_walk unchanged);
- every periodic checkpoint whose iteration is a multiple of --every (model_<iter>.pt).

Each checkpoint is exported to booster_deploy's format and played at SPEED_GRID plus its own target speed.
Outputs, per checkpoint: <videos>/<run name>/<checkpoint>/policy.pt, v<speed>.mp4, v<speed>.json, done.json.
Every video is also appended to <videos>/index.jsonl, and <videos>/summary.csv is rewritten on each pass.
Nothing is ever deleted. Videos are for watching; judge policies with final_test.py.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
SPEED_GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
STAGE_RE = re.compile(r"stage_(\d+)_(?:vmax(\d+\.\d+)_it(\d+)|k1_walk_baseline)\.pt$")
MODEL_RE = re.compile(r"model_(\d+)\.pt$")
SUMMARY_COLUMNS = ["run_dir", "run", "stage", "iteration", "stage_target_mps", "commanded_speed_mps", "steady_speed_mps",
                   "safety_stop_s", "survived_of_trials", "robust_mean_speed_mps", "seconds_run", "distance_m",
                   "min_trunk_height_m", "min_uprightness", "video", "checkpoint_sha256"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_json(path: str):
    with open(path) as f:
        return json.load(f)


def ramp_info(run_dir: str) -> tuple[float, float, list[int]]:
    """(start_speed, step, [promotion iterations]) from the run's ramp_events.jsonl."""
    start, step, hits = 1.0, 0.25, []
    path = os.path.join(run_dir, "ramp_events.jsonl")
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                e = json.loads(line)
                if e["event"] == "start":
                    start, step = e["start_speed"], e["step"]
                elif e["event"] == "target_hit":
                    hits.append(e["iteration"])
    return start, step, hits


def describe(run_dir: str, run: str, ckpt: str, every: int):
    """(stage, iteration, target, label) for a checkpoint file, or None to skip it."""
    name = os.path.basename(ckpt)
    m = STAGE_RE.search(name)
    if m:
        stage = int(m.group(1))
        if m.group(2) is None:
            return 0, 0, None, f"Booster's k1_walk, unchanged (stage 0 of run {run})"
        target, it = float(m.group(2)), int(m.group(3))
        return stage, it, target, (f"Booster k1_walk fine-tuned by Bronco Robotics | {run} | "
                                   f"stage {stage} (max target {target:.2f} m/s, iter {it})")
    m = MODEL_RE.search(name)
    if m:
        it = int(m.group(1))
        if it == 0 or it % every:
            return None
        start, step, hits = ramp_info(run_dir)
        target = start + step * sum(1 for h in hits if h < it)
        return None, it, target, (f"Booster k1_walk fine-tuned by Bronco Robotics | {run} | "
                                  f"iter {it} (training max target then {target:.2f} m/s)")
    return None


def robust_summary(r: dict) -> dict:
    keys = ("trials", "survived", "survival_rate", "mean_speed_survivors_mps", "perturbation", "gait_survivors_mean",
            "damping_profile")
    return {k: r.get(k) for k in keys} | {"per_trial": r["per_trial"]}


class Watcher:
    def __init__(self, args: argparse.Namespace):
        self.a = args
        # Imported here: they need booster_deploy's root as the working directory (main() changes to it).
        import eval_mujoco
        import export_stage
        import record_stage
        from run_state import sha256_file
        self.eval_mujoco, self.export_stage, self.record_stage, self.sha256_file = (
            eval_mujoco, export_stage, record_stage, sha256_file)
        self.index_path = os.path.join(args.videos, "index.jsonl")

    def evaluate(self, policy: str, speed: float, damping_profile: str | None) -> dict:
        r = self.eval_mujoco.evaluate(policy, speed, self.a.trials, self.a.seconds, damping_profile=damping_profile)
        return robust_summary(r)

    def process(self, run_dir: str, run: str, ckpt: str) -> None:
        info = describe(run_dir, run, ckpt, self.a.every)
        if info is None:
            return
        stage, it, target, label = info
        out_dir = os.path.join(self.a.videos, run, os.path.splitext(os.path.basename(ckpt))[0])
        if os.path.exists(os.path.join(out_dir, "done.json")):
            if self.a.trials:
                self.add_robust(out_dir)
            return
        if time.time() - os.path.getmtime(ckpt) < 30:  # may still be being written
            return
        os.makedirs(out_dir, exist_ok=True)
        policy = os.path.join(out_dir, "policy.pt")
        os.replace(self.export_stage.export(ckpt, out_dir), policy)
        speeds = sorted(set(SPEED_GRID + ([round(target, 2)] if target else [])))
        damping = None if stage == 0 else self.a.damping_profile  # stage 0 is Booster's k1_walk, played unchanged
        for speed in speeds:
            out = os.path.join(out_dir, f"v{speed:.2f}.mp4")
            m = self.record_stage.record(policy, speed, out, label, seconds=self.a.seconds, damping_profile=damping)
            if self.a.trials:
                m["robust"] = self.evaluate(policy, speed, damping)
            m.update({"run_dir": run_dir, "run": run, "training_checkpoint": ckpt,
                      "training_checkpoint_sha256": self.sha256_file(ckpt), "stage": stage, "iteration": it,
                      "stage_target_mps": target})
            with open(os.path.splitext(out)[0] + ".json", "w") as f:
                json.dump(m, f, indent=2)
            with open(self.index_path, "a") as f:
                f.write(json.dumps(m) + "\n")
            stop = "no stop" if m["safety_stop_s"] is None else f"STOP at {m['safety_stop_s']} s"
            rob = f", survived {m['robust']['survived']}/{m['robust']['trials']}" if "robust" in m else ""
            log(f"{run} {os.path.basename(ckpt)} @ {speed:.2f} m/s -> {m['steady_speed_mps']:.2f} m/s, {stop}{rob}")
        with open(os.path.join(out_dir, "done.json"), "w") as f:
            json.dump({"checkpoint": ckpt, "speeds": speeds, "label": label,
                       "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)

    def add_robust(self, out_dir: str) -> None:
        """Back-fill the perturbed-trial evaluation into videos recorded without it."""
        policy = os.path.join(out_dir, "policy.pt")
        for path in sorted(glob.glob(os.path.join(out_dir, "v*.json"))):
            m = read_json(path)
            if "robust" in m:
                continue
            m["robust"] = self.evaluate(policy, m["commanded_speed_mps"], m.get("damping_profile"))
            with open(path, "w") as f:
                json.dump(m, f, indent=2)
            log(f"robust {os.path.relpath(path, self.a.videos)}: survived {m['robust']['survived']}/{m['robust']['trials']}")

    def write_summary(self) -> None:
        rows = []
        for path in glob.glob(os.path.join(self.a.videos, "**", "v*.json"), recursive=True):
            m = read_json(path)
            if "run" not in m:
                continue
            r = m.get("robust") or {}
            m["survived_of_trials"] = f"{r['survived']}/{r['trials']}" if r else ""
            m["robust_mean_speed_mps"] = r.get("mean_speed_survivors_mps", "")
            rows.append(m)
        if not rows:
            return
        with open(os.path.join(self.a.videos, "summary.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
            w.writeheader()
            for r in sorted(rows, key=lambda r: (r["run"], r["iteration"] or 0, r["commanded_speed_mps"])):
                w.writerow(r)

    def one_pass(self) -> None:
        for run_dir in sorted(glob.glob(os.path.join(self.a.runs, self.a.run_glob))):
            run = os.path.basename(run_dir).split("_", 2)[-1]  # <date>_<time>_<run name>
            ckpts = sorted(glob.glob(os.path.join(run_dir, "stages", "stage_*.pt")))
            ckpts += sorted(glob.glob(os.path.join(run_dir, "model_*.pt")), key=lambda p: int(MODEL_RE.search(p).group(1)))
            for ckpt in ckpts:
                try:
                    self.process(run_dir, run, ckpt)
                except Exception:  # noqa: BLE001  (one bad checkpoint must not stop the watcher)
                    log(f"ERROR on {ckpt}:\n{traceback.format_exc()}")
        self.write_summary()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", default="/workspace/runs/k1_run_v3", help="folder holding the run folders")
    ap.add_argument("--videos", default="/workspace/videos/v3")
    ap.add_argument("--deploy_dir", default="/workspace/upstream/booster_deploy", help="booster_deploy's repository")
    ap.add_argument("--every", type=int, default=1000, help="also record model_<iter>.pt when iter is a multiple of this")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--poll", type=float, default=60.0, help="seconds between passes")
    ap.add_argument("--once", action="store_true", help="one pass, then exit")
    ap.add_argument("--trials", type=int, default=0, help="also run eval_mujoco.py with this many perturbed trials per video")
    ap.add_argument("--run_glob", default="*_*", help="which run folders under --runs to watch")
    ap.add_argument("--damping_profile", default="v3",
                    help="damping profile for trained policies (v3 for train_v3.py runs); stage 0 always plays unchanged")
    a = ap.parse_args()
    a.runs, a.videos = os.path.abspath(a.runs), os.path.abspath(a.videos)
    if not os.path.isdir(a.deploy_dir):
        sys.exit(f"--deploy_dir {a.deploy_dir} not found")
    os.chdir(a.deploy_dir)  # booster_deploy imports its task registry from the working directory
    sys.path.insert(0, HERE)
    watcher = Watcher(a)
    os.makedirs(a.videos, exist_ok=True)
    log(f"watching {a.runs}; videos in {a.videos}; speeds {SPEED_GRID} + each stage's own target")
    while True:
        watcher.one_pass()
        if a.once:
            break
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
