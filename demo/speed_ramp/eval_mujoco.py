"""Robustness check in booster_deploy's MuJoCo player: N perturbed trials per policy and speed.

Runs on: GPU host or laptop (recorder environment), from booster_deploy's repository root:
    python eval_mujoco.py --checkpoint <policy.pt> --speed 2.5 [--trials 20]

A single deterministic replay can land on either side of a fall (the same policy fell on a Mac
and survived on Linux), so a policy is judged by the share of perturbed trials it survives.
Trial k (seed k) adds, to the unchanged k1_walk player:
- a random offset of up to +/-0.05 rad on every joint at the start;
- one horizontal push of 0.3 m/s in a random direction at a random time between 3 and 7 s.
A trial survives if booster_deploy's fall detector does not stop the policy within --seconds.

Speeds, per trial, over the steady part (after the first 2 s):
- steady_speed_mps: straight-line displacement / time (any direction; what the training gate uses);
- forward_speed_mps: mean velocity along the trunk's heading, the quantity Isaac's training command
  tracks. Sideways drift and the push do not count toward it.
Energy and leg torque saturation are measured at every physics substep (booster_deploy runs PD at
2 ms, `decimation` substeps per 20 ms policy step), not once per policy step.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pkgutil
import platform
import sys

if platform.system() == "Linux":
    os.environ.setdefault("MUJOCO_GL", "egl")  # before mujoco is imported anywhere

import numpy as np
import torch

sys.path.append(".")
import tasks as tasks_pkg  # noqa: E402

for _mod in pkgutil.walk_packages(tasks_pkg.__path__, prefix="tasks."):
    __import__(_mod.name)
import mujoco  # noqa: E402
from booster_deploy.controllers.mujoco_controller import MujocoController  # noqa: E402
from booster_deploy.utils.registry import get_task  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from speed_metrics import epte_sp, heading_speed  # noqa: E402

JOINT_NOISE, PUSH_SPEED, PUSH_WINDOW = 0.05, 0.3, (3.0, 7.0)
SETTLE_S = 2.0  # speed, gait and energy measures start after this

# Joint damping profiles for booster_deploy's MuJoCo player. "v3": only the two arm joints whose explicit PD is
# unstable at 2 ms in some postures (damping x dt / inertia: elbow pitch 2.69, shoulder pitch 1.94) are lowered,
# to 0.7 and 1.0 (worst-case ratio ~0.95). Shoulder roll and elbow yaw keep Booster's 2.0.
DAMPING_PROFILES = {"v3": {"shoulder_pitch": 1.0, "elbow_pitch": 0.7}}


def set_damping_profile(cfg, profile: str | None) -> None:
    """Apply a damping profile to a booster_deploy task config (None keeps Booster's values)."""
    if not profile:
        return
    rules = DAMPING_PROFILES[profile]
    damping = list(cfg.robot.joint_damping)
    for i, n in enumerate(cfg.robot.joint_names):
        for key, kd in rules.items():
            if key in n:
                damping[i] = kd
    cfg.robot.joint_damping = damping


class SubstepMeter:
    """Integrates power, leg torque saturation and heading speed at every MuJoCo physics step.

    booster_deploy's ctrl_step computes PD torques and calls mujoco.mj_step `decimation` times per
    policy step; sampling d.ctrl once per policy step sees only the last substep, which misses
    fast arm chatter. Installed by wrapping mujoco.mj_step for the duration of one trial.
    """

    def __init__(self, m, limits: np.ndarray, legs: np.ndarray):
        self.m, self.limits, self.legs = m, limits, legs
        self.active = False
        self.energy = self.time = self.dist_fwd = 0.0
        self.sat_sum, self.n = 0.0, 0

    def sample(self, d) -> None:
        if not self.active:
            return
        dt = self.m.opt.timestep
        tau = np.asarray(d.ctrl, dtype=float)
        self.energy += float(np.abs(tau * d.qvel[6:]).sum()) * dt
        self.sat_sum += float((np.abs(tau[self.legs]) >= 0.98 * self.limits[self.legs]).mean())
        self.dist_fwd += heading_speed(d) * dt
        self.time += dt
        self.n += 1

    def __enter__(self):
        self._step = mujoco.mj_step

        def step(m, d, *a, **k):
            self._step(m, d, *a, **k)
            self.sample(d)

        mujoco.mj_step = step
        return self

    def __exit__(self, *exc):
        mujoco.mj_step = self._step


def trial(checkpoint: str | None, speed: float, seed: int, seconds: float, damping_profile: str | None = None) -> dict:
    rng = np.random.default_rng(seed)
    cfg = copy.deepcopy(get_task("k1_walk"))  # the registry object is shared
    cfg.policy.device = "cpu"
    set_damping_profile(cfg, damping_profile)
    if checkpoint:
        cfg.policy.checkpoint_path = os.path.abspath(checkpoint)
    ctrl = MujocoController(cfg)
    ctrl.update_vel_command = lambda: None
    ctrl.vel_command.lin_vel_x, ctrl.vel_command.lin_vel_y, ctrl.vel_command.ang_vel_yaw = speed, 0.0, 0.0
    d = ctrl.mj_data
    d.qpos[7:] += rng.uniform(-JOINT_NOISE, JOINT_NOISE, size=d.qpos[7:].shape)
    mujoco.mj_forward(ctrl.mj_model, d)
    dt = cfg.policy_dt
    push_step = int(rng.uniform(*PUSH_WINDOW) / dt)
    angle = rng.uniform(0, 2 * math.pi)
    m = ctrl.mj_model
    ground = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    feet = [[g for g in range(m.ngeom) if m.geom_bodyid[g] == mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
             and m.geom_contype[g] != 0] for b in ("left_ankle_roll_link", "right_ankle_roll_link")]
    limits = ctrl.robot.effort_limit.numpy()
    legs = np.array(["hip" in j or "knee" in j or "ankle" in j for j in ctrl.robot.cfg.joint_names])
    meter = SubstepMeter(m, limits, legs)
    settle = int(round(SETTLE_S / dt))  # gait, energy and speed measures cover the steady part only
    total = int(round(seconds / dt))
    contact_log, pitch, roll, speeds = [], [], [], []
    ctrl.update_state()
    ctrl.start()
    start_xy, xy_at_settle, stop_t, n = d.qpos[:2].copy(), None, None, 0
    with meter:
        for n in range(total):
            if n == push_step:
                d.qvel[0] += PUSH_SPEED * math.cos(angle)
                d.qvel[1] += PUSH_SPEED * math.sin(angle)
            if n == settle:
                xy_at_settle = d.qpos[:2].copy()
                meter.active = True
            ctrl.update_state()
            targets = ctrl.policy_step()
            if not ctrl.is_running:
                stop_t = n * dt
                break
            ctrl.ctrl_step(targets)
            speeds.append(heading_speed(d))
            if n >= settle:
                touch = [False, False]
                for c in d.contact[:d.ncon]:
                    for k in (0, 1):
                        if (c.geom1 == ground and c.geom2 in feet[k]) or (c.geom2 == ground and c.geom1 in feet[k]):
                            touch[k] = True
                contact_log.append(touch)
                w, x, y, z = d.qpos[3:7]
                roll.append(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
                pitch.append(math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))))
    end_xy = d.qpos[:2].copy()
    ran = (n + (0 if stop_t is not None else 1)) * dt
    steady = forward = None
    if xy_at_settle is not None and ran > SETTLE_S + 0.5:
        steady = float(np.linalg.norm(end_xy - xy_at_settle) / (ran - SETTLE_S))
        forward = meter.dist_fwd / meter.time if meter.time > 0 else None
    energy = meter.energy
    gait = None
    if len(contact_log) > 50 and xy_at_settle is not None:
        cl = np.array(contact_log)
        stance = []  # mean stance duration per foot (s)
        for k in (0, 1):
            runs, cur = [], 0
            for v in cl[:, k]:
                if v:
                    cur += 1
                elif cur:
                    runs.append(cur)
                    cur = 0
            stance.append(np.mean(runs) * dt if runs else float("nan"))
        dist = float(np.linalg.norm(end_xy - xy_at_settle))
        mass = float(m.body_mass.sum())
        gait = {
            "flight_fraction": float((~cl[:, 0] & ~cl[:, 1]).mean()),
            "duty_factor": float(cl.mean()),
            "stance_asymmetry": float(abs(stance[0] - stance[1]) / np.nanmean(stance)) if np.all(np.isfinite(stance)) else None,
            "cost_of_transport": energy / (mass * 9.81 * dist) if dist > 0.5 else None,
            "leg_torque_saturation": meter.sat_sum / meter.n if meter.n else None,
            "trunk_pitch_rms_deg": float(np.degrees(np.sqrt(np.mean(np.square(pitch))))),
            "trunk_roll_rms_deg": float(np.degrees(np.sqrt(np.mean(np.square(roll))))),
        }
    return {"seed": seed, "survived": stop_t is None, "stop_s": stop_t, "steady_speed_mps": steady,
            "forward_speed_mps": forward, "epte_sp": epte_sp(speeds, speed, total) if speed > 0 else None,
            "distance_m": float(np.linalg.norm(end_xy - start_xy)), "gait": gait}


def evaluate(checkpoint: str | None, speed: float, trials: int = 20, seconds: float = 10.0, seed0: int = 0,
             damping_profile: str | None = None) -> dict:
    torch.set_num_threads(1)
    res = [trial(checkpoint, speed, seed0 + k, seconds, damping_profile) for k in range(trials)]
    ok = [r for r in res if r["survived"]]
    speeds = [r["steady_speed_mps"] for r in ok if r["steady_speed_mps"] is not None]
    fwd = [r["forward_speed_mps"] for r in ok if r["forward_speed_mps"] is not None]
    gait = {}
    for key in ("flight_fraction", "duty_factor", "stance_asymmetry", "cost_of_transport", "leg_torque_saturation",
                "trunk_pitch_rms_deg", "trunk_roll_rms_deg"):
        vals = [r["gait"][key] for r in ok if r["gait"] and r["gait"][key] is not None]
        gait[key] = round(float(np.mean(vals)), 4) if vals else None
    return {
        "seed0": seed0,
        "damping_profile": damping_profile,
        "gait_survivors_mean": gait,
        "trials": trials,
        "survived": len(ok),
        "survival_rate": len(ok) / trials,
        "mean_speed_survivors_mps": round(float(np.mean(speeds)), 3) if speeds else None,
        "mean_forward_speed_survivors_mps": round(float(np.mean(fwd)), 3) if fwd else None,
        "perturbation": {"joint_noise_rad": JOINT_NOISE, "push_mps": PUSH_SPEED, "push_window_s": list(PUSH_WINDOW)},
        "per_trial": res,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--speed", type=float, required=True)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--seed0", type=int, default=0, help="first trial seed (validation 0-19; final test uses 1000+)")
    ap.add_argument("--damping_profile", default=None, help="joint damping profile (version 3: v3)")
    a = ap.parse_args()
    r = evaluate(a.checkpoint, a.speed, a.trials, a.seconds, a.seed0, a.damping_profile)
    print(json.dumps({k: v for k, v in r.items() if k != "per_trial"}, indent=2))
