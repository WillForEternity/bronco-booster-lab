"""Record one policy at one speed in booster_deploy's MuJoCo player, off-screen: a labeled MP4 and a metrics JSON.

Runs on: GPU host or laptop, in booster_deploy's environment (with opencv-python-headless and imageio-ffmpeg,
see requirements-laptop.txt), from booster_deploy's repository root:
    python <repo>/demo/speed_ramp/record_stage.py --checkpoint <policy.pt> --speed 2.0 --damping_profile v3 \
        --out videos/x.mp4 --label "Booster k1_walk fine-tuned by Bronco Robotics | seed 1 | stage 5"

Uses booster_deploy's k1_walk task and MujocoController unchanged (gains, observation, torque
limits, fall detector); only the viewer is replaced by an off-screen renderer. The player has no
randomness, so on one computer the same checkpoint, speed and versions give the same video.
"""

from __future__ import annotations

import argparse
import json
import os
import platform

from booster_player import fixed_speed_controller, k1_walk_cfg  # first: sets MUJOCO_GL before mujoco is imported

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

from k1_conventions import DAMPING_PROFILES
from run_state import sha256_file

WHITE, RED, GREY, GREEN = (255, 255, 255), (255, 70, 70), (190, 190, 190), (120, 230, 120)


def _band(img, y0, y1, alpha=0.55):
    img[y0:y1] = (img[y0:y1] * (1.0 - alpha)).astype(img.dtype)


def _text(img, s, y, color=WHITE, scale=0.55, thick=1):
    cv2.putText(img, s, (14, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def record(checkpoint: str | None, speed: float, out: str, label: str, seconds: float = 10.0, fps: int = 25,
           width: int = 1024, height: int = 576, damping_profile: str | None = None) -> dict:
    torch.set_num_threads(1)
    cfg = k1_walk_cfg(checkpoint, damping_profile)
    ctrl = fixed_speed_controller(cfg, speed)

    ctrl.mj_model.vis.global_.offwidth = max(width, ctrl.mj_model.vis.global_.offwidth)
    ctrl.mj_model.vis.global_.offheight = max(height, ctrl.mj_model.vis.global_.offheight)
    renderer = mujoco.Renderer(ctrl.mj_model, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = 2.6, 90.0, -12.0  # side view, following the robot

    dt = cfg.policy_dt
    every = max(1, round(1.0 / (fps * dt)))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    writer = imageio.get_writer(out, fps=fps, codec="libx264", pixelformat="yuv420p", quality=8,
                                macro_block_size=16)  # 1024 x 576 divides evenly, so frames are not resized

    pos, upright = [], []
    stop_t = None
    ctrl.update_state()
    ctrl.start()
    frame = None
    n_steps = int(round(seconds / dt))
    for n in range(n_steps):
        ctrl.update_state()
        targets = ctrl.policy_step()
        if not ctrl.is_running:  # booster_deploy's fall detector stopped the policy
            stop_t = n * dt
            break
        ctrl.ctrl_step(targets)
        q = ctrl.mj_data.qpos
        pos.append(q[:3].copy())
        w, x, y, z = q[3:7]
        upright.append(1.0 - 2.0 * (x * x + y * y))
        if n % every == 0:
            cam.lookat[:] = q[:3]
            renderer.update_scene(ctrl.mj_data, cam)
            frame = renderer.render().copy()
            p = np.array(pos)
            recent = p[-50:]
            v = np.linalg.norm(recent[-1, :2] - recent[0, :2]) / max((len(recent) - 1) * dt, dt) if len(recent) > 1 else 0.0
            _band(frame, 0, 88)
            _text(frame, "SIMULATION - MuJoCo replay in booster_deploy's k1_walk player (not a real robot)", 24, GREY, 0.5)
            _text(frame, label, 50, WHITE, 0.55)
            _text(frame, f"commanded {speed:.2f} m/s | measured {v:.2f} m/s (last 1 s) | t = {n * dt:4.1f} s", 76, GREEN, 0.6, 2)
            writer.append_data(frame)
    if stop_t is not None and frame is not None:
        for _ in range(int(1.5 * fps)):  # hold the last frame with the stop banner
            held = frame.copy()
            _band(held, height - 50, height, 0.7)
            _text(held, f"SAFETY STOP: Booster's fall detector stopped the policy at t = {stop_t:.1f} s", height - 24, RED, 0.65, 2)
            writer.append_data(held)
    writer.close()
    renderer.close()

    p = np.array(pos) if pos else np.zeros((1, 3))
    steady = p[100:] if len(p) > 101 else p  # skip the first 2 s (accelerating from standstill)
    steady_speed = (np.linalg.norm(steady[-1, :2] - steady[0, :2]) / ((len(steady) - 1) * dt)) if len(steady) > 1 else 0.0
    metrics = {
        "checkpoint": os.path.abspath(checkpoint) if checkpoint else "booster_deploy k1_walk.pt (Booster's)",
        "checkpoint_sha256": sha256_file(cfg.policy.checkpoint_path if os.path.isabs(cfg.policy.checkpoint_path)
                                    else os.path.join("tasks/locomotion", cfg.policy.checkpoint_path)),
        "label": label,
        "commanded_speed_mps": speed,
        "damping_profile": damping_profile,
        "seconds_requested": seconds,
        "seconds_run": round(len(pos) * dt, 2),
        "safety_stop_s": None if stop_t is None else round(stop_t, 2),
        "steady_speed_mps": round(float(steady_speed), 3),
        "distance_m": round(float(np.linalg.norm(p[-1, :2] - p[0, :2])), 2),
        "min_trunk_height_m": round(float(p[:, 2].min()), 3),
        "min_uprightness": round(float(min(upright)) if upright else 1.0, 3),
        "video": os.path.abspath(out),
        "fps": fps,
        "versions": {"mujoco": mujoco.__version__, "torch": torch.__version__, "python": platform.python_version(),
                     "platform": platform.platform()},
        "recorder_sha256": sha256_file(__file__),
    }
    with open(os.path.splitext(out)[0] + ".json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", default=None, help="exported policy; default is Booster's k1_walk")
    ap.add_argument("--speed", type=float, required=True)
    ap.add_argument("--out", required=True, help="MP4 path; the metrics JSON is written next to it")
    ap.add_argument("--label", default="Booster's k1_walk (unchanged)")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--damping_profile", choices=sorted(DAMPING_PROFILES), default=None,
                    help="v3 for policies from train_v3.py; omit for Booster's k1_walk")
    a = ap.parse_args()
    print(json.dumps(record(a.checkpoint, a.speed, a.out, a.label, a.seconds, damping_profile=a.damping_profile), indent=2))


if __name__ == "__main__":
    main()
