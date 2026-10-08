"""Play a K1 walking policy at a fixed forward speed in booster_deploy's MuJoCo player.

Runs on: laptop, inside booster_deploy's environment, from its repository root:
    mjpython <repo>/demo/speed_ramp/play_speed.py --speed 1.0             # Booster's k1_walk
    mjpython <repo>/demo/speed_ramp/play_speed.py --checkpoint /abs/path/policy.pt --speed 2.0 --damping_profile v3
(`mjpython` on macOS; `python` on Windows and Linux.)

Uses booster_deploy's k1_walk task unchanged (gains, observation, safety stop on falling); only the
policy file, the commanded speed and, with --damping_profile v3, two arm dampings differ. Policies
from train_v3.py were trained with the v3 damping (RECIPE.md, note 2), so play them with it. The
speed is set directly, so the player's 1.6 m/s clamp for typed commands does not apply; keep it at
or below the policy's trained v_max (note 4).
"""

import argparse
import pkgutil
import sys

sys.path.append(".")

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default=None, help="absolute path; default is Booster's k1_walk")
parser.add_argument("--speed", type=float, required=True)
parser.add_argument("--log", default=None, help="save states to <prefix>.npz every 100 steps")
parser.add_argument("--damping_profile", choices=["v3"], default=None,
                    help="v3 for policies from train_v3.py; omit for Booster's k1_walk")
args = parser.parse_args()

import tasks as tasks_pkg  # noqa: E402

for mod in pkgutil.walk_packages(tasks_pkg.__path__, prefix="tasks."):
    __import__(mod.name)
from booster_deploy.controllers.mujoco_controller import MujocoController  # noqa: E402
from booster_deploy.utils.registry import get_task  # noqa: E402

cfg = get_task("k1_walk")
cfg.policy.device = "cpu"
if args.checkpoint:
    cfg.policy.checkpoint_path = args.checkpoint
if args.log:
    cfg.mujoco.log_states = args.log
if args.damping_profile == "v3":  # same values as eval_mujoco.DAMPING_PROFILES["v3"]
    rules = {"shoulder_pitch": 1.0, "elbow_pitch": 0.7}
    cfg.robot.joint_damping = [next((kd for key, kd in rules.items() if key in n), d)
                               for n, d in zip(cfg.robot.joint_names, cfg.robot.joint_damping, strict=True)]
controller = MujocoController(cfg)
controller.vel_command.lin_vel_x = args.speed
controller.vel_command.lin_vel_y = 0.0
controller.vel_command.ang_vel_yaw = 0.0
controller.run()
