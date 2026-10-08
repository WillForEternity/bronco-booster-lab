"""Booster's k1_walk MuJoCo player (booster_deploy), set up the same way for every script that plays a policy.

Runs on: GPU host or laptop, in booster_deploy's environment, from booster_deploy's repository root. booster_deploy
is not installed as a package: its scripts import `tasks` and `booster_deploy` from the working directory.
Imported by eval_mujoco.py, record_stage.py and play_speed.py.
"""

from __future__ import annotations

import copy
import os
import pkgutil
import platform
import sys

if platform.system() == "Linux":
    os.environ.setdefault("MUJOCO_GL", "egl")  # headless rendering; must be set before mujoco is first imported

if not os.path.isdir(os.path.join("tasks", "locomotion")):
    raise ImportError("Run this from booster_deploy's repository root: it imports booster_deploy's `tasks` package "
                      f"from there (current directory: {os.getcwd()}).")
sys.path.append(os.getcwd())

import tasks as _tasks  # noqa: E402

for _mod in pkgutil.walk_packages(_tasks.__path__, prefix="tasks."):
    __import__(_mod.name)  # importing a task module registers it

from booster_deploy.controllers.mujoco_controller import MujocoController  # noqa: E402
from booster_deploy.utils.registry import get_task  # noqa: E402

from k1_conventions import damping_overrides  # noqa: E402

__all__ = ["MujocoController", "fixed_speed_controller", "k1_walk_cfg"]


def k1_walk_cfg(checkpoint: str | None = None, damping_profile: str | None = None):
    """A private copy of booster_deploy's k1_walk task config, on CPU, with an optional policy and damping profile.

    checkpoint: an exported policy (export_stage.py); None plays Booster's k1_walk. damping_profile: "v3" for
    policies from train_v3.py (recipe note 2); None keeps Booster's damping.
    """
    cfg = copy.deepcopy(get_task("k1_walk"))  # the registry object is shared
    cfg.policy.device = "cpu"
    overrides = damping_overrides(list(cfg.robot.joint_names), damping_profile)
    if overrides:
        cfg.robot.joint_damping = [overrides.get(name, kd)
                                   for name, kd in zip(cfg.robot.joint_names, cfg.robot.joint_damping, strict=True)]
    if checkpoint:
        cfg.policy.checkpoint_path = os.path.abspath(checkpoint)
    return cfg


def fixed_speed_controller(cfg, speed: float) -> MujocoController:
    """A MuJoCo controller commanded to walk straight at `speed` (m/s), ignoring commands typed on stdin."""
    ctrl = MujocoController(cfg)
    ctrl.update_vel_command = lambda: None
    ctrl.vel_command.lin_vel_x, ctrl.vel_command.lin_vel_y, ctrl.vel_command.ang_vel_yaw = speed, 0.0, 0.0
    return ctrl
