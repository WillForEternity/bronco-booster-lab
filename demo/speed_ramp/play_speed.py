"""Play a K1 walking policy at a fixed forward speed in booster_deploy's interactive MuJoCo viewer.

Runs on: laptop, in booster_deploy's environment, from its repository root:
    mjpython <repo>/demo/speed_ramp/play_speed.py --speed 1.0                     # Booster's k1_walk
    mjpython <repo>/demo/speed_ramp/play_speed.py --checkpoint policy.pt --speed 2.0 --damping_profile v3
(`mjpython` on macOS; `python` on Windows and Linux.)

Uses booster_deploy's k1_walk task unchanged (gains, observation, safety stop on falling); only the
policy file, the commanded speed and, with --damping_profile v3, two arm dampings differ. Policies
from train_v3.py were trained with the v3 damping (RECIPE.md, note 2), so play them with it. The
speed is set directly, so the player's 1.6 m/s clamp for typed commands does not apply; keep it at
or below the policy's trained v_max (note 4).
"""

from __future__ import annotations

import argparse

from k1_conventions import DAMPING_PROFILES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", default=None, help="exported policy; default is Booster's k1_walk")
    parser.add_argument("--speed", type=float, required=True, help="forward speed (m/s)")
    parser.add_argument("--log", default=None, help="save states to <prefix>.npz every 100 steps")
    parser.add_argument("--damping_profile", choices=sorted(DAMPING_PROFILES), default=None,
                        help="v3 for policies from train_v3.py; omit for Booster's k1_walk")
    args = parser.parse_args()

    from booster_player import MujocoController, k1_walk_cfg  # needs booster_deploy's root as working directory

    cfg = k1_walk_cfg(args.checkpoint, args.damping_profile)
    if args.log:
        cfg.mujoco.log_states = args.log
    controller = MujocoController(cfg)
    controller.vel_command.lin_vel_x = args.speed
    controller.vel_command.lin_vel_y = 0.0
    controller.vel_command.ang_vel_yaw = 0.0
    controller.run()


if __name__ == "__main__":
    main()
