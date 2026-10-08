"""Go/no-go gate: does Booster's k1_walk still walk in our Isaac Lab task, with no training?

Runs on: GPU host.
    python check_parity.py [--num_envs 512] [--no_delay] [--speeds 0.5,1.0,1.5]

Passes if, at each commanded speed, few robots fall and the steady forward speed is close to
the command (k1_walk walked at 0.97 m/s for a 1.0 m/s command in booster_deploy's MuJoCo player).
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

K1_WALK = "/workspace/upstream/booster_deploy/tasks/locomotion/robots/k1/models/k1_walk.pt"

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--no_delay", action="store_true", help="disable the actuator command delay")
parser.add_argument("--speeds", default="0.5,1.0,1.5")
parser.add_argument("--steps", type=int, default=900, help="policy steps per speed (50 per second)")
parser.add_argument("--checkpoint", default=K1_WALK, help="one or more comma-separated exported policies")
parser.add_argument("--friction", type=float, default=None, help="fix foot friction (static = dynamic)")
parser.add_argument("--ideal_pd", action="store_true", help="plain PD motors, like booster_deploy's MuJoCo player")
parser.add_argument("--tag", default="")
parser.add_argument("--v3", action="store_true", help="use the version-3 task (randomization included) and report mechanical power")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k1_speed_env as K  # noqa: E402

cfg = (K.make_env_cfg_v3(args.num_envs) if args.v3 else
       K.make_env_cfg("speed_only", args.num_envs, actuator_delay=not args.no_delay, ideal_pd=args.ideal_pd, friction=args.friction))
if args.v3:
    cfg.events.push = None  # measure steady power without pushes
    cfg.commands.base_velocity.frontier_frac = 1.0
    cfg.commands.base_velocity.frontier_width = 0.0
    cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
env = gym.make("Bronco-K1-SpeedRamp-v0", cfg=cfg)
u = env.unwrapped
robot = u.scene["robot"]
cmd = u.command_manager.get_term("base_velocity")
policies = {p: torch.jit.load(p, map_location=u.device).actor for p in args.checkpoint.split(",")}  # bare actor, as deployed

print("joints (sim order):", robot.joint_names)
print("bodies:", robot.body_names)
for name, act in robot.actuators.items():
    print(f"actuator {name}: joints {act.joint_names} stiffness {act.stiffness[0].tolist()} effort {act.effort_limit[0].tolist()}")

for (ckpt, policy), speed in [(pp, float(sp)) for pp in policies.items() for sp in args.speeds.split(",")]:
    with torch.inference_mode():  # Isaac's buffers become inference tensors once stepped here
        cmd.set_target(speed) if hasattr(cmd, "set_target") else cmd.set_v_max(speed)
        obs, _ = env.reset()
        fell = torch.zeros(u.num_envs, dtype=torch.bool, device=u.device)
        speed_sum, speed_n, pw_sum, pw_n = 0.0, 0, 0.0, 0
        for t in range(args.steps):
            obs, _, terminated, _, _ = env.step(policy(obs["policy"]))
            fell |= terminated
            if t >= 100:  # skip the first 2 s (accelerating from standstill)
                ok = ~fell
                speed_sum += K.forward_speed(robot)[ok].sum().item()
                speed_n += int(ok.sum().item())
                pw = (robot.data.applied_torque * robot.data.joint_vel).abs().sum(dim=1)
                pw_sum += pw[ok].sum().item()
                pw_n += int(ok.sum().item())
    mean = speed_sum / max(speed_n, 1)
    print(
        f"PARITY [{args.tag}] {'k1_walk' if ckpt.endswith('k1_walk.pt') else '/'.join(ckpt.split('/')[-3:-1])} speed={speed:.2f} m/s | fell {fell.float().mean().item() * 100:.1f}% of robots in {args.steps / 50:.0f} s"
        f" | steady forward speed {mean:.2f} m/s | trunk height {robot.data.root_pos_w[:, 2].median().item():.2f} m | mean mechanical power {pw_sum / max(pw_n, 1):.0f} W",
        flush=True,
    )

env.close()
app.close()
