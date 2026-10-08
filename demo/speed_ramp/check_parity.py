"""Go/no-go check before training: does Booster's k1_walk, untrained, still walk in the training task?

Runs on: GPU host, with the training environment active.
    python check_parity.py [--speeds 1.0,1.5] [--num_envs 512]

It builds the task train_v3.py trains (k1_speed_env.make_env_cfg, with its randomization: friction, motor gains,
trunk mass, command delay) with two changes so the measurement is steady: no pushes, and every robot commanded
exactly the speed under test. At each speed it plays k1_walk for --steps policy steps and reports the share of
robots that fell, the steady forward speed (after the first 2 s, robots that never fell), the trunk height and
the mean mechanical power.

A speed passes if at most --max_fall_rate of robots fell and the steady speed is within --speed_tolerance of the
command. The last line printed is the verdict ("PARITY passed" or "PARITY FAILED"); --out also writes the results
as JSON. (Isaac Sim's shutdown ends the process itself, so the exit status does not carry the verdict.) The
defaults are a sanity bar, not a calibration: the one recorded measurement is k1_walk at 1.0 m/s, which walked at
1.08 m/s in this task (RECIPE.md, note 2).
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

K1_WALK = "/workspace/upstream/booster_deploy/tasks/locomotion/robots/k1/models/k1_walk.pt"
POLICY_HZ = 50

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--speeds", default="1.0,1.5", help="comma-separated forward speeds (m/s)")
parser.add_argument("--steps", type=int, default=900, help="policy steps per speed (50 per second)")
parser.add_argument("--checkpoint", default=K1_WALK, help="exported policy to check (default: Booster's k1_walk)")
parser.add_argument("--max_fall_rate", type=float, default=0.05, help="pass bar: share of robots that fell")
parser.add_argument("--speed_tolerance", type=float, default=0.15, help="pass bar: |speed - command| / command")
parser.add_argument("--out", default=None, help="also write the results to this JSON file")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not os.path.isfile(args.checkpoint):
    sys.exit(f"--checkpoint {args.checkpoint} not found (run scripts/launch/runpod/setup_golden_env.sh first)")
speeds = [float(s) for s in args.speeds.split(",")]
if any(s <= 0 for s in speeds):
    sys.exit("--speeds must be positive")
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k1_speed_env as K  # noqa: E402

SETTLE_STEPS = 2 * POLICY_HZ  # skip the first 2 s (accelerating from standstill)


def main() -> list[dict]:
    cfg = K.make_env_cfg(args.num_envs)
    cfg.events.push = None  # measure steady walking, without pushes
    command = cfg.commands.base_velocity
    command.frontier_frac, command.frontier_width = 1.0, 0.0  # every robot gets exactly v_max ...
    command.resampling_time_range = (1.0e9, 1.0e9)  # ... for the whole check
    env = gym.make(K.TASK_ID, cfg=cfg)
    u = env.unwrapped
    robot = u.scene["robot"]
    cmd = u.command_manager.get_term("base_velocity")
    policy = torch.jit.load(args.checkpoint, map_location=u.device).actor  # the bare actor, as booster_deploy calls it

    results = []
    for speed in speeds:
        with torch.inference_mode():  # Isaac's buffers become inference tensors once stepped here
            cmd.set_v_max(speed)
            obs, _ = env.reset()
            fell = torch.zeros(u.num_envs, dtype=torch.bool, device=u.device)
            speed_sum, power_sum, n = 0.0, 0.0, 0
            for t in range(args.steps):
                obs, _, terminated, _, _ = env.step(policy(obs["policy"]))
                fell |= terminated
                if t >= SETTLE_STEPS:
                    ok = ~fell
                    speed_sum += K.forward_speed(robot)[ok].sum().item()
                    power = (robot.data.applied_torque * robot.data.joint_vel).abs().sum(dim=1)
                    power_sum += power[ok].sum().item()
                    n += int(ok.sum().item())
            fall_rate = fell.float().mean().item()
            height = robot.data.root_pos_w[:, 2].median().item()
        mean = speed_sum / n if n else float("nan")
        passed = fall_rate <= args.max_fall_rate and n > 0 and abs(mean - speed) <= args.speed_tolerance * speed
        results.append({"command_mps": speed, "passed": passed, "fall_rate": round(fall_rate, 4),
                        "steady_forward_speed_mps": round(mean, 3), "trunk_height_m": round(height, 3),
                        "mean_mechanical_power_w": round(power_sum / max(n, 1), 1)})
        print(f"PARITY {'PASS' if passed else 'FAIL'} command {speed:.2f} m/s | fell {fall_rate:.1%} of robots in "
              f"{args.steps / POLICY_HZ:.0f} s | steady forward speed {mean:.2f} m/s | trunk height {height:.2f} m | "
              f"mean mechanical power {power_sum / max(n, 1):.0f} W", flush=True)
    env.close()
    return results


if __name__ == "__main__":
    results = main()
    ok = all(r["passed"] for r in results)
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"passed": ok, "checkpoint": args.checkpoint, "max_fall_rate": args.max_fall_rate,
                       "speed_tolerance": args.speed_tolerance, "steps": args.steps, "speeds": results}, f, indent=2)
    print("PARITY " + ("passed" if ok else "FAILED: do not train until this is understood"), flush=True)
    app.close()
