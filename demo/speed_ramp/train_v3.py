"""Version 3: one recipe for a fast, natural K1 run (RECIPE.md; k1_speed_env.py, version 3 notes).

What it trains with:
- the version-3 task (energy-aware reward; arm-deviation penalty, recipe note 1; implicit arm and
  head PD with two lower arm dampings, recipe note 2; wider randomization: 0-40 ms delay, friction
  0.4-1.2, motor gains x0.8-1.2);
- symmetry-augmented PPO (k1_symmetry.py);
- a MuJoCo validation gate: after the Isaac gate passes, the candidate is exported and run in
  booster_deploy's MuJoCo player for 20 perturbed trials (seeds 0-19) at the new v_max. It is
  promoted only if >= --mj_min_survive trials survive at >= --mj_speed_ratio x v_max. A failed
  check is logged ("validation_failed") and retried after --mj_cooldown iterations. If the check
  itself breaks (crash, timeout, unreadable output), it is logged ("validation_error", with the
  evaluator's stderr in candidates/) and retried the same way; training continues. MuJoCo seeds
  1000+ are reserved for the final test (final_test.py) and never used here.

Runs on: GPU host.
    python train_v3.py --seed 1 --run_name seed1
    python train_v3.py --seed 1 --run_name seed1 --resume /workspace/runs/k1_run_v3/<run>   # after a crash

--resume continues a run in its own folder from its latest checkpoint (model_<it>.pt or a stage
checkpoint, whichever is later), with the optimizer state, and rebuilds the curriculum (v_max,
stage, level start, last gate attempt) from ramp_events.jsonl up to that iteration. Iterations
after the checkpoint are redone. The tracking window starts empty, so the Isaac gate cannot pass
for --window iterations, and the adaptive learning rate restarts from its configured value.

Each launch records the SHA-256 of every .py file next to this script (and the Git commit, if
any) in <run>/params/code_<timestamp>.json.

The speed curriculum (from version 2), with a stability-gated promotion:
- Robots train on a range of target speeds [0, v_max]; half are held near v_max (the frontier),
  and targets change mid-episode (k1_speed_env.SpeedCurriculumCommand).
- v_max is raised by --step only when ALL hold:
    * at least --min_iters iterations at the current v_max;
    * over the last --window iterations, frontier robots move at >= --track_ratio of their target;
    * over the same window, <= --max_fall_rate of finished frontier episodes ended in a fall,
      with at least --min_episodes finished frontier episodes counted.
- Training conditions vary: pushes, trunk mass, motor gains (plus friction and command delay).
Stage checkpoints: stages/stage_NN_vmax<v>_it<iter>.pt (stage 00 = Booster's k1_walk unchanged).
Events: ramp_events.jsonl ("start", "actor_unfrozen", "target_hit", "stalled", "finished").
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from datetime import datetime

from isaaclab.app import AppLauncher

K1_WALK = "/workspace/upstream/booster_deploy/tasks/locomotion/robots/k1/models/k1_walk.pt"

parser = argparse.ArgumentParser()
parser.add_argument("--run_name", required=True)
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--max_iterations", type=int, default=20000)
parser.add_argument("--save_interval", type=int, default=250)
parser.add_argument("--warmup", type=int, default=100, help="critic-only iterations before the actor trains")
parser.add_argument("--start_speed", type=float, default=1.0, help="initial v_max")
parser.add_argument("--step", type=float, default=0.25)
parser.add_argument("--min_iters", type=int, default=300)
parser.add_argument("--window", type=int, default=200)
parser.add_argument("--track_ratio", type=float, default=0.9)
parser.add_argument("--max_fall_rate", type=float, default=0.02)
parser.add_argument("--min_episodes", type=int, default=200)
parser.add_argument("--stall_iters", type=int, default=3000)
parser.add_argument("--checkpoint", default=K1_WALK)
parser.add_argument("--log_root", default="/workspace/runs/k1_run_v3")
parser.add_argument("--mj_trials", type=int, default=20)
parser.add_argument("--mj_min_survive", type=int, default=18)
parser.add_argument("--mj_speed_ratio", type=float, default=0.9)
parser.add_argument("--mj_cooldown", type=int, default=100)
parser.add_argument("--mj_timeout", type=int, default=600, help="seconds before a MuJoCo check counts as broken")
parser.add_argument("--resume", default=None, help="run folder to continue from its latest checkpoint")
parser.add_argument("--rec_python", default="/workspace/venv_rec/bin/python")
parser.add_argument("--deploy_dir", default="/workspace/upstream/booster_deploy")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import export_stage  # noqa: E402
import k1_speed_env as K  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlSymmetryCfg  # noqa: E402
from run_state import code_provenance, curriculum_state, latest_checkpoint  # noqa: E402


class CurriculumRunner(OnPolicyRunner):
    """RSL-RL runner plus critic warm-up and the stability-gated v_max curriculum."""

    def setup_curriculum(self, a, state: dict | None = None) -> None:
        state = state or {"stage": 0, "v_max": a.start_speed, "level_start": a.warmup, "last_mj_try": None,
                          "stall_logged": False}
        self.a = a
        self.cmd = self.env.unwrapped.command_manager.get_term("base_velocity")
        self.cmd.set_v_max(state["v_max"])
        self.win = deque(maxlen=a.window)
        self.stage = state["stage"]
        self.level_start = state["level_start"]
        self.stall_logged = state["stall_logged"]
        if state["last_mj_try"] is not None:
            self.last_mj_try = state["last_mj_try"]
        self.t0 = time.time()
        self._set_actor_frozen(self.current_learning_iteration < a.warmup)

    def event(self, kind: str, **fields) -> None:
        rec = {"event": kind, "wall_s": round(time.time() - self.t0, 1), "timesteps": getattr(self, "tot_timesteps", 0), **fields}
        with open(os.path.join(self.log_dir, "ramp_events.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[RAMP] {rec}", flush=True)

    def _set_actor_frozen(self, frozen: bool) -> None:
        policy = self.alg.policy
        for p in policy.actor.parameters():
            p.requires_grad_(not frozen)
        for name in ("std", "log_std"):
            if hasattr(policy, name):
                getattr(policy, name).requires_grad_(not frozen)
        if frozen:
            self.alg.schedule = "fixed"
        else:
            self.alg.schedule = "adaptive"
            self.alg.learning_rate = self.a.lr_after_warmup
            for group in self.alg.optimizer.param_groups:
                group["lr"] = self.alg.learning_rate

    def mujoco_check(self, it: int) -> dict:
        """Export the current actor and run eval_mujoco.py (perturbed trials, seeds 0-19) at v_max.

        Never raises: a broken check returns {"error": ...} so training continues.
        """
        cdir = os.path.join(self.log_dir, "candidates")
        os.makedirs(cdir, exist_ok=True)
        ckpt = os.path.join(cdir, f"cand_vmax{self.cmd.v_max:.2f}_it{it}.pt")
        log_path = ckpt.replace(".pt", "_eval.log")
        try:
            self.save(ckpt)
            policy = export_stage.export(ckpt, cdir)
            cmd = [self.a.rec_python, os.path.join(HERE, "eval_mujoco.py"), "--checkpoint", policy,
                   "--speed", f"{self.cmd.v_max:.2f}", "--trials", str(self.a.mj_trials), "--seed0", "0",
                   "--damping_profile", "v3"]
            env = dict(os.environ, MUJOCO_GL="egl", PYTHONWARNINGS="ignore")
            p = subprocess.run(cmd, cwd=self.a.deploy_dir, env=env, capture_output=True, text=True,
                               timeout=self.a.mj_timeout)
            with open(log_path, "w") as f:
                f.write(p.stderr)
            if p.returncode != 0:
                raise RuntimeError(f"eval_mujoco.py exited with {p.returncode}")
            out = p.stdout
            r = json.loads(out if out.startswith("{") else out[out.index("\n{") + 1:])
        except Exception as e:  # noqa: BLE001  (the gate must never take training down)
            return {"error": f"{type(e).__name__}: {e}"[:300], "candidate": os.path.basename(ckpt), "log": log_path}
        r["passed"] = (r["survived"] >= self.a.mj_min_survive and r["mean_speed_survivors_mps"] is not None
                       and r["mean_speed_survivors_mps"] >= self.a.mj_speed_ratio * self.cmd.v_max)
        r["candidate"] = os.path.basename(ckpt)
        return r

    def _window_stats(self) -> dict:
        s = {k: sum(w[k] for w in self.win) for k in ("ratio_sum", "ratio_n", "f_falls", "f_done", "a_falls", "a_done")}
        return {
            "frontier_tracking": s["ratio_sum"] / s["ratio_n"] if s["ratio_n"] else 0.0,
            "frontier_fall_rate": s["f_falls"] / s["f_done"] if s["f_done"] else 1.0,
            "frontier_episodes": int(s["f_done"]),
            "overall_fall_rate": s["a_falls"] / s["a_done"] if s["a_done"] else 1.0,
        }

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        super().log(locs, width, pad)
        it = locs["it"]
        self.win.append(self.cmd.pop_stats())
        st = self._window_stats()
        self.writer.add_scalar("Curriculum/v_max", self.cmd.v_max, it)
        self.writer.add_scalar("Curriculum/stage", self.stage, it)
        for k in ("frontier_tracking", "frontier_fall_rate", "overall_fall_rate"):
            self.writer.add_scalar(f"Curriculum/{k}", st[k], it)
        if it + 1 == self.a.warmup:
            self._set_actor_frozen(False)
            self.event("actor_unfrozen", iteration=it)
        if it < self.a.warmup:
            return
        on_level = it - self.level_start
        ready = (
            on_level >= self.a.min_iters
            and len(self.win) == self.win.maxlen
            and st["frontier_episodes"] >= self.a.min_episodes
            and st["frontier_tracking"] >= self.a.track_ratio
            and st["frontier_fall_rate"] <= self.a.max_fall_rate
        )
        if ready and it - getattr(self, "last_mj_try", -10**9) < self.a.mj_cooldown:
            ready = False
        mj = None
        if ready:
            self.last_mj_try = it
            mj = self.mujoco_check(it)
            if "error" in mj:
                self.event("validation_error", target=self.cmd.v_max, iteration=it, **mj)
                return
            mj_brief = {"mj_survived": mj["survived"], "mj_trials": mj["trials"], "mj_speed": mj["mean_speed_survivors_mps"],
                        "mj_gait": mj["gait_survivors_mean"], "candidate": mj["candidate"]}
            if not mj["passed"]:
                self.event("validation_failed", target=self.cmd.v_max, iteration=it, **mj_brief,
                           **{k: round(v, 4) if isinstance(v, float) else v for k, v in st.items()})
                ready = False
        if ready:
            self.stage += 1
            name = f"stage_{self.stage:02d}_vmax{self.cmd.v_max:.2f}_it{it}.pt"
            self.save(os.path.join(self.log_dir, "stages", name))
            self.event("target_hit", stage=self.stage, target=self.cmd.v_max, iteration=it, **mj_brief, iterations_on_target=on_level, checkpoint=name,
                       **{k: round(v, 4) if isinstance(v, float) else v for k, v in st.items()})
            self.cmd.set_v_max(self.cmd.v_max + self.a.step)
            self.win.clear()
            self.level_start = it
            self.stall_logged = False
        elif not self.stall_logged and on_level >= self.a.stall_iters:
            self.stall_logged = True
            self.event("stalled", target=self.cmd.v_max, iteration=it, **{k: round(v, 4) if isinstance(v, float) else v for k, v in st.items()})


env_cfg = K.make_env_cfg_v3(args.num_envs)
env_cfg.seed = args.seed
agent_cfg = K.K1SpeedPPOCfg()
agent_cfg.seed = args.seed
agent_cfg.max_iterations = args.max_iterations
agent_cfg.save_interval = args.save_interval
agent_cfg.experiment_name = "k1_run_v3"
agent_cfg.algorithm.symmetry_cfg = RslRlSymmetryCfg(use_data_augmentation=True, use_mirror_loss=False,
                                                    data_augmentation_func="k1_symmetry:compute_symmetric_states")
agent_cfg.run_name = args.run_name
args.lr_after_warmup = agent_cfg.algorithm.learning_rate

stamp = f"{datetime.now():%Y-%m-%d_%H-%M-%S}"
if args.resume:
    log_dir = os.path.abspath(args.resume)
    resume_ckpt, resume_it = latest_checkpoint(log_dir)
    with open(os.path.join(log_dir, "ramp_events.jsonl")) as f:
        resume_state = curriculum_state([json.loads(line) for line in f if line.strip()], args, resume_it)
    with open(os.path.join(log_dir, "params", f"resume_{stamp}.json"), "w") as f:
        json.dump({"args": vars(args), "checkpoint": resume_ckpt, "iteration": resume_it, "state": resume_state}, f,
                  indent=2, default=str)
else:
    log_dir = os.path.join(args.log_root, f"{stamp}_{args.run_name}")
    os.makedirs(os.path.join(log_dir, "stages"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    with open(os.path.join(log_dir, "params", "curriculum_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)
with open(os.path.join(log_dir, "params", f"code_{stamp}.json"), "w") as f:
    json.dump(code_provenance(), f, indent=2)

env = RslRlVecEnvWrapper(gym.make("Bronco-K1-SpeedRamp-v0", cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
runner = CurriculumRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

if args.resume:
    runner.load(resume_ckpt)  # policy, optimizer, iteration
    runner._prepare_logging_writer()  # save() needs the writer, which learn() would otherwise create
    runner.setup_curriculum(args, resume_state)
    runner.event("resumed", checkpoint=os.path.relpath(resume_ckpt, log_dir), iteration=resume_it, **resume_state)
else:
    k1_walk = torch.jit.load(args.checkpoint, map_location=agent_cfg.device)
    actor_sd = {k.removeprefix("actor."): v for k, v in k1_walk.state_dict().items() if k.startswith("actor.")}
    runner.alg.policy.actor.load_state_dict(actor_sd)
    runner._prepare_logging_writer()  # save() needs the writer, which learn() would otherwise create
    runner.save(os.path.join(log_dir, "stages", "stage_00_k1_walk_baseline.pt"))
    runner.setup_curriculum(args)
    runner.event("start", version=3, seed=args.seed, start_speed=args.start_speed, step=args.step,
                 min_iters=args.min_iters, window=args.window, track_ratio=args.track_ratio,
                 max_fall_rate=args.max_fall_rate, min_episodes=args.min_episodes, warmup=args.warmup,
                 source=args.checkpoint, mj_trials=args.mj_trials, mj_min_survive=args.mj_min_survive,
                 mj_speed_ratio=args.mj_speed_ratio)

runner.learn(num_learning_iterations=max(0, agent_cfg.max_iterations - runner.current_learning_iteration),
             init_at_random_ep_len=True)
runner.event("finished", final_v_max=runner.cmd.v_max, stages=runner.stage)
env.close()
app.close()
