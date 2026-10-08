"""Train the version-3 recipe: fine-tune Booster's k1_walk for speed (RECIPE.md).

Runs on: GPU host, with the training environment active (Docs/runbooks/golden_env.md):
    python train_v3.py --seed 1 --run_name seed1
    python train_v3.py --resume /workspace/runs/k1_run_v3/<run>      # continue a run after a crash

What it does:
- loads Booster's k1_walk actor and trains a new critic for --warmup iterations with the actor frozen;
- trains the version-3 task (k1_speed_env.py) with symmetry-augmented PPO (k1_symmetry.py);
- raises the target speed v_max by --step each time a level passes both gates (curriculum.py):
  * Isaac gate: tracking and fall rate at the frontier over the last --window iterations;
  * MuJoCo gate: the exported candidate runs --gate_trials perturbed trials (seeds 0-19) in booster_deploy's MuJoCo
    player at v_max, under the final test's rule (criteria.py): it passes with >= --gate_min_successes trials that
    survive at >= --gate_speed_ratio x v_max. Seeds 1000+ belong to the final test (final_test.py);
- stops when a level runs --patience iterations without promotion (recipe note 6), or at --max_iterations.

A failed MuJoCo check is logged ("validation_failed") and retried after --gate_cooldown iterations. A check that
breaks (crash, timeout, unreadable output) is logged ("validation_error", with the evaluator's output in
candidates/) and retried the same way: the gate never takes training down.

Outputs, in <log_root>/<date>_<run_name>/:
- stages/stage_NN_vmax<v>_it<iter>.pt, one per promoted level (stage 00 = Booster's k1_walk unchanged);
- model_<iter>.pt every --save_interval iterations, and at the end;
- ramp_events.jsonl: "start", "actor_unfrozen", "validation_failed", "validation_error", "target_hit",
  "stopped", "resumed", "finished";
- params/: the task and PPO settings, the arguments, and the SHA-256 of the code at each launch.

--resume continues a run in its own folder from its latest checkpoint (model_<it>.pt or a stage checkpoint,
whichever is later), with the optimizer state, and rebuilds the curriculum from ramp_events.jsonl up to that
iteration. The run's own arguments (params/curriculum_args.json) are used; passing a different value for one of
them is an error. Iterations after the checkpoint are redone. The tracking window starts empty, so the Isaac gate
cannot pass for --window iterations, and the adaptive learning rate restarts from its configured value.
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from curriculum import Rules, State, gate_due, out_of_patience, promote, state_from_events, window_stats  # noqa: E402
from run_state import code_provenance, latest_checkpoint, merge_resume_args  # noqa: E402

K1_WALK = "/workspace/upstream/booster_deploy/tasks/locomotion/robots/k1/models/k1_walk.pt"
# Arguments that define a run. --resume takes them from the run; the others (paths, timeouts, the iteration cap)
# may change between launches.
RUN_KEYS = ("seed", "num_envs", "save_interval", "warmup", "start_speed", "step", "min_iters", "window",
            "track_ratio", "max_fall_rate", "min_episodes", "patience", "checkpoint", "gate_trials",
            "gate_min_successes", "gate_speed_ratio", "gate_cooldown")

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--run_name", help="name of a new run (required unless --resume)")
parser.add_argument("--resume", default=None, help="run folder to continue from its latest checkpoint")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=20000, help="hard cap; the stopping rule usually ends first")
parser.add_argument("--save_interval", type=int, default=250)
parser.add_argument("--warmup", type=int, default=100, help="critic-only iterations before the actor trains")
parser.add_argument("--start_speed", type=float, default=1.0, help="first v_max (m/s)")
parser.add_argument("--step", type=float, default=0.25, help="v_max increase per promotion (m/s)")
parser.add_argument("--min_iters", type=int, default=300, help="minimum iterations per level")
parser.add_argument("--window", type=int, default=200, help="iterations the Isaac gate averages over")
parser.add_argument("--track_ratio", type=float, default=0.9, help="Isaac gate: minimum frontier speed / target")
parser.add_argument("--max_fall_rate", type=float, default=0.02, help="Isaac gate: maximum frontier fall rate")
parser.add_argument("--min_episodes", type=int, default=200, help="Isaac gate: minimum finished frontier episodes")
parser.add_argument("--patience", type=int, default=1000,
                    help="stop after this many iterations on one level without promotion (recipe note 6)")
parser.add_argument("--checkpoint", default=K1_WALK, help="Booster's exported k1_walk policy to start from")
parser.add_argument("--log_root", default="/workspace/runs/k1_run_v3")
parser.add_argument("--gate_trials", type=int, default=20)
parser.add_argument("--gate_min_successes", type=int, default=18)
parser.add_argument("--gate_speed_ratio", type=float, default=0.9)
parser.add_argument("--gate_cooldown", type=int, default=100, help="minimum iterations between MuJoCo checks")
parser.add_argument("--gate_timeout", type=int, default=600, help="seconds before a MuJoCo check counts as broken")
parser.add_argument("--rec_python", default="/workspace/venv_rec/bin/python", help="Python of the MuJoCo player env")
parser.add_argument("--deploy_dir", default="/workspace/upstream/booster_deploy")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.gate_min_successes > args.gate_trials:
    parser.error(f"--gate_min_successes {args.gate_min_successes} > --gate_trials {args.gate_trials}: no level could pass")

# Check the run folder and arguments before starting Isaac Sim, which takes a while.
stamp = f"{datetime.now():%Y-%m-%d_%H-%M-%S}"
if args.resume:
    log_dir = os.path.abspath(args.resume)
    try:
        with open(os.path.join(log_dir, "params", "curriculum_args.json")) as f:
            saved_args = json.load(f)
        args = argparse.Namespace(**merge_resume_args(saved_args, vars(args), vars(parser.parse_args([])), RUN_KEYS))
        resume_ckpt, resume_it = latest_checkpoint(log_dir)
        with open(os.path.join(log_dir, "ramp_events.jsonl")) as f:
            events = [json.loads(line) for line in f if line.strip()]
        resume_state = state_from_events(events, Rules.from_args(args), resume_it)
    except (OSError, ValueError) as e:
        sys.exit(f"--resume {args.resume}: {e}")
elif not args.run_name:
    parser.error("--run_name is required for a new run")
else:
    log_dir = os.path.join(args.log_root, f"{stamp}_{args.run_name}")
    if not os.path.isfile(args.checkpoint):
        sys.exit(f"--checkpoint {args.checkpoint} not found (run scripts/launch/runpod/setup_golden_env.sh first)")
if not os.path.isfile(args.rec_python):
    sys.exit(f"--rec_python {args.rec_python} not found (run scripts/launch/runpod/setup_recorder_env.sh first)")

args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlSymmetryCfg, RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import export_stage  # noqa: E402
import k1_speed_env as K  # noqa: E402
from criteria import stage_criteria  # noqa: E402


class StopTraining(Exception):
    """Raised from the runner's log() to end learn() early (the stopping rule)."""


def rounded(stats: dict) -> dict:
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in stats.items()}


class CurriculumRunner(OnPolicyRunner):
    """RSL-RL runner plus critic warm-up, the gated v_max curriculum and the stopping rule (curriculum.py)."""

    def setup_curriculum(self, a: argparse.Namespace, state: State, lr_after_warmup: float) -> None:
        self.a = a
        self.rules = Rules.from_args(a)
        self.state = state
        self.lr_after_warmup = lr_after_warmup
        self.cmd = self.env.unwrapped.command_manager.get_term("base_velocity")
        self.cmd.set_v_max(state.v_max)
        self.win = deque(maxlen=self.rules.window)
        self.t0 = time.time()
        self._set_actor_frozen(self.current_learning_iteration < self.rules.warmup)

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
            self.alg.learning_rate = self.lr_after_warmup
            for group in self.alg.optimizer.param_groups:
                group["lr"] = self.alg.learning_rate

    def mujoco_check(self, it: int) -> dict:
        """Export the current actor and score it in booster_deploy's MuJoCo player (seeds 0-19) at v_max.

        Never raises: a broken check returns {"error": ...} so training continues.
        """
        v_max = self.state.v_max
        cdir = os.path.join(self.log_dir, "candidates")
        os.makedirs(cdir, exist_ok=True)
        ckpt = os.path.join(cdir, f"cand_vmax{v_max:.2f}_it{it}.pt")
        stem = os.path.splitext(ckpt)[0]
        result_path, log_path = f"{stem}_eval.json", f"{stem}_eval.log"
        try:
            self.save(ckpt)
            policy = export_stage.export(ckpt, cdir)
            cmd = [self.a.rec_python, os.path.join(HERE, "eval_mujoco.py"), "--checkpoint", policy,
                   "--speed", f"{v_max:.2f}", "--trials", str(self.a.gate_trials), "--seed0", "0",
                   "--damping_profile", "v3", "--out", result_path]
            env = dict(os.environ, MUJOCO_GL="egl", PYTHONWARNINGS="ignore")
            p = subprocess.run(cmd, cwd=self.a.deploy_dir, env=env, capture_output=True, text=True,
                               timeout=self.a.gate_timeout)
            with open(log_path, "w") as f:
                f.write(p.stdout + p.stderr)
            if p.returncode != 0:
                raise RuntimeError(f"eval_mujoco.py exited with {p.returncode}")
            with open(result_path) as f:
                per_trial = json.load(f)["per_trial"]
            c = stage_criteria(per_trial, v_max, speed_ratio=self.a.gate_speed_ratio,
                               min_success=self.a.gate_min_successes)
        except Exception as e:  # noqa: BLE001  (the gate must never take training down)
            return {"error": f"{type(e).__name__}: {e}"[:300], "candidate": os.path.basename(ckpt), "log": log_path}
        return {"passed": c["passes"], "mj_successes": c["successes"], "mj_survived": c["survived"],
                "mj_trials": c["trials"], "mj_median_speed": c["median_speed_survivors_mps"],
                "mj_gait": c["gait_survivors_mean"], "candidate": os.path.basename(ckpt)}

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        super().log(locs, width, pad)
        it = locs["it"]
        self.win.append(self.cmd.pop_stats())
        st = window_stats(self.win)
        self.writer.add_scalar("Curriculum/v_max", self.state.v_max, it)
        self.writer.add_scalar("Curriculum/stage", self.state.stage, it)
        for k in ("frontier_tracking", "frontier_fall_rate", "overall_fall_rate"):
            self.writer.add_scalar(f"Curriculum/{k}", st[k], it)
        if it + 1 == self.rules.warmup:
            self._set_actor_frozen(False)
            self.event("actor_unfrozen", iteration=it)
        if it < self.rules.warmup:
            return
        if gate_due(self.state, self.rules, it, st, len(self.win)):
            self.state.last_gate_try = it
            mj = self.mujoco_check(it)
            if "error" in mj:
                self.event("validation_error", target=self.state.v_max, iteration=it, **mj)
            elif not mj.pop("passed"):
                self.event("validation_failed", target=self.state.v_max, iteration=it, **mj, **rounded(st))
            else:
                name = f"stage_{self.state.stage + 1:02d}_vmax{self.state.v_max:.2f}_it{it}.pt"
                self.save(os.path.join(self.log_dir, "stages", name))
                self.event("target_hit", stage=self.state.stage + 1, target=self.state.v_max, iteration=it, **mj,
                           iterations_on_target=it - self.state.level_start, checkpoint=name, **rounded(st))
                self.state = promote(self.state, self.rules, it)
                self.cmd.set_v_max(self.state.v_max)
                self.win.clear()
                return
        if out_of_patience(self.state, self.rules, it):
            self.event("stopped", reason="no promotion within --patience iterations (recipe note 6)",
                       target=self.state.v_max, iteration=it, level_start=self.state.level_start, **rounded(st))
            raise StopTraining


def main() -> None:
    env_cfg = K.make_env_cfg(args.num_envs)
    env_cfg.seed = args.seed
    agent_cfg = K.K1SpeedPPOCfg()
    agent_cfg.seed = args.seed
    agent_cfg.max_iterations = args.max_iterations
    agent_cfg.save_interval = args.save_interval
    agent_cfg.algorithm.symmetry_cfg = RslRlSymmetryCfg(use_data_augmentation=True, use_mirror_loss=False,
                                                        data_augmentation_func="k1_symmetry:compute_symmetric_states")
    agent_cfg.run_name = args.run_name or os.path.basename(log_dir)

    if args.resume:
        with open(os.path.join(log_dir, "params", f"resume_{stamp}.json"), "w") as f:
            json.dump({"args": vars(args), "checkpoint": resume_ckpt, "iteration": resume_it,
                       "state": vars(resume_state)}, f, indent=2, default=str)
    else:
        os.makedirs(os.path.join(log_dir, "stages"), exist_ok=True)
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        with open(os.path.join(log_dir, "params", "curriculum_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2, default=str)
    with open(os.path.join(log_dir, "params", f"code_{stamp}.json"), "w") as f:
        json.dump(code_provenance(), f, indent=2)

    env = RslRlVecEnvWrapper(gym.make(K.TASK_ID, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = CurriculumRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    lr = agent_cfg.algorithm.learning_rate

    if args.resume:
        runner.load(resume_ckpt)  # policy, optimizer, iteration
        runner._prepare_logging_writer()  # save() needs the writer, which learn() would otherwise create
        runner.setup_curriculum(args, resume_state, lr)
        runner.event("resumed", checkpoint=os.path.relpath(resume_ckpt, log_dir), iteration=resume_it,
                     **vars(resume_state))
    else:
        k1_walk = torch.jit.load(args.checkpoint, map_location=agent_cfg.device)
        actor_sd = {k.removeprefix("actor."): v for k, v in k1_walk.state_dict().items() if k.startswith("actor.")}
        runner.alg.policy.actor.load_state_dict(actor_sd)
        runner._prepare_logging_writer()  # save() needs the writer, which learn() would otherwise create
        runner.save(os.path.join(log_dir, "stages", "stage_00_k1_walk_baseline.pt"))
        state = State.fresh(Rules.from_args(args))
        runner.setup_curriculum(args, state, lr)
        runner.event("start", version=3, **{k: getattr(args, k) for k in RUN_KEYS})

    try:
        runner.learn(num_learning_iterations=max(0, agent_cfg.max_iterations - runner.current_learning_iteration),
                     init_at_random_ep_len=True)
    except StopTraining:
        runner.save(os.path.join(log_dir, f"model_{runner.current_learning_iteration}.pt"))
    runner.event("finished", iteration=runner.current_learning_iteration, final_v_max=runner.state.v_max,
                 stages=runner.state.stage)
    if runner.writer is not None:
        runner.writer.flush()  # Isaac Sim can exit before buffered TensorBoard events are written
    env.close()


if __name__ == "__main__":
    main()
    app.close()
