# K1 speed fine-tune: from Booster's k1_walk to a run

This folder fine-tunes Booster's `k1_walk` policy for speed with reinforcement learning (PPO) in Isaac Lab. It then checks every new speed stage in Booster's own MuJoCo player before promoting it. The recipe, and why each piece is there, is in [RECIPE.md](RECIPE.md).

Everything here is simulation only. Nothing in this repository sends commands to a physical robot.

## Files

| File | Runs on | What it does |
|---|---|---|
| `train_v3.py` | GPU host | The trainer: critic warm-up, speed curriculum, symmetry-augmented PPO, Isaac gate, MuJoCo gate, stopping rule. `--resume <run>` continues a run after a crash |
| `k1_speed_env.py` | GPU host | Isaac Lab task `Bronco-K1-SpeedRamp-v0`: robot, reward, curriculum command, randomization |
| `check_parity.py` | GPU host | Go/no-go check before training: does `k1_walk`, untrained, walk in the training task? |
| `launch_v3.sh` | GPU host | Starts one training process per seed, and the recorder, in the background |
| `record_watch.py` | GPU host | Records every new checkpoint, at a grid of speeds, while training runs |
| `eval_mujoco.py` | either | N perturbed trials of one policy at one speed in Booster's MuJoCo player: survival, speeds, gait, energy, EPTE-SP |
| `final_test.py` | either | The final test on held-out trial seeds, run once after training stops |
| `record_stage.py` | either | Plays one policy at one speed off-screen. Writes a labeled MP4 and a metrics JSON |
| `export_stage.py` | either | Converts a training checkpoint to `booster_deploy`'s TorchScript format |
| `play_speed.py` | laptop | Plays one policy at one speed in the interactive MuJoCo viewer |
| `sync_from_pod.sh` | laptop | Copies run records, stage checkpoints, final-test results, videos and logs from the pod into `artifacts/k1_speed_ramp/pod/`. Read-only on the pod |

Shared modules, imported by the scripts above. Each value or rule lives in exactly one of them:

| Module | What it holds |
|---|---|
| `k1_conventions.py` | K1 joint tables, `k1_walk`'s observation and action conventions, the actor's shape, damping profiles |
| `criteria.py` | The pass/fail rule for MuJoCo trials, used by both the training gate and the final test |
| `curriculum.py` | The curriculum's decisions: when the gate is due, promotion, the stopping rule, rebuilding state on resume |
| `run_state.py` | File hashes, code provenance, finding a run's latest checkpoint, the arguments for resuming |
| `booster_player.py` | Booster's `k1_walk` MuJoCo player, set up the same way for every script that plays a policy |
| `k1_symmetry.py` | Left-right mirror of observations and actions, for symmetry-augmented PPO |
| `speed_metrics.py` | Heading speed and EPTE-SP |

Only `k1_speed_env.py`, `train_v3.py` and `check_parity.py` need Isaac Lab. Everything that decides a result is in the shared modules, which the unit tests cover on a laptop.

Scripts that use Booster's MuJoCo player run from `booster_deploy`'s folder, in its environment: on a laptop, `booster_deploy/.venv` (see the [top-level README](../../README.md)); on the pod, `/workspace/venv_rec`.

## Train (GPU host)

Isaac Sim needs an NVIDIA RTX GPU, so training runs on a rented GPU (we used RunPod, one RTX PRO 6000, $2.09/hr). [Docs/runbooks/golden_env.md](../../Docs/runbooks/golden_env.md) describes the pod settings and the install problems the setup script works around.

```bash
# Runs on: GPU host
git clone https://github.com/WillForEternity/bronco-booster-lab.git /workspace/bronco-booster-lab
cd /workspace/bronco-booster-lab
bash scripts/launch/runpod/setup_golden_env.sh 2>&1 | tee /workspace/setup.log   # Isaac Sim, Isaac Lab, Booster repos (~4 min)
bash scripts/launch/runpod/setup_recorder_env.sh                                 # MuJoCo player env for the gate

export OMNI_KIT_ACCEPT_EULA=YES TERM=xterm-256color PYTHONUNBUFFERED=1
source /workspace/venv/bin/activate
cd demo/speed_ramp
python check_parity.py                          # ends with "PARITY passed"; don't train until it does
python train_v3.py --seed 1 --run_name seed1    # one seed in the foreground, or:
bash launch_v3.sh                               # seeds 1 and 2 and the recorder, in the background
```

`launch_v3.sh` writes logs and process IDs to `/workspace/logs/k1_run_v3/` (`tail -f /workspace/logs/k1_run_v3/train_seed1.log`), and refuses to start while training is already running.

Runs are written to `/workspace/runs/k1_run_v3/<date>_<run_name>/`:
- `stages/stage_NN_vmax<v>_it<iter>.pt`: one checkpoint per promoted speed stage (stage 00 is `k1_walk` unchanged);
- `model_<iter>.pt` every 250 iterations, and at the end;
- `ramp_events.jsonl`: every gate attempt and promotion, with its MuJoCo results, and why the run ended;
- `candidates/`: each MuJoCo gate candidate's trial results and log;
- `params/`: the exact task, PPO settings and arguments, and the code hashes of each launch.

A run ends by itself when a level goes 1,000 iterations without promotion (RECIPE.md, note 6). At about 950 iterations per hour per seed, our two seeds reached the 2.0 m/s stage after about 2,500 iterations and stopped at about 3,500. Stop the pod whenever it is idle: it bills while running.

If a run crashes, `python train_v3.py --resume /workspace/runs/k1_run_v3/<date>_seed1` continues it from its latest checkpoint, with the run's own settings. A run that ended by the stopping rule can't be resumed.

## Final test (after training stops)

```bash
# Runs on: GPU host
cd /workspace/upstream/booster_deploy
/workspace/venv_rec/bin/python /workspace/bronco-booster-lab/demo/speed_ramp/final_test.py \
    --runs /workspace/runs/k1_run_v3 --out /workspace/final_test_v3
```

It plays each promoted stage for 20 perturbed trials on seeds the training gate never used, and writes `summary.md` and `summary.json`. A stage passes with at least 18 of 20 trials surviving 10 s at ≥ 90% of its speed, the same rule as the training gate. Treat the highest passing stage as the result, not the fastest video.

## Play a policy (laptop)

Copy the pod's results to your laptop. Set `POD_HOST` and `POD_PORT` in `.env` first (see `.env.example`):

```bash
# Runs on: laptop, from this repository
demo/speed_ramp/sync_from_pod.sh
```

Export a stage checkpoint, then play it from `booster_deploy`'s folder:

```bash
# Runs on: laptop, from this repository
uv run --directory ../booster_deploy python "$PWD/demo/speed_ramp/export_stage.py" \
    "$PWD/artifacts/k1_speed_ramp/pod/runs/k1_run_v3/<run>/stages/<stage>.pt" --out_dir "$PWD/artifacts/k1_speed_ramp/export"
cd ../booster_deploy
uv run mjpython ../bronco-booster-lab/demo/speed_ramp/play_speed.py \
    --checkpoint ../bronco-booster-lab/artifacts/k1_speed_ramp/export/<stage>_policy.pt \
    --speed 2.0 --damping_profile v3     # Windows/Linux: python, not mjpython
```

Always pass `--damping_profile v3` for policies from `train_v3.py`: they were trained with two lower arm dampings (RECIPE.md, note 2). Keep `--speed` at or below the stage's trained v_max; faster commands are outside its training range and usually fall.

`record_stage.py` takes the same arguments plus `--out clip.mp4` and records off-screen instead. `eval_mujoco.py --trials 20` gives the survival rate.

## Results depend on the computer

Booster's MuJoCo player has no randomness: on one machine, the same policy, speed and versions give the same result every time. Across machines, stable policies match, but borderline ones can diverge as tiny floating-point differences grow over a 10-second rollout. For example, with identical files and versions (MuJoCo 3.15.0, PyTorch 2.14.1, NumPy 2.4.6):

| Policy, at 2.0 m/s | Pod (Linux, x86) | Mac (Apple chip) |
|---|---|---|
| v3 seed 2, stage 5 | 2.02 m/s, no stop | 2.03 m/s, no stop |
| v3 seed 1, stage 5 | 2.09 m/s, no stop | **stopped by the fall detector at 3.9 s** |

So judge a policy by its survival rate over perturbed trials (`eval_mujoco.py`, `final_test.py`), not by a single replay, and say which machine a clip was recorded on (each metrics JSON records `versions.platform`).
