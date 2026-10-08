# AGENTS.md

Instructions for AI coding agents (and humans) working in this repository.

## What this is

A baseline for Booster Robotics' K1 humanoid in simulation: Booster's `k1_walk` policy on a laptop (MuJoCo), a fine-tune of it for speed (Isaac Lab, GPU host), and playback of the result. See [README.md](README.md).

## Layout

The repository sits next to Booster's repositories, which are never copied into it:

```text
k1/
  bronco-booster-lab/   this repository
  booster_assets/       Booster's K1 model, pinned at 12a9751
  booster_deploy/       Booster's MuJoCo player and the laptop .venv, pinned at 7bb1462
```

| Path | What it is |
|---|---|
| `demo/speed_ramp/` | The fine-tuning pipeline; its `README.md` lists each file and where it runs, `RECIPE.md` explains the recipe |
| `scripts/launch/runpod/` | GPU host setup: `setup_golden_env.sh` (Isaac Sim, Isaac Lab, Booster repos), `setup_recorder_env.sh` (MuJoCo player env) |
| `Docs/runbooks/golden_env.md` | The GPU environment as built and checked |
| `tests/demo/` | Unit tests that run on a laptop without Isaac Lab or `booster_deploy` |
| `artifacts/`, `runs/` | Git-ignored: checkpoints, videos, synced pod outputs |

## Rules

- **Hardware is off.** Nothing here may send commands to a physical robot.
- **Never commit** Booster's code, robot models, motions or checkpoints, trained checkpoints, videos, keys, or pod addresses. `.gitignore` and the pre-commit hooks block model, checkpoint and video files; pod addresses go in the git-ignored `.env`.
- **Every command says where it runs:** laptop, GPU host, or robot.
- **Scripts that use Booster's MuJoCo player run from `booster_deploy/`**, in its environment (`uv run ...` there; `/workspace/venv_rec` on the GPU host).
- **Policies from `train_v3.py` are played with `--damping_profile v3`**, and at or below their trained v_max (RECIPE.md, notes 2 and 4).
- **Judge a policy by perturbed trials** (`eval_mujoco.py`, `final_test.py`), not one replay: results can differ between computers.
- **Never edit code a running training job reads from disk:** `train_v3.py`'s MuJoCo gate re-runs `eval_mujoco.py` at every check.
- Do not invent commands, files, or results that don't exist.

## Commands

| Command | Runs on | What it does |
|---|---|---|
| `uv run mjpython scripts/deploy.py --task k1_walk --mujoco` (in `booster_deploy/`) | laptop | Booster's interactive K1 walking demo (`python` instead of `mjpython` off macOS) |
| `uv run mjpython ../bronco-booster-lab/demo/speed_ramp/play_speed.py --speed 1.0` (in `booster_deploy/`) | laptop | Plays a policy at a fixed speed; add `--checkpoint <abs path> --damping_profile v3` for a fine-tuned one |
| `.venv/bin/ruff check .` | laptop | Lint (settings in `pyproject.toml`) |
| `.venv/bin/pytest` | laptop | Unit tests |
| `demo/speed_ramp/sync_from_pod.sh` | laptop | Copies pod outputs into `artifacts/k1_speed_ramp/pod/`; read-only on the pod |
| `bash scripts/launch/runpod/setup_golden_env.sh` | GPU host | Builds the Isaac Sim / Isaac Lab training environment |
| `bash scripts/launch/runpod/setup_recorder_env.sh [path]` | GPU host | Builds the MuJoCo player environment; refuses to touch an existing one |
| `python train_v3.py --seed 1 --run_name seed1` (in `demo/speed_ramp/`) | GPU host | Trains |

The full training, evaluation and playback steps are in [demo/speed_ramp/README.md](demo/speed_ramp/README.md).
