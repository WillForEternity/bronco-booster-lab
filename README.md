# Bronco Booster Lab: K1 baseline

A starting point from Bronco Robotics (Santa Clara University) for working with Booster Robotics' K1 humanoid in simulation. It covers three things:

1. **Walk:** run Booster's own `k1_walk` policy on your laptop, in Booster's MuJoCo player.
2. **Fine-tune:** train `k1_walk` to go faster with reinforcement learning in Isaac Lab, on a rented GPU.
3. **Play it back:** watch your fine-tuned policy run on your laptop.

Everything here is simulation only. Nothing in this repository sends commands to a physical robot.

## 1. Set up your laptop

Install Git and [uv](https://docs.astral.sh/uv/), which manages Python versions and packages. Skip anything you already have, then open a new terminal.

```bash
# macOS
xcode-select --install
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```powershell
# Windows
winget install --id Git.Git -e
winget install --id astral-sh.uv -e
```

Then clone this repository and Booster's two repositories **side by side** in one folder, at the commits this baseline was tested with:

```bash
# Runs on: laptop
mkdir k1 && cd k1
git clone https://github.com/WillForEternity/bronco-booster-lab.git
git clone https://github.com/BoosterRobotics/booster_assets.git
git clone https://github.com/BoosterRobotics/booster_deploy.git
git -C booster_assets checkout 12a9751
git -C booster_deploy checkout 7bb1462

cd booster_deploy
uv venv --python 3.11
uv pip install -r ../bronco-booster-lab/requirements-laptop.txt
uv pip install -e ../booster_assets
```

```text
k1/
  bronco-booster-lab/   this repository
  booster_assets/       Booster's K1 robot model (MJCF, URDF, meshes)
  booster_deploy/       Booster's policy player, with the Python environment (.venv) everything runs in
```

Keep Booster's repositories next to this one, not inside it: inside it, `uv run` would pick up this repository's settings instead of `booster_deploy`'s environment.

We skip Booster's `evdev` requirement: it's Linux-only and only reads gamepads.

## 2. Walk K1

```bash
# Runs on: laptop, in booster_deploy/
uv run mjpython scripts/deploy.py --task k1_walk --mujoco    # macOS
uv run python scripts/deploy.py --task k1_walk --mujoco      # Windows and Linux
```

A MuJoCo window opens with K1 standing. In the terminal, type a velocity command, `vx vy yaw`, and press Enter. For example, `0.8 0 0` walks forward at 0.8 m/s, `0.5 0 0.5` turns, and `0 0 0` stops. Booster's player caps forward speed at the policy's rated 1.6 m/s.

To hold one speed without typing, use our player (`k1_walk` is the default policy):

```bash
uv run mjpython ../bronco-booster-lab/demo/speed_ramp/play_speed.py --speed 1.0
```

## 3. Fine-tune it to run

Training needs Isaac Sim, which needs an NVIDIA RTX GPU on Linux or Windows, so it runs on a rented GPU, not on a laptop. [demo/speed_ramp/README.md](demo/speed_ramp/README.md) has the commands, from a fresh GPU host to a trained policy. [demo/speed_ramp/RECIPE.md](demo/speed_ramp/RECIPE.md) explains the recipe, and [Docs/runbooks/golden_env.md](Docs/runbooks/golden_env.md) describes the GPU environment.

In short, `train_v3.py` starts from `k1_walk` and raises the target speed in 0.25 m/s stages. A stage is promoted only when the policy tracks it in Isaac Lab **and** survives at least 18 of 20 perturbed trials in Booster's MuJoCo player. Our runs (October 7–8, 2026) passed the 2.0 m/s stage in both seeds, with a flight phase, which makes it a run rather than a fast walk. The held-out final test runs after training stops.

## 4. Play a fine-tuned policy

```bash
# Runs on: laptop, in booster_deploy/
uv run mjpython ../bronco-booster-lab/demo/speed_ramp/play_speed.py \
    --checkpoint /absolute/path/to/policy.pt --speed 2.0 --damping_profile v3
```

Use `python` instead of `mjpython` on Windows and Linux. `--damping_profile v3` matters: fine-tuned policies were trained with it ([RECIPE.md](demo/speed_ramp/RECIPE.md), note 2). Checkpoints are not stored in Git. Train your own, or ask a club lead for ours. [demo/speed_ramp/README.md](demo/speed_ramp/README.md#play-a-policy-laptop) shows how to export a training checkpoint into a playable `policy.pt`.

## Working in this repository

- **Claude Code** reads [AGENTS.md](AGENTS.md) (through `CLAUDE.md`) for this repository's rules and commands. Install it with `curl -fsSL https://claude.ai/install.sh | bash` (macOS and Linux) or `irm https://claude.ai/install.ps1 | iex` (Windows PowerShell), then run `claude` in this folder. It needs a Claude account.
- **Lint and tests** run on a laptop, in a separate environment in this folder:

  ```bash
  uv venv --python 3.11
  uv pip install --python .venv/bin/python -r requirements-dev.txt
  .venv/bin/ruff check .
  .venv/bin/pytest
  .venv/bin/pre-commit install    # once: runs the checks at every commit
  ```

## Tested versions

| Component | Version |
|---|---|
| `booster_assets` | `12a9751` |
| `booster_deploy` | `7bb1462` |
| `booster_train` (GPU host) | `8bb5b53` |
| MuJoCo, PyTorch, NumPy (laptop) | 3.15.0, 2.14.1, 2.4.6 ([requirements-laptop.txt](requirements-laptop.txt)) |
| Isaac Sim, Isaac Lab, RSL-RL (GPU host) | 5.1.0, 2.3.2, 3.1.2 |

The laptop steps were tested from a clean install on macOS (Apple chip). The GPU steps ran on Linux (Ubuntu 22.04). The Windows commands have not been tested yet.

## Ground rules

- No code from this repository runs on a physical robot without a Booster engineer's approval.
- Booster's code, robot models and checkpoints are fetched from Booster's repositories at pinned commits. They are never committed here, and neither are trained checkpoints or videos.
- Label results honestly: simulation or hardware, which simulator, and which starting checkpoint.

## License

Code: Apache-2.0 ([LICENSE](LICENSE)). Booster's repositories and other upstream projects keep their own licenses.

This is an independent student project. It is not affiliated with or endorsed by Booster Robotics.
