# GPU environment: RunPod

**Status:** passed the first checkpoint on October 7, 2026 (Booster's `list_envs.py` launches Isaac Sim and lists all eight Booster tasks). Not frozen: there is no container digest or manifest yet.

**Runs on:** GPU host.

## What was used

| Item | Value |
|---|---|
| Provider | RunPod, Secure Cloud, data center US-PA-1, $2.09/hr |
| GPU | 1× NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB), driver 595.91.07 |
| Pod image | `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04` (Ubuntu 22.04.5, Python 3.11.11) |
| Pod settings | 100 GB container disk, 200 GB persistent volume at `/workspace`, ports `22/tcp` and `6006/http`, env `NVIDIA_DRIVER_CAPABILITIES=all`, SSH key registered on the RunPod account |
| Python env | `/workspace/venv` (uv, Python 3.11) |
| PyTorch | 2.7.0+cu128 |
| Isaac Sim | 5.1.0.0 (pip, `isaacsim[all,extscache]`) |
| Isaac Lab | v2.3.2 (`isaaclab` 0.54.2), installed with `./isaaclab.sh --install rsl_rl`; `rsl-rl-lib` 3.1.2 |
| Booster repositories | `booster_assets` @ `12a9751`, `booster_train` @ `8bb5b53`, `booster_deploy` @ `7bb1462`, in `/workspace/upstream/` |

The install is scripted in [`scripts/launch/runpod/setup_golden_env.sh`](../../scripts/launch/runpod/setup_golden_env.sh). On a fresh pod it took about 4 minutes; Isaac Sim's 17 GB download took about 80 seconds.

Why pip and not NVIDIA's Isaac Sim container or Isaac Lab's Docker workflow? A RunPod pod is already a container, so it can't run Docker inside it, and NVIDIA's image doesn't start an SSH server. The pip install on RunPod's PyTorch image is the route that passed.

## Problems hit, and the fixes now in the script

1. **Image download stalled.** A recent image (`runpod/pytorch:1.4.0-cu1281-torch280-ubuntu2204`) that the host hadn't cached sat at 0% for ten minutes. RunPod's widely used template image was already cached and started in under a second. Prefer cached template images.
2. **`*.ubuntu.com` was unreachable from US-PA-1**, so `apt-get` hung. The script switches apt to `UBUNTU_MIRROR` (default: Princeton's mirror).
3. **The Vulkan driver file is mounted read-only** at `/etc/vulkan/icd.d/nvidia_icd.json`. Only write one if neither standard location has it.
4. **Isaac Lab's installer fails on an unknown `TERM`** inherited over SSH (`ansi+tabs`). The script sets `TERM=xterm-256color`.
5. **`flatdict==4.0.1` fails to build** with setuptools ≥ 81 (`pkg_resources` was removed). The installer kept going, then the core `isaaclab` package was silently missing. The script now builds `flatdict` first against `setuptools<81`, then checks that `isaaclab` imports.
6. **Output printed just before `simulation_app.close()` can be lost** when stdout is redirected, because Isaac Sim exits abruptly. Run Isaac scripts with `PYTHONUNBUFFERED=1`.

## Checks that passed

```text
vulkaninfo: deviceName = NVIDIA RTX PRO 6000 Blackwell Server Edition
torch 2.7.0+cu128, cuda 12.8, torch.cuda.is_available() = True
Isaac Sim startup: Graphics API Vulkan, driver 595.91.07, RTX PRO 6000 active
list_envs.py: Booster-K1-MJ_Dance_002-v0 (+ -Play), Booster-T1-MotionTracking-v0 (+ -Play),
              Booster-T2-MotionTracking-v0 (+ -Play), Booster-T2-MotionTracking-Rough-v0 (+ -Play)
```

## Using the environment

```bash
# Runs on: GPU host
export OMNI_KIT_ACCEPT_EULA=YES TERM=xterm-256color PYTHONUNBUFFERED=1
source /workspace/venv/bin/activate
cd /workspace/upstream/booster_train
python scripts/list_envs.py
```

Setting `OMNI_KIT_ACCEPT_EULA=YES` accepts the NVIDIA Omniverse license on your behalf, so read it before your first launch.

## The MuJoCo recorder environment

`booster_deploy`'s MuJoCo player runs in a second, CPU-only environment, `/workspace/venv_rec`: MuJoCo 3.15.0, PyTorch 2.14.1+cpu and NumPy 2.4.6. It was first built by hand on October 7. [`scripts/launch/runpod/setup_recorder_env.sh`](../../scripts/launch/runpod/setup_recorder_env.sh) rebuilds it from the package list read off the pod ([`recorder-requirements.txt`](../../scripts/launch/runpod/recorder-requirements.txt)). The script resolves on Linux but has not been run on a fresh pod yet. It refuses to modify an existing environment, because a running `train_v3.py` uses `/workspace/venv_rec` for every MuJoCo gate check. A laptop's `booster_deploy/.venv`, set up as in the README, has the same versions of these three packages.

Stop the pod whenever it is idle: it bills $2.09/hr while running. The `/workspace` volume survives a stop, but not a terminate.
