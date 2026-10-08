#!/usr/bin/env bash
# GPU training environment: Isaac Sim 5.1.0 (pip) + Isaac Lab 2.3.2 + pinned Booster repositories.
# Runs on: GPU host (RunPod pod, image runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04,
# env NVIDIA_DRIVER_CAPABILITIES=all, persistent volume at /workspace).
# Usage: bash setup_golden_env.sh 2>&1 | tee /workspace/setup.log
set -euo pipefail

WS=/workspace
ISAACLAB_TAG=v2.3.2
BOOSTER_ASSETS_COMMIT=12a97516f57469078707afb587a5f84c1cb110d1
BOOSTER_TRAIN_COMMIT=8bb5b53824e0fbcfe69a5cb78a81f4bddb3e2a6e
BOOSTER_DEPLOY_COMMIT=7bb1462e2ac742eba5c01659b815e2e99fbe617f

# Running this script accepts NVIDIA's Omniverse EULA; read it first.
export OMNI_KIT_ACCEPT_EULA=YES
# A TERM inherited over SSH (e.g. "ansi+tabs") breaks isaaclab.sh's terminal calls.
export TERM=xterm-256color

step() { echo; echo "=== $(date +%H:%M:%S) $*"; }
trap 'echo "SETUP FAILED at line $LINENO (exit $?)"' ERR

step "System packages (graphics libraries Isaac Sim needs)"
# *.ubuntu.com was unreachable from RunPod US-PA-1 on 2026-10-07; use a reachable mirror.
UBUNTU_MIRROR=${UBUNTU_MIRROR:-http://mirror.math.princeton.edu/pub/ubuntu}
sed -i -E "s#http://(archive|security)\.ubuntu\.com/ubuntu/?#${UBUNTU_MIRROR}/#g" /etc/apt/sources.list
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  cmake build-essential git libglu1-mesa libxt6 libxrandr2 libxi6 libxinerama1 \
  libxcursor1 libegl1 libvulkan1 vulkan-tools > /dev/null

step "Vulkan driver registration"
# The container toolkit mounts NVIDIA's GL/Vulkan libraries only when the graphics
# capability is enabled; register the ICD if the image does not.
if [ ! -e /etc/vulkan/icd.d/nvidia_icd.json ] && [ ! -e /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
  mkdir -p /etc/vulkan/icd.d
  cat > /etc/vulkan/icd.d/nvidia_icd.json <<'JSON'
{ "file_format_version": "1.0.0",
  "ICD": { "library_path": "libGLX_nvidia.so.0", "api_version": "1.3" } }
JSON
fi
vulkaninfo --summary 2>&1 | grep -E "deviceName|driverVersion|apiVersion" || echo "WARNING: vulkaninfo found no device"

step "Python 3.11 environment"
python -m pip install -q uv
[ -x "$WS/venv/bin/python" ] || uv venv --seed --python 3.11 "$WS/venv"
# shellcheck disable=SC1091
source "$WS/venv/bin/activate"

step "PyTorch 2.7.0 (CUDA 12.8 build; required for Blackwell GPUs)"
uv pip install -q torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

step "Isaac Sim 5.1.0"
uv pip install -q "isaacsim[all,extscache]==5.1.0" \
  --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match

step "Isaac Lab ${ISAACLAB_TAG}"
[ -d "$WS/IsaacLab" ] || git clone -q -c advice.detachedHead=false --depth 1 -b "$ISAACLAB_TAG" https://github.com/isaac-sim/IsaacLab.git "$WS/IsaacLab"
# flatdict 4.0.1 (an isaaclab dependency) imports pkg_resources at build time, which
# setuptools >= 81 removed; build it first against an older setuptools.
uv pip install -q "setuptools<81" wheel
uv pip install -q --no-build-isolation flatdict==4.0.1
(cd "$WS/IsaacLab" && ./isaaclab.sh --install rsl_rl)
python -c "import isaaclab, isaaclab_rl, isaaclab_tasks" || { echo "Isaac Lab import check failed"; exit 1; }

step "Booster repositories at pinned commits"
mkdir -p "$WS/upstream"
fetch() {  # fetch <repo> <commit>
  local dir="$WS/upstream/$1"
  [ -d "$dir" ] || git clone -q "https://github.com/BoosterRobotics/$1.git" "$dir"
  git -C "$dir" checkout -q "$2"
  echo "$1 @ $(git -C "$dir" log -1 --format='%h %cd')"
}
fetch booster_assets "$BOOSTER_ASSETS_COMMIT"
fetch booster_train "$BOOSTER_TRAIN_COMMIT"
fetch booster_deploy "$BOOSTER_DEPLOY_COMMIT"
python -m pip install -q -e "$WS/upstream/booster_assets"
python -m pip install -q -e "$WS/upstream/booster_train/source/booster_train"

step "Versions"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu ok', torch.cuda.is_available())"
pip show isaacsim isaaclab 2>/dev/null | grep -E "^(Name|Version)"
df -h "$WS" | tail -1

echo; echo "Setup finished. Next: first Isaac Lab launch (scripts in Docs/runbooks/golden_env.md)."
