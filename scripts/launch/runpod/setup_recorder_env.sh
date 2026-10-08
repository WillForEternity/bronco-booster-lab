#!/usr/bin/env bash
# MuJoCo recorder environment: booster_deploy's MuJoCo player, used by the training gate (train_v3.py),
# record_watch.py, eval_mujoco.py and final_test.py. Rebuilds what /workspace/venv_rec held on 2026-10-07.
# Runs on: GPU host, after setup_golden_env.sh (which fetches the pinned Booster repositories).
# Usage: bash setup_recorder_env.sh [target]      # default target: /workspace/venv_rec
#
# It never touches an existing environment: a running training job calls /workspace/venv_rec for every
# MuJoCo gate check, so changing it mid-run would change the gate. Build a new one elsewhere instead.
set -euo pipefail

TARGET=${1:-/workspace/venv_rec}
WS=/workspace
HERE=$(cd "$(dirname "$0")" && pwd)

if [ -e "$TARGET" ]; then
  echo "$TARGET already exists; refusing to modify it. Pass a new path, e.g. /workspace/venv_rec2." >&2
  exit 1
fi
[ -d "$WS/upstream/booster_assets" ] && [ -d "$WS/upstream/booster_deploy" ] \
  || { echo "Run setup_golden_env.sh first (needs $WS/upstream/booster_assets and booster_deploy)." >&2; exit 1; }

command -v uv >/dev/null || python -m pip install -q uv
uv venv --python 3.11 "$TARGET"
# torch==2.14.1+cpu comes from PyTorch's CPU index; everything else from PyPI.
uv pip install --python "$TARGET/bin/python" -q -r "$HERE/recorder-requirements.txt" \
  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
uv pip install --python "$TARGET/bin/python" -q -e "$WS/upstream/booster_assets"

# booster_deploy is not installed: its scripts run from its repository root, which puts it on sys.path.
(cd "$WS/upstream/booster_deploy" && MUJOCO_GL=egl "$TARGET/bin/python" -c "
import sys, pkgutil; sys.path.append('.')
import tasks, mujoco, torch
for m in pkgutil.walk_packages(tasks.__path__, prefix='tasks.'): __import__(m.name)
from booster_deploy.utils.registry import get_task
get_task('k1_walk')
print('recorder env ok: mujoco', mujoco.__version__, 'torch', torch.__version__)
")
