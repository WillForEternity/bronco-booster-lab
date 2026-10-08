#!/usr/bin/env bash
# Launch version-3 training (one process per seed) and the video recorder in the background, detached from the
# terminal so they survive a dropped SSH session. Runs on: GPU host, after both setup scripts.
#   bash demo/speed_ramp/launch_v3.sh                 # seeds 1 and 2
#   SEEDS="1 2 3" bash demo/speed_ramp/launch_v3.sh
# Logs and process IDs: $LOGS (default /workspace/logs/k1_run_v3). Follow with: tail -f $LOGS/train_seed1.log
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
SEEDS=${SEEDS:-"1 2"}
TRAIN_PY=${TRAIN_PY:-/workspace/venv/bin/python}
REC_PY=${REC_PY:-/workspace/venv_rec/bin/python}
RUNS=${RUNS:-/workspace/runs/k1_run_v3}
VIDEOS=${VIDEOS:-/workspace/videos/v3}
LOGS=${LOGS:-/workspace/logs/k1_run_v3}

for py in "$TRAIN_PY" "$REC_PY"; do
  [[ -x "$py" ]] || { echo "$py not found: run scripts/launch/runpod/setup_golden_env.sh and setup_recorder_env.sh first." >&2; exit 1; }
done
if pgrep -f "train_v3.py" > /dev/null || pgrep -f "record_watch.py" > /dev/null; then
  echo "train_v3.py or record_watch.py is already running (pgrep -af 'train_v3|record_watch'); not starting another." >&2
  exit 1
fi
mkdir -p "$LOGS" "$VIDEOS"

for seed in $SEEDS; do
  setsid nohup env TERM=xterm-256color OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
    "$TRAIN_PY" "$HERE/train_v3.py" --seed "$seed" --run_name "seed$seed" --log_root "$RUNS" --rec_python "$REC_PY" \
    > "$LOGS/train_seed$seed.log" 2>&1 < /dev/null &
  echo $! > "$LOGS/train_seed$seed.pid"
  echo "seed $seed: pid $!, log $LOGS/train_seed$seed.log"
  sleep 3  # stagger Isaac Sim start-ups
done

setsid nohup env PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore \
  "$REC_PY" "$HERE/record_watch.py" --runs "$RUNS" --videos "$VIDEOS" --trials 20 --damping_profile v3 \
  > "$LOGS/recorder.log" 2>&1 < /dev/null &
echo $! > "$LOGS/recorder.pid"
echo "recorder: pid $!, log $LOGS/recorder.log"
