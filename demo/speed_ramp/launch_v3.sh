#!/usr/bin/env bash
# Launch version-3 training (two seeds) and its recorder on the pod, in the background. Runs on: GPU host.
# Logs: train_v3_seed<N>.log next to this script, and /workspace/videos/v3/recorder.log.
cd "$(dirname "$0")"
for sd in 1 2; do
  setsid nohup env TERM=xterm-256color OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
    /workspace/venv/bin/python train_v3.py --seed $sd --run_name seed$sd --num_envs 4096 \
    > train_v3_seed$sd.log 2>&1 < /dev/null &
  sleep 3
done
mkdir -p /workspace/videos/v3
setsid nohup env PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore /workspace/venv_rec/bin/python record_watch.py \
  --runs /workspace/runs/k1_run_v3 --videos /workspace/videos/v3 --trials 20 --damping_profile v3 \
  > /workspace/videos/v3/recorder.log 2>&1 < /dev/null &
