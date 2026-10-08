#!/usr/bin/env bash
# Copy the speed-ramp videos, metrics, checkpoints, run records and code from the GPU pod to this laptop.
# Runs on: laptop. Read-only on the pod, and safe to re-run: it only adds or updates local files, never deletes.
#   demo/speed_ramp/sync_from_pod.sh            # videos + metrics + run records + stage checkpoints
#   demo/speed_ramp/sync_from_pod.sh --all      # also every periodic training checkpoint (model_*.pt)
#
# Every run family under the pod's /workspace/runs/ is copied. Local folders (kept as earlier syncs made them):
#   k1_speed_ramp (version 1) -> runs/      k1_run_v3 -> runs_v3/      any other family <name> -> runs_<name>/
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
# The pod's address comes from the environment, or else from the git-ignored .env (see .env.example).
env_value() { [[ -f "$REPO/.env" ]] || return 0; sed -n "s/^$1=//p" "$REPO/.env" | tail -n 1; }
POD_HOST=${POD_HOST:-$(env_value POD_HOST)}
POD_PORT=${POD_PORT:-$(env_value POD_PORT)}
: "${POD_HOST:?set POD_HOST in .env, e.g. root@<pod-ip>}" "${POD_PORT:?set POD_PORT in .env (the SSH port of the pod)}"
POD_KEY=${POD_KEY:-$HOME/.ssh/runpod_ed25519}
POD_CODE=${POD_CODE:-$(env_value POD_CODE)}
POD_CODE=${POD_CODE:-/workspace/bronco-booster-lab/demo/speed_ramp}   # where this folder lives on the pod
DEST="$REPO/artifacts/k1_speed_ramp/pod"   # artifacts/ is git-ignored
SSH=(ssh -i "$POD_KEY" -p "$POD_PORT" -o ConnectTimeout=20)
RSYNC=(rsync -a -e "${SSH[*]}")

mkdir -p "$DEST"
# Videos, per-video metrics, policies, index.jsonl, summary.csv, recorder logs (if record_watch.py ran).
if "${SSH[@]}" "$POD_HOST" 'test -d /workspace/videos'; then
  "${RSYNC[@]}" "$POD_HOST:/workspace/videos/" "$DEST/videos/"
fi

# Run records: params, ramp_events.jsonl, TensorBoard events, stage checkpoints, gate candidates' results.
FILTER=(--include='*/' --include='params/***' --include='stages/***' --include='ramp_events.jsonl'
        --include='events.out.tfevents.*' --include='git/***' --exclude='*')
[[ "${1:-}" == "--all" ]] && FILTER=()
for family in $("${SSH[@]}" "$POD_HOST" 'ls /workspace/runs'); do
  case "$family" in
    k1_speed_ramp) local_dir="$DEST/runs" ;;
    k1_run_v3) local_dir="$DEST/runs_v3" ;;
    *) local_dir="$DEST/runs_$family" ;;
  esac
  "${RSYNC[@]}" "${FILTER[@]}" "$POD_HOST:/workspace/runs/$family/" "$local_dir/"
done

# Final-test results, once final_test.py has run on the pod.
if "${SSH[@]}" "$POD_HOST" 'test -d /workspace/final_test_v3'; then
  "${RSYNC[@]}" --exclude='policies/' "$POD_HOST:/workspace/final_test_v3/" "$DEST/final_test_v3/"
fi

# The code that produced them, as it is on the pod (scripts and logs; no caches).
if "${SSH[@]}" "$POD_HOST" "test -d '$POD_CODE'"; then
  "${RSYNC[@]}" --exclude='__pycache__' "$POD_HOST:$POD_CODE/" "$DEST/code_on_pod/"
fi

echo "Synced to $DEST"
echo "  videos:  $(find "$DEST" -path '*/videos/*.mp4' | wc -l | tr -d ' ') mp4 files"
echo "  stages:  $(find "$DEST" -path '*/stages/stage_*.pt' | wc -l | tr -d ' ') stage checkpoints"
echo "  code:    compare with demo/speed_ramp/ using: diff -rq $DEST/code_on_pod demo/speed_ramp"
