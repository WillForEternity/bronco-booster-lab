#!/usr/bin/env bash
# Copy training results from the GPU pod to this laptop: run records, stage checkpoints, final-test results,
# videos, logs, and the code as it is on the pod. Runs on: laptop, from anywhere.
# Read-only on the pod, and safe to re-run: it only adds or updates local files, never deletes.
#   demo/speed_ramp/sync_from_pod.sh            # everything except periodic checkpoints
#   demo/speed_ramp/sync_from_pod.sh --all      # also every periodic training checkpoint (model_*.pt)
#
# The pod's address comes from the environment or the git-ignored .env (see .env.example): POD_HOST, POD_PORT,
# and optionally POD_KEY (default ~/.ssh/runpod_ed25519) and POD_CODE. Local copy, mirroring the pod's folders:
#   artifacts/k1_speed_ramp/pod/{runs/<family>/<run>, final_test_v3, videos, logs, code_on_pod}
set -euo pipefail

case "${1:-}" in
  "") ALL=0 ;;
  --all) ALL=1 ;;
  *) echo "usage: $0 [--all]" >&2; exit 2 ;;
esac

REPO=$(cd "$(dirname "$0")/../.." && pwd)
env_value() { [[ -f "$REPO/.env" ]] || return 0; sed -n "s/^$1=//p" "$REPO/.env" | tail -n 1; }
POD_HOST=${POD_HOST:-$(env_value POD_HOST)}
POD_PORT=${POD_PORT:-$(env_value POD_PORT)}
: "${POD_HOST:?set POD_HOST in .env, e.g. root@<pod-ip>}" "${POD_PORT:?set POD_PORT in .env (the SSH port of the pod)}"
POD_KEY=${POD_KEY:-$(env_value POD_KEY)}
POD_KEY=${POD_KEY:-$HOME/.ssh/runpod_ed25519}
POD_CODE=${POD_CODE:-$(env_value POD_CODE)}
POD_CODE=${POD_CODE:-/workspace/bronco-booster-lab/demo/speed_ramp}
DEST="$REPO/artifacts/k1_speed_ramp/pod"  # artifacts/ is git-ignored

SSH=(ssh -i "$POD_KEY" -p "$POD_PORT" -o ConnectTimeout=20)
# rsync takes the remote shell as one string; single quotes keep a key path with spaces intact.
RSYNC=(rsync -a -e "ssh -i '$POD_KEY' -p '$POD_PORT' -o ConnectTimeout=20")
pod() { "${SSH[@]}" "$POD_HOST" "$@"; }
pull() { "${RSYNC[@]}" "${@:3}" "$POD_HOST:$1" "$2"; }  # pull <pod dir/> <local dir/> [rsync options]

mkdir -p "$DEST"

# Run records: params, ramp_events.jsonl, TensorBoard events, stage checkpoints, gate candidates' results.
RUN_FILTER=(--include='*/' --include='params/***' --include='stages/***' --include='ramp_events.jsonl'
            --include='events.out.tfevents.*' --include='candidates/*.json' --include='candidates/*.log' --exclude='*')
(( ALL )) && RUN_FILTER=()
if pod 'test -d /workspace/runs'; then
  pull /workspace/runs/ "$DEST/runs/" ${RUN_FILTER[@]+"${RUN_FILTER[@]}"}  # empty-array-safe on macOS's bash 3.2
fi

# Final-test results (without the exported policies), videos with their metrics, logs.
pod 'test -d /workspace/final_test_v3' && pull /workspace/final_test_v3/ "$DEST/final_test_v3/" --exclude='policies/'
pod 'test -d /workspace/videos' && pull /workspace/videos/ "$DEST/videos/"
pod 'test -d /workspace/logs' && pull /workspace/logs/ "$DEST/logs/"

# The code that produced them, as it is on the pod (no caches).
pod "test -d '$POD_CODE'" && pull "$POD_CODE/" "$DEST/code_on_pod/" --exclude='__pycache__'

echo "Synced to $DEST"
echo "  stage checkpoints: $(find "$DEST/runs" -path '*/stages/stage_*.pt' 2>/dev/null | wc -l | tr -d ' ')"
echo "  videos:            $(find "$DEST/videos" -name '*.mp4' 2>/dev/null | wc -l | tr -d ' ')"
echo "  compare the pod's code with yours: diff -rq '$DEST/code_on_pod' '$REPO/demo/speed_ramp'"
