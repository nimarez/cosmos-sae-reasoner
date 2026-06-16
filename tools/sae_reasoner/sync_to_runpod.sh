#!/usr/bin/env bash
set -uo pipefail

REMOTE_HOST="${COSMOS_SAE_RUNPOD_HOST:-root@64.119.209.250}"
REMOTE_PORT="${COSMOS_SAE_RUNPOD_PORT:-11792}"
REMOTE_DIR="${COSMOS_SAE_RUNPOD_DIR:-/workspace/cosmos-sae-reasoner/tools/sae_reasoner/}"
SSH_KEY="${COSMOS_SAE_RUNPOD_SSH_KEY:-$HOME/.runpod/ssh/runpodctl-ssh-key}"
INTERVAL_SECONDS="${COSMOS_SAE_SYNC_INTERVAL:-3}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="$ROOT/tools/sae_reasoner/"

sync_once() {
  rsync -az --delete \
    --exclude '__pycache__/' \
    --exclude '.ipynb_checkpoints/' \
    --exclude '.cache/' \
    --exclude 'outputs/' \
    -e "ssh -i $SSH_KEY -p $REMOTE_PORT" \
    "$SRC" "$REMOTE_HOST:$REMOTE_DIR"
  echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') synced" >&2
}

echo "syncing $SRC -> $REMOTE_HOST:$REMOTE_DIR every ${INTERVAL_SECONDS}s" >&2
sync_once

if command -v fswatch >/dev/null 2>&1; then
  fswatch -o "$SRC" | while read -r _event_count; do
    if ! sync_once; then
      echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') sync failed; retrying" >&2
    fi
  done
else
  last_signature=""
  while true; do
    signature="$(find "$SRC" -type f \
      ! -path '*/__pycache__/*' \
      ! -path '*/.ipynb_checkpoints/*' \
      ! -path '*/.cache/*' \
      ! -path '*/outputs/*' \
      -print0 | xargs -0 stat -f '%m %z %N' 2>/dev/null | shasum | awk '{print $1}' || true)"
    if [[ "$signature" != "$last_signature" ]]; then
      if sync_once; then
        last_signature="$signature"
      else
        echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') sync failed; retrying" >&2
      fi
    fi
    sleep "$INTERVAL_SECONDS"
  done
fi
