#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

POD_ID="${COSMOS_SAE_STREAMLIT_POD_ID:-}"
POD_NAME="${COSMOS_SAE_STREAMLIT_POD_NAME:-cosmos-reasoner-streamlit-3090}"
GPU_ID="${COSMOS_SAE_STREAMLIT_GPU_ID:-NVIDIA GeForce RTX 3090}"
DATA_CENTER_IDS="${COSMOS_SAE_STREAMLIT_DATA_CENTER_IDS:-EU-CZ-1}"
IMAGE="${COSMOS_SAE_STREAMLIT_IMAGE:-runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404}"
VOLUME_GB="${COSMOS_SAE_STREAMLIT_VOLUME_GB:-200}"
CONTAINER_DISK_GB="${COSMOS_SAE_STREAMLIT_CONTAINER_DISK_GB:-80}"
REMOTE_DIR="${COSMOS_SAE_STREAMLIT_REMOTE_DIR:-/workspace/cosmos-sae-reasoner}"
REMOTE_PORT="${COSMOS_SAE_STREAMLIT_PORT:-8501}"
SSH_KEY="${COSMOS_SAE_RUNPOD_SSH_KEY:-$HOME/.runpod/ssh/runpodctl-ssh-key}"
TERMINATE_AFTER="${COSMOS_SAE_STREAMLIT_TERMINATE_AFTER:-}"
SYNC_ENV_FILE="${COSMOS_SAE_STREAMLIT_ENV_FILE:-}"
MANIFEST="${COSMOS_SAE_STREAMLIT_MANIFEST:-outputs/sae_reasoner/manifests/physical_ai_instruct_10m_with_synhuman_20260616_2018.jsonl}"
MODEL_ID="${COSMOS_SAE_MODEL_ID:-nvidia/Cosmos3-Nano}"

if ! command -v runpodctl >/dev/null 2>&1; then
  echo "runpodctl is required. Install/configure it first with runpodctl doctor." >&2
  exit 2
fi
if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is required." >&2
  exit 2
fi
if [[ ! -f "$SSH_KEY" ]]; then
  echo "RunPod SSH key not found: $SSH_KEY" >&2
  exit 2
fi

if [[ -z "$TERMINATE_AFTER" ]]; then
  TERMINATE_AFTER="$(date -u -v+24H '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null || date -u -d '+24 hours' '+%Y-%m-%dT%H:%M:%SZ')"
fi

json_get() {
  python3 -c 'import json,sys; obj=json.load(sys.stdin); key=sys.argv[1]; print(obj.get(key, ""))' "$1"
}

pod_json() {
  runpodctl pod get "$1" -o json
}

wait_for_ssh() {
  local pod_id="$1"
  local json ip port
  for _ in $(seq 1 90); do
    json="$(pod_json "$pod_id")"
    if ! grep -q '"pod not ready"' <<<"$json"; then
      ip="$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("ssh", {}).get("ip", ""))' <<<"$json")"
      port="$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("ssh", {}).get("port", ""))' <<<"$json")"
      if [[ -n "$ip" && -n "$port" ]]; then
        printf '%s %s\n' "$ip" "$port"
        return 0
      fi
    fi
    sleep 10
  done
  echo "timed out waiting for SSH on pod $pod_id" >&2
  return 1
}

if [[ -n "$POD_ID" ]]; then
  status="$(pod_json "$POD_ID" | json_get desiredStatus)"
  if [[ "$status" != "RUNNING" ]]; then
    echo "starting existing pod $POD_ID" >&2
    runpodctl pod start "$POD_ID" -o json >/dev/null
  fi
else
  echo "creating $POD_NAME on $GPU_ID in $DATA_CENTER_IDS" >&2
  create_args=(
    pod create
    --name "$POD_NAME"
    --image "$IMAGE"
    --gpu-id "$GPU_ID"
    --gpu-count 1
    --volume-in-gb "$VOLUME_GB"
    --container-disk-in-gb "$CONTAINER_DISK_GB"
    --ports "22/tcp,${REMOTE_PORT}/http"
    --terminate-after "$TERMINATE_AFTER"
    -o json
  )
  if [[ -n "$DATA_CENTER_IDS" ]]; then
    create_args+=(--data-center-ids "$DATA_CENTER_IDS")
  fi
  created="$(runpodctl "${create_args[@]}")"
  POD_ID="$(json_get id <<<"$created")"
  if [[ -z "$POD_ID" ]]; then
    echo "runpodctl did not return a pod id" >&2
    echo "$created" >&2
    exit 1
  fi
fi

read -r POD_IP POD_SSH_PORT < <(wait_for_ssh "$POD_ID")
SSH=(ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p "$POD_SSH_PORT" "root@$POD_IP")
RSYNC_RSH="ssh -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $POD_SSH_PORT"

echo "pod $POD_ID SSH: ssh -i $SSH_KEY root@$POD_IP -p $POD_SSH_PORT" >&2

"${SSH[@]}" "mkdir -p '$REMOTE_DIR/outputs/sae_reasoner/manifests' '$REMOTE_DIR/outputs/sae_reasoner/app_runs'"

rsync -az --no-owner --no-group --no-perms --omit-dir-times \
  --exclude '__pycache__/' \
  --exclude '.ipynb_checkpoints/' \
  --exclude 'outputs/' \
  -e "$RSYNC_RSH" \
  tools pyproject.toml uv.lock "root@$POD_IP:$REMOTE_DIR/"

if [[ -f "$MANIFEST" ]]; then
  rsync -az --no-owner --no-group --no-perms --omit-dir-times \
    -e "$RSYNC_RSH" \
    "$MANIFEST" "root@$POD_IP:$REMOTE_DIR/outputs/sae_reasoner/manifests/"
else
  echo "manifest not found locally, skipping manifest sync: $MANIFEST" >&2
fi

if [[ -n "$SYNC_ENV_FILE" ]]; then
  if [[ ! -f "$SYNC_ENV_FILE" ]]; then
    echo "COSMOS_SAE_STREAMLIT_ENV_FILE does not exist: $SYNC_ENV_FILE" >&2
    exit 2
  fi
  echo "syncing env file to remote .env: $SYNC_ENV_FILE" >&2
  rsync -az --no-owner --no-group --no-perms --omit-dir-times \
    -e "$RSYNC_RSH" \
    "$SYNC_ENV_FILE" "root@$POD_IP:$REMOTE_DIR/.env"
fi

"${SSH[@]}" "cd '$REMOTE_DIR' && tmux kill-session -t setup_sae_streamlit 2>/dev/null || true"
"${SSH[@]}" "cd '$REMOTE_DIR' && tmux new-session -d -s setup_sae_streamlit 'PYTHON_VERSION=3.12 UV_TORCH_BACKEND=auto COSMOS_SAE_USE_SYSTEM_TORCH=1 bash tools/sae_reasoner/runpod_setup_uv.sh > outputs/sae_reasoner/app_runs/setup_streamlit.log 2>&1'"

echo "waiting for setup smoke check" >&2
for _ in $(seq 1 180); do
  if "${SSH[@]}" "cd '$REMOTE_DIR' && ! tmux has-session -t setup_sae_streamlit 2>/dev/null"; then
    break
  fi
  sleep 10
done
if "${SSH[@]}" "cd '$REMOTE_DIR' && tmux has-session -t setup_sae_streamlit 2>/dev/null"; then
  echo "setup is still running. Log: $REMOTE_DIR/outputs/sae_reasoner/app_runs/setup_streamlit.log" >&2
else
  "${SSH[@]}" "cd '$REMOTE_DIR' && tail -n 30 outputs/sae_reasoner/app_runs/setup_streamlit.log"
fi

if ! "${SSH[@]}" "cd '$REMOTE_DIR' && [ -x .venv/bin/streamlit ] && .venv/bin/python -m tools.sae_reasoner --help >/dev/null"; then
  echo "remote setup did not produce a usable Streamlit/SAE environment" >&2
  "${SSH[@]}" "cd '$REMOTE_DIR' && tail -n 80 outputs/sae_reasoner/app_runs/setup_streamlit.log" >&2 || true
  exit 1
fi

remote_manifest="$REMOTE_DIR/$MANIFEST"
start_cmd=$(cat <<EOF
set -euo pipefail
cd '$REMOTE_DIR'
set -a
[ -f .env ] && source .env
set +a
export HF_HOME=/workspace/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub
export COSMOS_SAE_MEDIA_CACHE=/workspace/.cache/sae_reasoner/media
export COSMOS_SAE_MODEL_ID='$MODEL_ID'
export COSMOS_SAE_MANIFEST='$remote_manifest'
.venv/bin/streamlit run tools/sae_reasoner/apps/reasoner_streamlit.py --server.address 0.0.0.0 --server.port '$REMOTE_PORT'
EOF
)

"${SSH[@]}" "cd '$REMOTE_DIR' && tmux kill-session -t reasoner_streamlit 2>/dev/null || true"
"${SSH[@]}" "cd '$REMOTE_DIR' && cat > /tmp/run_reasoner_streamlit.sh <<'SH'
$start_cmd
SH
chmod +x /tmp/run_reasoner_streamlit.sh
tmux new-session -d -s reasoner_streamlit 'bash /tmp/run_reasoner_streamlit.sh > outputs/sae_reasoner/app_runs/reasoner_streamlit.log 2>&1'
tmux ls"

echo
echo "pod_id=$POD_ID"
echo "ssh=ssh -i $SSH_KEY root@$POD_IP -p $POD_SSH_PORT"
echo "streamlit_url=http://$POD_IP:$REMOTE_PORT"
echo "remote_log=$REMOTE_DIR/outputs/sae_reasoner/app_runs/reasoner_streamlit.log"
