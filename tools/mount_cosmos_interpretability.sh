#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mount_dir="${repo_root}/s3/cosmos-interpretability"
cache_dir="${repo_root}/.rclone-cache/cosmos-interpretability"
log_dir="${repo_root}/.rclone-logs"
log_file="${log_dir}/cosmos-interpretability.log"
rclone_bin="${repo_root}/.rclone-bin/rclone"

if [[ ! -x "${rclone_bin}" ]]; then
  rclone_bin="$(command -v rclone)"
fi

mkdir -p "${mount_dir}" "${cache_dir}" "${log_dir}"

if mount | grep -F " on ${mount_dir} " >/dev/null 2>&1; then
  echo "Already mounted: ${mount_dir}"
  exit 0
fi

aws_exports="$(aws configure export-credentials --format env)"
eval "${aws_exports}"
unset aws_exports

exec "${rclone_bin}" mount cosmos-interpretability:cosmos-interpretability "${mount_dir}" \
  --read-only \
  --vfs-cache-mode full \
  --vfs-cache-max-size 20G \
  --vfs-cache-max-age 24h \
  --cache-dir "${cache_dir}" \
  --dir-cache-time 30m \
  --poll-interval 0 \
  --no-modtime \
  --no-checksum \
  --vfs-fast-fingerprint \
  --vfs-read-chunk-size 16M \
  --vfs-read-chunk-size-limit 256M \
  --volname cosmos-interpretability \
  --daemon \
  --daemon-wait 10s \
  --log-file "${log_file}" \
  --log-level INFO
