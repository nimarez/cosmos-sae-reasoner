from __future__ import annotations

import argparse
import os
import re
import time
from dataclasses import dataclass

from tools.sae_reasoner.storage import parse_s3_uri, s3_client


@dataclass(frozen=True)
class S3CollectionStats:
    shard_count: int
    total_bytes: int
    latest_shard_index: int | None


def collect_s3_stats(uri: str) -> S3CollectionStats:
    bucket, prefix = parse_s3_uri(uri)
    normalized_prefix = prefix.rstrip("/")
    if normalized_prefix:
        normalized_prefix += "/"
    client = s3_client()
    paginator = client.get_paginator("list_objects_v2")
    shard_count = 0
    total_bytes = 0
    latest_shard_index: int | None = None
    for page in paginator.paginate(Bucket=bucket, Prefix=normalized_prefix):
        for item in page.get("Contents", []):
            key = item.get("Key", "")
            if not key.endswith(".pt"):
                continue
            shard_count += 1
            total_bytes += int(item.get("Size") or 0)
            name = key.rsplit("/", 1)[-1]
            match = re.match(r"^(\d+)_", name)
            if match:
                index = int(match.group(1))
                latest_shard_index = index if latest_shard_index is None else max(latest_shard_index, index)
    return S3CollectionStats(
        shard_count=shard_count,
        total_bytes=total_bytes,
        latest_shard_index=latest_shard_index,
    )


def init_wandb(args: argparse.Namespace):
    project = args.wandb_project or os.environ.get("WANDB_PROJECT")
    mode = args.wandb_mode or os.environ.get("WANDB_MODE")
    if not project and mode != "offline":
        return None
    try:
        import wandb
    except Exception as exc:
        raise RuntimeError("W&B logging requested but wandb is not installed.") from exc
    tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
    return wandb.init(
        project=project or "cosmos-sae-reasoner",
        entity=args.wandb_entity or os.environ.get("WANDB_ENTITY"),
        name=args.wandb_run_name,
        tags=tags,
        job_type="collect_activations_monitor",
        mode=mode,
        config={
            "activation_dir": args.activation_dir,
            "interval_seconds": args.interval_seconds,
            "target_examples": args.target_examples,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monitor an S3 activation collection prefix.")
    parser.add_argument("--activation-dir", required=True, help="S3 prefix containing activation .pt shards.")
    parser.add_argument("--target-examples", type=int, default=None)
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-tags", default="")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    run = init_wandb(args)
    start = time.time()
    try:
        while True:
            stats = collect_s3_stats(args.activation_dir)
            elapsed = max(1e-9, time.time() - start)
            metric = {
                "collected_examples": stats.shard_count,
                "latest_shard_index": stats.latest_shard_index if stats.latest_shard_index is not None else -1,
                "total_activation_gb": stats.total_bytes / 1_000_000_000,
                "elapsed_seconds": elapsed,
                "examples_per_min": 60.0 * stats.shard_count / elapsed,
            }
            if args.target_examples:
                metric["target_examples"] = args.target_examples
                metric["progress_frac"] = stats.shard_count / args.target_examples
            print(metric, flush=True)
            if run is not None:
                run.log(metric, step=stats.shard_count)
            if args.once:
                break
            time.sleep(args.interval_seconds)
    finally:
        if run is not None:
            run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
