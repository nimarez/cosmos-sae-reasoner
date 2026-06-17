from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

from tools.sae_reasoner.storage import s3_client


def main() -> int:
    parser = argparse.ArgumentParser(description="Wait for activation collection to finish, then launch a planned SAE sweep.")
    parser.add_argument("--activation-dir", required=True, help="Activation S3 prefix.")
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--collect-log", type=Path, required=True)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--stage", default="lr_batch")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wait-seconds", type=int, default=120)
    parser.add_argument("--python", default=".venv/bin/python")
    args = parser.parse_args()

    print(json.dumps({"event": "wait_start", "expected_count": args.expected_count, "activation_dir": args.activation_dir}), flush=True)
    while True:
        counts = count_activation_objects(args.activation_dir)
        print(json.dumps({"event": "wait_status", **counts}), flush=True)
        if is_complete(counts, args.expected_count, args.collect_log):
            break
        time.sleep(max(1, args.wait_seconds))

    print(json.dumps({"event": "sweep_plan_start", "stage": args.stage, "sweep_root": str(args.sweep_root)}), flush=True)
    plan_cmd = [
        args.python,
        "-m",
        "tools.sae_reasoner.scripts.plan_training_sweep",
        "--activation-dir",
        args.activation_dir,
        "--output-root",
        str(args.sweep_root),
        "--stage",
        args.stage,
    ]
    if args.wandb_project:
        plan_cmd.extend(["--wandb-project", args.wandb_project])
    subprocess.run(plan_cmd, check=True)

    script = args.sweep_root / f"run_{args.stage}.sh"
    print(json.dumps({"event": "sweep_run_start", "script": str(script)}), flush=True)
    subprocess.run(["bash", str(script)], check=True)
    print(json.dumps({"event": "sweep_run_complete", "script": str(script)}), flush=True)
    return 0


def is_complete(counts: dict[str, int], expected_count: int, collect_log: Path) -> bool:
    enough_objects = counts["pt"] >= expected_count and counts["sidecars"] >= expected_count
    return enough_objects and (log_has_collect_complete(collect_log) or not collect_log.exists())


def log_has_collect_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        for line in reversed(path.read_text(errors="ignore").splitlines()[-200:]):
            if '"event": "collect_complete"' in line or '"event":"collect_complete"' in line:
                return True
    except OSError:
        return False
    return False


def count_activation_objects(uri: str) -> dict[str, int]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise ValueError("--activation-dir must be an S3 URI for wait_for_collection_then_sweep")
    prefix = parsed.path.lstrip("/").rstrip("/") + "/"
    pt = 0
    sidecars = 0
    for page in s3_client().get_paginator("list_objects_v2").paginate(Bucket=parsed.netloc, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            pt += int(key.endswith(".pt"))
            sidecars += int("/metadata/" in key and key.endswith(".json"))
    return {"pt": pt, "sidecars": sidecars}


if __name__ == "__main__":
    raise SystemExit(main())
