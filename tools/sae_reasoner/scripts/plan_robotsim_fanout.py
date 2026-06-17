from __future__ import annotations

import argparse
import json
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkerPlan:
    worker_index: int
    num_workers: int
    output: str
    manifest_s3_uri: str
    command: list[str]


def main() -> int:
    parser = argparse.ArgumentParser(description="Write per-worker RobotSim tar materialization commands.")
    parser.add_argument("--s3-uri", required=True, help="S3 output prefix for materialized RobotSim media.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/sae_reasoner/robotsim_fanout"))
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--max-records-per-worker", type=int, default=0, help="0 means unlimited.")
    parser.add_argument("--max-shards", type=int, default=0, help="0 means all assigned shards.")
    parser.add_argument("--max-shard-gb", type=float, default=0.0, help="0 means no shard size filter.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard-list", default=None, help="Optional local/S3 shard JSONL shared by all workers.")
    parser.add_argument("--python", default=".venv/bin/python")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stream-tars", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.num_workers < 1:
        raise ValueError("--num-workers must be at least 1")

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_dir = args.output_root / "manifests"
    script_dir = args.output_root / "scripts"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    script_dir.mkdir(parents=True, exist_ok=True)

    plans = [make_worker_plan(args, worker_index=idx, manifest_dir=manifest_dir) for idx in range(args.num_workers)]
    commands_path = args.output_root / "worker_commands.jsonl"
    commands_path.write_text("".join(json.dumps(asdict(plan), sort_keys=True) + "\n" for plan in plans), encoding="utf-8")

    for plan in plans:
        script = render_worker_script(plan.command, env_file=args.env_file)
        path = script_dir / f"run_worker_{plan.worker_index:03d}.sh"
        path.write_text(script, encoding="utf-8")
        path.chmod(0o755)

    all_script = args.output_root / "run_all_local_parallel.sh"
    all_script.write_text(render_all_script(plans), encoding="utf-8")
    all_script.chmod(0o755)

    print(
        json.dumps(
            {
                "worker_commands": str(commands_path),
                "scripts": str(script_dir),
                "run_all_local_parallel": str(all_script),
                "num_workers": len(plans),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def make_worker_plan(args: argparse.Namespace, *, worker_index: int, manifest_dir: Path) -> WorkerPlan:
    output = manifest_dir / f"robotsim_worker_{worker_index:03d}.jsonl"
    manifest_s3_uri = s3_child_uri(args.s3_uri, f"manifests/worker_{worker_index:03d}.jsonl")
    command = [
        args.python,
        "-m",
        "tools.sae_reasoner",
        "build-corpus-manifest",
        "--source",
        "recipe",
        "--recipe",
        "physicalai-robotsim",
        "--s3-uri",
        args.s3_uri,
        "--manifest-s3-uri",
        manifest_s3_uri,
        "--max-records",
        str(args.max_records_per_worker),
        "--max-shards",
        str(args.max_shards),
        "--max-shard-gb",
        str(args.max_shard_gb),
        "--seed",
        str(args.seed),
        "--worker-index",
        str(worker_index),
        "--num-workers",
        str(args.num_workers),
        "--output",
        str(output),
    ]
    if args.shard_list:
        command.extend(["--shard-list", args.shard_list])
    if args.resume:
        command.append("--resume")
    if args.stream_tars:
        command.append("--stream-tars")
    return WorkerPlan(
        worker_index=worker_index,
        num_workers=args.num_workers,
        output=str(output),
        manifest_s3_uri=manifest_s3_uri,
        command=command,
    )


def render_worker_script(command: list[str], *, env_file: str) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"set -a; [ -f {shlex.quote(env_file)} ] && source {shlex.quote(env_file)}; set +a",
        "if [ -d /workspace ]; then",
        "  export COSMOS_SAE_TAR_TMPDIR=\"${COSMOS_SAE_TAR_TMPDIR:-/workspace/tmp/sae_hf_tar}\"",
        "  export TMPDIR=\"${TMPDIR:-/workspace/tmp}\"",
        "  mkdir -p \"$COSMOS_SAE_TAR_TMPDIR\" \"$TMPDIR\"",
        "fi",
        " ".join(shlex.quote(part) for part in command),
        "",
    ]
    return "\n".join(lines)


def render_all_script(plans: list[WorkerPlan]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "script_dir=\"$(cd \"$(dirname \"$0\")\" && pwd)\"",
        "pids=()",
    ]
    for plan in plans:
        lines.append(
            f"bash \"$script_dir/scripts/run_worker_{plan.worker_index:03d}.sh\" "
            f"> \"$script_dir/worker_{plan.worker_index:03d}.log\" 2>&1 &"
        )
        lines.append("pids+=(\"$!\")")
    lines.extend(
        [
            "for pid in \"${pids[@]}\"; do",
            "  wait \"$pid\"",
            "done",
            "",
        ]
    )
    return "\n".join(lines)


def s3_child_uri(parent: str, child: str) -> str:
    if not parent.startswith("s3://"):
        raise ValueError("--s3-uri must be an S3 URI")
    return f"{parent.rstrip('/')}/{child.lstrip('/')}"


if __name__ == "__main__":
    raise SystemExit(main())
