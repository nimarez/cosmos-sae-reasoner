from __future__ import annotations

import argparse
import json
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ActivationWorkerPlan:
    worker_index: int
    num_workers: int
    command: list[str]


def main() -> int:
    parser = argparse.ArgumentParser(description="Write per-GPU activation collection worker commands.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/sae_reasoner/activation_fanout"))
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--model-id", default="nvidia/Cosmos3-Nano")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--init-mode", default="pretrained")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--phase", choices=["prefill", "decode", "both"], default="prefill")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--activation-dtype", choices=["auto", "float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--python", default=".venv/bin/python")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", default=None)
    parser.add_argument("--wandb-tags", default="sae,activation-collection,fanout")
    parser.add_argument("--run-prefix", default="collect_activations")
    args = parser.parse_args()

    if args.num_workers < 1:
        raise ValueError("--num-workers must be at least 1")

    args.output_root.mkdir(parents=True, exist_ok=True)
    script_dir = args.output_root / "scripts"
    script_dir.mkdir(parents=True, exist_ok=True)

    plans = [make_worker_plan(args, worker_index=idx) for idx in range(args.num_workers)]
    commands_path = args.output_root / "worker_commands.jsonl"
    commands_path.write_text("".join(json.dumps(asdict(plan), sort_keys=True) + "\n" for plan in plans), encoding="utf-8")

    for plan in plans:
        path = script_dir / f"run_worker_{plan.worker_index:03d}.sh"
        path.write_text(render_worker_script(plan.command, env_file=args.env_file), encoding="utf-8")
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


def make_worker_plan(args: argparse.Namespace, *, worker_index: int) -> ActivationWorkerPlan:
    command = [
        args.python,
        "-m",
        "tools.sae_reasoner",
        "collect-activations",
        "--model-id",
        args.model_id,
        "--dtype",
        args.dtype,
        "--init-mode",
        args.init_mode,
        "--manifest",
        args.manifest,
        "--layer",
        str(args.layer),
        "--output-dir",
        args.output_dir,
        "--phase",
        args.phase,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--activation-dtype",
        args.activation_dtype,
        "--batch-size",
        str(args.batch_size),
        "--worker-index",
        str(worker_index),
        "--num-workers",
        str(args.num_workers),
        "--wandb-run-name",
        f"{args.run_prefix}_worker_{worker_index:03d}",
        "--wandb-tags",
        f"{args.wandb_tags},worker-{worker_index:03d}",
    ]
    if args.device:
        command.extend(["--device", args.device])
    if args.max_examples is not None:
        command.extend(["--max-examples", str(args.max_examples)])
    if args.resume:
        command.append("--resume")
    if args.wandb_project:
        command.extend(["--wandb-project", args.wandb_project])
    if args.wandb_entity:
        command.extend(["--wandb-entity", args.wandb_entity])
    if args.wandb_mode:
        command.extend(["--wandb-mode", args.wandb_mode])
    return ActivationWorkerPlan(worker_index=worker_index, num_workers=args.num_workers, command=command)


def render_worker_script(command: list[str], *, env_file: str) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"set -a; [ -f {shlex.quote(env_file)} ] && source {shlex.quote(env_file)}; set +a",
        " ".join(shlex.quote(part) for part in command),
        "",
    ]
    return "\n".join(lines)


def render_all_script(plans: list[ActivationWorkerPlan]) -> str:
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


if __name__ == "__main__":
    raise SystemExit(main())
