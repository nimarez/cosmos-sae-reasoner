from __future__ import annotations

import argparse
import json
import shlex
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SweepConfig:
    run_id: str
    stage: str
    activation_dir: str
    output: str
    analysis_output: str
    expansion_factor: int = 8
    top_k: int = 32
    topk_activation: str = "relu_topk"
    init_method: str = "kaiming"
    activation_norm: str = "sqrt_d"
    batch_size: int = 1024
    lr: float = 3e-4
    steps: int = 2000
    warmup_steps: int = 0
    lr_schedule: str = "constant"
    max_grad_norm: float = 0.0
    train_splits: str = "sae_train"
    val_splits: str = "sae_val"
    log_every: int = 50


def main() -> int:
    parser = argparse.ArgumentParser(description="Write staged SAE sweep configs and runnable shell scripts.")
    parser.add_argument("--activation-dir", required=True, help="Activation directory or S3 prefix.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/sae_reasoner/sweeps/bridgecaps_prefill"))
    parser.add_argument("--run-prefix", default="bridgecaps_l18")
    parser.add_argument("--stage", choices=["lr_batch", "capacity", "topk", "ablations", "all"], default="lr_batch")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--analysis-batch-size", type=int, default=4096)
    parser.add_argument("--python", default=".venv/bin/python", help="Python executable written into generated run scripts.")
    parser.add_argument("--best-batch-size", type=int, default=1024, help="Used for stages after lr_batch.")
    parser.add_argument("--best-lr", type=float, default=3e-4, help="Used for stages after lr_batch.")
    parser.add_argument("--best-expansion-factor", type=int, default=8, help="Used for stages after capacity.")
    parser.add_argument("--best-top-k", type=int, default=32, help="Used for ablation stage.")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-tags", default="sae,sweep,bridge-captions,prefill")
    args = parser.parse_args()

    configs = list(build_configs(args))
    args.output_root.mkdir(parents=True, exist_ok=True)
    config_path = args.output_root / f"{args.stage}_configs.jsonl"
    script_path = args.output_root / f"run_{args.stage}.sh"
    config_path.write_text("".join(json.dumps(asdict(config), sort_keys=True) + "\n" for config in configs), encoding="utf-8")
    script_path.write_text(render_script(configs, args), encoding="utf-8")
    script_path.chmod(0o755)
    print(json.dumps({"configs": str(config_path), "script": str(script_path), "num_runs": len(configs)}, indent=2))
    return 0


def build_configs(args: argparse.Namespace) -> Iterable[SweepConfig]:
    stages = ["lr_batch", "capacity", "topk", "ablations"] if args.stage == "all" else [args.stage]
    for stage in stages:
        yield from configs_for_stage(stage, args)


def configs_for_stage(stage: str, args: argparse.Namespace) -> Iterable[SweepConfig]:
    if stage == "lr_batch":
        for batch_size in [512, 1024, 2048]:
            for lr in [1e-4, 3e-4, 1e-3]:
                yield make_config(args, stage, f"bs{batch_size}_lr{lr:g}", batch_size=batch_size, lr=lr)
        return
    if stage == "capacity":
        for expansion_factor in [4, 8, 16]:
            yield make_config(
                args,
                stage,
                f"exp{expansion_factor}_bs{args.best_batch_size}_lr{args.best_lr:g}",
                expansion_factor=expansion_factor,
                batch_size=args.best_batch_size,
                lr=args.best_lr,
            )
        return
    if stage == "topk":
        for top_k in [16, 32, 64]:
            yield make_config(
                args,
                stage,
                f"exp{args.best_expansion_factor}_k{top_k}_bs{args.best_batch_size}_lr{args.best_lr:g}",
                expansion_factor=args.best_expansion_factor,
                top_k=top_k,
                batch_size=args.best_batch_size,
                lr=args.best_lr,
            )
        return
    if stage == "ablations":
        base = {
            "expansion_factor": args.best_expansion_factor,
            "top_k": args.best_top_k,
            "batch_size": args.best_batch_size,
            "lr": args.best_lr,
        }
        yield make_config(args, stage, "baseline", **base)
        yield make_config(args, stage, "init_data", init_method="data", **base)
        yield make_config(args, stage, "raw_topk", topk_activation="topk", **base)
        yield make_config(args, stage, "cosine", lr_schedule="cosine", **base)
        yield make_config(args, stage, "gradclip1", max_grad_norm=1.0, **base)
        return
    raise ValueError(f"unknown stage: {stage}")


def make_config(args: argparse.Namespace, stage: str, suffix: str, **overrides) -> SweepConfig:
    run_id = f"{args.run_prefix}_{stage}_{suffix}".replace(".", "p")
    output = args.output_root / "saes" / f"{run_id}.pt"
    analysis_output = args.output_root / "analysis" / f"{run_id}.json"
    data = {
        "run_id": run_id,
        "stage": stage,
        "activation_dir": args.activation_dir,
        "output": str(output),
        "analysis_output": str(analysis_output),
        "steps": args.steps,
        "log_every": args.log_every,
    }
    data.update(overrides)
    return SweepConfig(**data)


def render_script(configs: list[SweepConfig], args: argparse.Namespace) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
    ]
    for config in configs:
        lines.extend(render_run(config, args))
        lines.append("")
    return "\n".join(lines)


def render_run(config: SweepConfig, args: argparse.Namespace) -> list[str]:
    train = [
        args.python,
        "-m",
        "tools.sae_reasoner",
        "train-sae",
        "--activation-dir",
        config.activation_dir,
        "--output",
        config.output,
        "--expansion-factor",
        str(config.expansion_factor),
        "--top-k",
        str(config.top_k),
        "--topk-activation",
        config.topk_activation,
        "--init-method",
        config.init_method,
        "--activation-norm",
        config.activation_norm,
        "--recon-loss",
        "mse",
        "--feature-l1-coeff",
        "0.0",
        "--steps",
        str(config.steps),
        "--batch-size",
        str(config.batch_size),
        "--lr",
        str(config.lr),
        "--warmup-steps",
        str(config.warmup_steps),
        "--lr-schedule",
        config.lr_schedule,
        "--max-grad-norm",
        str(config.max_grad_norm),
        "--train-splits",
        config.train_splits,
        "--val-splits",
        config.val_splits,
        "--val-batch-size",
        str(config.batch_size),
        "--log-every",
        str(config.log_every),
        "--wandb-run-name",
        config.run_id,
        "--wandb-tags",
        f"{args.wandb_tags},{config.stage}",
    ]
    if args.wandb_project:
        train.extend(["--wandb-project", args.wandb_project])
    analyze = [
        args.python,
        "-m",
        "tools.sae_reasoner",
        "analyze-sae",
        "--activation-dir",
        config.activation_dir,
        "--sae",
        config.output,
        "--output",
        config.analysis_output,
        "--splits",
        "sae_train,sae_val",
        "--batch-size",
        str(args.analysis_batch_size),
        "--wandb-run-name",
        f"{config.run_id}_analysis",
        "--wandb-tags",
        f"{args.wandb_tags},{config.stage},analysis",
    ]
    if args.wandb_project:
        analyze.extend(["--wandb-project", args.wandb_project])
    return [
        f"echo {shlex.quote('start ' + config.run_id)}",
        " ".join(shlex.quote(part) for part in train),
        " ".join(shlex.quote(part) for part in analyze),
        f"echo {shlex.quote('done ' + config.run_id)}",
    ]


if __name__ == "__main__":
    raise SystemExit(main())
