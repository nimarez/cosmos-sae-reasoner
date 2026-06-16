from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import ensure_dir, write_jsonl
from .manifest import iter_manifest, load_manifest, make_sample_manifest


class RuntimeLoadError(RuntimeError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        if isinstance(exc, RuntimeLoadError) or exc.__class__.__name__ == "RuntimeLoadError":
            print(f"runtime error: {exc}")
            return 2
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tools.sae_reasoner")
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("prepare-snapshot", help="download/cache the HF snapshot")
    p.add_argument("--model-id", default="nvidia/Cosmos3-Nano")
    p.add_argument("--local-dir", default=None)
    p.add_argument("--include-weights", action="store_true")
    p.set_defaults(func=cmd_prepare_snapshot)

    p = sub.add_parser("make-sample-jsonl", help="write a tiny sample manifest")
    p.add_argument("--output", type=Path, default=Path("outputs/sae_reasoner/sample_manifest.jsonl"))
    p.set_defaults(func=cmd_make_sample_jsonl)

    p = sub.add_parser("build-corpus-manifest", help="stream external corpus metadata into a neutral manifest")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source", choices=["recipe", "hf-files", "hf-dataset", "hf-tar-s3", "s3-prefix", "jsonl"], default="recipe")
    p.add_argument("--recipe", default="robotics-bridge-captions")
    p.add_argument("--hf-repo-id", default=None)
    p.add_argument("--hf-split", default="train")
    p.add_argument("--s3-uri", default=None)
    p.add_argument("--input-uri", default=None)
    p.add_argument("--include-glob", action="append", default=[])
    p.add_argument("--member-glob", action="append", default=[], help="Tar member glob for --source hf-tar-s3, e.g. '*.mp4'.")
    p.add_argument("--prompt", default=None)
    p.add_argument("--media-type", choices=["auto", "text", "image", "video"], default="auto")
    p.add_argument("--max-records", type=int, default=1000)
    p.add_argument("--max-shards", type=int, default=1, help="Maximum tar shards to download for --source hf-tar-s3. Use 0 for all matched shards.")
    p.add_argument("--max-shard-gb", type=float, default=1.0, help="Skip tar shards larger than this for --source hf-tar-s3. Use 0 for no size filter.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split-ratios", default="sae_train=0.85,sae_val=0.10,feature_labeling=0.025,steering_eval=0.025")
    p.add_argument("--manifest-s3-uri", default=None, help="Optional S3 URI to upload the generated manifest JSONL.")
    p.add_argument("--id-field", default=None)
    p.add_argument("--text-field", default=None)
    p.add_argument("--prompt-field", default=None)
    p.add_argument("--media-field", default=None)
    p.add_argument("--shuffle-buffer", type=int, default=10000)
    p.add_argument("--min-text-chars", type=int, default=64)
    p.add_argument("--max-text-chars", type=int, default=8000)
    p.set_defaults(func=cmd_build_corpus_manifest)

    p = sub.add_parser("inspect-model", help="load model and print hook points")
    add_model_args(p)
    p.set_defaults(func=cmd_inspect_model)

    p = sub.add_parser("collect-activations", help="collect all-token prefill/decode activations")
    add_model_args(p)
    add_prompt_args(p)
    p.add_argument("--manifest", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument(
        "--output-dir",
        required=True,
        help="Local directory or S3 URI such as s3://bucket/prefix. S3 credentials are read from AWS env vars.",
    )
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--resume", action="store_true", help="Skip records whose activation shard and metadata sidecar already exist.")
    p.add_argument("--phase", choices=["prefill", "decode", "both"], default="both")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument(
        "--activation-dtype",
        choices=["auto", "float32", "bfloat16", "float16"],
        default="bfloat16",
        help="Saved activation dtype. Training casts sampled mini-batches to float32.",
    )
    add_wandb_args(p)
    p.set_defaults(func=cmd_collect_activations)

    p = sub.add_parser("train-sae", help="train a SAE from activation shards")
    p.add_argument("--activation-dir", required=True, help="Local activation directory or S3 prefix.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expansion-factor", type=int, default=16)
    p.add_argument("--top-k", type=int, default=32)
    p.add_argument("--topk-activation", choices=["topk", "relu_topk", "batch_topk"], default="topk")
    p.add_argument("--batch-topk-momentum", type=float, default=0.01, help="EMA momentum for BatchTopK inference threshold.")
    p.add_argument("--init-method", choices=["data", "kaiming"], default="data")
    p.add_argument("--init-blend", type=float, default=0.8, help="Data-point init blend p. Used when --init-method=data.")
    p.add_argument("--activation-norm", choices=["sqrt_d", "none"], default="sqrt_d", help="Scale activations so average L2 norm is sqrt(hidden_dim).")
    p.add_argument(
        "--matryoshka-prefixes",
        default="",
        help=(
            "Optional comma-separated nested dictionary cutoffs for Matryoshka SAE training. "
            "Use absolute feature counts or fractions such as 0.125,0.25,0.5. Empty disables it."
        ),
    )
    p.add_argument("--matryoshka-loss-coeff", type=float, default=1.0, help="Coefficient for summed Matryoshka prefix reconstruction losses.")
    p.add_argument("--recon-loss", choices=["mse", "l1", "smooth_l1"], default="mse")
    p.add_argument("--feature-l1-coeff", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--lr-schedule", choices=["constant", "cosine"], default="cosine")
    p.add_argument("--max-grad-norm", type=float, default=1.0, help="Clip gradient norm when >0. Always logs unclipped grad_norm.")
    p.add_argument("--token-kinds", default="", help="Optional comma-separated token kinds to train on. Empty means all.")
    p.add_argument("--phases", default="", help="Optional comma-separated phases to train on. Empty means all.")
    p.add_argument("--train-splits", default="sae_train", help="Comma-separated manifest splits used for training. Empty means all splits.")
    p.add_argument("--val-splits", default="sae_val", help="Comma-separated manifest splits used for reconstruction validation. Empty disables validation.")
    p.add_argument("--val-batch-size", type=int, default=None, help="Validation sample size per metric row. Defaults to --batch-size.")
    p.add_argument("--log-every", type=int, default=10, help="Emit training metrics every N steps.")
    add_wandb_args(p)
    p.set_defaults(func=cmd_train_sae)

    p = sub.add_parser("find-features", help="rank top activating records/features")
    p.add_argument("--activation-dir", required=True, help="Local activation directory or S3 prefix.")
    p.add_argument("--sae", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--feature-ids", type=str, default="")
    p.add_argument("--feature-rank", choices=["absolute", "positive"], default="absolute")
    p.add_argument("--token-kinds", default="", help="Optional comma-separated token kinds to include. Empty means all.")
    p.add_argument("--phases", default="", help="Optional comma-separated token phases to include. Empty means all.")
    p.set_defaults(func=cmd_find_features)

    p = sub.add_parser("render-feature-report", help="render feature examples as a standalone HTML report")
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("outputs/sae_reasoner/reports/features.html"))
    p.add_argument("--title", default="Cosmos SAE Feature Browser")
    p.set_defaults(func=cmd_render_feature_report)

    p = sub.add_parser("find-neighbors", help="find nearest activation-token neighbors by cosine similarity")
    p.add_argument("--activation-dir", required=True, help="Local activation directory or S3 prefix.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-tokens", type=int, default=5000)
    p.add_argument("--num-queries", type=int, default=40)
    p.add_argument("--neighbors", type=int, default=8)
    p.add_argument("--query-kinds", default="", help="Comma-separated token kinds, e.g. image,video,text. Empty means all.")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_find_neighbors)

    p = sub.add_parser("render-neighbor-report", help="render nearest-token activation neighbors as standalone HTML")
    p.add_argument("--neighbors", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("outputs/sae_reasoner/reports/neighbors.html"))
    p.add_argument("--title", default="Cosmos Activation Nearest Neighbors")
    p.set_defaults(func=cmd_render_neighbor_report)

    p = sub.add_parser("steer", help="baseline vs steered generation")
    add_model_args(p)
    add_prompt_args(p)
    p.add_argument("--sae", type=Path, required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--feature-id", type=int, required=True)
    p.add_argument("--multiplier", type=float, required=True)
    p.add_argument("--scope", choices=["prefill", "decode", "both"], default="decode")
    p.add_argument("--prompt", default=None, help="Text-only prompt. Use --manifest for multimodal steering.")
    p.add_argument("--manifest", type=Path, default=None, help="Optional manifest JSONL containing the record to steer.")
    p.add_argument("--record-id", default=None, help="Record id inside --manifest. Defaults to the first manifest record.")
    p.add_argument("--steer-token-kinds", default="", help="Optional comma-separated prefill token kinds to edit, e.g. video,image,special.")
    p.add_argument("--steer-roles", default="", help="Optional comma-separated prefill chat roles to edit, e.g. user,assistant.")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--output", type=Path, default=Path("outputs/sae_reasoner/reports/steer_result.json"))
    p.set_defaults(func=cmd_steer)
    return parser


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-id", default="nvidia/Cosmos3-Nano")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"])
    parser.add_argument(
        "--init-mode",
        default="pretrained",
        choices=["pretrained", "random", "meta"],
        help=(
            "pretrained loads checkpoint weights; random builds full-size random weights from config; "
            "meta builds the full architecture on meta tensors for inspection only"
        ),
    )


def add_prompt_args(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(prompt_format="chat")
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system prompt passed through the model processor chat template.",
    )


def add_wandb_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--wandb-project", default=None, help="Optional W&B project. Also read from WANDB_PROJECT.")
    parser.add_argument("--wandb-entity", default=None, help="Optional W&B entity. Also read from WANDB_ENTITY.")
    parser.add_argument("--wandb-run-name", default=None, help="Optional W&B run name.")
    parser.add_argument("--wandb-tags", default="", help="Comma-separated W&B tags.")
    parser.add_argument("--wandb-mode", default=None, choices=["online", "offline", "disabled"], help="W&B mode. Also read from WANDB_MODE.")


def cmd_prepare_snapshot(args: argparse.Namespace) -> int:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeLoadError("Missing huggingface_hub. Install it before preparing the snapshot.") from exc
    ignore_patterns = None if args.include_weights else ["*.safetensors", "*.bin", "*.pt", "*.pth", "*.ckpt"]
    path = snapshot_download(
        repo_id=args.model_id,
        local_dir=args.local_dir,
        ignore_patterns=ignore_patterns,
    )
    print(json.dumps({"snapshot_path": path, "include_weights": bool(args.include_weights)}, indent=2))
    return 0


def cmd_make_sample_jsonl(args: argparse.Namespace) -> int:
    samples = make_sample_manifest(args.output)
    print(json.dumps({"output": str(args.output), "num_records": len(samples)}, indent=2))
    return 0


def cmd_build_corpus_manifest(args: argparse.Namespace) -> int:
    from .corpus import BuildCorpusConfig, build_corpus_manifest, parse_split_ratios

    records = build_corpus_manifest(
        BuildCorpusConfig(
            output=args.output,
            source=args.source,
            recipe=args.recipe,
            hf_repo_id=args.hf_repo_id,
            hf_split=args.hf_split,
            s3_uri=args.s3_uri,
            input_uri=args.input_uri,
            include_globs=tuple(args.include_glob),
            member_globs=tuple(args.member_glob),
            prompt=args.prompt,
            media_type=args.media_type,
            max_records=args.max_records,
            max_shards=None if args.max_shards == 0 else args.max_shards,
            max_shard_gb=None if args.max_shard_gb == 0 else args.max_shard_gb,
            seed=args.seed,
            split_ratios=parse_split_ratios(args.split_ratios),
            id_field=args.id_field,
            text_field=args.text_field,
            prompt_field=args.prompt_field,
            media_field=args.media_field,
            shuffle_buffer=args.shuffle_buffer,
            min_text_chars=args.min_text_chars,
            max_text_chars=args.max_text_chars,
        )
    )
    manifest_uri = None
    if args.manifest_s3_uri:
        manifest_uri = upload_file_to_s3(args.output, args.manifest_s3_uri)
    split_counts: dict[str, int] = {}
    media_counts: dict[str, int] = {}
    for record in records:
        metadata = record.get("metadata") or {}
        split = metadata.get("split", "unknown")
        split_counts[split] = split_counts.get(split, 0) + 1
        media_type = record.get("media_type", "unknown")
        media_counts[media_type] = media_counts.get(media_type, 0) + 1
    print(
        json.dumps(
            {
                "output": str(args.output),
                "manifest_uri": manifest_uri,
                "num_records": len(records),
                "split_counts": split_counts,
                "media_counts": media_counts,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def upload_file_to_s3(path: Path, uri: str) -> str:
    from .storage import parse_s3_uri

    try:
        import boto3
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeLoadError("S3 manifest upload requires boto3. Install the sae dependency group.") from exc
    bucket, key = parse_s3_uri(uri)
    endpoint_url = __import__("os").environ.get("AWS_ENDPOINT_URL_S3") or __import__("os").environ.get("AWS_ENDPOINT_URL")
    kwargs = {"endpoint_url": endpoint_url} if endpoint_url else {}
    boto3.client("s3", **kwargs).upload_file(
        str(path),
        bucket,
        key,
        ExtraArgs={"ContentType": "application/jsonl; charset=utf-8"},
    )
    return f"s3://{bucket}/{key}"


def cmd_inspect_model(args: argparse.Namespace) -> int:
    from .runtime import CosmosReasonerRuntime

    runtime = CosmosReasonerRuntime(
        args.model_id,
        device=args.device,
        dtype=args.dtype,
        init_mode=args.init_mode,
    ).load()
    print(json.dumps(runtime.describe(), indent=2))
    return 0


def cmd_collect_activations(args: argparse.Namespace) -> int:
    import torch

    from .runtime import CosmosReasonerRuntime
    from .storage import make_activation_store

    store = make_activation_store(args.output_dir)
    wandb_run = init_basic_wandb_run(
        args,
        job_type="collect_activations",
        config={
            "model_id": args.model_id,
            "device": args.device,
            "dtype": args.dtype,
            "init_mode": args.init_mode,
            "manifest": args.manifest,
            "layer": args.layer,
            "output_dir": args.output_dir,
            "max_examples": args.max_examples,
            "resume": args.resume,
            "phase": args.phase,
            "max_new_tokens": args.max_new_tokens,
            "activation_dtype": args.activation_dtype,
            "prompt_format": args.prompt_format,
            "system_prompt": args.system_prompt,
        },
    )
    runtime = CosmosReasonerRuntime(
        args.model_id,
        device=args.device,
        dtype=args.dtype,
        init_mode=args.init_mode,
    ).load()
    metadata: list[dict[str, Any]] = []
    start = time.time()
    total_tokens = 0
    total_activation_bytes = 0
    skipped_examples = 0
    for idx, record in enumerate(iter_manifest(args.manifest)):
        if args.max_examples is not None and idx >= args.max_examples:
            break
        shard_name = shard_name_for_record(idx, record.id)
        sidecar_name = activation_sidecar_name(shard_name)
        if args.resume and store.exists(shard_name) and store.exists(sidecar_name):
            sidecar = json.loads(store.read_text(sidecar_name))
            metadata.append(sidecar)
            skipped_examples += 1
            print(
                json.dumps(
                    {
                        "event": "collect_skip",
                        "record_index": idx,
                        "skipped_examples": skipped_examples,
                        "record_id": record.id,
                        "shard": shard_name,
                    }
                ),
                flush=True,
            )
            continue
        record_start = time.time()
        hidden, meta = runtime.collect_activations(
            record,
            layer=args.layer,
            phase=args.phase,
            max_new_tokens=args.max_new_tokens,
            prompt_format=args.prompt_format,
            system_prompt=args.system_prompt,
            activation_dtype=args.activation_dtype,
        )
        shard_uri = store.write_torch(shard_name, {"activations": hidden, "meta": meta})
        meta["shard"] = shard_name
        meta["shard_uri"] = shard_uri
        compact_meta = compact_activation_meta(meta)
        metadata.append(compact_meta)
        store.write_text(sidecar_name, json.dumps(compact_meta, ensure_ascii=True, sort_keys=True) + "\n")
        elapsed = max(1e-9, time.time() - start)
        record_seconds = max(1e-9, time.time() - record_start)
        activation_bytes = int(hidden.numel() * hidden.element_size())
        total_tokens += int(meta["num_tokens"])
        total_activation_bytes += activation_bytes
        metric = collect_metric(
            idx=idx,
            meta=meta,
            activation_bytes=activation_bytes,
            total_tokens=total_tokens,
            total_activation_bytes=total_activation_bytes,
            elapsed_seconds=elapsed,
            record_seconds=record_seconds,
        )
        print(json.dumps({"event": "collect_metric", **metric, "collected": record.id, "shard": shard_name, "uri": shard_uri}), flush=True)
        if wandb_run is not None:
            wandb_run.log(metric, step=int(metric["collected_examples"]))
    metadata_text = "".join(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n" for record in metadata)
    metadata_uri = store.write_text("metadata.jsonl", metadata_text)
    if wandb_run is not None:
        wandb_run.summary.update(
            {
                "final_collected_examples": len(metadata),
                "final_skipped_examples": skipped_examples,
                "final_total_tokens": total_tokens,
                "final_total_activation_gb": total_activation_bytes / 1_000_000_000,
                "metadata_uri": metadata_uri,
            }
        )
        wandb_run.finish()
    return 0


def collect_metric(
    *,
    idx: int,
    meta: dict[str, Any],
    activation_bytes: int,
    total_tokens: int,
    total_activation_bytes: int,
    elapsed_seconds: float,
    record_seconds: float,
) -> dict[str, float | int | str]:
    token_kind_counts = meta.get("token_kind_counts") or {}
    token_phase_counts = meta.get("token_phase_counts") or {}
    tokens = int(meta.get("num_tokens") or 0)
    metric: dict[str, float | int | str] = {
        "collected_examples": idx + 1,
        "tokens": tokens,
        "total_tokens": total_tokens,
        "record_seconds": record_seconds,
        "elapsed_seconds": elapsed_seconds,
        "examples_per_min": 60.0 * float(idx + 1) / elapsed_seconds,
        "tokens_per_sec": float(total_tokens) / elapsed_seconds,
        "activation_bytes": activation_bytes,
        "total_activation_gb": total_activation_bytes / 1_000_000_000,
        "activation_dtype": str(meta.get("activation_dtype") or ""),
    }
    for kind, count in token_kind_counts.items():
        metric[f"token_kind/{kind}"] = int(count)
    for phase, count in token_phase_counts.items():
        metric[f"token_phase/{phase}"] = int(count)
    return metric


def cmd_train_sae(args: argparse.Namespace) -> int:
    from .sae import save_sae, train_sae_from_tensor

    train_data = load_activation_dataset(
        args.activation_dir,
        token_kinds=parse_kind_filter(args.token_kinds),
        phases=parse_kind_filter(args.phases),
        splits=parse_kind_filter(args.train_splits),
    )
    if train_data.activations is None:
        raise ValueError(f"no training activation shards found in {args.activation_dir}")
    val_splits = parse_kind_filter(args.val_splits)
    val_data = (
        load_activation_dataset(
            args.activation_dir,
            token_kinds=parse_kind_filter(args.token_kinds),
            phases=parse_kind_filter(args.phases),
            splits=val_splits,
            allow_empty=True,
        )
        if val_splits
        else None
    )
    activations = train_data.activations
    validation_activations = val_data.activations if val_data is not None else None
    wandb_run = init_wandb_run(
        args,
        num_activations=int(activations.shape[0]),
        hidden_dim=int(activations.shape[1]),
        train_group_counts=train_data.group_counts,
        val_group_counts=val_data.group_counts if val_data is not None else {},
    )

    def log_progress(metric: dict[str, float]) -> None:
        record = {"event": "train_metric", **metric}
        print(json.dumps(record), flush=True)
        if wandb_run is not None:
            wandb_run.log(metric, step=int(metric["step"]))

    sae, metrics = train_sae_from_tensor(
        activations,
        validation_activations=validation_activations,
        token_groups=train_data.token_groups,
        validation_token_groups=val_data.token_groups if val_data is not None else None,
        expansion_factor=args.expansion_factor,
        top_k=args.top_k,
        topk_activation=args.topk_activation,
        batch_topk_momentum=args.batch_topk_momentum,
        init_method=args.init_method,
        init_blend=args.init_blend,
        activation_norm=args.activation_norm,
        matryoshka_prefixes=args.matryoshka_prefixes,
        matryoshka_loss_coeff=args.matryoshka_loss_coeff,
        recon_loss=args.recon_loss,
        feature_l1_coeff=args.feature_l1_coeff,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        lr_schedule=args.lr_schedule,
        max_grad_norm=args.max_grad_norm if args.max_grad_norm > 0 else None,
        val_batch_size=args.val_batch_size,
        log_every=args.log_every,
        progress_callback=log_progress,
    )
    ensure_dir(args.output.parent)
    save_sae(
        str(args.output),
        sae,
        metadata={
            "metrics": metrics,
            "recon_loss": args.recon_loss,
            "feature_l1_coeff": args.feature_l1_coeff,
            "topk_activation": args.topk_activation,
            "batch_topk_momentum": args.batch_topk_momentum,
            "init_method": args.init_method,
            "init_blend": args.init_blend,
            "activation_norm": args.activation_norm,
            "matryoshka_prefixes": args.matryoshka_prefixes,
            "matryoshka_loss_coeff": args.matryoshka_loss_coeff,
            "token_kinds": args.token_kinds,
            "phases": args.phases,
            "train_splits": args.train_splits,
            "val_splits": args.val_splits,
            "val_batch_size": args.val_batch_size,
            "lr": args.lr,
            "warmup_steps": args.warmup_steps,
            "lr_schedule": args.lr_schedule,
            "max_grad_norm": args.max_grad_norm,
            "wandb": wandb_run_metadata(wandb_run),
            "train_group_counts": train_data.group_counts,
            "val_group_counts": val_data.group_counts if val_data is not None else {},
        },
    )
    write_jsonl(args.output.with_suffix(".metrics.jsonl"), metrics)
    if wandb_run is not None:
        wandb_run.summary.update(
            {
                "final_loss": metrics[-1]["loss"] if metrics else None,
                "final_recon_loss": metrics[-1]["recon_loss"] if metrics else None,
                "final_explained_variance": metrics[-1].get("explained_variance") if metrics else None,
                "final_val_explained_variance": metrics[-1].get("val_explained_variance") if metrics else None,
                "final_grad_norm": metrics[-1].get("grad_norm") if metrics else None,
                "sae_output": str(args.output),
            }
        )
        wandb_run.finish()
    print(json.dumps({"output": str(args.output), "num_activations": int(activations.shape[0]), "metrics": metrics[-1]}, indent=2))
    return 0


def init_wandb_run(
    args: argparse.Namespace,
    *,
    num_activations: int,
    hidden_dim: int,
    train_group_counts: dict[str, int] | None = None,
    val_group_counts: dict[str, int] | None = None,
) -> Any | None:
    config = {
        "activation_dir": str(args.activation_dir),
        "output": str(args.output),
        "num_activations": num_activations,
        "hidden_dim": hidden_dim,
        "expansion_factor": args.expansion_factor,
        "top_k": args.top_k,
        "topk_activation": args.topk_activation,
        "batch_topk_momentum": args.batch_topk_momentum,
        "init_method": args.init_method,
        "init_blend": args.init_blend,
        "activation_norm": args.activation_norm,
        "matryoshka_prefixes": args.matryoshka_prefixes,
        "matryoshka_loss_coeff": args.matryoshka_loss_coeff,
        "recon_loss": args.recon_loss,
        "feature_l1_coeff": args.feature_l1_coeff,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "warmup_steps": args.warmup_steps,
        "lr_schedule": args.lr_schedule,
        "max_grad_norm": args.max_grad_norm,
        "token_kinds": args.token_kinds,
        "phases": args.phases,
        "train_splits": args.train_splits,
        "val_splits": args.val_splits,
        "val_batch_size": args.val_batch_size,
        "log_every": args.log_every,
        "train_group_counts": train_group_counts or {},
        "val_group_counts": val_group_counts or {},
    }
    return init_basic_wandb_run(args, job_type="train_sae", config=config)


def init_basic_wandb_run(args: argparse.Namespace, *, job_type: str, config: dict[str, Any]) -> Any | None:
    project = args.wandb_project or __import__("os").environ.get("WANDB_PROJECT")
    mode = args.wandb_mode or __import__("os").environ.get("WANDB_MODE")
    if not project and mode != "offline":
        return None
    try:
        import wandb
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeLoadError("W&B logging requested but wandb is not installed. Run `uv pip install wandb`.") from exc
    tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
    return wandb.init(
        project=project or "cosmos-sae-reasoner",
        entity=args.wandb_entity or __import__("os").environ.get("WANDB_ENTITY"),
        name=args.wandb_run_name,
        tags=tags,
        job_type=job_type,
        mode=mode,
        config=config,
    )


def wandb_run_metadata(run: Any | None) -> dict[str, Any] | None:
    if run is None:
        return None
    return {
        "project": getattr(run, "project", None),
        "entity": getattr(run, "entity", None),
        "name": getattr(run, "name", None),
        "id": getattr(run, "id", None),
        "url": getattr(run, "url", None),
    }


def cmd_find_features(args: argparse.Namespace) -> int:
    import torch

    from .sae import load_sae
    from .storage import iter_activation_payloads

    sae = load_sae(str(args.sae))
    feature_ids = parse_feature_ids(args.feature_ids, sae.config.feature_dim)
    token_kinds = parse_kind_filter(args.token_kinds)
    phases = parse_kind_filter(args.phases)
    records: list[dict[str, Any]] = []
    for shard_name, payload in iter_activation_payloads(args.activation_dir):
        if shard_name == "metadata.pt":
            continue
        acts = payload["activations"].float()
        meta = payload.get("meta", {})
        token_map = meta.get("token_map") or []
        with torch.no_grad():
            features = sae.encode(acts)
        eligible = [
            idx
            for idx in range(features.shape[0])
            if token_matches_filters(token_info_for_index(token_map, idx), token_kinds=token_kinds, phases=phases)
        ]
        if not eligible:
            continue
        eligible_idx = torch.tensor(eligible, dtype=torch.long)
        for feature_id in feature_ids:
            values = features[eligible_idx, feature_id]
            scores = values.abs() if args.feature_rank == "absolute" else values
            top_scores, top_idx = torch.topk(scores, k=min(args.top_n, scores.numel()))
            for score, local_token_idx in zip(top_scores.tolist(), top_idx.tolist()):
                value = float(values[int(local_token_idx)].item())
                if args.feature_rank == "positive" and value <= 0:
                    continue
                token_idx = int(eligible_idx[int(local_token_idx)].item())
                token_info = token_info_for_index(token_map, token_idx)
                records.append(
                    {
                        "feature_id": feature_id,
                        "activation": value,
                        "activation_score": float(score),
                        "token_index": int(token_idx),
                        "token_info": token_info,
                        "record_id": meta.get("id"),
                        "prompt": meta.get("prompt"),
                        "media_type": meta.get("media_type"),
                        "media_path": meta.get("media_path"),
                        "tags": meta.get("tags", []),
                        "metadata": meta.get("metadata", {}),
                        "shard": shard_name,
                    }
                )
    records.sort(key=lambda r: r.get("activation_score", abs(float(r.get("activation", 0.0)))), reverse=True)
    write_jsonl(args.output, records[: max(args.top_n, len(feature_ids) * args.top_n)])
    print(json.dumps({"output": str(args.output), "num_records": len(records)}, indent=2))
    return 0


def compact_activation_meta(meta: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in meta.items() if key != "token_map"}


def activation_sidecar_name(shard_name: str) -> str:
    return f"metadata/{shard_name}.json"


def token_info_for_index(token_map: list[dict[str, Any]], token_index: int) -> dict[str, Any] | None:
    if 0 <= token_index < len(token_map):
        return token_map[token_index]
    return None


def load_activation_examples(path: str | Path, *, max_tokens: int, seed: int) -> tuple[list[dict[str, Any]], torch.Tensor]:
    import torch
    from .storage import iter_activation_payloads

    if max_tokens <= 1:
        raise ValueError("--max-tokens must be greater than 1")
    rng = random.Random(seed)
    examples: list[dict[str, Any]] = []
    vectors: list[torch.Tensor] = []
    seen = 0
    for shard_name, payload in iter_activation_payloads(path):
        if "activations" not in payload:
            continue
        activations = payload["activations"].float()
        acts = activations.reshape(-1, activations.shape[-1])
        meta = payload.get("meta", {})
        token_map = meta.get("token_map") or []
        for token_index in range(acts.shape[0]):
            seen += 1
            replacement = len(examples) if len(examples) < max_tokens else rng.randrange(seen)
            if replacement >= max_tokens:
                continue
            example = activation_example(meta, shard_name, token_map, token_index)
            vector = acts[token_index].detach().clone()
            if replacement == len(examples):
                examples.append(example)
                vectors.append(vector)
            else:
                examples[replacement] = example
                vectors[replacement] = vector
    if not vectors:
        raise ValueError(f"no activation shards found in {path}")
    return examples, torch.stack(vectors, dim=0)


def activation_example(meta: dict[str, Any], shard: str, token_map: list[dict[str, Any]], token_index: int) -> dict[str, Any]:
    return {
        "record_id": meta.get("id"),
        "prompt": meta.get("prompt"),
        "media_type": meta.get("media_type"),
        "media_path": meta.get("media_path"),
        "tags": meta.get("tags", []),
        "metadata": meta.get("metadata", {}),
        "shard": shard,
        "token_index": int(token_index),
        "token_info": token_info_for_index(token_map, token_index),
    }


def parse_kind_filter(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def cmd_render_feature_report(args: argparse.Namespace) -> int:
    from .visualize import render_feature_report

    render_feature_report(args.features, args.output, title=args.title)
    print(json.dumps({"output": str(args.output)}, indent=2))
    return 0


def cmd_find_neighbors(args: argparse.Namespace) -> int:
    import torch

    examples, matrix = load_activation_examples(args.activation_dir, max_tokens=args.max_tokens, seed=args.seed)
    if len(examples) < 2:
        raise ValueError(f"need at least 2 activation tokens, found {len(examples)}")
    allowed_kinds = parse_kind_filter(args.query_kinds)
    query_candidates = [
        idx for idx, example in enumerate(examples) if not allowed_kinds or (example.get("token_info") or {}).get("kind") in allowed_kinds
    ]
    if not query_candidates:
        raise ValueError(f"no query tokens matched kinds={sorted(allowed_kinds)}")
    rng = random.Random(args.seed)
    rng.shuffle(query_candidates)
    query_indices = query_candidates[: min(args.num_queries, len(query_candidates))]
    normalized = torch.nn.functional.normalize(matrix.float(), dim=1)
    records: list[dict[str, Any]] = []
    for query_index in query_indices:
        sims = normalized @ normalized[query_index]
        top_values, top_indices = torch.topk(sims, k=min(args.neighbors + 1, sims.numel()))
        neighbors = []
        for value, neighbor_index in zip(top_values.tolist(), top_indices.tolist()):
            neighbor_index = int(neighbor_index)
            if neighbor_index == query_index:
                continue
            neighbors.append({**examples[neighbor_index], "similarity": float(value)})
            if len(neighbors) >= args.neighbors:
                break
        records.append({"query": examples[query_index], "neighbors": neighbors})
    write_jsonl(args.output, records)
    print(json.dumps({"output": str(args.output), "num_queries": len(records), "sampled_tokens": len(examples)}, indent=2))
    return 0


def cmd_render_neighbor_report(args: argparse.Namespace) -> int:
    from .visualize import render_neighbor_report

    render_neighbor_report(args.neighbors, args.output, title=args.title)
    print(json.dumps({"output": str(args.output)}, indent=2))
    return 0


def cmd_steer(args: argparse.Namespace) -> int:
    from .runtime import CosmosReasonerRuntime
    from .runtime.hooks import FeatureSteeringHook
    from .sae import load_sae

    record = load_steering_record(args)
    if record is None and not args.prompt:
        raise ValueError("steer requires either --prompt or --manifest")
    if record is None and (parse_kind_filter(args.steer_token_kinds) or parse_kind_filter(args.steer_roles)):
        raise ValueError("--steer-token-kinds/--steer-roles require --manifest so token maps can be built")
    runtime = CosmosReasonerRuntime(
        args.model_id,
        device=args.device,
        dtype=args.dtype,
        init_mode=args.init_mode,
    ).load()
    sae = load_sae(str(args.sae), map_location="cpu")
    if record is None:
        baseline: Any = runtime.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            prompt_format=args.prompt_format,
            system_prompt=args.system_prompt,
        )
    else:
        baseline = runtime.generate_for_record(
            record,
            max_new_tokens=args.max_new_tokens,
            prompt_format=args.prompt_format,
            system_prompt=args.system_prompt,
        )
    hook = FeatureSteeringHook(
        sae=sae,
        feature_id=args.feature_id,
        multiplier=args.multiplier,
        scope=args.scope,
        token_kinds=frozenset(parse_kind_filter(args.steer_token_kinds)),
        roles=frozenset(parse_kind_filter(args.steer_roles)),
    )
    if record is None:
        steered: Any = runtime.generate(
            prompt=args.prompt,
            layer=args.layer,
            edit_fn=hook,
            max_new_tokens=args.max_new_tokens,
            prompt_format=args.prompt_format,
            system_prompt=args.system_prompt,
        )
    else:
        steered = runtime.generate_for_record(
            record,
            layer=args.layer,
            edit_fn=hook,
            max_new_tokens=args.max_new_tokens,
            prompt_format=args.prompt_format,
            system_prompt=args.system_prompt,
        )
    result = {
        "prompt": args.prompt,
        "record_id": record.id if record is not None else None,
        "media_type": record.media_type if record is not None else None,
        "media_path": record.media_path if record is not None else None,
        "layer": args.layer,
        "feature_id": args.feature_id,
        "multiplier": args.multiplier,
        "scope": args.scope,
        "steer_token_kinds": sorted(parse_kind_filter(args.steer_token_kinds)),
        "steer_roles": sorted(parse_kind_filter(args.steer_roles)),
        "prompt_format": args.prompt_format,
        "system_prompt": args.system_prompt,
        "baseline": baseline,
        "steered": steered,
    }
    ensure_dir(args.output.parent)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output)}, indent=2))
    return 0


def load_steering_record(args: argparse.Namespace):
    if args.manifest is None:
        return None
    for record in iter_manifest(args.manifest):
        if args.record_id is None or record.id == args.record_id:
            return record
    raise ValueError(f"record id {args.record_id!r} not found in {args.manifest}")


@dataclass(frozen=True)
class ActivationDataset:
    activations: Any | None
    token_groups: list[tuple[str, ...]]
    group_counts: dict[str, int]


def load_activation_matrix(
    path: str | Path,
    *,
    token_kinds: set[str] | None = None,
    phases: set[str] | None = None,
    splits: set[str] | None = None,
    allow_empty: bool = False,
) -> torch.Tensor | None:
    return load_activation_dataset(
        path,
        token_kinds=token_kinds,
        phases=phases,
        splits=splits,
        allow_empty=allow_empty,
    ).activations


def load_activation_dataset(
    path: str | Path,
    *,
    token_kinds: set[str] | None = None,
    phases: set[str] | None = None,
    splits: set[str] | None = None,
    allow_empty: bool = False,
) -> ActivationDataset:
    import torch
    from .storage import iter_activation_payloads

    shards = []
    token_group_rows: list[tuple[str, ...]] = []
    token_kinds = token_kinds or set()
    phases = phases or set()
    splits = splits or set()
    for _shard_name, payload in iter_activation_payloads(path):
        if "activations" in payload:
            meta = payload.get("meta") or {}
            if splits and activation_split(meta) not in splits:
                continue
            acts = payload["activations"].reshape(-1, payload["activations"].shape[-1])
            token_map = meta.get("token_map") or []
            keep = list(range(acts.shape[0]))
            if token_kinds or phases:
                keep = [
                    idx
                    for idx in range(acts.shape[0])
                    if token_matches_filters(token_info_for_index(token_map, idx), token_kinds=token_kinds, phases=phases)
                ]
                if not keep:
                    continue
                acts = acts[torch.tensor(keep, dtype=torch.long)]
            shards.append(acts)
            token_group_rows.extend(token_metric_groups(token_info_for_index(token_map, idx)) for idx in keep)
    if not shards:
        if allow_empty:
            return ActivationDataset(activations=None, token_groups=[], group_counts={})
        raise ValueError(f"no activation shards found in {path}")
    return ActivationDataset(
        activations=torch.cat(shards, dim=0),
        token_groups=token_group_rows,
        group_counts=count_group_memberships(token_group_rows),
    )


def activation_split(meta: dict[str, Any]) -> str:
    metadata = meta.get("metadata") or {}
    return str(metadata.get("split") or "sae_train")


def token_matches_filters(token: dict[str, Any] | None, *, token_kinds: set[str], phases: set[str]) -> bool:
    token = token or {}
    if token_kinds and token.get("kind") not in token_kinds:
        return False
    if phases and token.get("phase") not in phases:
        return False
    return True


def token_metric_groups(token: dict[str, Any] | None) -> tuple[str, ...]:
    token = token or {}
    kind = str(token.get("kind") or "unknown")
    phase = str(token.get("phase") or "unknown")
    role = str(token.get("role") or "unknown")
    groups = {
        "all",
        f"kind:{kind}",
        f"phase:{phase}",
        f"phase_kind:{phase}:{kind}",
    }
    if role != "unknown":
        groups.add(f"role:{role}")
        groups.add(f"role_kind:{role}:{kind}")
    if kind in {"image", "video"}:
        groups.add("media")
        groups.add(f"media:{kind}")
    if kind == "special":
        groups.add("special")
    return tuple(sorted(groups))


def count_group_memberships(rows: list[tuple[str, ...]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for groups in rows:
        for group in groups:
            counts[group] = counts.get(group, 0) + 1
    return dict(sorted(counts.items()))


def shard_path_for_record(output_dir: Path, idx: int, record_id: str) -> Path:
    return output_dir / shard_name_for_record(idx, record_id)


def shard_name_for_record(idx: int, record_id: str) -> str:
    leaf = record_id.rstrip("/").rsplit("/", 1)[-1] or record_id
    stem = Path(leaf).stem or leaf
    safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in stem).strip("._")
    safe_stem = (safe_stem or "record")[:80]
    digest = hashlib.sha1(record_id.encode("utf-8")).hexdigest()[:10]
    return f"{idx:06d}_{safe_stem}_{digest}.pt"


def parse_feature_ids(raw: str, feature_dim: int) -> list[int]:
    if not raw:
        return list(range(min(feature_dim, 128)))
    out = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value < 0 or value >= feature_dim:
            raise ValueError(f"feature id {value} outside [0, {feature_dim})")
        out.append(value)
    return out
