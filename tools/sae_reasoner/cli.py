from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any
from urllib.parse import quote

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
    p.add_argument("--source", choices=["recipe", "hf-files", "hf-dataset", "s3-prefix", "jsonl"], default="recipe")
    p.add_argument("--recipe", default="physicalai-driving")
    p.add_argument("--hf-repo-id", default=None)
    p.add_argument("--hf-split", default="train")
    p.add_argument("--s3-uri", default=None)
    p.add_argument("--input-uri", default=None)
    p.add_argument("--include-glob", action="append", default=[])
    p.add_argument("--prompt", default=None)
    p.add_argument("--media-type", choices=["auto", "text", "image", "video"], default="auto")
    p.add_argument("--max-records", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split-ratios", default="sae_train=0.90,feature_labeling=0.05,steering_eval=0.05")
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

    p = sub.add_parser("collect-activations", help="collect prefill activations")
    add_model_args(p)
    add_prompt_args(p)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument(
        "--output-dir",
        required=True,
        help="Local directory or S3 URI such as s3://bucket/prefix. S3 credentials are read from AWS env vars.",
    )
    p.add_argument("--max-examples", type=int, default=None)
    p.set_defaults(func=cmd_collect_activations)

    p = sub.add_parser("train-sae", help="train a SAE from activation shards")
    p.add_argument("--activation-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expansion-factor", type=int, default=16)
    p.add_argument("--top-k", type=int, default=32)
    p.add_argument("--recon-loss", choices=["mse", "l1", "smooth_l1"], default="mse")
    p.add_argument("--feature-l1-coeff", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.set_defaults(func=cmd_train_sae)

    p = sub.add_parser("find-features", help="rank top activating records/features")
    p.add_argument("--activation-dir", type=Path, required=True)
    p.add_argument("--sae", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--feature-ids", type=str, default="")
    p.set_defaults(func=cmd_find_features)

    p = sub.add_parser("render-feature-report", help="render feature examples as a standalone HTML report")
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("outputs/sae_reasoner/reports/features.html"))
    p.add_argument("--title", default="Cosmos SAE Feature Browser")
    p.set_defaults(func=cmd_render_feature_report)

    p = sub.add_parser("find-neighbors", help="find nearest activation-token neighbors by cosine similarity")
    p.add_argument("--activation-dir", type=Path, required=True)
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
    p.add_argument("--mode", choices=["multiply", "clamp"], default="multiply")
    p.add_argument("--prompt", required=True)
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
            prompt=args.prompt,
            media_type=args.media_type,
            max_records=args.max_records,
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
                "num_records": len(records),
                "split_counts": split_counts,
                "media_counts": media_counts,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


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
    runtime = CosmosReasonerRuntime(
        args.model_id,
        device=args.device,
        dtype=args.dtype,
        init_mode=args.init_mode,
    ).load()
    metadata: list[dict[str, Any]] = []
    for idx, record in enumerate(iter_manifest(args.manifest)):
        if args.max_examples is not None and idx >= args.max_examples:
            break
        hidden, meta = runtime.collect_prefill(
            record,
            layer=args.layer,
            prompt_format=args.prompt_format,
            system_prompt=args.system_prompt,
        )
        shard_name = shard_name_for_record(idx, record.id)
        shard_uri = store.write_torch(shard_name, {"activations": hidden, "meta": meta})
        meta["shard"] = shard_name
        meta["shard_uri"] = shard_uri
        metadata.append(compact_activation_meta(meta))
        print(json.dumps({"collected": record.id, "tokens": meta["num_tokens"], "shard": shard_name, "uri": shard_uri}))
    metadata_text = "".join(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n" for record in metadata)
    store.write_text("metadata.jsonl", metadata_text)
    return 0


def cmd_train_sae(args: argparse.Namespace) -> int:
    from .sae import save_sae, train_sae_from_tensor

    activations = load_activation_matrix(args.activation_dir)
    sae, metrics = train_sae_from_tensor(
        activations,
        expansion_factor=args.expansion_factor,
        top_k=args.top_k,
        recon_loss=args.recon_loss,
        feature_l1_coeff=args.feature_l1_coeff,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    ensure_dir(args.output.parent)
    save_sae(
        str(args.output),
        sae,
        metadata={
            "metrics": metrics,
            "recon_loss": args.recon_loss,
            "feature_l1_coeff": args.feature_l1_coeff,
        },
    )
    write_jsonl(args.output.with_suffix(".metrics.jsonl"), metrics)
    print(json.dumps({"output": str(args.output), "num_activations": int(activations.shape[0]), "metrics": metrics[-1]}, indent=2))
    return 0


def cmd_find_features(args: argparse.Namespace) -> int:
    import torch

    from .sae import load_sae

    sae = load_sae(str(args.sae))
    feature_ids = parse_feature_ids(args.feature_ids, sae.config.feature_dim)
    records: list[dict[str, Any]] = []
    for shard_path in sorted(args.activation_dir.glob("*.pt")):
        if shard_path.name == "metadata.pt":
            continue
        payload = torch.load(shard_path, map_location="cpu")
        acts = payload["activations"].float()
        meta = payload.get("meta", {})
        token_map = meta.get("token_map") or []
        with torch.no_grad():
            features = sae.encode(acts)
        for feature_id in feature_ids:
            values = features[:, feature_id]
            top_vals, top_idx = torch.topk(values, k=min(args.top_n, values.numel()))
            for value, token_idx in zip(top_vals.tolist(), top_idx.tolist()):
                if value <= 0:
                    continue
                records.append(
                    {
                        "feature_id": feature_id,
                        "activation": float(value),
                        "token_index": int(token_idx),
                        "token_info": token_info_for_index(token_map, token_idx),
                        "record_id": meta.get("id"),
                        "prompt": meta.get("prompt"),
                        "media_type": meta.get("media_type"),
                        "media_path": meta.get("media_path"),
                        "tags": meta.get("tags", []),
                        "metadata": meta.get("metadata", {}),
                        "shard": shard_path.name,
                    }
                )
    records.sort(key=lambda r: r["activation"], reverse=True)
    write_jsonl(args.output, records[: max(args.top_n, len(feature_ids) * args.top_n)])
    print(json.dumps({"output": str(args.output), "num_records": len(records)}, indent=2))
    return 0


def compact_activation_meta(meta: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in meta.items() if key != "token_map"}


def token_info_for_index(token_map: list[dict[str, Any]], token_index: int) -> dict[str, Any] | None:
    if 0 <= token_index < len(token_map):
        return token_map[token_index]
    return None


def load_activation_examples(path: Path, *, max_tokens: int, seed: int) -> tuple[list[dict[str, Any]], torch.Tensor]:
    import torch

    if max_tokens <= 1:
        raise ValueError("--max-tokens must be greater than 1")
    rng = random.Random(seed)
    examples: list[dict[str, Any]] = []
    vectors: list[torch.Tensor] = []
    seen = 0
    for shard_path in sorted(path.glob("*.pt")):
        payload = torch.load(shard_path, map_location="cpu")
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
            example = activation_example(meta, shard_path.name, token_map, token_index)
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

    runtime = CosmosReasonerRuntime(
        args.model_id,
        device=args.device,
        dtype=args.dtype,
        init_mode=args.init_mode,
    ).load()
    sae = load_sae(str(args.sae), map_location="cpu")
    baseline = runtime.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        prompt_format=args.prompt_format,
        system_prompt=args.system_prompt,
    )
    hook = FeatureSteeringHook(
        sae=sae,
        feature_id=args.feature_id,
        multiplier=args.multiplier,
        scope=args.scope,
        mode=args.mode,
    )
    steered = runtime.generate(
        prompt=args.prompt,
        layer=args.layer,
        edit_fn=hook,
        max_new_tokens=args.max_new_tokens,
        prompt_format=args.prompt_format,
        system_prompt=args.system_prompt,
    )
    result = {
        "prompt": args.prompt,
        "layer": args.layer,
        "feature_id": args.feature_id,
        "multiplier": args.multiplier,
        "scope": args.scope,
        "mode": args.mode,
        "prompt_format": args.prompt_format,
        "system_prompt": args.system_prompt,
        "baseline": baseline,
        "steered": steered,
    }
    ensure_dir(args.output.parent)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output)}, indent=2))
    return 0


def load_activation_matrix(path: Path) -> torch.Tensor:
    import torch

    shards = []
    for shard_path in sorted(path.glob("*.pt")):
        payload = torch.load(shard_path, map_location="cpu")
        if "activations" in payload:
            shards.append(payload["activations"].float())
    if not shards:
        raise ValueError(f"no activation shards found in {path}")
    return torch.cat([x.reshape(-1, x.shape[-1]) for x in shards], dim=0)


def shard_path_for_record(output_dir: Path, idx: int, record_id: str) -> Path:
    return output_dir / shard_name_for_record(idx, record_id)


def shard_name_for_record(idx: int, record_id: str) -> str:
    safe_id = quote(record_id, safe="")
    return f"{idx:06d}_{safe_id}.pt"


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
