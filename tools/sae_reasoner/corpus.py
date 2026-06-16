from __future__ import annotations

import fnmatch
import json
import os
import random
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from string import Formatter
from typing import Any, Iterable
from urllib.parse import quote, urlparse
from urllib.request import urlopen

from .artifacts import write_jsonl
from .manifest import MediaType, ManifestRecord
from .storage import parse_s3_uri

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm"}


@dataclass(frozen=True)
class CorpusRecipe:
    name: str
    source: str
    repo_id: str | None = None
    include_globs: tuple[str, ...] = ()
    member_globs: tuple[str, ...] = ()
    prompt: str = ""
    tags: tuple[str, ...] = ()
    notes: str = ""


RECIPES: dict[str, CorpusRecipe] = {
    "physicalai-driving": CorpusRecipe(
        name="physicalai-driving",
        source="hf-files",
        repo_id="nvidia/PhysicalAI-WorldModel-Synthetic-Autonomous-Driving-Scenarios",
        include_globs=("*/*/video/*.mp4",),
        prompt=(
            "Describe the traffic scene, the relevant agents, and the likely next physical events. "
            "Use visible evidence and avoid speculating beyond the clip."
        ),
        tags=("cosmos3", "physicalai", "driving", "video", "sae_train"),
        notes="Direct MP4 files with sidecar descriptions in the HF repo.",
    ),
    "physicalai-vantage": CorpusRecipe(
        name="physicalai-vantage",
        source="hf-files",
        repo_id="nvidia/PhysicalAI-VANTAGE-Bench",
        include_globs=("data/*/sequence_*/images/*.jpg", "data/*/sequence_*/images/*.jpeg", "data/*/sequence_*/images/*.png"),
        prompt=(
            "Describe the scene with attention to spatial relations, visible objects, and physical affordances. "
            "If a task is implied by the path, answer in that style without using hidden annotations."
        ),
        tags=("physicalai", "vantage", "image", "spatial", "sae_train"),
        notes="Evaluation-style image tasks. Keep held-out portions separate from training if used for eval.",
    ),
    "robotics-bridge": CorpusRecipe(
        name="robotics-bridge",
        source="hf-files",
        repo_id="nvidia/BridgeData2_LeRobot_v3",
        include_globs=("videos/*/*/*.mp4",),
        prompt=(
            "Describe the robot manipulation episode, including the visible objects, end-effector motion, "
            "contact events, and likely task outcome."
        ),
        tags=("physicalai", "robotics", "bridge", "lerobot", "video", "sae_train"),
        notes="Loose MP4 files from the BridgeData2 LeRobot release.",
    ),
    "robotics-bridge-captions": CorpusRecipe(
        name="robotics-bridge-captions",
        source="hf-files",
        repo_id="nvidia/BridgeData2-Subset-Synthetic-Captions",
        include_globs=("sft_dataset_bridge/train/videos/*.mp4", "sft_dataset_bridge/val/videos/*.mp4"),
        prompt=(
            "Describe the robot manipulation clip in detail. Focus on object state, gripper pose, "
            "physical interactions, and what action should happen next."
        ),
        tags=("physicalai", "robotics", "bridge", "synthetic_captions", "video", "sae_train"),
        notes="BridgeData2 subset with loose MP4 clips and synthetic captions.",
    ),
    "robotics-libero": CorpusRecipe(
        name="robotics-libero",
        source="hf-files",
        repo_id="nvidia/LIBERO_LeRobot_v3",
        include_globs=("*/videos/*/*/*.mp4",),
        prompt=(
            "Describe the robot tabletop task, visible objects, gripper trajectory, contact dynamics, "
            "and the likely next step needed to complete the instruction."
        ),
        tags=("physicalai", "robotics", "libero", "lerobot", "video", "sae_train"),
        notes="Loose MP4 files from the LIBERO LeRobot release.",
    ),
    "physicalai-robotsim": CorpusRecipe(
        name="physicalai-robotsim",
        source="hf-tar-s3",
        repo_id="nvidia/PhysicalAI-WorldModel-Synthetic-Embodied-Robot-Scenes",
        include_globs=("data/*/*/*.tar",),
        member_globs=("*.mp4",),
        prompt=(
            "Describe the robot simulation clip, including the embodiment, scene objects, robot motion, "
            "contacts, collisions, and likely physical outcome."
        ),
        tags=("cosmos3", "physicalai", "robotics", "robotsim", "video", "sae_train"),
        notes="Cosmos RobotSim SDG tar shards. Materialize a bounded sample to S3 before activation collection.",
    ),
}


@dataclass(frozen=True)
class BuildCorpusConfig:
    output: Path
    source: str
    recipe: str | None = None
    hf_repo_id: str | None = None
    hf_split: str = "train"
    s3_uri: str | None = None
    input_uri: str | None = None
    include_globs: tuple[str, ...] = ()
    member_globs: tuple[str, ...] = ()
    prompt: str | None = None
    media_type: str = "auto"
    max_records: int = 1000
    max_shards: int | None = 1
    max_shard_gb: float | None = 1.0
    seed: int = 0
    split_ratios: tuple[tuple[str, float], ...] = (("sae_train", 0.9), ("feature_labeling", 0.05), ("steering_eval", 0.05))
    id_field: str | None = None
    text_field: str | None = None
    prompt_field: str | None = None
    media_field: str | None = None
    shuffle_buffer: int = 10000
    min_text_chars: int = 64
    max_text_chars: int = 8000


def build_corpus_manifest(config: BuildCorpusConfig) -> list[dict[str, Any]]:
    if config.source == "recipe":
        if not config.recipe:
            raise ValueError("--recipe is required when --source recipe")
        recipe = RECIPES[config.recipe]
        if recipe.source == "hf-files":
            records = build_from_hf_files(
                repo_id=required(recipe.repo_id, "recipe repo_id"),
                include_globs=recipe.include_globs,
                prompt=config.prompt or recipe.prompt,
                tags=recipe.tags,
                max_records=config.max_records,
                seed=config.seed,
                split_ratios=config.split_ratios,
                media_type=config.media_type,
            )
        elif recipe.source == "hf-tar-s3":
            records = build_from_hf_tar_s3(
                repo_id=required(recipe.repo_id, "recipe repo_id"),
                shard_globs=recipe.include_globs,
                member_globs=config.member_globs or recipe.member_globs,
                prompt=config.prompt or recipe.prompt,
                tags=recipe.tags,
                s3_uri=required(config.s3_uri, "--s3-uri"),
                max_records=config.max_records,
                max_shards=config.max_shards,
                max_shard_gb=config.max_shard_gb,
                seed=config.seed,
                split_ratios=config.split_ratios,
                media_type=config.media_type,
            )
        else:
            raise ValueError(f"recipe {recipe.name!r} has unsupported source {recipe.source!r}")
    elif config.source == "hf-files":
        records = build_from_hf_files(
            repo_id=required(config.hf_repo_id, "--hf-repo-id"),
            include_globs=config.include_globs,
            prompt=required(config.prompt, "--prompt"),
            tags=("hf", "external"),
            max_records=config.max_records,
            seed=config.seed,
            split_ratios=config.split_ratios,
            media_type=config.media_type,
        )
    elif config.source == "hf-dataset":
        records = build_from_hf_dataset(config)
    elif config.source == "hf-tar-s3":
        records = build_from_hf_tar_s3(
            repo_id=required(config.hf_repo_id, "--hf-repo-id"),
            shard_globs=config.include_globs,
            member_globs=config.member_globs or ("*.mp4",),
            prompt=required(config.prompt, "--prompt"),
            tags=("hf", "external", "materialized"),
            s3_uri=required(config.s3_uri, "--s3-uri"),
            max_records=config.max_records,
            max_shards=config.max_shards,
            max_shard_gb=config.max_shard_gb,
            seed=config.seed,
            split_ratios=config.split_ratios,
            media_type=config.media_type,
        )
    elif config.source == "s3-prefix":
        records = build_from_s3_prefix(config)
    elif config.source == "jsonl":
        records = build_from_jsonl_uri(required(config.input_uri, "--input-uri"), config)
    else:
        raise ValueError(f"unsupported source {config.source!r}")
    write_jsonl(config.output, records)
    return records


def build_from_hf_files(
    *,
    repo_id: str,
    include_globs: tuple[str, ...],
    prompt: str,
    tags: tuple[str, ...],
    max_records: int,
    seed: int,
    split_ratios: tuple[tuple[str, float], ...],
    media_type: str,
) -> list[dict[str, Any]]:
    try:
        from huggingface_hub import HfApi
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("HF file manifests require huggingface_hub.") from exc
    files = HfApi().list_repo_files(repo_id, repo_type="dataset")
    candidates = [path for path in files if match_any(path, include_globs) and infer_media_type(path, media_type) != "text"]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    records: list[dict[str, Any]] = []
    for idx, path in enumerate(candidates[:max_records]):
        split = split_for_index(idx, max_records, split_ratios)
        inferred = infer_media_type(path, media_type)
        record_tags = sorted(set(tags + (split,) + tags_from_path(path)))
        records.append(
            make_manifest_dict(
                record_id=f"hf:{repo_id}:{path}",
                media_type=inferred,
                prompt=render_template(prompt, path=path, stem=Path(path).stem, repo_id=repo_id),
                media_path=f"hf://dataset/{repo_id}/{path}",
                tags=record_tags,
                metadata={
                    "source": "hf-files",
                    "source_uri": f"hf://dataset/{repo_id}/{path}",
                    "repo_id": repo_id,
                    "path": path,
                    "split": split,
                },
            )
        )
    return records


def build_from_hf_dataset(config: BuildCorpusConfig) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("HF streaming datasets require the datasets package.") from exc
    repo_id = required(config.hf_repo_id, "--hf-repo-id")
    dataset = load_dataset(repo_id, split=config.hf_split, streaming=True)
    if config.shuffle_buffer > 0:
        dataset = dataset.shuffle(seed=config.seed, buffer_size=config.shuffle_buffer)
    records: list[dict[str, Any]] = []
    for row_idx, row in enumerate(dataset):
        if len(records) >= config.max_records:
            break
        record = manifest_from_dataset_row(row_idx, row, config, repo_id)
        if record is not None:
            records.append(record)
    return records


def manifest_from_dataset_row(row_idx: int, row: dict[str, Any], config: BuildCorpusConfig, repo_id: str) -> dict[str, Any] | None:
    split = split_for_index(row_idx, config.max_records, config.split_ratios)
    record_id = str(row.get(config.id_field, f"hf:{repo_id}:{config.hf_split}:{row_idx}")) if config.id_field else f"hf:{repo_id}:{config.hf_split}:{row_idx}"
    media_path = value_as_path(row.get(config.media_field)) if config.media_field else None
    if media_path:
        media_type = infer_media_type(media_path, config.media_type)
        prompt = prompt_from_row(row, config)
        if not prompt:
            return None
    else:
        media_type = "text"
        prompt = prompt_from_row(row, config)
        if not prompt:
            return None
        prompt = prompt[: config.max_text_chars]
        if len(prompt) < config.min_text_chars:
            return None
    return make_manifest_dict(
        record_id=record_id,
        media_type=media_type,
        prompt=prompt,
        media_path=media_path,
        tags=sorted({"hf", "external", split, media_type}),
        metadata={"source": "hf-dataset", "repo_id": repo_id, "split": split, "row_index": row_idx},
    )


def build_from_hf_tar_s3(
    *,
    repo_id: str,
    shard_globs: tuple[str, ...],
    member_globs: tuple[str, ...],
    prompt: str,
    tags: tuple[str, ...],
    s3_uri: str,
    max_records: int,
    max_shards: int | None,
    max_shard_gb: float | None,
    seed: int,
    split_ratios: tuple[tuple[str, float], ...],
    media_type: str,
) -> list[dict[str, Any]]:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("HF tar materialization requires huggingface_hub.") from exc
    try:
        import boto3
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("HF tar materialization to S3 requires boto3.") from exc

    bucket, prefix = parse_s3_uri(s3_uri)
    endpoint_url = os.environ.get("AWS_ENDPOINT_URL_S3") or os.environ.get("AWS_ENDPOINT_URL")
    kwargs = {"endpoint_url": endpoint_url} if endpoint_url else {}
    s3 = boto3.client("s3", **kwargs)
    api = HfApi()
    max_shard_bytes = int(max_shard_gb * 1_000_000_000) if max_shard_gb and max_shard_gb > 0 else None
    shards = list_hf_tar_shards(api, repo_id=repo_id, shard_globs=shard_globs, max_shard_bytes=max_shard_bytes)
    rng = random.Random(seed)
    rng.shuffle(shards)
    if max_shards is not None and max_shards > 0:
        shards = shards[:max_shards]

    records: list[dict[str, Any]] = []
    repo_slug = repo_id.replace("/", "--")
    for shard_idx, shard in enumerate(shards):
        if len(records) >= max_records:
            break
        shard_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=shard)
        with tarfile.open(shard_path, mode="r:*") as tar:
            members = [
                member
                for member in tar.getmembers()
                if member.isfile() and match_any(member.name, member_globs) and infer_media_type(member.name, media_type) != "text"
            ]
            rng.shuffle(members)
            for member in members:
                if len(records) >= max_records:
                    break
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                split = split_for_index(len(records), max_records, split_ratios)
                key = s3_key_for_tar_member(prefix, repo_slug, shard, member.name)
                with extracted:
                    s3.upload_fileobj(
                        extracted,
                        bucket,
                        key,
                        ExtraArgs={"ContentType": content_type_for_path(member.name)},
                    )
                media_path = f"s3://{bucket}/{key}"
                record_tags = sorted(set(tags + (split,) + tags_from_path(shard) + tags_from_path(member.name)))
                records.append(
                    make_manifest_dict(
                        record_id=f"hf-tar:{repo_id}:{shard}:{member.name}",
                        media_type=infer_media_type(member.name, media_type),
                        prompt=render_template(prompt, path=member.name, stem=Path(member.name).stem, repo_id=repo_id, shard=shard),
                        media_path=media_path,
                        tags=record_tags,
                        metadata={
                            "source": "hf-tar-s3",
                            "repo_id": repo_id,
                            "shard": shard,
                            "member": member.name,
                            "source_uri": f"hf://dataset/{repo_id}/{shard}#{member.name}",
                            "bucket": bucket,
                            "key": key,
                            "split": split,
                            "shard_index": shard_idx,
                        },
                    )
                )
    return records


def list_hf_tar_shards(api: Any, *, repo_id: str, shard_globs: tuple[str, ...], max_shard_bytes: int | None) -> list[str]:
    try:
        items = api.list_repo_tree(repo_id, repo_type="dataset", recursive=True, expand=True)
    except Exception:
        files = api.list_repo_files(repo_id, repo_type="dataset")
        return [path for path in files if match_any(path, shard_globs) and path.endswith(".tar")]
    shards: list[str] = []
    for item in items:
        path = getattr(item, "path", None)
        size = getattr(item, "size", None)
        if not path or not path.endswith(".tar") or not match_any(path, shard_globs):
            continue
        if max_shard_bytes is not None and size is not None and size > max_shard_bytes:
            continue
        shards.append(path)
    return shards


def build_from_s3_prefix(config: BuildCorpusConfig) -> list[dict[str, Any]]:
    s3_uri = required(config.s3_uri, "--s3-uri")
    bucket, prefix = parse_s3_uri(s3_uri)
    try:
        import boto3
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("S3 prefix manifests require boto3.") from exc
    endpoint_url = __import__("os").environ.get("AWS_ENDPOINT_URL_S3") or __import__("os").environ.get("AWS_ENDPOINT_URL")
    kwargs = {"endpoint_url": endpoint_url} if endpoint_url else {}
    client = boto3.client("s3", **kwargs)
    rng = random.Random(config.seed)
    candidates: list[str] = []
    seen = 0
    token = None
    while True:
        request: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if token:
            request["ContinuationToken"] = token
        response = client.list_objects_v2(**request)
        for obj in response.get("Contents", []):
            key = obj["Key"]
            if not match_any(key, config.include_globs) or infer_media_type(key, config.media_type) == "text":
                continue
            seen += 1
            if len(candidates) < config.max_records:
                candidates.append(key)
            else:
                replacement = rng.randrange(seen)
                if replacement < config.max_records:
                    candidates[replacement] = key
        if not response.get("IsTruncated"):
            break
        token = response.get("NextContinuationToken")
    rng.shuffle(candidates)
    prompt = required(config.prompt, "--prompt")
    records: list[dict[str, Any]] = []
    for idx, key in enumerate(candidates):
        split = split_for_index(idx, len(candidates), config.split_ratios)
        records.append(
            make_manifest_dict(
                record_id=f"s3:{bucket}:{key}",
                media_type=infer_media_type(key, config.media_type),
                prompt=render_template(prompt, path=key, stem=Path(key).stem, repo_id=bucket),
                media_path=f"s3://{bucket}/{key}",
                tags=sorted({"s3", "external", split} | set(tags_from_path(key))),
                metadata={"source": "s3-prefix", "source_uri": f"s3://{bucket}/{key}", "bucket": bucket, "key": key, "split": split},
            )
        )
    return records


def s3_key_for_tar_member(prefix: str, repo_slug: str, shard: str, member: str) -> str:
    shard_stem = Path(shard).stem
    parts = [
        "media",
        repo_slug,
        safe_s3_path_part(shard_stem),
        *safe_s3_member_parts(member),
    ]
    suffix = "/".join(part for part in parts if part)
    return f"{prefix.rstrip('/')}/{suffix}" if prefix else suffix


def safe_s3_member_parts(path: str) -> list[str]:
    parts: list[str] = []
    for part in PurePosixPath(path.replace("\\", "/")).parts:
        if part in {"", ".", "..", "/"}:
            continue
        parts.append(safe_s3_path_part(part))
    return parts


def safe_s3_path_part(value: str) -> str:
    return quote(value, safe="._-")


def content_type_for_path(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".mp4", ".m4v"}:
        return "video/mp4"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    return "application/octet-stream"


def build_from_jsonl_uri(uri: str, config: BuildCorpusConfig) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, obj in enumerate(iter_jsonl_uri(uri)):
        if len(records) >= config.max_records:
            break
        obj = dict(obj)
        obj.setdefault("id", f"jsonl:{idx}")
        metadata = dict(obj.get("metadata") or {})
        metadata.setdefault("source", "jsonl")
        metadata.setdefault("source_uri", uri)
        metadata.setdefault("row_index", idx)
        obj["metadata"] = metadata
        record = ManifestRecord.from_json(obj)
        records.append(record.to_json())
    return records


def iter_jsonl_uri(uri: str) -> Iterable[dict[str, Any]]:
    parsed = urlparse(uri)
    if parsed.scheme == "":
        stream: Iterable[str] = Path(uri).open("r", encoding="utf-8")
    elif parsed.scheme in {"http", "https"}:
        stream = (line.decode("utf-8") for line in urlopen(uri))
    elif parsed.scheme == "s3":
        bucket, key = parse_s3_uri(uri)
        try:
            import boto3
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("S3 JSONL input requires boto3.") from exc
        endpoint_url = __import__("os").environ.get("AWS_ENDPOINT_URL_S3") or __import__("os").environ.get("AWS_ENDPOINT_URL")
        kwargs = {"endpoint_url": endpoint_url} if endpoint_url else {}
        body = boto3.client("s3", **kwargs).get_object(Bucket=bucket, Key=key)["Body"]
        stream = (line.decode("utf-8") for line in body.iter_lines())
    else:
        raise ValueError(f"unsupported JSONL URI scheme for {uri!r}")
    for line_no, line in enumerate(stream, start=1):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise ValueError(f"{uri}:{line_no}: expected object JSONL row")
        yield obj


def make_manifest_dict(
    *,
    record_id: str,
    media_type: MediaType,
    prompt: str,
    media_path: str | None,
    tags: list[str],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    record = ManifestRecord(
        id=record_id,
        media_type=media_type,
        prompt=prompt,
        media_path=media_path,
        tags=tuple(tags),
        metadata=metadata,
    )
    return record.to_json()


def prompt_from_row(row: dict[str, Any], config: BuildCorpusConfig) -> str | None:
    if config.prompt_field and row.get(config.prompt_field) is not None:
        return str(row[config.prompt_field])
    if config.prompt:
        return render_template(config.prompt, **{key: stringify_value(value) for key, value in row.items()})
    if config.text_field and row.get(config.text_field) is not None:
        return str(row[config.text_field])
    return None


def value_as_path(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("path", "url", "uri", "filename"):
            if value.get(key):
                return str(value[key])
    path = getattr(value, "filename", None)
    return str(path) if path else None


def infer_media_type(path: str, requested: str) -> MediaType:
    if requested in {"text", "image", "video"}:
        return requested  # type: ignore[return-value]
    suffix = Path(urlparse(path).path).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    return "text"


def match_any(path: str, patterns: tuple[str, ...]) -> bool:
    return not patterns or any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def split_for_index(index: int, total: int, ratios: tuple[tuple[str, float], ...]) -> str:
    if not ratios:
        return "sae_train"
    if total <= 0:
        return ratios[0][0]
    position = (index + 0.5) / total
    acc = 0.0
    for name, ratio in ratios:
        acc += ratio
        if position <= acc:
            return name
    return ratios[-1][0]


def parse_split_ratios(raw: str) -> tuple[tuple[str, float], ...]:
    pairs: list[tuple[str, float]] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        name, sep, value = part.partition("=")
        if not sep or not name:
            raise ValueError(f"invalid split ratio {part!r}; expected name=float")
        pairs.append((name, float(value)))
    total = sum(value for _name, value in pairs)
    if total <= 0:
        raise ValueError("split ratios must sum to a positive value")
    return tuple((name, value / total) for name, value in pairs)


def tags_from_path(path: str) -> tuple[str, ...]:
    parts = [part for part in Path(path).parts if part and part not in {".", "/"}]
    useful = []
    for part in parts[:4]:
        clean = part.lower().replace("_", "-")
        if clean and not clean.endswith((".jpg", ".jpeg", ".png", ".mp4", ".json")):
            useful.append(clean)
    return tuple(useful)


def render_template(template: str, **values: str) -> str:
    safe_values = {key: values.get(key, "") for _literal, key, _fmt, _conv in Formatter().parse(template) if key}
    safe_values.update(values)
    return template.format_map(_SafeFormatDict(safe_values))


def stringify_value(value: Any) -> str:
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=True, sort_keys=True)[:1000]


def required(value: str | None, name: str) -> str:
    if not value:
        raise ValueError(f"{name} is required")
    return value


class _SafeFormatDict(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return ""
