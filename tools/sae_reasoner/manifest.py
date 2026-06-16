from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal
from urllib.parse import urlparse
from urllib.request import urlopen

from .artifacts import iter_jsonl, repo_root, write_jsonl
from .storage import parse_s3_uri

MediaType = Literal["text", "image", "video"]


@dataclass(frozen=True)
class ManifestRecord:
    id: str
    media_type: MediaType
    prompt: str
    media_path: str | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_json(cls, obj: dict[str, Any], *, base_dir: Path | None = None) -> "ManifestRecord":
        missing = [key for key in ("id", "media_type", "prompt") if key not in obj]
        if missing:
            raise ValueError(f"manifest record missing required fields: {', '.join(missing)}")
        media_type = obj["media_type"]
        if media_type not in {"text", "image", "video"}:
            raise ValueError(f"unsupported media_type={media_type!r}; expected text, image, or video")
        media_path = obj.get("media_path")
        resolved_media_path: str | None = None
        if media_type != "text":
            if not media_path:
                raise ValueError(f"record {obj['id']!r} requires media_path for media_type={media_type!r}")
            candidate = Path(media_path)
            if is_remote_media_path(str(media_path)):
                resolved_media_path = str(media_path)
            else:
                if not candidate.is_absolute() and base_dir is not None:
                    candidate = base_dir / candidate
                if not candidate.exists():
                    raise FileNotFoundError(f"record {obj['id']!r} media_path does not exist: {candidate}")
                resolved_media_path = str(candidate.resolve())
        tags = obj.get("tags", [])
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise ValueError(f"record {obj['id']!r} tags must be a list of strings")
        metadata = obj.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError(f"record {obj['id']!r} metadata must be an object")
        return cls(
            id=str(obj["id"]),
            media_type=media_type,
            prompt=str(obj["prompt"]),
            media_path=resolved_media_path if resolved_media_path else (str(media_path) if media_path else None),
            tags=tuple(tags),
            metadata=metadata,
        )

    def to_json(self) -> dict[str, Any]:
        obj: dict[str, Any] = {
            "id": self.id,
            "media_type": self.media_type,
            "prompt": self.prompt,
            "tags": list(self.tags),
        }
        if self.media_path is not None:
            obj["media_path"] = self.media_path
        if self.metadata:
            obj["metadata"] = self.metadata
        return obj


def load_manifest(path: Path | str) -> list[ManifestRecord]:
    return list(iter_manifest(path))


def iter_manifest(path: Path | str) -> Iterator[ManifestRecord]:
    base = repo_root()
    for obj in iter_manifest_json(path):
        yield ManifestRecord.from_json(obj, base_dir=base)


def iter_manifest_json(path: Path | str) -> Iterator[dict[str, Any]]:
    raw = str(path)
    parsed = urlparse(raw)
    if parsed.scheme in {"http", "https"}:
        for line_no, line in enumerate((line.decode("utf-8") for line in urlopen(raw)), start=1):
            line = line.strip()
            if not line:
                continue
            obj = __import__("json").loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{raw}:{line_no}: expected object record")
            yield obj
        return
    if parsed.scheme == "s3":
        try:
            import boto3
        except Exception as exc:  # pragma: no cover - depends on optional env
            raise RuntimeError("S3 manifest input requires boto3.") from exc
        bucket, key = parse_s3_uri(raw)
        endpoint_url = __import__("os").environ.get("AWS_ENDPOINT_URL_S3") or __import__("os").environ.get("AWS_ENDPOINT_URL")
        kwargs = {"endpoint_url": endpoint_url} if endpoint_url else {}
        body = boto3.client("s3", **kwargs).get_object(Bucket=bucket, Key=key)["Body"]
        for line_no, line in enumerate((line.decode("utf-8") for line in body.iter_lines()), start=1):
            line = line.strip()
            if not line:
                continue
            obj = __import__("json").loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{raw}:{line_no}: expected object record")
            yield obj
        return
    yield from iter_jsonl(Path(raw))


def make_sample_manifest(output: Path) -> list[ManifestRecord]:
    samples = [
        ManifestRecord(
            id="robot_caption",
            media_type="image",
            media_path="cookbooks/cosmos3/reasoner/assets/robot_153.jpg",
            prompt="Caption the image in detail. Focus on the robot, objects, and likely task.",
            tags=("robotics", "caption", "spatial"),
        ),
        ManifestRecord(
            id="robot_planning",
            media_type="image",
            media_path="cookbooks/cosmos3/reasoner/assets/robot_planning.png",
            prompt="What task is the robot likely performing, and what should happen next?",
            tags=("robotics", "planning", "next_action"),
        ),
        ManifestRecord(
            id="grounding_2d",
            media_type="image",
            media_path="cookbooks/cosmos3/reasoner/assets/grounding_2d.png",
            prompt="Identify the main relevant objects and describe their spatial relations.",
            tags=("grounding", "spatial"),
        ),
        ManifestRecord(
            id="physical_plausibility",
            media_type="video",
            media_path="cookbooks/cosmos3/reasoner/assets/physical_plausibility.mp4",
            prompt="Is the physical motion in this video plausible? Explain using visible evidence.",
            tags=("video", "physics", "plausibility"),
        ),
        ManifestRecord(
            id="temporal_localization",
            media_type="video",
            media_path="cookbooks/cosmos3/reasoner/assets/temporal_localization_1.mp4",
            prompt="Describe the key temporal events in order and mention when the main action occurs.",
            tags=("video", "temporal"),
        ),
        ManifestRecord(
            id="text_control_contact",
            media_type="text",
            prompt=(
                "A robot arm moves a block next to a bowl, pauses, then pushes it. "
                "What physical concepts would you track to decide whether contact occurred?"
            ),
            tags=("text", "control", "physics"),
        ),
    ]
    write_jsonl(output, [sample.to_json() for sample in samples])
    return samples


def is_remote_media_path(path: str) -> bool:
    return urlparse(path).scheme in {"hf", "http", "https", "s3"}
