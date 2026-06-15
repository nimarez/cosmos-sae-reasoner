from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

from .artifacts import ensure_dir, repo_root
from .storage import parse_s3_uri


def materialize_media_path(media_path: str, *, cache_dir: Path | None = None) -> str:
    """Return a local path for local, HF, HTTP(S), or S3 media.

    Large corpora should keep manifests as remote URIs. This downloads only the
    record currently being processed and reuses the cached file on later runs.
    """

    parsed = urlparse(media_path)
    if parsed.scheme == "":
        return media_path
    cache = cache_dir or default_media_cache_dir()
    if parsed.scheme == "hf":
        return materialize_hf_uri(media_path, cache)
    if parsed.scheme == "s3":
        return materialize_s3_uri(media_path, cache)
    if parsed.scheme in {"http", "https"}:
        return materialize_http_uri(media_path, cache)
    raise ValueError(f"unsupported media URI scheme for {media_path!r}")


def default_media_cache_dir() -> Path:
    raw = os.environ.get("COSMOS_SAE_MEDIA_CACHE")
    if raw:
        return ensure_dir(Path(raw).expanduser())
    return ensure_dir(repo_root() / ".cache" / "sae_reasoner" / "media")


def materialize_hf_uri(uri: str, cache_dir: Path) -> str:
    parsed = urlparse(uri)
    if parsed.netloc != "dataset":
        raise ValueError("HF media URIs must use hf://dataset/<namespace>/<repo>/<path>")
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 3:
        raise ValueError("HF media URIs must use hf://dataset/<namespace>/<repo>/<path>")
    repo_id = "/".join(parts[:2])
    filename = "/".join(parts[2:])
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError("HF media download requires huggingface_hub.") from exc
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=str(cache_dir / "hf" / repo_id.replace("/", "--")),
    )


def materialize_s3_uri(uri: str, cache_dir: Path) -> str:
    bucket, key = parse_s3_uri(uri)
    local_path = cache_dir / "s3" / bucket / key
    if local_path.exists():
        return str(local_path)
    ensure_dir(local_path.parent)
    try:
        import boto3
    except Exception as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError("S3 media download requires boto3.") from exc
    endpoint_url = os.environ.get("AWS_ENDPOINT_URL_S3") or os.environ.get("AWS_ENDPOINT_URL")
    kwargs = {"endpoint_url": endpoint_url} if endpoint_url else {}
    boto3.client("s3", **kwargs).download_file(bucket, key, str(local_path))
    return str(local_path)


def materialize_http_uri(uri: str, cache_dir: Path) -> str:
    parsed = urlparse(uri)
    suffix = Path(parsed.path).suffix
    local_path = cache_dir / "http" / f"{hashlib.sha256(uri.encode('utf-8')).hexdigest()}{suffix}"
    if not local_path.exists():
        ensure_dir(local_path.parent)
        urlretrieve(uri, local_path)
    return str(local_path)
