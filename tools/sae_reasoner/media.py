from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse
from urllib.request import urlretrieve

from .artifacts import ensure_dir, repo_root
from .storage import parse_s3_uri, s3_client


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
    if parsed.scheme == "hf-tar-range":
        return materialize_hf_tar_range_uri(media_path, cache)
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


def materialize_hf_tar_range_uri(uri: str, cache_dir: Path) -> str:
    parsed = urlparse(uri)
    if parsed.netloc != "dataset":
        raise ValueError("HF tar range URIs must use hf-tar-range://dataset/<namespace>/<repo>/<shard>?offset=N&size=N&name=...")
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 3:
        raise ValueError("HF tar range URIs must use hf-tar-range://dataset/<namespace>/<repo>/<shard>?offset=N&size=N&name=...")
    query = parse_qs(parsed.query)
    try:
        offset = int(query["offset"][0])
        size = int(query["size"][0])
    except Exception as exc:
        raise ValueError(f"HF tar range URI is missing integer offset/size: {uri!r}") from exc
    if offset < 0 or size <= 0:
        raise ValueError(f"HF tar range URI has invalid offset/size: {uri!r}")
    repo_id = "/".join(parts[:2])
    shard = "/".join(parts[2:])
    name = unquote(query.get("name", [Path(shard).name])[0])
    suffix = Path(name).suffix or ".bin"
    digest = hashlib.sha256(uri.encode("utf-8")).hexdigest()
    local_path = cache_dir / "hf-tar-range" / repo_id.replace("/", "--") / f"{digest}{suffix}"
    if local_path.exists() and local_path.stat().st_size == size:
        return str(local_path)
    ensure_dir(local_path.parent)
    tmp = local_path.with_name(local_path.name + ".tmp")
    try:
        from huggingface_hub import hf_hub_url
        import requests
    except Exception as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError("HF tar range media download requires huggingface_hub and requests.") from exc
    url = hf_hub_url(repo_id=repo_id, filename=shard, repo_type="dataset")
    end = offset + size - 1
    headers = {"Range": f"bytes={offset}-{end}", **hf_auth_headers()}
    data = http_get_with_retries(requests, url, headers=headers, expected_size=size)
    tmp.write_bytes(data)
    os.replace(tmp, local_path)
    return str(local_path)


def materialize_s3_uri(uri: str, cache_dir: Path) -> str:
    bucket, key = parse_s3_uri(uri)
    local_path = cache_dir / "s3" / bucket / key
    if local_path.exists():
        return str(local_path)
    ensure_dir(local_path.parent)
    s3_client().download_file(bucket, key, str(local_path))
    return str(local_path)


def materialize_http_uri(uri: str, cache_dir: Path) -> str:
    parsed = urlparse(uri)
    suffix = Path(parsed.path).suffix
    local_path = cache_dir / "http" / f"{hashlib.sha256(uri.encode('utf-8')).hexdigest()}{suffix}"
    if not local_path.exists():
        ensure_dir(local_path.parent)
        urlretrieve(uri, local_path)
    return str(local_path)


def http_get_with_retries(requests_module, url: str, *, headers: dict[str, str], expected_size: int, attempts: int = 5) -> bytes:
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests_module.get(url, headers=headers, timeout=(30, 300))
            response.raise_for_status()
            data = response.content
            if len(data) != expected_size:
                raise RuntimeError(f"expected {expected_size} bytes, got {len(data)}")
            return data
        except Exception as exc:  # pragma: no cover - retry timing is hard to exercise deterministically
            last_exc = exc
            if attempt + 1 == attempts:
                break
            time.sleep(min(30.0, 2.0**attempt))
    raise RuntimeError(f"failed to download HF tar range after {attempts} attempts: {url}") from last_exc


def hf_auth_headers() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}
