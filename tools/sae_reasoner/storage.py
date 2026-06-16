from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

from .artifacts import ensure_dir


class ActivationStore(Protocol):
    def write_torch(self, name: str, payload: dict) -> str:
        ...

    def write_text(self, name: str, text: str) -> str:
        ...

    def read_text(self, name: str) -> str:
        ...

    def exists(self, name: str) -> bool:
        ...


@dataclass(frozen=True)
class LocalActivationStore:
    root: Path

    def __post_init__(self) -> None:
        ensure_dir(self.root)

    def write_torch(self, name: str, payload: dict) -> str:
        import torch

        path = self.root / name
        ensure_dir(path.parent)
        torch.save(payload, path)
        return str(path)

    def write_text(self, name: str, text: str) -> str:
        path = self.root / name
        ensure_dir(path.parent)
        path.write_text(text, encoding="utf-8")
        return str(path)

    def read_text(self, name: str) -> str:
        return (self.root / name).read_text(encoding="utf-8")

    def exists(self, name: str) -> bool:
        return (self.root / name).exists()


@dataclass(frozen=True)
class S3ActivationStore:
    bucket: str
    prefix: str

    def __post_init__(self) -> None:
        if not self.bucket:
            raise ValueError("S3 bucket must not be empty")

    def write_torch(self, name: str, payload: dict) -> str:
        import torch

        body = io.BytesIO()
        torch.save(payload, body)
        body.seek(0)
        key = self.key(name)
        self.client().put_object(Bucket=self.bucket, Key=key, Body=body.getvalue())
        return f"s3://{self.bucket}/{key}"

    def write_text(self, name: str, text: str) -> str:
        key = self.key(name)
        self.client().put_object(
            Bucket=self.bucket,
            Key=key,
            Body=text.encode("utf-8"),
            ContentType="application/jsonl; charset=utf-8",
        )
        return f"s3://{self.bucket}/{key}"

    def read_text(self, name: str) -> str:
        key = self.key(name)
        return self.client().get_object(Bucket=self.bucket, Key=key)["Body"].read().decode("utf-8")

    def exists(self, name: str) -> bool:
        key = self.key(name)
        try:
            self.client().head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code")
            if str(code) in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise

    def key(self, name: str) -> str:
        clean_name = name.lstrip("/")
        if not self.prefix:
            return clean_name
        return f"{self.prefix.rstrip('/')}/{clean_name}"

    @staticmethod
    def client():
        return s3_client()


def make_activation_store(uri: str) -> ActivationStore:
    if uri.startswith("s3://"):
        bucket, prefix = parse_s3_uri(uri)
        return S3ActivationStore(bucket=bucket, prefix=prefix)
    return LocalActivationStore(root=Path(uri))


def iter_activation_payloads(uri: str | Path) -> Iterator[tuple[str, dict]]:
    import torch

    raw = str(uri)
    if raw.startswith("s3://"):
        bucket, prefix = parse_s3_uri(raw)
        client = s3_client()
        for key in iter_s3_keys(bucket, prefix, suffix=".pt"):
            body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
            yield Path(key).name, torch.load(io.BytesIO(body), map_location="cpu")
        return
    path = Path(raw)
    for shard_path in sorted(path.glob("*.pt")):
        yield shard_path.name, torch.load(shard_path, map_location="cpu")


def iter_s3_keys(bucket: str, prefix: str, *, suffix: str = "") -> Iterator[str]:
    client = s3_client()
    paginator = client.get_paginator("list_objects_v2")
    normalized_prefix = prefix.rstrip("/")
    if normalized_prefix:
        normalized_prefix += "/"
    for page in paginator.paginate(Bucket=bucket, Prefix=normalized_prefix):
        for item in page.get("Contents", []):
            key = item.get("Key", "")
            if key and (not suffix or key.endswith(suffix)):
                yield key


def is_s3_uri(uri: str | Path) -> bool:
    return str(uri).startswith("s3://")


def join_uri(base: str | Path, name: str) -> str:
    raw = str(base)
    if is_s3_uri(raw):
        return f"{raw.rstrip('/')}/{name}"
    return str(Path(raw) / name)


def save_torch_uri(uri: str | Path, payload: dict) -> str:
    """Serialize a torch payload to a local path (atomically) or an S3 object."""
    import torch

    raw = str(uri)
    if is_s3_uri(raw):
        bucket, key = parse_s3_uri(raw)
        body = io.BytesIO()
        torch.save(payload, body)
        s3_client().put_object(Bucket=bucket, Key=key, Body=body.getvalue())
        return raw
    path = Path(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return str(path)


def load_torch_uri(uri: str | Path, map_location: str = "cpu") -> dict:
    import torch

    raw = str(uri)
    if is_s3_uri(raw):
        bucket, key = parse_s3_uri(raw)
        body = s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()
        return torch.load(io.BytesIO(body), map_location=map_location)
    return torch.load(raw, map_location=map_location)


def list_uri_names(dir_uri: str | Path, *, prefix: str = "", suffix: str = "") -> list[str]:
    """Immediate child file names under a local directory or S3 'directory' prefix."""
    raw = str(dir_uri)
    if is_s3_uri(raw):
        bucket, key_prefix = parse_s3_uri(raw)
        base = key_prefix.rstrip("/")
        names: list[str] = []
        for key in iter_s3_keys(bucket, base, suffix=suffix):
            name = key[len(base) :].lstrip("/") if base else key
            if "/" in name:  # skip nested objects, keep immediate children only
                continue
            if name.startswith(prefix):
                names.append(name)
        return names
    path = Path(raw)
    if not path.is_dir():
        return []
    return sorted(p.name for p in path.iterdir() if p.is_file() and p.name.startswith(prefix) and p.name.endswith(suffix))


def uri_exists(uri: str | Path) -> bool:
    raw = str(uri)
    if is_s3_uri(raw):
        bucket, key = parse_s3_uri(raw)
        try:
            s3_client().head_object(Bucket=bucket, Key=key)
            return True
        except Exception as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code")
            if str(code) in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise
    return Path(raw).exists()


def read_text_uri(uri: str | Path) -> str:
    raw = str(uri)
    if raw.startswith("s3://"):
        bucket, key = parse_s3_uri(raw)
        return s3_client().get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
    return Path(raw).read_text(encoding="utf-8")


def s3_client():
    try:
        import boto3
    except Exception as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError("S3 access requires boto3. Install the sae dependency group.") from exc
    endpoint_url = os.environ.get("AWS_ENDPOINT_URL_S3") or os.environ.get("AWS_ENDPOINT_URL")
    kwargs = {"endpoint_url": endpoint_url} if endpoint_url else {}
    return boto3.client("s3", **kwargs)


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"not an S3 URI: {uri}")
    rest = uri[len("s3://") :]
    bucket, sep, prefix = rest.partition("/")
    if not bucket:
        raise ValueError(f"S3 URI is missing bucket: {uri}")
    return bucket, prefix if sep else ""
