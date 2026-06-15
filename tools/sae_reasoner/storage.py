from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .artifacts import ensure_dir


class ActivationStore(Protocol):
    def write_torch(self, name: str, payload: dict) -> str:
        ...

    def write_text(self, name: str, text: str) -> str:
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

    def key(self, name: str) -> str:
        clean_name = name.lstrip("/")
        if not self.prefix:
            return clean_name
        return f"{self.prefix.rstrip('/')}/{clean_name}"

    @staticmethod
    def client():
        try:
            import boto3
        except Exception as exc:  # pragma: no cover - depends on optional env
            raise RuntimeError("S3 output requires boto3. Install the sae dependency group.") from exc
        endpoint_url = os.environ.get("AWS_ENDPOINT_URL_S3") or os.environ.get("AWS_ENDPOINT_URL")
        kwargs = {"endpoint_url": endpoint_url} if endpoint_url else {}
        return boto3.client("s3", **kwargs)


def make_activation_store(uri: str) -> ActivationStore:
    if uri.startswith("s3://"):
        bucket, prefix = parse_s3_uri(uri)
        return S3ActivationStore(bucket=bucket, prefix=prefix)
    return LocalActivationStore(root=Path(uri))


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"not an S3 URI: {uri}")
    rest = uri[len("s3://") :]
    bucket, sep, prefix = rest.partition("/")
    if not bucket:
        raise ValueError(f"S3 URI is missing bucket: {uri}")
    return bucket, prefix if sep else ""
