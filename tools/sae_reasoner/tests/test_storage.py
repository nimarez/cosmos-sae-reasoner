from pathlib import Path
import types

import torch

from tools.sae_reasoner.storage import (
    LocalActivationStore,
    S3ActivationStore,
    iter_activation_payloads,
    parse_s3_uri,
    read_text_uri,
    s3_client,
    s3_client_kwargs,
)


def test_parse_s3_uri():
    assert parse_s3_uri("s3://bucket/prefix/path") == ("bucket", "prefix/path")
    assert parse_s3_uri("s3://bucket") == ("bucket", "")


def test_s3_store_key_joining():
    store = S3ActivationStore(bucket="bucket", prefix="prefix/path/")
    assert store.key("/000001_example.pt") == "prefix/path/000001_example.pt"


def test_s3_client_kwargs_supports_r2_env(monkeypatch):
    for key in (
        "AWS_ENDPOINT_URL_S3",
        "AWS_ENDPOINT_URL",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "R2_ENDPOINT_URL",
        "CLOUDFLARE_R2_ENDPOINT_URL",
        "R2_ACCOUNT_ID",
        "CLOUDFLARE_ACCOUNT_ID",
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
        "R2_REGION",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("R2_ACCOUNT_ID", "abc123")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "key")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "secret")

    assert s3_client_kwargs() == {
        "endpoint_url": "https://abc123.r2.cloudflarestorage.com",
        "aws_access_key_id": "key",
        "aws_secret_access_key": "secret",
        "region_name": "auto",
    }


def test_s3_client_forwards_r2_kwargs(monkeypatch):
    captured = {}
    for key in (
        "AWS_ENDPOINT_URL_S3",
        "AWS_ENDPOINT_URL",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "R2_REGION",
        "R2_ACCOUNT_ID",
        "CLOUDFLARE_ACCOUNT_ID",
        "CLOUDFLARE_R2_ENDPOINT_URL",
    ):
        monkeypatch.delenv(key, raising=False)

    def fake_client(service, **kwargs):
        captured["service"] = service
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setenv("R2_ENDPOINT_URL", "https://example.r2.cloudflarestorage.com")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "key")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setitem(__import__("sys").modules, "boto3", types.SimpleNamespace(client=fake_client))

    s3_client()

    assert captured == {
        "service": "s3",
        "kwargs": {
            "endpoint_url": "https://example.r2.cloudflarestorage.com",
            "aws_access_key_id": "key",
            "aws_secret_access_key": "secret",
            "region_name": "auto",
        },
    }


def test_local_activation_store_roundtrip(tmp_path: Path):
    store = LocalActivationStore(tmp_path)
    uri = store.write_torch("x.pt", {"activations": torch.ones(2, 3)})
    payload = torch.load(uri, map_location="cpu")
    assert payload["activations"].shape == (2, 3)
    meta_uri = store.write_text("metadata.jsonl", "{}\n")
    assert Path(meta_uri).read_text(encoding="utf-8") == "{}\n"
    assert store.exists("metadata.jsonl")
    assert store.read_text("metadata.jsonl") == "{}\n"
    assert not store.exists("missing.json")


def test_iter_activation_payloads_reads_s3_prefix(monkeypatch):
    body = torch_payload({"activations": torch.ones(2, 3)})

    class FakeBody:
        def __init__(self, data: bytes):
            self.data = data

        def read(self):
            return self.data

    class FakePaginator:
        def paginate(self, Bucket, Prefix):
            assert Bucket == "bucket"
            assert Prefix == "prefix/run/"
            return [{"Contents": [{"Key": "prefix/run/000000_rec.pt"}, {"Key": "prefix/run/metadata.jsonl"}]}]

    class FakeS3:
        def get_paginator(self, name):
            assert name == "list_objects_v2"
            return FakePaginator()

        def get_object(self, Bucket, Key):
            assert Bucket == "bucket"
            assert Key == "prefix/run/000000_rec.pt"
            return {"Body": FakeBody(body)}

    monkeypatch.setitem(__import__("sys").modules, "boto3", types.SimpleNamespace(client=lambda *_args, **_kwargs: FakeS3()))

    rows = list(iter_activation_payloads("s3://bucket/prefix/run"))

    assert rows[0][0] == "000000_rec.pt"
    assert rows[0][1]["activations"].shape == (2, 3)


def test_read_text_uri_reads_s3_object(monkeypatch):
    class FakeBody:
        def read(self):
            return b"hello\n"

    class FakeS3:
        def get_object(self, Bucket, Key):
            assert Bucket == "bucket"
            assert Key == "path/metadata.jsonl"
            return {"Body": FakeBody()}

    monkeypatch.setitem(__import__("sys").modules, "boto3", types.SimpleNamespace(client=lambda *_args, **_kwargs: FakeS3()))

    assert read_text_uri("s3://bucket/path/metadata.jsonl") == "hello\n"


def torch_payload(payload: dict) -> bytes:
    import io

    body = io.BytesIO()
    torch.save(payload, body)
    return body.getvalue()
