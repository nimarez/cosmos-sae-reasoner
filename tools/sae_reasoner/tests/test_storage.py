from pathlib import Path

import torch

from tools.sae_reasoner.storage import LocalActivationStore, S3ActivationStore, parse_s3_uri


def test_parse_s3_uri():
    assert parse_s3_uri("s3://bucket/prefix/path") == ("bucket", "prefix/path")
    assert parse_s3_uri("s3://bucket") == ("bucket", "")


def test_s3_store_key_joining():
    store = S3ActivationStore(bucket="bucket", prefix="prefix/path/")
    assert store.key("/000001_example.pt") == "prefix/path/000001_example.pt"


def test_local_activation_store_roundtrip(tmp_path: Path):
    store = LocalActivationStore(tmp_path)
    uri = store.write_torch("x.pt", {"activations": torch.ones(2, 3)})
    payload = torch.load(uri, map_location="cpu")
    assert payload["activations"].shape == (2, 3)
    meta_uri = store.write_text("metadata.jsonl", "{}\n")
    assert Path(meta_uri).read_text(encoding="utf-8") == "{}\n"

