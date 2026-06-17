import sys
import types
from pathlib import Path

from tools.sae_reasoner.media import materialize_media_path


def test_materialize_hf_tar_range_uri_downloads_requested_bytes(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)

    class FakeResponse:
        content = b"mp4data"

        def raise_for_status(self):
            return None

    class FakeRequests:
        @staticmethod
        def get(url, headers=None, timeout=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["timeout"] = timeout
            return FakeResponse()

    fake_hf = types.SimpleNamespace(
        hf_hub_url=lambda repo_id, filename, repo_type: f"https://hf.test/{repo_id}/{filename}",
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
    monkeypatch.setitem(sys.modules, "requests", FakeRequests)

    uri = "hf-tar-range://dataset/org/repo/data/shard.tar?offset=512&size=7&name=videos%2Fclip.mp4"
    local = Path(materialize_media_path(uri, cache_dir=tmp_path))

    assert local.exists()
    assert local.read_bytes() == b"mp4data"
    assert local.suffix == ".mp4"
    assert captured["url"] == "https://hf.test/org/repo/data/shard.tar"
    assert captured["headers"] == {"Range": "bytes=512-518"}


def test_materialize_hf_tar_range_uri_uses_existing_cache(monkeypatch, tmp_path: Path):
    calls = {"get": 0}
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)

    class FakeResponse:
        content = b"mp4data"

        def raise_for_status(self):
            return None

    class FakeRequests:
        @staticmethod
        def get(url, headers=None, timeout=None):
            calls["get"] += 1
            return FakeResponse()

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_url=lambda repo_id, filename, repo_type: f"https://hf.test/{repo_id}/{filename}"),
    )
    monkeypatch.setitem(sys.modules, "requests", FakeRequests)

    uri = "hf-tar-range://dataset/org/repo/data/shard.tar?offset=512&size=7&name=videos%2Fclip.mp4"
    first = materialize_media_path(uri, cache_dir=tmp_path)
    second = materialize_media_path(uri, cache_dir=tmp_path)

    assert first == second
    assert calls["get"] == 1
