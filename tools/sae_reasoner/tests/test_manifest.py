import sys
import types
from pathlib import Path

from tools.sae_reasoner.manifest import ManifestRecord, load_manifest, make_sample_manifest


def test_make_sample_manifest(tmp_path: Path):
    out = tmp_path / "sample.jsonl"
    records = make_sample_manifest(out)
    assert out.exists()
    assert len(records) >= 3
    loaded = load_manifest(out)
    assert [r.id for r in loaded] == [r.id for r in records]
    assert all(set(record.to_json()) <= {"id", "media_type", "prompt", "media_path", "tags"} for record in loaded)


def test_manifest_resolves_relative_media_path(tmp_path: Path, monkeypatch):
    media = tmp_path / "image.png"
    media.write_bytes(b"not actually decoded in this test")
    record = ManifestRecord.from_json(
        {
            "id": "relative-image",
            "media_type": "image",
            "media_path": "image.png",
            "prompt": "describe this",
        },
        base_dir=tmp_path,
    )
    monkeypatch.chdir("/")
    assert record.media_path == str(media.resolve())
    assert Path(record.media_path).exists()


def test_manifest_accepts_remote_media_path():
    record = ManifestRecord.from_json(
        {
            "id": "remote-video",
            "media_type": "video",
            "media_path": "hf://dataset/nvidia/example/videos/clip.mp4",
            "prompt": "describe this",
            "tags": ["video"],
            "metadata": {"source": "hf-files"},
        }
    )

    assert record.media_path == "hf://dataset/nvidia/example/videos/clip.mp4"
    assert record.metadata == {"source": "hf-files"}
    assert record.to_json()["metadata"] == {"source": "hf-files"}

    range_record = ManifestRecord.from_json(
        {
            "id": "remote-range-video",
            "media_type": "video",
            "media_path": "hf-tar-range://dataset/nvidia/example/data/shard.tar?offset=512&size=10&name=clip.mp4",
            "prompt": "describe this",
        }
    )
    assert range_record.media_path.startswith("hf-tar-range://dataset/")


def test_load_manifest_from_s3_uri(monkeypatch):
    class FakeBody:
        def iter_lines(self):
            return iter(
                [
                    b'{"id":"remote","media_type":"video","media_path":"s3://bucket/media/clip.mp4","prompt":"describe"}'
                ]
            )

    class FakeS3Client:
        def get_object(self, Bucket, Key):
            assert Bucket == "bucket"
            assert Key == "manifests/run.jsonl"
            return {"Body": FakeBody()}

    fake_boto3 = types.SimpleNamespace(client=lambda *_args, **_kwargs: FakeS3Client())
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)

    records = load_manifest("s3://bucket/manifests/run.jsonl")

    assert len(records) == 1
    assert records[0].id == "remote"
    assert records[0].media_path == "s3://bucket/media/clip.mp4"
