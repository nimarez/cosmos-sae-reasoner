import sys
import tarfile
import types
from pathlib import Path

from tools.sae_reasoner.corpus import (
    BuildCorpusConfig,
    build_corpus_manifest,
    build_from_hf_files,
    build_from_hf_tar_s3,
    build_from_s3_prefix,
    parse_split_ratios,
    s3_key_for_tar_member,
)


def test_parse_split_ratios_normalizes():
    ratios = parse_split_ratios("train=9,label=1")
    assert ratios == (("train", 0.9), ("label", 0.1))


def test_build_from_hf_files_uses_remote_media(monkeypatch):
    class FakeHfApi:
        def list_repo_files(self, repo_id: str, repo_type: str):
            assert repo_id == "org/repo"
            assert repo_type == "dataset"
            return [
                "README.md",
                "data/a/video/clip1.mp4",
                "data/a/video/clip2.mp4",
                "data/a/description/clip1.json",
            ]

    fake_module = types.SimpleNamespace(HfApi=lambda: FakeHfApi())
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    records = build_from_hf_files(
        repo_id="org/repo",
        include_globs=("data/*/video/*.mp4",),
        prompt="Describe {stem} from {repo_id}.",
        tags=("test",),
        max_records=1,
        seed=0,
        split_ratios=(("sae_train", 1.0),),
        media_type="auto",
    )

    assert len(records) == 1
    assert records[0]["media_type"] == "video"
    assert records[0]["media_path"].startswith("hf://dataset/org/repo/data/a/video/")
    assert records[0]["metadata"]["source"] == "hf-files"
    assert "sae_train" in records[0]["tags"]


def test_robotics_recipe_uses_loose_videos(monkeypatch, tmp_path: Path):
    class FakeHfApi:
        def list_repo_files(self, repo_id: str, repo_type: str):
            assert repo_id == "nvidia/BridgeData2-Subset-Synthetic-Captions"
            assert repo_type == "dataset"
            return [
                "README.md",
                "sft_dataset_bridge/train/videos/episode_000015_clip000.mp4",
                "sft_dataset_bridge/train/captions/episode_000015_clip000/caption.txt",
                "sft_dataset_bridge/val/videos_5frames/episode_000015_clip000.mp4",
            ]

    fake_module = types.SimpleNamespace(HfApi=lambda: FakeHfApi())
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    output = tmp_path / "robotics.jsonl"
    records = build_corpus_manifest(
        BuildCorpusConfig(
            output=output,
            source="recipe",
            recipe="robotics-bridge-captions",
            max_records=10,
            split_ratios=(("sae_train", 1.0),),
        )
    )

    assert len(records) == 1
    assert records[0]["media_type"] == "video"
    assert records[0]["media_path"] == (
        "hf://dataset/nvidia/BridgeData2-Subset-Synthetic-Captions/"
        "sft_dataset_bridge/train/videos/episode_000015_clip000.mp4"
    )
    assert "robotics" in records[0]["tags"]


def test_s3_key_for_tar_member_sanitizes_traversal():
    key = s3_key_for_tar_member("prefix", "org--repo", "data/a/shard-000.tar", "../clips/a b.mp4")

    assert key == "prefix/media/org--repo/shard-000/clips/a%20b.mp4"
    assert ".." not in key


def test_build_from_hf_tar_s3_materializes_members(monkeypatch, tmp_path: Path):
    tar_path = tmp_path / "shard.tar"
    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(b"fake mp4")
    text_path = tmp_path / "note.txt"
    text_path.write_text("skip", encoding="utf-8")
    with tarfile.open(tar_path, "w") as tar:
        tar.add(clip_path, arcname="videos/clip 1.mp4")
        tar.add(text_path, arcname="videos/note.txt")

    class FakeHfApi:
        def list_repo_files(self, repo_id: str, repo_type: str):
            assert repo_id == "org/repo"
            assert repo_type == "dataset"
            return ["README.md", "data/task/source/shard.tar"]

    def fake_download(repo_id: str, repo_type: str, filename: str):
        assert repo_id == "org/repo"
        assert repo_type == "dataset"
        assert filename == "data/task/source/shard.tar"
        return str(tar_path)

    uploads: list[tuple[str, str, bytes]] = []

    class FakeS3Client:
        def upload_fileobj(self, fileobj, bucket, key, ExtraArgs=None):
            uploads.append((bucket, key, fileobj.read()))

    fake_hf = types.SimpleNamespace(HfApi=lambda: FakeHfApi(), hf_hub_download=fake_download)
    fake_boto3 = types.SimpleNamespace(client=lambda *_args, **_kwargs: FakeS3Client())
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)

    records = build_from_hf_tar_s3(
        repo_id="org/repo",
        shard_globs=("data/*/*/*.tar",),
        member_globs=("*.mp4",),
        prompt="Describe {stem} from {shard}.",
        tags=("robotics",),
        s3_uri="s3://bucket/prefix",
        max_records=1,
        max_shards=1,
        max_shard_gb=1.0,
        seed=0,
        split_ratios=(("sae_train", 1.0),),
        media_type="auto",
    )

    assert len(records) == 1
    assert len(uploads) == 1
    assert uploads[0] == ("bucket", "prefix/media/org--repo/shard/videos/clip%201.mp4", b"fake mp4")
    assert records[0]["media_path"] == "s3://bucket/prefix/media/org--repo/shard/videos/clip%201.mp4"
    assert records[0]["metadata"]["source"] == "hf-tar-s3"
    assert records[0]["metadata"]["member"] == "videos/clip 1.mp4"


def test_build_corpus_manifest_from_jsonl(tmp_path: Path):
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        '{"id":"one","media_type":"text","prompt":"A long enough text prompt for a smoke record."}\n',
        encoding="utf-8",
    )
    output = tmp_path / "out.jsonl"

    records = build_corpus_manifest(
        BuildCorpusConfig(
            output=output,
            source="jsonl",
            input_uri=str(input_path),
            max_records=10,
        )
    )

    assert output.exists()
    assert records[0]["id"] == "one"
    assert records[0]["metadata"]["source"] == "jsonl"


def test_build_from_s3_prefix_reservoir_samples(monkeypatch):
    class FakeS3Client:
        def __init__(self):
            self.calls = 0

        def list_objects_v2(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return {
                    "IsTruncated": True,
                    "NextContinuationToken": "next",
                    "Contents": [{"Key": f"clips/a_{idx}.mp4"} for idx in range(5)],
                }
            return {
                "IsTruncated": False,
                "Contents": [{"Key": f"clips/b_{idx}.mp4"} for idx in range(5)],
            }

    fake_client = FakeS3Client()
    fake_boto3 = types.SimpleNamespace(client=lambda *_args, **_kwargs: fake_client)
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)

    records = build_from_s3_prefix(
        BuildCorpusConfig(
            output=Path("unused.jsonl"),
            source="s3-prefix",
            s3_uri="s3://bucket/clips/",
            include_globs=("*.mp4",),
            prompt="Describe {stem}.",
            max_records=3,
            split_ratios=(("sae_train", 1.0),),
        )
    )

    assert len(records) == 3
    assert all(record["media_path"].startswith("s3://bucket/clips/") for record in records)
    assert fake_client.calls == 2
