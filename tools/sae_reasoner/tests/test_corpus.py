import io
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
    download_hf_tar_shard,
    list_hf_tar_shards_from_uri,
    materialize_hf_tar_stream,
    parse_split_ratios,
    partition_shards_for_worker,
    s3_key_for_tar_member,
    split_for_record_id,
)


def test_parse_split_ratios_normalizes():
    ratios = parse_split_ratios("train=9,label=1")
    assert ratios == (("train", 0.9), ("label", 0.1))


def test_split_for_record_id_is_extension_stable():
    ratios = parse_split_ratios("train=0.85,val=0.10,label=0.05")
    first = {f"rec-{idx}": split_for_record_id(f"rec-{idx}", 7, ratios) for idx in range(100)}
    extended = {f"rec-{idx}": split_for_record_id(f"rec-{idx}", 7, ratios) for idx in range(1000)}

    assert {key: extended[key] for key in first} == first
    assert split_for_record_id("rec-1", 8, ratios) in {"train", "val", "label"}


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

    caption_path = tmp_path / "caption.txt"
    caption_path.write_text("A real per-clip robot caption.", encoding="utf-8")

    def fake_download(repo_id: str, repo_type: str, filename: str, **_kwargs):
        assert repo_id == "nvidia/BridgeData2-Subset-Synthetic-Captions"
        assert repo_type == "dataset"
        assert filename == "sft_dataset_bridge/train/captions/episode_000015_clip000/caption.txt"
        return str(caption_path)

    fake_module = types.SimpleNamespace(HfApi=lambda: FakeHfApi(), hf_hub_download=fake_download)
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
    assert records[0]["prompt"] == "A real per-clip robot caption."
    assert records[0]["metadata"]["prompt_source"] == "sidecar"
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

    def fake_download(repo_id: str, repo_type: str, filename: str, **_kwargs):
        assert repo_id == "org/repo"
        assert repo_type == "dataset"
        assert filename == "data/task/source/shard.tar"
        return str(tar_path)

    uploads: list[tuple[str, str, bytes]] = []

    class FakeS3Client:
        def upload_fileobj(self, fileobj, bucket, key, ExtraArgs=None):
            uploads.append((bucket, key, fileobj.read()))

    fake_hf = types.SimpleNamespace(HfApi=lambda: FakeHfApi(), hf_hub_download=fake_download, hf_hub_url=lambda **_kwargs: "https://hf.test")
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


def test_hf_tar_s3_resume_skips_existing_upload(monkeypatch, tmp_path: Path):
    tar_path = tmp_path / "shard.tar"
    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(b"fake mp4")
    with tarfile.open(tar_path, "w") as tar:
        tar.add(clip_path, arcname="videos/clip.mp4")

    class FakeHfApi:
        def list_repo_files(self, repo_id: str, repo_type: str):
            return ["data/task/source/shard.tar"]

    def fake_download(repo_id: str, repo_type: str, filename: str, **_kwargs):
        return str(tar_path)

    uploads: list[str] = []

    class FakeS3Client:
        def head_object(self, Bucket, Key):
            return {"ContentLength": 8}

        def upload_fileobj(self, fileobj, bucket, key, ExtraArgs=None):
            uploads.append(key)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(HfApi=lambda: FakeHfApi(), hf_hub_download=fake_download, hf_hub_url=lambda **_kwargs: "https://hf.test"),
    )
    monkeypatch.setitem(sys.modules, "boto3", types.SimpleNamespace(client=lambda *_args, **_kwargs: FakeS3Client()))

    records = build_from_hf_tar_s3(
        repo_id="org/repo",
        shard_globs=("data/*/*/*.tar",),
        member_globs=("*.mp4",),
        prompt="Describe {stem}.",
        tags=("robotics",),
        s3_uri="s3://bucket/prefix",
        max_records=0,
        max_shards=None,
        max_shard_gb=None,
        seed=0,
        split_ratios=(("sae_train", 1.0),),
        media_type="auto",
        resume=True,
    )

    assert len(records) == 1
    assert uploads == []
    assert records[0]["metadata"]["uploaded"] is False
    assert records[0]["metadata"]["skipped_existing"] is True


def test_hf_tar_s3_stream_materializes_members(monkeypatch, tmp_path: Path):
    tar_bytes = io.BytesIO()
    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(b"streamed mp4")
    with tarfile.open(fileobj=tar_bytes, mode="w") as tar:
        tar.add(clip_path, arcname="videos/clip.mp4")
    tar_payload = tar_bytes.getvalue()

    class FakeResponse:
        def __init__(self, status_code: int, body: bytes = b""):
            self.status_code = status_code
            self.raw = io.BytesIO(body)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

        def close(self):
            self.raw.close()

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"status {self.status_code}")

    requests_calls: list[str] = []

    class FakeRequests:
        @staticmethod
        def get(url, headers=None, stream=False, timeout=None):
            requests_calls.append(url)
            if url.endswith("/collision/isaaclab/shard.tar") and "/data/" not in url:
                return FakeResponse(404)
            return FakeResponse(200, tar_payload)

    uploads: list[tuple[str, str, bytes]] = []

    class FakeS3Client:
        def upload_fileobj(self, fileobj, bucket, key, ExtraArgs=None):
            uploads.append((bucket, key, fileobj.read()))

    def fake_hf_hub_url(*, repo_id: str, filename: str, repo_type: str):
        return f"https://hf.test/{repo_type}/{repo_id}/{filename}"

    monkeypatch.setitem(sys.modules, "requests", FakeRequests)

    records = materialize_hf_tar_stream(
        hf_hub_url=fake_hf_hub_url,
        repo_id="org/repo",
        shard="collision/isaaclab/shard.tar",
        member_globs=("*.mp4",),
        prompt="Describe {stem}.",
        tags=("robotics",),
        bucket="bucket",
        prefix="prefix",
        repo_slug="org--repo",
        s3=FakeS3Client(),
        max_records=0,
        seed=0,
        split_ratios=(("sae_train", 1.0),),
        media_type="auto",
        resume=False,
        shard_index=0,
        worker_index=1,
        num_workers=4,
    )

    assert len(records) == 1
    assert requests_calls == [
        "https://hf.test/dataset/org/repo/collision/isaaclab/shard.tar",
        "https://hf.test/dataset/org/repo/data/collision/isaaclab/shard.tar",
    ]
    assert uploads == [("bucket", "prefix/media/org--repo/shard/videos/clip.mp4", b"streamed mp4")]
    assert records[0]["metadata"]["shard"] == "data/collision/isaaclab/shard.tar"
    assert records[0]["metadata"]["worker_index"] == 1


def test_hf_tar_s3_worker_partition_is_stable():
    shards = [f"data/task/source/shard-{idx:03d}.tar" for idx in range(12)]
    worker_parts = [partition_shards_for_worker(shards, worker_index=idx, num_workers=3) for idx in range(3)]
    flattened = sorted(shard for part in worker_parts for shard in part)

    assert flattened == sorted(shards)
    assert all(not (set(worker_parts[left]) & set(worker_parts[right])) for left in range(3) for right in range(left + 1, 3))
    assert partition_shards_for_worker(shards, worker_index=1, num_workers=3) == worker_parts[1]


def test_hf_tar_s3_shard_list_uri_filters_rows(tmp_path: Path):
    shard_list = tmp_path / "shards.jsonl"
    shard_list.write_text(
        "\n".join(
            [
                '{"shard_path":"data/a/b/keep-small.tar","tar_bytes":10}',
                '{"path":"data/a/b/skip-large.tar","tar_bytes":2000}',
                '{"shard_path":"collision/isaaclab/robotsim-v1.0-collision-isaaclab-000000.tar","tar_bytes":10}',
                '"data/a/b/keep-string.tar"',
                '{"path":"data/a/b/not-text.txt","tar_bytes":10}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    shards = list_hf_tar_shards_from_uri(str(shard_list), shard_globs=("*/*/*.tar", "data/*/*/*.tar"), max_shard_bytes=100)

    assert shards == [
        "data/a/b/keep-small.tar",
        "collision/isaaclab/robotsim-v1.0-collision-isaaclab-000000.tar",
        "data/a/b/keep-string.tar",
    ]


def test_download_hf_tar_shard_retries_public_manifest_path():
    calls: list[str] = []

    def fake_download(*, repo_id: str, repo_type: str, filename: str):
        calls.append(filename)
        if filename == "collision/isaaclab/shard.tar":
            raise FileNotFoundError(filename)
        return f"/cache/{filename}"

    shard, path = download_hf_tar_shard(fake_download, repo_id="org/repo", shard="collision/isaaclab/shard.tar")

    assert shard == "data/collision/isaaclab/shard.tar"
    assert path == "/cache/data/collision/isaaclab/shard.tar"
    assert calls == ["collision/isaaclab/shard.tar", "data/collision/isaaclab/shard.tar"]


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
