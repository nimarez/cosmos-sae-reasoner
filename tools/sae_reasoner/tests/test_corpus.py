import io
import sqlite3
import sys
import tarfile
import types
from pathlib import Path

from tools.sae_reasoner.corpus import (
    BuildCorpusConfig,
    build_corpus_manifest,
    build_from_hf_conversation_tar_s3,
    build_from_hf_files,
    build_from_hf_tar_s3,
    build_from_hf_tar_range,
    build_from_s3_prefix,
    download_hf_tar_shard,
    hf_tar_range_uri,
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


def test_drivesim_recipe_uses_description_json_caption(monkeypatch, tmp_path: Path):
    class FakeHfApi:
        def list_repo_files(self, repo_id: str, repo_type: str):
            assert repo_id == "nvidia/PhysicalAI-WorldModel-Synthetic-Autonomous-Driving-Scenarios"
            assert repo_type == "dataset"
            return [
                "emergency/run_001/video/front_tele.mp4",
                "emergency/run_001/description/front_tele.json",
            ]

    description_path = tmp_path / "front_tele.json"
    description_path.write_text(
        '{"t2w_windows": [{"qwen2p5_7b_caption": "A captioned driving scene."}]}',
        encoding="utf-8",
    )

    def fake_download(repo_id: str, repo_type: str, filename: str, **_kwargs):
        assert filename == "emergency/run_001/description/front_tele.json"
        return str(description_path)

    fake_module = types.SimpleNamespace(HfApi=lambda: FakeHfApi(), hf_hub_download=fake_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    output = tmp_path / "drivesim.jsonl"
    records = build_corpus_manifest(
        BuildCorpusConfig(
            output=output,
            source="recipe",
            recipe="physicalai-drivesim",
            max_records=10,
            split_ratios=(("sae_train", 1.0),),
        )
    )

    assert len(records) == 1
    assert records[0]["prompt"] == "A captioned driving scene."
    assert records[0]["metadata"]["prompt_source"] == "sidecar_json"
    assert records[0]["metadata"]["prompt_sidecar"] == "emergency/run_001/description/front_tele.json"


def test_lerobot_recipe_uses_task_metadata(monkeypatch, tmp_path: Path):
    class FakeHfApi:
        def list_repo_files(self, repo_id: str, repo_type: str):
            assert repo_id == "nvidia/BridgeData2_LeRobot_v3"
            assert repo_type == "dataset"
            return [
                "videos/observation.images.image_0/chunk-000/file-000.mp4",
                "meta/episodes/chunk-000/file-000.parquet",
            ]

    class FakeFrame:
        def iterrows(self):
            yield 0, {
                "episode_index": 17,
                "tasks": ["open the drawer", "place the cup"],
                "videos/observation.images.image_0/chunk_index": 0,
                "videos/observation.images.image_0/file_index": 0,
            }

    class FakePandas:
        @staticmethod
        def read_parquet(path):
            assert Path(path).name == "file-000.parquet"
            return FakeFrame()

    parquet_path = tmp_path / "file-000.parquet"
    parquet_path.write_bytes(b"fake parquet")

    def fake_download(repo_id: str, repo_type: str, filename: str, **_kwargs):
        assert filename == "meta/episodes/chunk-000/file-000.parquet"
        return str(parquet_path)

    fake_hf = types.SimpleNamespace(HfApi=lambda: FakeHfApi(), hf_hub_download=fake_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
    monkeypatch.setitem(sys.modules, "pandas", FakePandas)

    output = tmp_path / "bridge.jsonl"
    records = build_corpus_manifest(
        BuildCorpusConfig(
            output=output,
            source="recipe",
            recipe="robotics-bridge",
            max_records=10,
            split_ratios=(("sae_train", 1.0),),
        )
    )

    assert len(records) == 1
    assert "LeRobot task annotations in this video shard:" in records[0]["prompt"]
    assert "- open the drawer" in records[0]["prompt"]
    assert "- place the cup" in records[0]["prompt"]
    assert records[0]["metadata"]["source"] == "hf-lerobot-v3"
    assert records[0]["metadata"]["prompt_source"] == "lerobot_tasks"
    assert records[0]["metadata"]["lerobot_episode_count"] == 1


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


def test_build_from_hf_tar_range_points_at_byte_ranges(monkeypatch, tmp_path: Path):
    tar_path = tmp_path / "shard.tar"
    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(b"fake range mp4")
    sidecar_path = tmp_path / "clip.json"
    sidecar_path.write_text('{"caption":"A robot closes a cabinet.","task":"close cabinet"}', encoding="utf-8")
    with tarfile.open(tar_path, "w") as tar:
        tar.add(clip_path, arcname="videos/clip.mp4")
        tar.add(sidecar_path, arcname="videos/clip.json")

    class FakeHfApi:
        def list_repo_files(self, repo_id: str, repo_type: str):
            assert repo_id == "org/range"
            assert repo_type == "dataset"
            return ["data/shard.tar"]

    def fake_download(repo_id: str, repo_type: str, filename: str, local_dir: str):
        assert repo_id == "org/range"
        assert repo_type == "dataset"
        assert filename == "data/shard.tar"
        return str(tar_path)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(HfApi=lambda: FakeHfApi(), hf_hub_download=fake_download),
    )

    records = build_from_hf_tar_range(
        repo_id="org/range",
        shard_globs=("data/*.tar",),
        member_globs=("*.mp4",),
        prompt="Describe {stem}.",
        tags=("physicalai",),
        max_records=1,
        max_shards=1,
        max_shard_gb=None,
        seed=0,
        split_ratios=(("sae_train", 1.0),),
        media_type="auto",
        sidecar_suffix=".json",
    )

    assert len(records) == 1
    record = records[0]
    assert record["media_path"].startswith("hf-tar-range://dataset/org/range/data/shard.tar?")
    assert "offset=" in record["media_path"]
    assert "size=" in record["media_path"]
    assert record["prompt"] == "A robot closes a cabinet."
    assert record["metadata"]["source"] == "hf-tar-range"
    assert record["metadata"]["content_byte_size"] == len(b"fake range mp4")
    assert record["metadata"]["robotsim_sidecar"] == {"task": "close cabinet"}


def test_phyxsim_recipe_uses_matching_caption_tar(monkeypatch, tmp_path: Path):
    video_tar_path = tmp_path / "videos-ball_mixer-00000.tar"
    caption_tar_path = tmp_path / "captions-ball_mixer-00000.tar"
    clip_path = tmp_path / "camera_e.mp4"
    clip_path.write_bytes(b"fake phyxsim mp4")
    caption_path = tmp_path / "camera_e.json"
    caption_path.write_text(
        '{"Qwen3-VL-30B-A3B-Instruct": {"long": "Objects collide and settle in a box."}}',
        encoding="utf-8",
    )
    with tarfile.open(video_tar_path, "w") as tar:
        tar.add(clip_path, arcname="ball_mixer_abc_0/camera_e.mp4")
    with tarfile.open(caption_tar_path, "w") as tar:
        tar.add(caption_path, arcname="ball_mixer_abc_0/camera_e.json")

    class FakeHfApi:
        def list_repo_files(self, repo_id: str, repo_type: str):
            assert repo_id == "nvidia/PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes"
            assert repo_type == "dataset"
            return [
                "videos/ball_mixer/videos-ball_mixer-00000.tar",
                "videos/wrecking_ball/videos-wrecking_ball-00058.tar",
                "captions/ball_mixer/captions-ball_mixer-00000.tar",
            ]

    def fake_download(repo_id: str, repo_type: str, filename: str, local_dir: str):
        if filename == "videos/ball_mixer/videos-ball_mixer-00000.tar":
            return str(video_tar_path)
        if filename == "captions/ball_mixer/captions-ball_mixer-00000.tar":
            return str(caption_tar_path)
        assert "wrecking_ball" not in filename
        raise AssertionError(filename)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(HfApi=lambda: FakeHfApi(), hf_hub_download=fake_download),
    )

    output = tmp_path / "phyxsim.jsonl"
    records = build_corpus_manifest(
        BuildCorpusConfig(
            output=output,
            source="recipe",
            recipe="physicalai-phyxsim-range",
            max_records=1,
            max_shards=1,
            max_shard_gb=0,
            split_ratios=(("sae_train", 1.0),),
        )
    )

    assert len(records) == 1
    assert records[0]["prompt"] == "Objects collide and settle in a box."
    assert records[0]["metadata"]["prompt_source"] == "phyxsim_caption_sidecar"
    assert records[0]["metadata"]["sidecar_member"] == "ball_mixer_abc_0/camera_e.json"


def test_hf_tar_range_uri_encodes_member_name():
    uri = hf_tar_range_uri(repo_id="org/repo", shard="data/shard.tar", member_name="videos/clip 1.mp4", offset=512, size=7)
    assert uri == "hf-tar-range://dataset/org/repo/data/shard.tar?offset=512&size=7&name=videos%2Fclip+1.mp4"


def test_robotsim_recipe_materializes_json_sidecars(monkeypatch, tmp_path: Path):
    tar_path = tmp_path / "shard.tar"
    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(b"fake mp4")
    sidecar_path = tmp_path / "clip.json"
    sidecar_path.write_text(
        '{"caption":"A simulated robot opens a drawer.","simulation_tool":"isaaclab","nb_frames":32}',
        encoding="utf-8",
    )
    with tarfile.open(tar_path, "w") as tar:
        tar.add(clip_path, arcname="videos/clip.mp4")
        tar.add(sidecar_path, arcname="videos/clip.json")

    class FakeHfApi:
        def list_repo_files(self, repo_id: str, repo_type: str):
            assert repo_id == "nvidia/PhysicalAI-WorldModel-Synthetic-Embodied-Robot-Scenes"
            assert repo_type == "dataset"
            return ["data/task/source/shard.tar"]

    def fake_download(repo_id: str, repo_type: str, filename: str, **_kwargs):
        assert filename == "data/task/source/shard.tar"
        return str(tar_path)

    uploads: list[tuple[str, str, bytes]] = []
    sidecar_uploads: list[tuple[str, str, bytes]] = []

    class FakeS3Client:
        def upload_fileobj(self, fileobj, bucket, key, ExtraArgs=None):
            uploads.append((bucket, key, fileobj.read()))

        def put_object(self, Bucket, Key, Body, ContentType=None):
            sidecar_uploads.append((Bucket, Key, Body))

    fake_hf = types.SimpleNamespace(
        HfApi=lambda: FakeHfApi(),
        hf_hub_download=fake_download,
        hf_hub_url=lambda **_kwargs: "https://hf.test",
    )
    fake_boto3 = types.SimpleNamespace(client=lambda *_args, **_kwargs: FakeS3Client())
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)

    output = tmp_path / "manifest.jsonl"
    records = build_corpus_manifest(
        BuildCorpusConfig(
            output=output,
            source="recipe",
            recipe="physicalai-robotsim",
            s3_uri="s3://bucket/prefix",
            max_records=1,
            max_shards=1,
            max_shard_gb=0,
            split_ratios=(("sae_train", 1.0),),
        )
    )

    assert len(records) == 1
    assert uploads == [("bucket", "prefix/media/nvidia--PhysicalAI-WorldModel-Synthetic-Embodied-Robot-Scenes/shard/videos/clip.mp4", b"fake mp4")]
    assert sidecar_uploads == [
        (
            "bucket",
            "prefix/sidecars/nvidia--PhysicalAI-WorldModel-Synthetic-Embodied-Robot-Scenes/shard/videos/clip.json",
            b'{"caption": "A simulated robot opens a drawer.", "nb_frames": 32, "simulation_tool": "isaaclab"}\n',
        )
    ]
    assert records[0]["prompt"] == "A simulated robot opens a drawer."
    assert records[0]["metadata"]["prompt_source"] == "robotsim_sidecar"
    assert records[0]["metadata"]["sidecar_path"] == (
        "s3://bucket/prefix/sidecars/nvidia--PhysicalAI-WorldModel-Synthetic-Embodied-Robot-Scenes/shard/videos/clip.json"
    )
    assert records[0]["metadata"]["robotsim_sidecar"] == {
        "nb_frames": 32,
        "simulation_tool": "isaaclab",
    }


def test_physical_ai_instruct_composes_weighted_instruction_manifest(monkeypatch, tmp_path: Path):
    from tools.sae_reasoner import corpus

    seen: list[tuple[str, int]] = []

    def fake_build_recipe_records(recipe, *, config, max_records, seed):
        seen.append((recipe.name, max_records))
        return [
            {
                "id": f"{recipe.name}:{idx}",
                "media_type": "video",
                "media_path": f"hf://dataset/org/repo/{recipe.name}/{idx}.mp4",
                "prompt": f"base prompt {recipe.name} {idx}",
                "tags": ["base"],
                "metadata": {"source": recipe.source, "split": "sae_train"},
            }
            for idx in range(max_records)
        ]

    monkeypatch.setattr(corpus, "build_recipe_records", fake_build_recipe_records)
    output = tmp_path / "physical.jsonl"

    records = build_corpus_manifest(
        BuildCorpusConfig(
            output=output,
            source="recipe",
            recipe="physical-ai-instruct",
            max_records=9,
            split_ratios=(("sae_train", 1.0),),
        )
    )

    assert len(records) == 9
    assert ("physicalai-drivesim", 2) in seen
    assert all(record["prompt"].startswith("You are a Physical AI reasoning assistant.") for record in records)
    assert all(record["metadata"]["source"] == "physical-ai-instruct" for record in records)
    assert all(record["metadata"]["corpus_bucket"] for record in records)
    assert all(record["metadata"]["source_dataset"] for record in records)
    assert all(record["metadata"]["estimated_tokens"] > 0 for record in records)
    assert "physical-ai-instruct" in records[0]["tags"]


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


def test_hf_tar_s3_stream_keeps_scanning_for_selected_sidecars(monkeypatch, tmp_path: Path):
    tar_bytes = io.BytesIO()
    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(b"streamed mp4")
    sidecar_path = tmp_path / "clip.json"
    sidecar_path.write_text('{"caption":"Caption after media.","task":"pick"}', encoding="utf-8")
    with tarfile.open(fileobj=tar_bytes, mode="w") as tar:
        tar.add(clip_path, arcname="videos/clip.mp4")
        tar.add(sidecar_path, arcname="videos/clip.json")
    tar_payload = tar_bytes.getvalue()

    class FakeResponse:
        status_code = 200

        def __init__(self, body: bytes):
            self.raw = io.BytesIO(body)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.raw.close()

        def close(self):
            self.raw.close()

        def raise_for_status(self):
            return None

    class FakeRequests:
        @staticmethod
        def get(url, headers=None, stream=False, timeout=None):
            return FakeResponse(tar_payload)

    class FakeS3Client:
        def upload_fileobj(self, fileobj, bucket, key, ExtraArgs=None):
            fileobj.read()

    monkeypatch.setitem(sys.modules, "requests", FakeRequests)

    records = materialize_hf_tar_stream(
        hf_hub_url=lambda **_kwargs: "https://hf.test/tar",
        repo_id="org/repo",
        shard="data/task/source/shard.tar",
        member_globs=("*.mp4",),
        prompt="Generic.",
        tags=("robotics",),
        bucket="bucket",
        prefix="prefix",
        repo_slug="org--repo",
        s3=FakeS3Client(),
        max_records=1,
        seed=0,
        split_ratios=(("sae_train", 1.0),),
        media_type="auto",
        resume=False,
        shard_index=0,
        worker_index=0,
        num_workers=1,
        sidecar_suffix=".json",
    )

    assert len(records) == 1
    assert records[0]["prompt"] == "Caption after media."
    assert records[0]["metadata"]["robotsim_sidecar"] == {"task": "pick"}


def test_hf_tar_s3_worker_partition_is_stable():
    shards = [f"data/task/source/shard-{idx:03d}.tar" for idx in range(12)]
    worker_parts = [partition_shards_for_worker(shards, worker_index=idx, num_workers=3) for idx in range(3)]
    flattened = sorted(shard for part in worker_parts for shard in part)

    assert flattened == sorted(shards)
    assert all(not (set(worker_parts[left]) & set(worker_parts[right])) for left in range(3) for right in range(left + 1, 3))
    assert partition_shards_for_worker(shards, worker_index=1, num_workers=3) == worker_parts[1]


def test_build_from_hf_conversation_tar_s3_materializes_indexed_media(monkeypatch, tmp_path: Path):
    jsonl_path = tmp_path / "breakfast_actions.jsonl"
    jsonl_path.write_text(
        (
            '{"id":"row-1","messages":['
            '{"role":"user","content":['
            '{"type":"video","video":"P19_cam01_P19_tea.mp4","duration":4.0},'
            '{"type":"text","text":"What is the person doing?"}]},'
            '{"role":"assistant","content":[{"type":"text","text":"Making tea."}]}'
            ']}\n'
        ),
        encoding="utf-8",
    )
    index_path = tmp_path / "index.sqlite"
    with sqlite3.connect(index_path) as conn:
        conn.execute(
            "CREATE TABLE samples (tar_file_id INTEGER, sample_key TEXT, sample_index INTEGER, byte_offset INTEGER, byte_size INTEGER)"
        )
        conn.execute(
            "CREATE TABLE sample_parts (tar_file_id INTEGER, sample_index INTEGER, part_name TEXT, content_byte_offset INTEGER, content_byte_size INTEGER)"
        )
        conn.execute("INSERT INTO samples VALUES (0, 'P19_cam01_P19_tea', 7, 0, 123)")
        conn.execute("INSERT INTO sample_parts VALUES (0, 7, 'mp4', 512, 8)")

    class FakeHfApi:
        def list_repo_files(self, repo_id: str, repo_type: str):
            assert repo_id == "nvidia/Nemotron-VLM-Dataset-v2"
            assert repo_type == "dataset"
            return [
                "breakfast_actions/breakfast_actions.jsonl",
                "breakfast_actions/media/.nv-meta/index.sqlite",
                "breakfast_actions/media/shard_000000.tar",
                "activity_net_1/activity_net_1.jsonl",
            ]

    def fake_download(repo_id: str, repo_type: str, filename: str, **_kwargs):
        if filename == "breakfast_actions/breakfast_actions.jsonl":
            return str(jsonl_path)
        if filename == "breakfast_actions/media/.nv-meta/index.sqlite":
            return str(index_path)
        raise AssertionError(filename)

    uploads: list[tuple[str, str, bytes, str | None]] = []

    class FakeS3Client:
        def put_object(self, Bucket, Key, Body, ContentType=None):
            uploads.append((Bucket, Key, Body, ContentType))

    def fake_hf_hub_url(*, repo_id: str, filename: str, repo_type: str):
        return f"https://hf.test/{repo_id}/{filename}"

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(HfApi=lambda: FakeHfApi(), hf_hub_download=fake_download, hf_hub_url=fake_hf_hub_url),
    )
    monkeypatch.setitem(sys.modules, "boto3", types.SimpleNamespace(client=lambda *_args, **_kwargs: FakeS3Client()))
    monkeypatch.setattr("tools.sae_reasoner.corpus.http_range", lambda session, url, start, end: b"fake mp4")

    records = build_from_hf_conversation_tar_s3(
        repo_id="nvidia/Nemotron-VLM-Dataset-v2",
        include_globs=("breakfast_actions/*.jsonl", "activity_net_1/*.jsonl"),
        tags=("nemotron",),
        s3_uri="s3://bucket/prefix",
        max_records=10,
        seed=0,
        split_ratios=(("sae_train", 1.0),),
        media_type="auto",
    )

    assert len(records) == 1
    assert uploads == [
        (
            "bucket",
            "prefix/media/nvidia--Nemotron-VLM-Dataset-v2/breakfast_actions/P19_cam01_P19_tea.mp4",
            b"fake mp4",
            "video/mp4",
        )
    ]
    assert records[0]["prompt"] == "What is the person doing?"
    assert records[0]["media_type"] == "video"
    assert records[0]["media_path"] == (
        "s3://bucket/prefix/media/nvidia--Nemotron-VLM-Dataset-v2/breakfast_actions/P19_cam01_P19_tea.mp4"
    )
    assert records[0]["metadata"]["source"] == "hf-conversation-tar-s3"
    assert records[0]["metadata"]["assistant_text"] == "Making tea."
    assert records[0]["metadata"]["shard"] == "breakfast_actions/media/shard_000000.tar"


def test_nemotron_recipe_uses_indexed_conversation_rows(monkeypatch, tmp_path: Path):
    jsonl_path = tmp_path / "wiki_en.jsonl"
    jsonl_path.write_text(
        (
            '{"messages":[{"role":"user","content":['
            '{"type":"image","image":"32747872_214332.png"},'
            '{"type":"text","text":"Transcribe the document."}]}]}\n'
        ),
        encoding="utf-8",
    )
    index_path = tmp_path / "index.sqlite"
    with sqlite3.connect(index_path) as conn:
        conn.execute(
            "CREATE TABLE samples (tar_file_id INTEGER, sample_key TEXT, sample_index INTEGER, byte_offset INTEGER, byte_size INTEGER)"
        )
        conn.execute(
            "CREATE TABLE sample_parts (tar_file_id INTEGER, sample_index INTEGER, part_name TEXT, content_byte_offset INTEGER, content_byte_size INTEGER)"
        )
        conn.execute("INSERT INTO samples VALUES (2, '32747872_214332', 3, 0, 42)")
        conn.execute("INSERT INTO sample_parts VALUES (2, 3, 'png', 1024, 7)")

    class FakeHfApi:
        def list_repo_files(self, repo_id: str, repo_type: str):
            return [
                "wiki_en/wiki_en.jsonl",
                "wiki_en/media/.nv-meta/index.sqlite",
                "wiki_en/media/shard_000002.tar",
                "nextqa/nextqa.jsonl",
            ]

    def fake_download(repo_id: str, repo_type: str, filename: str, **_kwargs):
        if filename == "wiki_en/wiki_en.jsonl":
            return str(jsonl_path)
        if filename == "wiki_en/media/.nv-meta/index.sqlite":
            return str(index_path)
        raise AssertionError(filename)

    class FakeS3Client:
        def put_object(self, Bucket, Key, Body, ContentType=None):
            pass

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(HfApi=lambda: FakeHfApi(), hf_hub_download=fake_download, hf_hub_url=lambda **_kwargs: "https://hf.test"),
    )
    monkeypatch.setitem(sys.modules, "boto3", types.SimpleNamespace(client=lambda *_args, **_kwargs: FakeS3Client()))
    monkeypatch.setattr("tools.sae_reasoner.corpus.http_range", lambda session, url, start, end: b"pngdata")

    output = tmp_path / "nemotron.jsonl"
    records = build_corpus_manifest(
        BuildCorpusConfig(
            output=output,
            source="recipe",
            recipe="nemotron-vlm-v2",
            s3_uri="s3://bucket/prefix",
            max_records=5,
            split_ratios=(("sae_train", 1.0),),
        )
    )

    assert len(records) == 1
    assert records[0]["prompt"] == "Transcribe the document."
    assert records[0]["media_type"] == "image"
    assert records[0]["metadata"]["subset"] == "wiki_en"


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
