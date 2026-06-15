import sys
import types
from pathlib import Path

from tools.sae_reasoner.corpus import (
    BuildCorpusConfig,
    build_corpus_manifest,
    build_from_hf_files,
    parse_split_ratios,
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
