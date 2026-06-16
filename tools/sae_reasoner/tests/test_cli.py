from pathlib import Path

import torch

from tools.sae_reasoner.cli import build_parser, load_activation_examples, shard_name_for_record, shard_path_for_record


def test_shard_path_escapes_dataset_style_ids(tmp_path: Path):
    shard = shard_path_for_record(tmp_path, 7, "../split/example_001")
    assert shard.parent == tmp_path
    assert shard.name == "000007_..%2Fsplit%2Fexample_001.pt"
    assert "/" not in shard.name


def test_shard_name_escapes_dataset_style_ids():
    assert shard_name_for_record(2, "split/example_001") == "000002_split%2Fexample_001.pt"


def test_prompt_format_is_chat_only_by_command():
    parser = build_parser()
    collect_args = parser.parse_args(
        [
            "collect-activations",
            "--manifest",
            "manifest.jsonl",
            "--layer",
            "18",
            "--output-dir",
            "outputs/acts",
        ]
    )
    steer_args = parser.parse_args(
        [
            "steer",
            "--sae",
            "sae.pt",
            "--layer",
            "18",
            "--feature-id",
            "1",
            "--multiplier",
            "5",
            "--prompt",
            "hello",
        ]
    )
    assert collect_args.prompt_format == "chat"
    assert steer_args.prompt_format == "chat"
    assert "--prompt-format" not in parser.format_help()


def test_build_corpus_manifest_parser_defaults():
    parser = build_parser()
    args = parser.parse_args(["build-corpus-manifest", "--output", "manifest.jsonl"])

    assert args.source == "recipe"
    assert args.recipe == "robotics-bridge-captions"
    assert args.split_ratios == "sae_train=0.90,feature_labeling=0.05,steering_eval=0.05"


def test_build_corpus_manifest_parser_hf_tar_s3_options():
    parser = build_parser()
    args = parser.parse_args(
        [
            "build-corpus-manifest",
            "--source",
            "hf-tar-s3",
            "--hf-repo-id",
            "org/repo",
            "--s3-uri",
            "s3://bucket/prefix",
            "--output",
            "manifest.jsonl",
            "--include-glob",
            "data/*/*.tar",
            "--member-glob",
            "*.mp4",
            "--max-shards",
            "2",
            "--max-shard-gb",
            "0.5",
            "--manifest-s3-uri",
            "s3://bucket/manifests/run.jsonl",
        ]
    )

    assert args.source == "hf-tar-s3"
    assert args.member_glob == ["*.mp4"]
    assert args.max_shards == 2
    assert args.max_shard_gb == 0.5
    assert args.manifest_s3_uri == "s3://bucket/manifests/run.jsonl"


def test_render_feature_report_parser_defaults():
    parser = build_parser()
    args = parser.parse_args(["render-feature-report", "--features", "features.jsonl"])

    assert str(args.output) == "outputs/sae_reasoner/reports/features.html"


def test_find_neighbors_parser_defaults():
    parser = build_parser()
    args = parser.parse_args(["find-neighbors", "--activation-dir", "acts", "--output", "neighbors.jsonl"])

    assert args.max_tokens == 5000
    assert args.num_queries == 40
    assert args.neighbors == 8


def test_load_activation_examples_reservoir_samples(tmp_path: Path):
    shard = tmp_path / "000000_rec.pt"
    torch.save(
        {
            "activations": torch.eye(4),
            "meta": {
                "id": "rec",
                "media_type": "image",
                "prompt": "Describe.",
                "token_map": [{"index": i, "kind": "image", "token_id": i, "token_text": "<|image_pad|>"} for i in range(4)],
            },
        },
        shard,
    )

    examples, matrix = load_activation_examples(tmp_path, max_tokens=3, seed=0)

    assert len(examples) == 3
    assert matrix.shape == (3, 4)
    assert all(example["record_id"] == "rec" for example in examples)
    assert all((example["token_info"] or {})["kind"] == "image" for example in examples)
