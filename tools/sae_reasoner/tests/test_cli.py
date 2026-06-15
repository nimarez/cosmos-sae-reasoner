from pathlib import Path

from tools.sae_reasoner.cli import build_parser, shard_name_for_record, shard_path_for_record


def test_shard_path_escapes_dataset_style_ids(tmp_path: Path):
    shard = shard_path_for_record(tmp_path, 7, "../split/example_001")
    assert shard.parent == tmp_path
    assert shard.name == "000007_..%2Fsplit%2Fexample_001.pt"
    assert "/" not in shard.name


def test_shard_name_escapes_dataset_style_ids():
    assert shard_name_for_record(2, "split/example_001") == "000002_split%2Fexample_001.pt"


def test_prompt_format_defaults_by_command():
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
    assert collect_args.prompt_format == "raw"
    assert steer_args.prompt_format == "chat"


def test_build_corpus_manifest_parser_defaults():
    parser = build_parser()
    args = parser.parse_args(["build-corpus-manifest", "--output", "manifest.jsonl"])

    assert args.source == "recipe"
    assert args.recipe == "physicalai-driving"
    assert args.split_ratios == "sae_train=0.90,feature_labeling=0.05,steering_eval=0.05"


def test_render_feature_report_parser_defaults():
    parser = build_parser()
    args = parser.parse_args(["render-feature-report", "--features", "features.jsonl"])

    assert str(args.output) == "outputs/sae_reasoner/reports/features.html"
