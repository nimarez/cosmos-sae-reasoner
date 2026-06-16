from pathlib import Path

import torch

from tools.sae_reasoner.cli import (
    activation_record_assigned,
    activation_meta_bytes,
    activation_meta_tokens,
    activation_sidecar_name,
    activation_worker_metadata_name,
    build_parser,
    collect_top_feature_records,
    cmd_collect_activations,
    cmd_steer,
    collect_metric,
    feature_frequency_summary,
    load_activation_dataset,
    load_activation_examples,
    load_activation_matrix,
    shard_name_for_record,
    shard_path_for_record,
)
from tools.sae_reasoner.sae import SAEConfig, TopKSAE


def test_shard_path_escapes_dataset_style_ids(tmp_path: Path):
    shard = shard_path_for_record(tmp_path, 7, "../split/example_001")
    assert shard.parent == tmp_path
    assert shard.name.startswith("000007_example_001_")
    assert shard.name.endswith(".pt")
    assert "/" not in shard.name


def test_shard_name_escapes_dataset_style_ids():
    name = shard_name_for_record(2, "hf:nvidia/repo/path/to/episode_034240_clip000.mp4")

    assert name.startswith("000002_episode_034240_clip000_")
    assert name.endswith(".pt")
    assert "%2F" not in name


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
    assert collect_args.phase == "prefill"
    assert collect_args.max_new_tokens == 128
    assert collect_args.activation_dtype == "bfloat16"
    assert collect_args.resume is False
    assert collect_args.worker_index == 0
    assert collect_args.num_workers == 1
    assert collect_args.wandb_project is None
    assert collect_args.wandb_mode is None
    assert steer_args.prompt_format == "chat"
    assert steer_args.manifest is None
    assert steer_args.steer_token_kinds == ""
    assert "--prompt-format" not in parser.format_help()


def test_activation_worker_partition_helpers():
    assert activation_record_assigned(5, worker_index=1, num_workers=4) is True
    assert activation_record_assigned(5, worker_index=2, num_workers=4) is False
    assert activation_worker_metadata_name(worker_index=0, num_workers=1) == "metadata.jsonl"
    assert activation_worker_metadata_name(worker_index=3, num_workers=8) == "metadata/worker_003.jsonl"


def test_train_sae_parser_supports_top_feature_wandb_options():
    parser = build_parser()
    args = parser.parse_args(
        [
            "train-sae",
            "--activation-dir",
            "acts",
            "--output",
            "sae.pt",
        ]
    )

    assert args.feature_report_splits == "feature_labeling"
    assert args.feature_report_top_n == 20
    assert args.feature_report_max_features == 128
    assert args.feature_report_rank == "absolute"
    assert args.wandb_top_feature_table is True


def test_collect_resume_skips_existing_shard_and_sidecar(monkeypatch, tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        '{"id":"rec","media_type":"text","prompt":"Describe contact.","tags":[],"metadata":{"split":"sae_train"}}\n',
        encoding="utf-8",
    )
    output_dir = tmp_path / "acts"
    shard_name = shard_name_for_record(0, "rec")
    sidecar_name = activation_sidecar_name(shard_name)
    (output_dir / "metadata").mkdir(parents=True)
    torch.save({"activations": torch.ones(1, 2), "meta": {}}, output_dir / shard_name)
    (output_dir / sidecar_name).write_text(
        '{"id":"rec","num_tokens":1,"hidden_dim":2,"metadata":{"split":"sae_train"},"shard":"'
        + shard_name
        + '"}\n',
        encoding="utf-8",
    )

    class FakeRuntime:
        def __init__(self, *_args, **_kwargs):
            pass

        def load(self):
            return self

        def collect_activations(self, *_args, **_kwargs):
            raise AssertionError("resume should skip existing shard and sidecar")

    monkeypatch.setattr("tools.sae_reasoner.runtime.CosmosReasonerRuntime", FakeRuntime)
    args = build_parser().parse_args(
        [
            "collect-activations",
            "--manifest",
            str(manifest),
            "--layer",
            "18",
            "--output-dir",
            str(output_dir),
            "--resume",
        ]
    )

    assert cmd_collect_activations(args) == 0
    assert '"id": "rec"' in (output_dir / "metadata.jsonl").read_text(encoding="utf-8")


def test_collect_fanout_processes_assigned_records_only(monkeypatch, tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                '{"id":"rec0","media_type":"text","prompt":"zero","tags":[],"metadata":{"split":"sae_train"}}',
                '{"id":"rec1","media_type":"text","prompt":"one","tags":[],"metadata":{"split":"sae_val"}}',
                '{"id":"rec2","media_type":"text","prompt":"two","tags":[],"metadata":{"split":"sae_train"}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "acts"
    seen: list[str] = []

    class FakeRuntime:
        def __init__(self, *_args, **_kwargs):
            pass

        def load(self):
            return self

        def collect_activations(self, record, **_kwargs):
            seen.append(record.id)
            hidden = torch.ones(2, 4, dtype=torch.bfloat16)
            return hidden, {
                "id": record.id,
                "num_tokens": 2,
                "hidden_dim": 4,
                "activation_dtype": "bfloat16",
                "metadata": dict(record.metadata),
            }

    monkeypatch.setattr("tools.sae_reasoner.runtime.CosmosReasonerRuntime", FakeRuntime)
    args = build_parser().parse_args(
        [
            "collect-activations",
            "--manifest",
            str(manifest),
            "--layer",
            "18",
            "--output-dir",
            str(output_dir),
            "--worker-index",
            "1",
            "--num-workers",
            "2",
        ]
    )

    assert cmd_collect_activations(args) == 0
    assert seen == ["rec1"]
    assert not (output_dir / shard_name_for_record(0, "rec0")).exists()
    assert (output_dir / shard_name_for_record(1, "rec1")).exists()
    assert not (output_dir / shard_name_for_record(2, "rec2")).exists()
    assert not (output_dir / "metadata.jsonl").exists()
    worker_metadata = output_dir / "metadata" / "worker_001.jsonl"
    assert worker_metadata.exists()
    assert '"id": "rec1"' in worker_metadata.read_text(encoding="utf-8")


def test_build_corpus_manifest_parser_defaults():
    parser = build_parser()
    args = parser.parse_args(["build-corpus-manifest", "--output", "manifest.jsonl"])

    assert args.source == "recipe"
    assert args.recipe == "robotics-bridge-captions"
    assert args.split_ratios == "sae_train=0.85,sae_val=0.10,feature_labeling=0.025,steering_eval=0.025"


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
            "--shard-list",
            "s3://bucket/manifests/shards.jsonl",
            "--worker-index",
            "3",
            "--num-workers",
            "8",
            "--resume",
            "--stream-tars",
        ]
    )

    assert args.source == "hf-tar-s3"
    assert args.member_glob == ["*.mp4"]
    assert args.max_shards == 2
    assert args.max_shard_gb == 0.5
    assert args.manifest_s3_uri == "s3://bucket/manifests/run.jsonl"
    assert args.shard_list == "s3://bucket/manifests/shards.jsonl"
    assert args.worker_index == 3
    assert args.num_workers == 8
    assert args.resume is True
    assert args.stream_tars is True


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
    assert args.query_kinds == ""


def test_train_sae_parser_defaults_to_all_tokens():
    parser = build_parser()
    args = parser.parse_args(["train-sae", "--activation-dir", "acts", "--output", "sae.pt"])

    assert args.token_kinds == ""
    assert args.phases == ""
    assert args.topk_activation == "relu_topk"
    assert args.batch_topk_momentum == 0.01
    assert args.init_method == "kaiming"
    assert args.init_blend == 0.8
    assert args.activation_norm == "sqrt_d"
    assert args.matryoshka_prefixes == ""
    assert args.matryoshka_loss_coeff == 1.0
    assert args.train_splits == "sae_train"
    assert args.val_splits == "sae_val"
    assert args.val_batch_size is None
    assert args.log_every == 10
    assert args.warmup_steps == 0
    assert args.lr_schedule == "constant"
    assert args.max_grad_norm == 0.0
    assert args.wandb_project is None
    assert args.wandb_mode is None


def test_analyze_sae_parser_defaults_to_train_and_val_splits():
    parser = build_parser()
    args = parser.parse_args(["analyze-sae", "--activation-dir", "acts", "--sae", "sae.pt", "--output", "analysis.json"])

    assert args.splits == "sae_train,sae_val"
    assert args.batch_size == 4096
    assert args.token_kinds == ""
    assert args.phases == ""


def test_collect_metric_flattens_progress_counts():
    metric = collect_metric(
        idx=4,
        meta={
            "num_tokens": 100,
            "activation_dtype": "bfloat16",
            "token_kind_counts": {"video": 64, "text": 30, "special": 6},
            "token_phase_counts": {"prefill": 90, "decode": 10},
        },
        activation_bytes=2000,
        total_tokens=500,
        total_activation_bytes=10000,
        elapsed_seconds=10.0,
        record_seconds=1.5,
    )

    assert metric["collected_examples"] == 5
    assert metric["examples_per_min"] == 30.0
    assert metric["tokens_per_sec"] == 50.0
    assert metric["total_activation_gb"] == 0.00001
    assert metric["token_kind/video"] == 64
    assert metric["token_phase/decode"] == 10
    assert metric["activation_dtype"] == "bfloat16"


def test_activation_meta_accounting_supports_resume_sidecars():
    meta = {"num_tokens": 100, "hidden_dim": 4096, "activation_dtype": "bfloat16"}

    assert activation_meta_tokens(meta) == 100
    assert activation_meta_bytes(meta) == 819200


def test_feature_frequency_summary_reports_dead_fraction():
    class DummyConfig:
        input_dim = 4
        expansion_factor = 2
        top_k = 2
        topk_activation = "relu_topk"
        init_method = "kaiming"
        input_scale = 1.0

    class DummySAE:
        config = DummyConfig()

    class DummyData:
        group_counts = {"all": 10}

    fire_counts = torch.tensor([0, 2, 5, 10])
    fire_rate = fire_counts.double() / 10

    summary = feature_frequency_summary(
        fire_counts=fire_counts,
        fire_rate=fire_rate,
        num_tokens=10,
        data=DummyData(),
        sae=DummySAE(),
        activation_dir="acts",
        sae_path="sae.pt",
        filters={"splits": "sae_train"},
        per_feature_path="analysis.features.jsonl",
    )

    assert summary["dead_features"] == 1
    assert summary["live_features"] == 3
    assert summary["live_feature_frac"] == 0.75
    assert summary["dead_feature_frac"] == 0.25
    assert summary["feature_usage_thresholds"]["ge_1"]["count"] == 3
    assert summary["feature_usage_thresholds"]["ge_2"]["count"] == 3
    assert summary["feature_usage_thresholds"]["ge_10"]["count"] == 1
    assert summary["feature_usage_thresholds"]["ge_1e_2"]["frac"] == 0.75
    assert summary["highest_fire_rate_features"][0]["feature_id"] == 3


def test_collect_top_feature_records_uses_split_and_topk(tmp_path: Path):
    shard = tmp_path / "000000_demo.pt"
    torch.save(
        {
            "activations": torch.tensor([[1.0, 0.0], [3.0, 2.0], [2.0, 1.0]]),
            "meta": {
                "id": "rec-1",
                "prompt": "demo prompt",
                "media_type": "text",
                "media_path": None,
                "tags": ["demo"],
                "metadata": {"split": "feature_labeling"},
                "token_map": [
                    {"kind": "text", "phase": "prefill", "role": "user"},
                    {"kind": "text", "phase": "prefill", "role": "user"},
                    {"kind": "text", "phase": "decode", "role": "assistant"},
                ],
            },
        },
        shard,
    )
    sae = TopKSAE(SAEConfig(input_dim=2, expansion_factor=2, top_k=4, topk_activation="topk"))
    with torch.no_grad():
        sae.encoder.weight.zero_()
        sae.encoder.bias.zero_()
        sae.encoder.weight[0, 0] = 1.0
        sae.encoder.weight[1, 1] = 1.0
        sae.pre_bias.zero_()
    records = collect_top_feature_records(
        activation_dir=tmp_path,
        sae=sae,
        feature_ids=[0, 1],
        top_n=2,
        feature_rank="absolute",
        token_kinds=set(),
        phases=set(),
        splits={"feature_labeling"},
    )

    assert len(records) == 4
    feature0 = [row for row in records if row["feature_id"] == 0]
    feature1 = [row for row in records if row["feature_id"] == 1]
    assert [row["token_index"] for row in feature0] == [1, 2]
    assert [row["token_index"] for row in feature1] == [1, 2]
    assert all(row["record_id"] == "rec-1" for row in records)


def test_steer_parser_supports_manifest_and_token_scope():
    parser = build_parser()
    args = parser.parse_args(
        [
            "steer",
            "--sae",
            "sae.pt",
            "--layer",
            "18",
            "--feature-id",
            "1",
            "--multiplier",
            "3",
            "--manifest",
            "manifest.jsonl",
            "--record-id",
            "clip_001",
            "--scope",
            "prefill",
            "--steer-token-kinds",
            "video,special",
            "--steer-roles",
            "user",
        ]
    )

    assert str(args.manifest) == "manifest.jsonl"
    assert args.record_id == "clip_001"
    assert args.scope == "prefill"
    assert args.steer_token_kinds == "video,special"
    assert args.steer_roles == "user"


def test_text_only_steer_rejects_token_filters_before_loading_runtime(monkeypatch):
    parser = build_parser()
    args = parser.parse_args(
        [
            "steer",
            "--sae",
            "sae.pt",
            "--layer",
            "18",
            "--feature-id",
            "1",
            "--multiplier",
            "3",
            "--prompt",
            "hello",
            "--steer-token-kinds",
            "text",
        ]
    )

    class ForbiddenRuntime:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("runtime should not load for invalid text-only filters")

    monkeypatch.setattr("tools.sae_reasoner.runtime.CosmosReasonerRuntime", ForbiddenRuntime)
    try:
        cmd_steer(args)
    except ValueError as exc:
        assert "require --manifest" in str(exc)
    else:
        raise AssertionError("expected ValueError")


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


def test_load_activation_matrix_filters_by_kind_and_phase(tmp_path: Path):
    torch.save(
        {
            "activations": torch.arange(12, dtype=torch.float32).reshape(3, 4),
            "meta": {
                "token_map": [
                    {"kind": "video", "phase": "prefill"},
                    {"kind": "text", "phase": "prefill"},
                    {"kind": "text", "phase": "decode"},
                ]
            },
        },
        tmp_path / "000000_rec.pt",
    )

    all_rows = load_activation_matrix(tmp_path)
    decode_text = load_activation_matrix(tmp_path, token_kinds={"text"}, phases={"decode"})

    assert all_rows.shape == (3, 4)
    assert decode_text.tolist() == [[8.0, 9.0, 10.0, 11.0]]


def test_load_activation_dataset_keeps_token_group_alignment(tmp_path: Path):
    torch.save(
        {
            "activations": torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
            "meta": {
                "token_map": [
                    {"kind": "video", "phase": "prefill", "role": "user"},
                    {"kind": "special", "phase": "prefill", "role": None},
                    {"kind": "text", "phase": "decode", "role": "assistant"},
                ]
            },
        },
        tmp_path / "000000_rec.pt",
    )

    data = load_activation_dataset(tmp_path, phases={"prefill"})

    assert data.activations.tolist() == [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]
    assert data.activations.dtype == torch.bfloat16
    assert data.group_counts["phase:prefill"] == 2
    assert data.group_counts["kind:video"] == 1
    assert data.group_counts["media:video"] == 1
    assert data.group_counts["special"] == 1


def test_load_activation_matrix_filters_by_split(tmp_path: Path):
    torch.save(
        {
            "activations": torch.ones(2, 3),
            "meta": {"metadata": {"split": "sae_train"}},
        },
        tmp_path / "000000_train.pt",
    )
    torch.save(
        {
            "activations": torch.full((2, 3), 2.0),
            "meta": {"metadata": {"split": "feature_labeling"}},
        },
        tmp_path / "000001_val.pt",
    )

    train = load_activation_matrix(tmp_path, splits={"sae_train"})
    val = load_activation_matrix(tmp_path, splits={"feature_labeling"})
    missing = load_activation_matrix(tmp_path, splits={"steering_eval"}, allow_empty=True)

    assert train.tolist() == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    assert val.tolist() == [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]
    assert missing is None
