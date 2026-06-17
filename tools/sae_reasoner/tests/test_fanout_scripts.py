from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.sae_reasoner.scripts.combine_manifest_shards import combine_manifests, expand_inputs
from tools.sae_reasoner.scripts.enrich_robotsim_manifest_sidecars import (
    enrich_records,
    parse_pax_headers,
    sidecar_s3_key,
    sidecar_member_for_media_member,
    source_shard_and_member,
)
from tools.sae_reasoner.scripts.plan_activation_fanout import make_worker_plan as make_activation_worker_plan
from tools.sae_reasoner.scripts.plan_robotsim_fanout import make_worker_plan, render_all_script


def test_plan_robotsim_fanout_worker_command(tmp_path: Path):
    args = argparse.Namespace(
        s3_uri="s3://bucket/cosmos/robotsim/full",
        max_records_per_worker=0,
        max_shards=0,
        max_shard_gb=0.0,
        seed=123,
        shard_list="s3://bucket/shards/robotsim.jsonl",
        python="python",
        resume=True,
        stream_tars=True,
        num_workers=4,
    )

    plan = make_worker_plan(args, worker_index=2, manifest_dir=tmp_path)

    assert plan.manifest_s3_uri == "s3://bucket/cosmos/robotsim/full/manifests/worker_002.jsonl"
    assert plan.output == str(tmp_path / "robotsim_worker_002.jsonl")
    assert "--worker-index" in plan.command
    assert plan.command[plan.command.index("--worker-index") + 1] == "2"
    assert "--num-workers" in plan.command
    assert plan.command[plan.command.index("--num-workers") + 1] == "4"
    assert "--shard-list" in plan.command
    assert "--resume" in plan.command
    assert "--stream-tars" in plan.command


def test_plan_activation_fanout_worker_command():
    args = argparse.Namespace(
        manifest="s3://bucket/manifests/robotsim.jsonl",
        output_dir="s3://bucket/activations/robotsim_l18",
        num_workers=8,
        model_id="nvidia/Cosmos3-Nano",
        device=None,
        dtype="bfloat16",
        init_mode="pretrained",
        layer=18,
        phase="prefill",
        max_new_tokens=128,
        activation_dtype="bfloat16",
        batch_size=4,
        max_examples=None,
        python="python",
        resume=True,
        wandb_project="sae",
        wandb_entity=None,
        wandb_mode="offline",
        wandb_tags="sae,robotsim",
        run_prefix="robotsim_l18",
    )

    plan = make_activation_worker_plan(args, worker_index=5)

    assert plan.worker_index == 5
    assert plan.num_workers == 8
    assert plan.command[:4] == ["python", "-m", "tools.sae_reasoner", "collect-activations"]
    assert "--worker-index" in plan.command
    assert plan.command[plan.command.index("--worker-index") + 1] == "5"
    assert "--num-workers" in plan.command
    assert plan.command[plan.command.index("--num-workers") + 1] == "8"
    assert "--batch-size" in plan.command
    assert plan.command[plan.command.index("--batch-size") + 1] == "4"
    assert "--resume" in plan.command
    assert "--wandb-run-name" in plan.command
    assert plan.command[plan.command.index("--wandb-run-name") + 1] == "robotsim_l18_worker_005"


def test_render_all_script_runs_from_output_root():
    args = argparse.Namespace(
        s3_uri="s3://bucket/root",
        max_records_per_worker=10,
        max_shards=1,
        max_shard_gb=1.0,
        seed=0,
        shard_list=None,
        python="python",
        resume=False,
        stream_tars=False,
        num_workers=1,
    )
    plan = make_worker_plan(args, worker_index=0, manifest_dir=Path("manifests"))

    script = render_all_script([plan])

    assert 'script_dir="$(cd "$(dirname "$0")" && pwd)"' in script
    assert 'bash "$script_dir/scripts/run_worker_000.sh" > "$script_dir/worker_000.log" 2>&1 &' in script


def test_render_worker_script_uses_workspace_tmp():
    from tools.sae_reasoner.scripts.plan_robotsim_fanout import render_worker_script

    script = render_worker_script(["python", "-m", "tools.sae_reasoner"], env_file=".env")

    assert "COSMOS_SAE_TAR_TMPDIR" in script
    assert "/workspace/tmp/sae_hf_tar" in script
    assert 'mkdir -p "$COSMOS_SAE_TAR_TMPDIR" "$TMPDIR"' in script


def test_combine_manifest_shards_dedupes_and_sorts(tmp_path: Path):
    first = tmp_path / "worker_001.jsonl"
    second = tmp_path / "worker_000.jsonl"
    first.write_text(
        json.dumps({"id": "b", "media_path": "s3://bucket/b.mp4"}) + "\n"
        + json.dumps({"id": "a", "media_path": "s3://bucket/a-old.mp4"}) + "\n",
        encoding="utf-8",
    )
    second.write_text(
        json.dumps({"id": "a", "media_path": "s3://bucket/a.mp4"}) + "\n"
        + json.dumps({"id": "c", "media_path": "s3://bucket/c.mp4"}) + "\n",
        encoding="utf-8",
    )

    records = combine_manifests([str(first), str(second)], dedupe=True)

    assert [record["id"] for record in records] == ["a", "b", "c"]
    assert records[0]["media_path"] == "s3://bucket/a.mp4"


def test_expand_inputs_reads_local_prefix(tmp_path: Path):
    (tmp_path / "worker_000.jsonl").write_text('{"id":"a"}\n', encoding="utf-8")
    (tmp_path / "notes.txt").write_text("skip", encoding="utf-8")

    assert list(expand_inputs([], [str(tmp_path)])) == [str(tmp_path / "worker_000.jsonl")]


def test_enrich_robotsim_manifest_uses_sidecar_caption_without_touching_media():
    records = [
        {
            "id": "hf-tar:nvidia/dataset:data/shard.tar:path/clip.mp4",
            "media_type": "video",
            "media_path": "s3://bucket/materialized/path/clip.mp4",
            "prompt": "Generic prompt.",
            "tags": ["robotsim"],
            "metadata": {
                "repo_id": "nvidia/dataset",
                "shard": "data/shard.tar",
                "member": "path/clip.mp4",
                "key": "materialized/path/clip.mp4",
            },
        }
    ]
    sidecar = {
        "caption": "A robot reaches toward a cup.",
        "simulation_tool": "dreamzero",
        "nb_frames": 32,
        "nested_state": {"joint_state": [1, 2, 3]},
    }

    enriched, stats = enrich_records(
        records,
        repo_id="nvidia/dataset",
        caption_field="caption",
        metadata_mode="compact",
        missing="error",
        sidecar_lookup={("data/shard.tar", "path/clip.json"): sidecar},
    )

    assert stats["enriched_records"] == 1
    assert enriched[0]["media_path"] == "s3://bucket/materialized/path/clip.mp4"
    assert enriched[0]["prompt"] == "A robot reaches toward a cup."
    assert enriched[0]["metadata"]["prompt_source"] == "robotsim_sidecar"
    assert enriched[0]["metadata"]["sidecar_member"] == "path/clip.json"
    assert enriched[0]["metadata"]["robotsim_sidecar"] == {
        "simulation_tool": "dreamzero",
        "nb_frames": 32,
    }


def test_enrich_robotsim_manifest_uploads_full_sidecar():
    records = [
        {
            "id": "rec",
            "media_type": "video",
            "media_path": "s3://bucket/materialized/path/clip.mp4",
            "prompt": "Generic prompt.",
            "metadata": {"shard": "data/shard.tar", "member": "path/clip.mp4"},
        }
    ]
    uploads: list[tuple[str, str, bytes]] = []

    enriched, stats = enrich_records(
        records,
        repo_id="nvidia/dataset",
        caption_field="caption",
        metadata_mode="none",
        sidecar_s3_uri="s3://sidecar-bucket/robotsim/sidecars",
        missing="error",
        sidecar_lookup={("data/shard.tar", "path/clip.json"): {"caption": "Caption.", "state": [1, 2, 3]}},
        sidecar_uploader=lambda bucket, key, body: uploads.append((bucket, key, body)),
    )

    assert stats["uploaded_sidecars"] == 1
    assert uploads == [
        (
            "sidecar-bucket",
            "robotsim/sidecars/nvidia--dataset/shard/path/clip.json",
            b'{"caption": "Caption.", "state": [1, 2, 3]}\n',
        )
    ]
    assert enriched[0]["metadata"]["sidecar_path"] == (
        "s3://sidecar-bucket/robotsim/sidecars/nvidia--dataset/shard/path/clip.json"
    )
    assert "robotsim_sidecar" not in enriched[0]["metadata"]


def test_enrich_robotsim_manifest_missing_keep_preserves_record():
    records = [
        {
            "id": "rec",
            "media_type": "video",
            "media_path": "s3://bucket/clip.mp4",
            "prompt": "Generic prompt.",
            "metadata": {"shard": "data/shard.tar", "member": "path/clip.mp4"},
        }
    ]

    enriched, stats = enrich_records(
        records,
        repo_id="nvidia/dataset",
        caption_field="caption",
        metadata_mode="compact",
        missing="keep",
        sidecar_lookup={},
    )

    assert stats["missing_sidecars"] == 1
    assert enriched == records


def test_enrich_robotsim_manifest_worker_partition():
    records = [
        {"id": "0", "prompt": "zero", "metadata": {"shard": "s.tar", "member": "a.mp4"}},
        {"id": "1", "prompt": "one", "metadata": {"shard": "s.tar", "member": "b.mp4"}},
        {"id": "2", "prompt": "two", "metadata": {"shard": "s.tar", "member": "c.mp4"}},
    ]

    enriched, stats = enrich_records(
        records,
        repo_id="nvidia/dataset",
        caption_field="caption",
        metadata_mode="none",
        missing="error",
        worker_index=1,
        num_workers=2,
        sidecar_lookup={("s.tar", "b.json"): {"caption": "caption one"}},
    )

    assert stats["selected_records"] == 1
    assert [record["id"] for record in enriched] == ["1"]
    assert enriched[0]["prompt"] == "caption one"


def test_robotsim_sidecar_member_and_pax_helpers():
    assert sidecar_member_for_media_member("path/to/rgb.mp4") == "path/to/rgb.json"
    assert sidecar_member_for_media_member("path/to/rgb.json") is None
    assert parse_pax_headers("28 path=long/name/file.json\n") == {"path": "long/name/file.json"}
    assert sidecar_s3_key(
        "root/sidecars",
        repo_id="nvidia/dataset",
        shard="data/collision/shard:1.tar",
        sidecar_member="path/clip%20.json",
    ) == "root/sidecars/nvidia--dataset/shard%3A1/path/clip%2520.json"


def test_robotsim_source_uri_fallback_parses_two_part_repo_id():
    record = {
        "metadata": {
            "source_uri": (
                "hf://dataset/nvidia/PhysicalAI-WorldModel-Synthetic-Embodied-Robot-Scenes/"
                "data/collision/isaaclab/shard.tar#collision/isaaclab/clip.mp4"
            )
        }
    }

    assert source_shard_and_member(record) == (
        "data/collision/isaaclab/shard.tar",
        "collision/isaaclab/clip.mp4",
    )
