from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import os
import random
import shutil
import sqlite3
import tarfile
import tempfile
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from string import Formatter
from typing import Any
from urllib.parse import quote, urlencode, urlparse
from urllib.request import urlopen

from .artifacts import write_jsonl
from .manifest import MediaType, ManifestRecord
from .storage import parse_s3_uri, read_text_uri, s3_client

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm"}

GENERIC_TAR_GLOBS = ("*.tar", "*/*.tar", "*/*/*.tar", "*/*/*/*.tar", "data/*.tar", "data/*/*.tar", "data/*/*/*.tar", "shards/*.tar")


@dataclass(frozen=True)
class CorpusRecipe:
    name: str
    source: str
    repo_id: str | None = None
    include_globs: tuple[str, ...] = ()
    member_globs: tuple[str, ...] = ()
    prompt: str = ""
    prompt_sidecar: str | None = None
    prompt_sidecar_field: str | None = None
    prompt_sidecar_strategy: str | None = None
    tar_sidecar_suffix: str | None = None
    tar_sidecar_caption_field: str = "caption"
    tar_sidecar_metadata_mode: str = "compact"
    tar_sidecar_shard_strategy: str | None = None
    tar_sidecar_prompt_source: str = "robotsim_sidecar"
    tar_sidecar_metadata_key: str = "robotsim_sidecar"
    tags: tuple[str, ...] = ()
    notes: str = ""


RECIPES: dict[str, CorpusRecipe] = {
    "physicalai-driving": CorpusRecipe(
        name="physicalai-driving",
        source="hf-files",
        repo_id="nvidia/PhysicalAI-WorldModel-Synthetic-Autonomous-Driving-Scenarios",
        include_globs=("*/*/video/*.mp4",),
        prompt=(
            "Describe the traffic scene, the relevant agents, and the likely next physical events. "
            "Use visible evidence and avoid speculating beyond the clip."
        ),
        prompt_sidecar_strategy="drivesim_description",
        prompt_sidecar_field="t2w_windows[].qwen2p5_7b_caption",
        tags=("cosmos3", "physicalai", "driving", "video"),
        notes="Direct MP4 files with sidecar descriptions in the HF repo.",
    ),
    "physicalai-vantage": CorpusRecipe(
        name="physicalai-vantage",
        source="hf-files",
        repo_id="nvidia/PhysicalAI-VANTAGE-Bench",
        include_globs=("data/*/sequence_*/images/*.jpg", "data/*/sequence_*/images/*.jpeg", "data/*/sequence_*/images/*.png"),
        prompt=(
            "Describe the scene with attention to spatial relations, visible objects, and physical affordances. "
            "If a task is implied by the path, answer in that style without using hidden annotations."
        ),
        tags=("physicalai", "vantage", "image", "spatial"),
        notes="Evaluation-style image tasks. Keep held-out portions separate from training if used for eval.",
    ),
    "robotics-bridge": CorpusRecipe(
        name="robotics-bridge",
        source="lerobot-v3",
        repo_id="nvidia/BridgeData2_LeRobot_v3",
        include_globs=("videos/*/*/*.mp4",),
        prompt=(
            "Describe the robot manipulation episode, including the visible objects, end-effector motion, "
            "contact events, and likely task outcome."
        ),
        tags=("physicalai", "robotics", "bridge", "lerobot", "video"),
        notes="Loose MP4 files from the BridgeData2 LeRobot release, annotated with task metadata from meta/episodes parquet files.",
    ),
    "robotics-bridge-captions": CorpusRecipe(
        name="robotics-bridge-captions",
        source="hf-files",
        repo_id="nvidia/BridgeData2-Subset-Synthetic-Captions",
        include_globs=("sft_dataset_bridge/train/videos/*.mp4", "sft_dataset_bridge/val/videos/*.mp4"),
        prompt=(
            "Describe the robot manipulation clip in detail. Focus on object state, gripper pose, "
            "physical interactions, and what action should happen next."
        ),
        prompt_sidecar="sft_dataset_bridge/{hf_split}/captions/{stem}/caption.txt",
        tags=("physicalai", "robotics", "bridge", "synthetic_captions", "video"),
        notes="BridgeData2 subset with loose MP4 clips and synthetic captions.",
    ),
    "robotics-libero": CorpusRecipe(
        name="robotics-libero",
        source="lerobot-v3",
        repo_id="nvidia/LIBERO_LeRobot_v3",
        include_globs=("*/videos/*/*/*.mp4",),
        prompt=(
            "Describe the robot tabletop task, visible objects, gripper trajectory, contact dynamics, "
            "and the likely next step needed to complete the instruction."
        ),
        tags=("physicalai", "robotics", "libero", "lerobot", "video"),
        notes="Loose MP4 files from the LIBERO LeRobot release, annotated with task metadata from meta/episodes parquet files.",
    ),
    "physicalai-robotsim": CorpusRecipe(
        name="physicalai-robotsim",
        source="hf-tar-s3",
        repo_id="nvidia/PhysicalAI-WorldModel-Synthetic-Embodied-Robot-Scenes",
        include_globs=("*/*/*.tar", "data/*/*/*.tar"),
        member_globs=("*.mp4",),
        prompt=(
            "Describe the robot simulation clip, including the embodiment, scene objects, robot motion, "
            "contacts, collisions, and likely physical outcome."
        ),
        tar_sidecar_suffix=".json",
        tar_sidecar_prompt_source="robotsim_sidecar",
        tar_sidecar_metadata_key="robotsim_sidecar",
        tags=("cosmos3", "physicalai", "robotics", "robotsim", "video"),
        notes="Cosmos RobotSim SDG tar shards. Materialize media and paired JSON metadata to S3-compatible storage.",
    ),
    "physicalai-robotsim-range": CorpusRecipe(
        name="physicalai-robotsim-range",
        source="hf-tar-range",
        repo_id="nvidia/PhysicalAI-WorldModel-Synthetic-Embodied-Robot-Scenes",
        include_globs=("*/*/*.tar", "data/*/*/*.tar"),
        member_globs=("*.mp4",),
        prompt="Analyze the robot simulation clip for physical state, contact, task progress, and likely next action.",
        tar_sidecar_suffix=".json",
        tar_sidecar_prompt_source="robotsim_sidecar",
        tar_sidecar_metadata_key="robotsim_sidecar",
        tags=("cosmos3", "physicalai", "robotics", "robotsim", "video", "hf-tar-range"),
        notes="RobotSim direct HF tar-range manifest; media is cached locally during activation collection.",
    ),
    "physicalai-phyxsim-range": CorpusRecipe(
        name="physicalai-phyxsim-range",
        source="hf-tar-range",
        repo_id="nvidia/PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes",
        include_globs=("videos/*/*.tar",),
        member_globs=("*.mp4",),
        prompt="Analyze the physical interaction clip for object motion, collisions, gravity, and physical plausibility.",
        tar_sidecar_suffix=".json",
        tar_sidecar_caption_field="Qwen3-VL-30B-A3B-Instruct.long|caption_physics.long",
        tar_sidecar_shard_strategy="phyxsim_caption_tar",
        tar_sidecar_prompt_source="phyxsim_caption_sidecar",
        tar_sidecar_metadata_key="phyxsim_sidecar",
        tags=("cosmos3", "physicalai", "physics", "phyxsim", "video", "hf-tar-range"),
        notes="PhyxSim direct HF tar-range manifest with captions from matching caption tar shards.",
    ),
    "physicalai-drivesim": CorpusRecipe(
        name="physicalai-drivesim",
        source="hf-files",
        repo_id="nvidia/PhysicalAI-WorldModel-Synthetic-Autonomous-Driving-Scenarios",
        include_globs=("*/*/video/*.mp4",),
        prompt="Analyze the driving scene for agents, hazards, ego-vehicle decision points, and likely next events.",
        prompt_sidecar_strategy="drivesim_description",
        prompt_sidecar_field="t2w_windows[].qwen2p5_7b_caption",
        tags=("cosmos3", "physicalai", "driving", "drivesim", "video"),
        notes="DriveSim direct loose MP4 manifest with VLM captions from matching description JSON files.",
    ),
    "physicalai-warehouse-range": CorpusRecipe(
        name="physicalai-warehouse-range",
        source="hf-tar-range",
        repo_id="nvidia/PhysicalAI-WorldModel-Synthetic-Warehouse-Operations-Scenes",
        include_globs=GENERIC_TAR_GLOBS,
        member_globs=("*.mp4",),
        prompt="Analyze the warehouse scene for safety, worker/forklift interactions, anomaly risk, and likely next events.",
        tags=("cosmos3", "physicalai", "warehouse", "safety", "video", "hf-tar-range"),
        notes="Warehouse direct HF tar-range manifest.",
    ),
    "physicalai-synhuman-range": CorpusRecipe(
        name="physicalai-synhuman-range",
        source="hf-tar-range",
        repo_id="nvidia/PhysicalAI-WorldModel-Synthetic-Digital-Human-Scenes",
        include_globs=GENERIC_TAR_GLOBS,
        member_globs=("video/*.mp4", "*.mp4"),
        prompt="Analyze the human-scene video for pose, motion, camera movement, interactions, and likely next events.",
        tags=("cosmos3", "physicalai", "human", "synhuman", "video", "hf-tar-range"),
        notes="SynHuman direct HF tar-range manifest.",
    ),
    "nemotron-vlm-v2": CorpusRecipe(
        name="nemotron-vlm-v2",
        source="hf-conversation-tar-s3",
        repo_id="nvidia/Nemotron-VLM-Dataset-v2",
        include_globs=("*/*.jsonl",),
        tags=("nemotron-vlm-v2", "vlm", "conversation"),
        notes=(
            "Conversation JSONL rows whose image/video media is bundled in indexed tar shards. "
            "Rows from subsets without bundled .nv-meta indexes are skipped."
        ),
    ),
    "physical-ai-instruct": CorpusRecipe(
        name="physical-ai-instruct",
        source="physical-ai-instruct",
        tags=("cosmos3", "physicalai", "instruction", "steering"),
        notes="Composite Physical AI instruction corpus for SAE activation collection.",
    ),
}


@dataclass(frozen=True)
class BuildCorpusConfig:
    output: Path
    source: str
    recipe: str | None = None
    hf_repo_id: str | None = None
    hf_split: str = "train"
    s3_uri: str | None = None
    input_uri: str | None = None
    include_globs: tuple[str, ...] = ()
    member_globs: tuple[str, ...] = ()
    prompt: str | None = None
    media_type: str = "auto"
    max_records: int = 1000
    max_shards: int | None = 1
    max_shard_gb: float | None = 1.0
    shard_list_uri: str | None = None
    worker_index: int = 0
    num_workers: int = 1
    resume: bool = False
    stream_tars: bool = False
    seed: int = 0
    split_ratios: tuple[tuple[str, float], ...] = (("sae_train", 0.85), ("sae_val", 0.10), ("feature_labeling", 0.025), ("steering_eval", 0.025))
    id_field: str | None = None
    text_field: str | None = None
    prompt_field: str | None = None
    media_field: str | None = None
    shuffle_buffer: int = 10000
    min_text_chars: int = 64
    max_text_chars: int = 8000
    target_tokens: int = 0
    estimated_image_tokens: int = 1024
    estimated_video_tokens: int = 4096
    estimated_text_tokens: int = 256


def build_corpus_manifest(config: BuildCorpusConfig) -> list[dict[str, Any]]:
    if config.source == "recipe":
        if not config.recipe:
            raise ValueError("--recipe is required when --source recipe")
        recipe = RECIPES[config.recipe]
        if recipe.source == "hf-files":
            records = build_from_hf_files(
                repo_id=required(recipe.repo_id, "recipe repo_id"),
                include_globs=recipe.include_globs,
                prompt=config.prompt or recipe.prompt,
                prompt_sidecar=None if config.prompt else recipe.prompt_sidecar,
                prompt_sidecar_field=None if config.prompt else recipe.prompt_sidecar_field,
                prompt_sidecar_strategy=None if config.prompt else recipe.prompt_sidecar_strategy,
                tags=recipe.tags,
                max_records=config.max_records,
                seed=config.seed,
                split_ratios=config.split_ratios,
                media_type=config.media_type,
            )
        elif recipe.source == "lerobot-v3":
            records = build_from_lerobot_v3(
                repo_id=required(recipe.repo_id, "recipe repo_id"),
                include_globs=recipe.include_globs,
                prompt=config.prompt or recipe.prompt,
                tags=recipe.tags,
                max_records=config.max_records,
                seed=config.seed,
                split_ratios=config.split_ratios,
                media_type=config.media_type,
                include_task_context=config.prompt is None,
            )
        elif recipe.source == "hf-tar-s3":
            records = build_from_hf_tar_s3(
                repo_id=required(recipe.repo_id, "recipe repo_id"),
                shard_globs=recipe.include_globs,
                member_globs=config.member_globs or recipe.member_globs,
                prompt=config.prompt or recipe.prompt,
                tags=recipe.tags,
                s3_uri=required(config.s3_uri, "--s3-uri"),
                max_records=config.max_records,
                max_shards=config.max_shards,
                max_shard_gb=config.max_shard_gb,
                seed=config.seed,
                split_ratios=config.split_ratios,
                media_type=config.media_type,
                shard_list_uri=config.shard_list_uri,
                worker_index=config.worker_index,
                num_workers=config.num_workers,
                resume=config.resume,
                stream_tars=config.stream_tars,
                sidecar_suffix=recipe.tar_sidecar_suffix,
                sidecar_caption_field=recipe.tar_sidecar_caption_field,
                sidecar_metadata_mode=recipe.tar_sidecar_metadata_mode,
                sidecar_prompt_source=recipe.tar_sidecar_prompt_source,
                sidecar_metadata_key=recipe.tar_sidecar_metadata_key,
                sidecar_s3_uri=s3_child_uri(required(config.s3_uri, "--s3-uri"), "sidecars") if recipe.tar_sidecar_suffix else None,
            )
        elif recipe.source == "hf-tar-range":
            records = build_from_hf_tar_range(
                repo_id=required(recipe.repo_id, "recipe repo_id"),
                shard_globs=recipe.include_globs,
                member_globs=config.member_globs or recipe.member_globs,
                prompt=config.prompt or recipe.prompt,
                tags=recipe.tags,
                max_records=config.max_records,
                max_shards=config.max_shards,
                max_shard_gb=config.max_shard_gb,
                seed=config.seed,
                split_ratios=config.split_ratios,
                media_type=config.media_type,
                shard_list_uri=config.shard_list_uri,
                worker_index=config.worker_index,
                num_workers=config.num_workers,
                sidecar_suffix=recipe.tar_sidecar_suffix,
                sidecar_caption_field=recipe.tar_sidecar_caption_field,
                sidecar_metadata_mode=recipe.tar_sidecar_metadata_mode,
                sidecar_shard_strategy=recipe.tar_sidecar_shard_strategy,
                sidecar_prompt_source=recipe.tar_sidecar_prompt_source,
                sidecar_metadata_key=recipe.tar_sidecar_metadata_key,
            )
        elif recipe.source == "hf-conversation-tar-s3":
            records = build_from_hf_conversation_tar_s3(
                repo_id=required(recipe.repo_id, "recipe repo_id"),
                include_globs=config.include_globs or recipe.include_globs,
                tags=recipe.tags,
                s3_uri=required(config.s3_uri, "--s3-uri"),
                max_records=config.max_records,
                seed=config.seed,
                split_ratios=config.split_ratios,
                media_type=config.media_type,
                worker_index=config.worker_index,
                num_workers=config.num_workers,
                resume=config.resume,
            )
        elif recipe.source == "physical-ai-instruct":
            records = build_physical_ai_instruct_manifest(config)
        else:
            raise ValueError(f"recipe {recipe.name!r} has unsupported source {recipe.source!r}")
    elif config.source == "hf-files":
        records = build_from_hf_files(
            repo_id=required(config.hf_repo_id, "--hf-repo-id"),
            include_globs=config.include_globs,
            prompt=required(config.prompt, "--prompt"),
            prompt_sidecar=None,
            prompt_sidecar_field=None,
            prompt_sidecar_strategy=None,
            tags=("hf", "external"),
            max_records=config.max_records,
            seed=config.seed,
            split_ratios=config.split_ratios,
            media_type=config.media_type,
        )
    elif config.source == "hf-dataset":
        records = build_from_hf_dataset(config)
    elif config.source == "hf-tar-s3":
        records = build_from_hf_tar_s3(
            repo_id=required(config.hf_repo_id, "--hf-repo-id"),
            shard_globs=config.include_globs,
            member_globs=config.member_globs or ("*.mp4",),
            prompt=required(config.prompt, "--prompt"),
            tags=("hf", "external", "materialized"),
            s3_uri=required(config.s3_uri, "--s3-uri"),
            max_records=config.max_records,
            max_shards=config.max_shards,
            max_shard_gb=config.max_shard_gb,
            seed=config.seed,
            split_ratios=config.split_ratios,
            media_type=config.media_type,
            shard_list_uri=config.shard_list_uri,
            worker_index=config.worker_index,
            num_workers=config.num_workers,
            resume=config.resume,
            stream_tars=config.stream_tars,
            sidecar_suffix=None,
            sidecar_caption_field="caption",
            sidecar_metadata_mode="compact",
            sidecar_prompt_source="tar_sidecar",
            sidecar_metadata_key="tar_sidecar",
            sidecar_s3_uri=None,
        )
    elif config.source == "hf-tar-range":
        records = build_from_hf_tar_range(
            repo_id=required(config.hf_repo_id, "--hf-repo-id"),
            shard_globs=config.include_globs,
            member_globs=config.member_globs or ("*.mp4",),
            prompt=required(config.prompt, "--prompt"),
            tags=("hf", "external", "range"),
            max_records=config.max_records,
            max_shards=config.max_shards,
            max_shard_gb=config.max_shard_gb,
            seed=config.seed,
            split_ratios=config.split_ratios,
            media_type=config.media_type,
            shard_list_uri=config.shard_list_uri,
            worker_index=config.worker_index,
            num_workers=config.num_workers,
            sidecar_suffix=None,
            sidecar_caption_field="caption",
            sidecar_metadata_mode="compact",
            sidecar_shard_strategy=None,
            sidecar_prompt_source="tar_sidecar",
            sidecar_metadata_key="tar_sidecar",
        )
    elif config.source == "hf-conversation-tar-s3":
        records = build_from_hf_conversation_tar_s3(
            repo_id=required(config.hf_repo_id, "--hf-repo-id"),
            include_globs=config.include_globs,
            tags=("hf", "external", "conversation", "materialized"),
            s3_uri=required(config.s3_uri, "--s3-uri"),
            max_records=config.max_records,
            seed=config.seed,
            split_ratios=config.split_ratios,
            media_type=config.media_type,
            worker_index=config.worker_index,
            num_workers=config.num_workers,
            resume=config.resume,
        )
    elif config.source == "s3-prefix":
        records = build_from_s3_prefix(config)
    elif config.source == "jsonl":
        records = build_from_jsonl_uri(required(config.input_uri, "--input-uri"), config)
    elif config.source == "physical-ai-instruct":
        records = build_physical_ai_instruct_manifest(config)
    else:
        raise ValueError(f"unsupported source {config.source!r}")
    write_jsonl(config.output, records)
    return records


@dataclass(frozen=True)
class PhysicalAiSourceSpec:
    recipe: str
    bucket: str
    source_dataset: str
    weight: float


PHYSICAL_AI_INSTRUCT_SOURCES = (
    PhysicalAiSourceSpec("physicalai-robotsim-range", "robot_simulation", "robotsim", 0.25),
    PhysicalAiSourceSpec("robotics-bridge-captions", "robot_manipulation_real", "bridge_captions", 0.15),
    PhysicalAiSourceSpec("robotics-bridge", "robot_manipulation_real", "bridge", 0.05),
    PhysicalAiSourceSpec("robotics-libero", "robot_manipulation_real", "libero", 0.05),
    PhysicalAiSourceSpec("physicalai-phyxsim-range", "physics_simulation", "phyxsim", 0.15),
    PhysicalAiSourceSpec("physicalai-drivesim", "driving", "drivesim", 0.15),
    PhysicalAiSourceSpec("physicalai-warehouse-range", "warehouse_safety", "warehouse", 0.10),
    PhysicalAiSourceSpec("physicalai-synhuman-range", "human_scene", "synhuman", 0.10),
)

PHYSICAL_AI_PROMPT_FAMILIES: dict[str, tuple[tuple[str, str], ...]] = {
    "robot_manipulation_real": (
        ("state", "Describe the current robot manipulation state: objects, gripper pose, object state, contacts, and task progress."),
        ("next_action", "What should the robot do next to make progress? Ground the answer in visible object state and gripper position."),
        ("contact", "Identify visible contact events or near-contact relationships, and explain what object state has changed or is likely to change."),
        ("failure", "Is the manipulation attempt succeeding, failing, or ambiguous? Cite the visual evidence."),
        ("plan", "Give a short physical action plan for the next few steps, focusing on feasible end-effector motion."),
    ),
    "robot_simulation": (
        ("state", "Describe the robot embodiment, scene objects, object states, contacts, and task progress in the simulation."),
        ("next_action", "What action or motion should happen next? Explain using the robot pose, objects, and current physical constraints."),
        ("contact", "Which contacts, collisions, or support relationships matter in this clip? Describe their physical consequences."),
        ("affordance", "Which visible objects or regions are actionable for the robot, and what affordances do they provide?"),
        ("failure", "Does the simulated behavior look successful, failed, or unstable? Explain the physical evidence."),
    ),
    "physics_simulation": (
        ("physics", "Is the motion physically plausible? Discuss gravity, collisions, momentum, support, and object permanence."),
        ("state", "Describe the current physical state of the objects, including motion, contact, and likely forces."),
        ("temporal", "What is the key physical interaction in the clip, and when does it occur relative to the sequence?"),
        ("next_event", "What physical event is likely to happen next? Base the answer on visible trajectories and constraints."),
    ),
    "driving": (
        ("state", "Describe the driving scene: ego context, surrounding agents, lanes, traffic controls, weather, and visibility."),
        ("hazard", "Identify any hazards or safety-critical agents, and explain why they matter for the ego vehicle."),
        ("next_action", "What should the ego vehicle do next? Consider traffic rules, agent intent, and likely trajectories."),
        ("temporal", "What event or agent motion is most important over time in this clip?"),
    ),
    "warehouse_safety": (
        ("state", "Describe the warehouse scene, including workers, forklifts, shelves, boxes, paths, and event state."),
        ("hazard", "Identify any safety hazard, near miss, blocked path, fire, collision risk, or worker-forklift interaction."),
        ("next_action", "What should the relevant agent do next to remain safe or complete the task?"),
        ("failure", "Is the scene normal, anomalous, or unsafe? Explain the visual evidence."),
    ),
    "human_scene": (
        ("state", "Describe the human-scene state: people, poses, motion, camera movement, and interaction context."),
        ("temporal", "What motion or interaction changes over time, and what is likely to happen next?"),
        ("affordance", "Which objects, paths, or spaces are actionable for the visible humans?"),
        ("physics", "Comment on physical plausibility, body motion, camera motion, and scene consistency."),
    ),
    "general_grounding": (
        ("grounding", "Answer the visual question with careful grounding in the image or video. Avoid unsupported claims."),
        ("spatial", "Describe the spatial relations needed to answer the question, then give the concise answer."),
    ),
}


def build_physical_ai_instruct_manifest(config: BuildCorpusConfig) -> list[dict[str, Any]]:
    target_tokens = config.target_tokens if config.target_tokens > 0 else 0
    total_weight = sum(spec.weight for spec in PHYSICAL_AI_INSTRUCT_SOURCES)
    records: list[dict[str, Any]] = []
    for source_index, spec in enumerate(PHYSICAL_AI_INSTRUCT_SOURCES):
        source_token_budget = int(target_tokens * spec.weight / total_weight) if target_tokens else 0
        source_max_records = (
            physical_source_record_budget(config, source_token_budget)
            if target_tokens
            else physical_source_record_count(config, weight=spec.weight, total_weight=total_weight)
        )
        if source_max_records <= 0:
            continue
        recipe = RECIPES[spec.recipe]
        base_records = build_recipe_records(
            recipe,
            config=config,
            max_records=source_max_records,
            seed=config.seed + 1009 * (source_index + 1),
        )
        source_records = instructionize_physical_records(
            base_records,
            bucket=spec.bucket,
            source_dataset=spec.source_dataset,
            seed=config.seed + 9173 * (source_index + 1),
            split_ratios=config.split_ratios,
            config=config,
            target_tokens=source_token_budget,
        )
        records.extend(source_records)
    rng = random.Random(config.seed)
    rng.shuffle(records)
    if config.max_records > 0:
        records = records[: config.max_records]
    return records


def physical_source_record_budget(config: BuildCorpusConfig, token_budget: int) -> int:
    if token_budget <= 0:
        return max(0, config.max_records)
    estimate = max(1, config.estimated_video_tokens)
    return max(1, int(token_budget / estimate) + 1)


def physical_source_record_count(config: BuildCorpusConfig, *, weight: float, total_weight: float) -> int:
    if config.max_records <= 0:
        return 0
    return max(1, int(config.max_records * weight / total_weight) + 1)


def build_recipe_records(recipe: CorpusRecipe, *, config: BuildCorpusConfig, max_records: int, seed: int) -> list[dict[str, Any]]:
    if recipe.source == "hf-files":
        return build_from_hf_files(
            repo_id=required(recipe.repo_id, "recipe repo_id"),
            include_globs=recipe.include_globs,
            prompt=config.prompt or recipe.prompt,
            prompt_sidecar=None if config.prompt else recipe.prompt_sidecar,
            prompt_sidecar_field=None if config.prompt else recipe.prompt_sidecar_field,
            prompt_sidecar_strategy=None if config.prompt else recipe.prompt_sidecar_strategy,
            tags=recipe.tags,
            max_records=max_records,
            seed=seed,
            split_ratios=config.split_ratios,
            media_type=config.media_type,
        )
    if recipe.source == "lerobot-v3":
        return build_from_lerobot_v3(
            repo_id=required(recipe.repo_id, "recipe repo_id"),
            include_globs=recipe.include_globs,
            prompt=config.prompt or recipe.prompt,
            tags=recipe.tags,
            max_records=max_records,
            seed=seed,
            split_ratios=config.split_ratios,
            media_type=config.media_type,
            include_task_context=config.prompt is None,
        )
    if recipe.source == "hf-tar-range":
        return build_from_hf_tar_range(
            repo_id=required(recipe.repo_id, "recipe repo_id"),
            shard_globs=recipe.include_globs,
            member_globs=config.member_globs or recipe.member_globs,
            prompt=config.prompt or recipe.prompt,
            tags=recipe.tags,
            max_records=max_records,
            max_shards=config.max_shards,
            max_shard_gb=config.max_shard_gb,
            seed=seed,
            split_ratios=config.split_ratios,
            media_type=config.media_type,
            shard_list_uri=config.shard_list_uri,
            worker_index=config.worker_index,
            num_workers=config.num_workers,
            sidecar_suffix=recipe.tar_sidecar_suffix,
            sidecar_caption_field=recipe.tar_sidecar_caption_field,
            sidecar_metadata_mode=recipe.tar_sidecar_metadata_mode,
            sidecar_shard_strategy=recipe.tar_sidecar_shard_strategy,
            sidecar_prompt_source=recipe.tar_sidecar_prompt_source,
            sidecar_metadata_key=recipe.tar_sidecar_metadata_key,
        )
    raise ValueError(f"physical-ai-instruct cannot use recipe source {recipe.source!r} for {recipe.name!r}")


def instructionize_physical_records(
    records: list[dict[str, Any]],
    *,
    bucket: str,
    source_dataset: str,
    seed: int,
    split_ratios: tuple[tuple[str, float], ...],
    config: BuildCorpusConfig,
    target_tokens: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    output: list[dict[str, Any]] = []
    token_total = 0
    for record in records:
        family, instruction = rng.choice(PHYSICAL_AI_PROMPT_FAMILIES.get(bucket, PHYSICAL_AI_PROMPT_FAMILIES["general_grounding"]))
        next_record = physical_instruction_record(
            record,
            bucket=bucket,
            source_dataset=source_dataset,
            prompt_family=family,
            instruction=instruction,
            seed=seed,
            split_ratios=split_ratios,
            config=config,
        )
        output.append(next_record)
        token_total += int((next_record.get("metadata") or {}).get("estimated_tokens") or 0)
        if target_tokens > 0 and token_total >= target_tokens:
            break
    return output


def physical_instruction_record(
    record: dict[str, Any],
    *,
    bucket: str,
    source_dataset: str,
    prompt_family: str,
    instruction: str,
    seed: int,
    split_ratios: tuple[tuple[str, float], ...],
    config: BuildCorpusConfig,
) -> dict[str, Any]:
    base_id = str(record["id"])
    record_id = f"{base_id}:physical-ai-instruct:{prompt_family}"
    split = split_for_record_id(record_id, seed, split_ratios)
    metadata = dict(record.get("metadata") or {})
    base_prompt = str(record.get("prompt") or "").strip()
    prompt = render_physical_instruction_prompt(instruction, base_prompt=base_prompt, bucket=bucket)
    estimated_tokens = estimate_manifest_tokens({**record, "prompt": prompt}, config)
    metadata.update(
        {
            "source": "physical-ai-instruct",
            "base_source": metadata.get("source"),
            "base_record_id": base_id,
            "base_prompt": base_prompt,
            "corpus_bucket": bucket,
            "prompt_family": prompt_family,
            "source_dataset": source_dataset,
            "split": split,
            "estimated_tokens": estimated_tokens,
        }
    )
    tags = sorted(set(record.get("tags") or []) | {"physical-ai-instruct", bucket, source_dataset, prompt_family, split})
    return make_manifest_dict(
        record_id=record_id,
        media_type=record["media_type"],
        prompt=prompt,
        media_path=record.get("media_path"),
        tags=tags,
        metadata=metadata,
    )


def render_physical_instruction_prompt(instruction: str, *, base_prompt: str, bucket: str) -> str:
    header = "You are a Physical AI reasoning assistant. Answer using only visible evidence from the media."
    parts = [header, instruction]
    if base_prompt:
        parts.append(f"Available caption or dataset context:\n{base_prompt}")
    if bucket in {"robot_simulation", "robot_manipulation_real"}:
        parts.append("Be specific about contacts, object state, embodiment motion, and task progress.")
    elif bucket == "driving":
        parts.append("Be specific about agent intent, hazards, traffic context, and ego-vehicle implications.")
    elif bucket == "physics_simulation":
        parts.append("Be specific about physical causes, constraints, collisions, and likely state changes.")
    return "\n\n".join(parts)


def estimate_manifest_tokens(record: dict[str, Any], config: BuildCorpusConfig) -> int:
    prompt_tokens = max(config.estimated_text_tokens, len(str(record.get("prompt") or "")) // 4)
    media_type = record.get("media_type")
    if media_type == "video":
        return prompt_tokens + config.estimated_video_tokens
    if media_type == "image":
        return prompt_tokens + config.estimated_image_tokens
    return prompt_tokens


def build_from_hf_files(
    *,
    repo_id: str,
    include_globs: tuple[str, ...],
    prompt: str,
    tags: tuple[str, ...],
    max_records: int,
    seed: int,
    split_ratios: tuple[tuple[str, float], ...],
    media_type: str,
    prompt_sidecar: str | None = None,
    prompt_sidecar_field: str | None = None,
    prompt_sidecar_strategy: str | None = None,
) -> list[dict[str, Any]]:
    try:
        from huggingface_hub import HfApi
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("HF file manifests require huggingface_hub.") from exc
    files = HfApi().list_repo_files(repo_id, repo_type="dataset")
    candidates = [path for path in files if match_any(path, include_globs) and infer_media_type(path, media_type) != "text"]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    records: list[dict[str, Any]] = []
    for idx, path in enumerate(candidates[:max_records]):
        record_id = f"hf:{repo_id}:{path}"
        split = split_for_record_id(record_id, seed, split_ratios)
        inferred = infer_media_type(path, media_type)
        template_values = hf_template_values(path=path, repo_id=repo_id)
        sidecar_path = prompt_sidecar_path_for_media(
            path,
            prompt_sidecar=prompt_sidecar,
            prompt_sidecar_strategy=prompt_sidecar_strategy,
            template_values=template_values,
        )
        prompt_text = render_template(prompt, **template_values)
        prompt_source = "template"
        if sidecar_path and sidecar_path in files:
            from huggingface_hub import hf_hub_download

            sidecar_file = Path(hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=sidecar_path))
            sidecar_prompt = prompt_from_sidecar_file(sidecar_file, field=prompt_sidecar_field)
            if sidecar_prompt:
                prompt_text = sidecar_prompt
                prompt_source = "sidecar_json" if prompt_sidecar_field else "sidecar"
        record_tags = sorted(set(tags + (split,) + tags_from_path(path)))
        records.append(
            make_manifest_dict(
                record_id=f"hf:{repo_id}:{path}",
                media_type=inferred,
                prompt=prompt_text,
                media_path=f"hf://dataset/{repo_id}/{path}",
                tags=record_tags,
                metadata={
                    "source": "hf-files",
                    "source_uri": f"hf://dataset/{repo_id}/{path}",
                    "repo_id": repo_id,
                    "path": path,
                    "prompt_source": prompt_source,
                    "prompt_sidecar": sidecar_path,
                    "split": split,
                },
            )
        )
    return records


def prompt_sidecar_path_for_media(
    path: str,
    *,
    prompt_sidecar: str | None,
    prompt_sidecar_strategy: str | None,
    template_values: dict[str, str],
) -> str | None:
    if prompt_sidecar_strategy == "drivesim_description":
        media_path = PurePosixPath(path)
        if media_path.parent.name != "video":
            return None
        return str(media_path.parent.parent / "description" / f"{media_path.stem}.json")
    if prompt_sidecar_strategy:
        raise ValueError(f"unsupported prompt sidecar strategy: {prompt_sidecar_strategy!r}")
    return render_template(prompt_sidecar, **template_values) if prompt_sidecar else None


def prompt_from_sidecar_file(path: Path, *, field: str | None) -> str | None:
    if field:
        data = json.loads(path.read_text(encoding="utf-8"))
        return extract_json_text(data, field)
    prompt = path.read_text(encoding="utf-8").strip()
    return prompt or None


def extract_json_text(data: Any, field: str) -> str | None:
    for candidate in field.split("|"):
        values = extract_json_values(data, candidate.strip())
        text_values = [stringify_caption_value(value) for value in values]
        text_values = [value for value in text_values if value]
        if text_values:
            return "\n".join(text_values)
    return None


def extract_json_values(data: Any, field: str) -> list[Any]:
    if not field:
        return []
    values = [data]
    for token in field.split("."):
        expand_list = token.endswith("[]")
        key = token[:-2] if expand_list else token
        next_values: list[Any] = []
        for value in values:
            if isinstance(value, dict) and key in value:
                child = value[key]
            else:
                continue
            if expand_list:
                if isinstance(child, list):
                    next_values.extend(child)
            else:
                next_values.append(child)
        values = next_values
        if not values:
            break
    return values


def stringify_caption_value(value: Any) -> str | None:
    if isinstance(value, str):
        value = value.strip()
        return value or None
    if isinstance(value, (int, float, bool)):
        return str(value)
    return None


def build_from_lerobot_v3(
    *,
    repo_id: str,
    include_globs: tuple[str, ...],
    prompt: str,
    tags: tuple[str, ...],
    max_records: int,
    seed: int,
    split_ratios: tuple[tuple[str, float], ...],
    media_type: str,
    include_task_context: bool = True,
) -> list[dict[str, Any]]:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("LeRobot manifests require huggingface_hub.") from exc
    pd = None
    if include_task_context:
        try:
            import pandas as pd
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("LeRobot manifests require pandas and a parquet engine.") from exc

    files = HfApi().list_repo_files(repo_id, repo_type="dataset")
    candidates = [path for path in files if match_any(path, include_globs) and infer_media_type(path, media_type) != "text"]
    task_index = load_lerobot_video_tasks(repo_id=repo_id, files=files, hf_hub_download=hf_hub_download, pandas=pd) if pd is not None else {}
    rng = random.Random(seed)
    rng.shuffle(candidates)
    records: list[dict[str, Any]] = []
    for path in candidates[:max_records]:
        record_id = f"hf:{repo_id}:{path}"
        split = split_for_record_id(record_id, seed, split_ratios)
        inferred = infer_media_type(path, media_type)
        template_values = hf_template_values(path=path, repo_id=repo_id)
        base_prompt = render_template(prompt, **template_values)
        annotations = task_index.get(path, [])
        task_texts = unique_lerobot_tasks(annotations)
        if task_texts:
            task_block = "\n".join(f"- {task}" for task in task_texts[:24])
            prompt_text = f"{base_prompt}\n\nLeRobot task annotations in this video shard:\n{task_block}"
            prompt_source = "lerobot_tasks"
        else:
            prompt_text = base_prompt
            prompt_source = "template"
        record_tags = sorted(set(tags + (split,) + tags_from_path(path)))
        records.append(
            make_manifest_dict(
                record_id=record_id,
                media_type=inferred,
                prompt=prompt_text,
                media_path=f"hf://dataset/{repo_id}/{path}",
                tags=record_tags,
                metadata={
                    "source": "hf-lerobot-v3",
                    "source_uri": f"hf://dataset/{repo_id}/{path}",
                    "repo_id": repo_id,
                    "path": path,
                    "prompt_source": prompt_source,
                    "lerobot_episode_count": len(annotations),
                    "lerobot_tasks": task_texts[:24],
                    "split": split,
                },
            )
        )
    return records


def load_lerobot_video_tasks(*, repo_id: str, files: list[str], hf_hub_download: Any, pandas: Any) -> dict[str, list[dict[str, Any]]]:
    episode_files = sorted(path for path in files if path.endswith(".parquet") and lerobot_meta_kind(path) == "episodes")
    task_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode_file in episode_files:
        prefix = lerobot_dataset_prefix(episode_file)
        local_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=episode_file)
        frame = pandas.read_parquet(local_path)
        for row in dataframe_dict_rows(frame):
            for video_path in lerobot_video_paths_for_episode(row, prefix=prefix):
                task_index[video_path].append(
                    {
                        "episode_index": row.get("episode_index"),
                        "tasks": normalize_lerobot_tasks(row.get("tasks")),
                    }
                )
    return task_index


def lerobot_meta_kind(path: str) -> str | None:
    parts = PurePosixPath(path).parts
    for idx, part in enumerate(parts):
        if part == "meta" and idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def lerobot_dataset_prefix(path: str) -> str:
    parts = PurePosixPath(path).parts
    for idx, part in enumerate(parts):
        if part == "meta":
            return "/".join(parts[:idx])
    return ""


def dataframe_dict_rows(frame: Any) -> Iterable[dict[str, Any]]:
    for _idx, row in frame.iterrows():
        yield dict(row)


def lerobot_video_paths_for_episode(row: dict[str, Any], *, prefix: str) -> list[str]:
    paths: list[str] = []
    for key, chunk_value in row.items():
        if not isinstance(key, str) or not key.startswith("videos/") or not key.endswith("/chunk_index"):
            continue
        video_key = key[: -len("/chunk_index")]
        file_value = row.get(f"{video_key}/file_index")
        chunk_index = int_or_none(chunk_value)
        file_index = int_or_none(file_value)
        if chunk_index is None or file_index is None:
            continue
        rel_path = f"{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        paths.append(f"{prefix}/{rel_path}" if prefix else rel_path)
    return paths


def int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_lerobot_tasks(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, Iterable):
        tasks = []
        for item in value:
            if isinstance(item, str) and item.strip():
                tasks.append(item.strip())
        return tasks
    return []


def unique_lerobot_tasks(annotations: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    tasks: list[str] = []
    for annotation in annotations:
        for task in normalize_lerobot_tasks(annotation.get("tasks")):
            if task not in seen:
                seen.add(task)
                tasks.append(task)
    return tasks


def build_from_hf_dataset(config: BuildCorpusConfig) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("HF streaming datasets require the datasets package.") from exc
    repo_id = required(config.hf_repo_id, "--hf-repo-id")
    dataset = load_dataset(repo_id, split=config.hf_split, streaming=True)
    if config.shuffle_buffer > 0:
        dataset = dataset.shuffle(seed=config.seed, buffer_size=config.shuffle_buffer)
    records: list[dict[str, Any]] = []
    for row_idx, row in enumerate(dataset):
        if len(records) >= config.max_records:
            break
        record = manifest_from_dataset_row(row_idx, row, config, repo_id)
        if record is not None:
            records.append(record)
    return records


def manifest_from_dataset_row(row_idx: int, row: dict[str, Any], config: BuildCorpusConfig, repo_id: str) -> dict[str, Any] | None:
    record_id = str(row.get(config.id_field, f"hf:{repo_id}:{config.hf_split}:{row_idx}")) if config.id_field else f"hf:{repo_id}:{config.hf_split}:{row_idx}"
    split = split_for_record_id(record_id, config.seed, config.split_ratios)
    media_path = value_as_path(row.get(config.media_field)) if config.media_field else None
    if media_path:
        media_type = infer_media_type(media_path, config.media_type)
        prompt = prompt_from_row(row, config)
        if not prompt:
            return None
    else:
        media_type = "text"
        prompt = prompt_from_row(row, config)
        if not prompt:
            return None
        prompt = prompt[: config.max_text_chars]
        if len(prompt) < config.min_text_chars:
            return None
    return make_manifest_dict(
        record_id=record_id,
        media_type=media_type,
        prompt=prompt,
        media_path=media_path,
        tags=sorted({"hf", "external", split, media_type}),
        metadata={"source": "hf-dataset", "repo_id": repo_id, "split": split, "row_index": row_idx},
    )


@dataclass(frozen=True)
class NemotronMediaRef:
    kind: str
    ref: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class NemotronIndexedPart:
    sample_key: str
    tar_file_id: int
    sample_index: int
    part_name: str
    content_byte_offset: int
    content_byte_size: int

    @property
    def shard(self) -> str:
        return f"shard_{self.tar_file_id:06d}.tar"

    @property
    def filename(self) -> str:
        return f"{self.sample_key}.{self.part_name.lstrip('.')}"


def build_from_hf_conversation_tar_s3(
    *,
    repo_id: str,
    include_globs: tuple[str, ...],
    tags: tuple[str, ...],
    s3_uri: str,
    max_records: int,
    seed: int,
    split_ratios: tuple[tuple[str, float], ...],
    media_type: str,
    worker_index: int = 0,
    num_workers: int = 1,
    resume: bool = False,
) -> list[dict[str, Any]]:
    try:
        import requests
        from huggingface_hub import HfApi, hf_hub_download, hf_hub_url
    except Exception as exc:  # pragma: no cover - optional dependencies
        raise RuntimeError("HF conversation tar materialization requires requests and huggingface_hub.") from exc

    validate_worker_partition(worker_index=worker_index, num_workers=num_workers)
    bucket, prefix = parse_s3_uri(s3_uri)
    s3 = s3_client()
    api = HfApi()
    files = api.list_repo_files(repo_id, repo_type="dataset")
    indexed_subsets = nemotron_indexed_subsets(files)
    jsonl_paths = [
        path
        for path in files
        if path.endswith(".jsonl") and match_any(path, include_globs) and nemotron_subset_for_path(path) in indexed_subsets
    ]
    rng = random.Random(seed)
    rng.shuffle(jsonl_paths)

    session = requests.Session()
    records: list[dict[str, Any]] = []
    indexes: dict[str, Path] = {}
    repo_slug = repo_id.replace("/", "--")
    for jsonl_path in jsonl_paths:
        if record_limit_reached(len(records), max_records):
            break
        subset = required(nemotron_subset_for_path(jsonl_path), f"subset for {jsonl_path}")
        if subset not in indexes:
            indexes[subset] = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename=f"{subset}/media/.nv-meta/index.sqlite",
                )
            )
        local_jsonl = Path(hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=jsonl_path))
        with local_jsonl.open("r", encoding="utf-8") as handle:
            for row_idx, line in enumerate(handle):
                if record_limit_reached(len(records), max_records):
                    break
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    continue
                media = nemotron_first_media_ref(row)
                if media is None:
                    continue
                record_id = f"hf-conversation-tar:{repo_id}:{jsonl_path}:{row.get('id', row_idx)}:{media.ref}"
                if shard_worker_index(record_id, num_workers=num_workers) != worker_index:
                    continue
                inferred_type = infer_media_type(media.ref, media_type)
                if inferred_type == "text":
                    inferred_type = "video" if media.kind == "video" else "image" if media.kind == "image" else "text"
                if inferred_type == "text":
                    continue
                prompt = nemotron_prompt_from_messages(row.get("messages"))
                if not prompt:
                    continue
                part = lookup_nemotron_media_part(indexes[subset], media.ref)
                if part is None:
                    continue
                if part.content_byte_size <= 0:
                    continue
                shard = f"{subset}/media/{part.shard}"
                media_name = nemotron_media_output_name(media.ref, part)
                key = s3_key_for_nemotron_media(prefix, repo_slug=repo_slug, subset=subset, media_name=media_name)
                uploaded = False
                skipped_existing = False
                if resume and s3_object_exists_with_size(s3, bucket, key, part.content_byte_size):
                    skipped_existing = True
                else:
                    url = hf_hub_url(repo_id=repo_id, filename=shard, repo_type="dataset")
                    body = http_range(session, url, part.content_byte_offset, part.content_byte_offset + part.content_byte_size - 1)
                    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType=content_type_for_path(media_name))
                    uploaded = True
                split = split_for_record_id(record_id, seed, split_ratios)
                record_tags = sorted(set(tags + (split, inferred_type, subset) + tags_from_path(jsonl_path)))
                records.append(
                    make_manifest_dict(
                        record_id=record_id,
                        media_type=inferred_type,
                        prompt=prompt,
                        media_path=f"s3://{bucket}/{key}",
                        tags=record_tags,
                        metadata={
                            "source": "hf-conversation-tar-s3",
                            "repo_id": repo_id,
                            "jsonl_path": jsonl_path,
                            "row_index": row_idx,
                            "source_id": row.get("id"),
                            "source_uri": f"hf://dataset/{repo_id}/{jsonl_path}#{row_idx}",
                            "media_ref": media.ref,
                            "media_kind": media.kind,
                            "media_metadata": media.metadata,
                            "subset": subset,
                            "shard": shard,
                            "sample_key": part.sample_key,
                            "sample_index": part.sample_index,
                            "part_name": part.part_name,
                            "content_byte_offset": part.content_byte_offset,
                            "content_byte_size": part.content_byte_size,
                            "bucket": bucket,
                            "key": key,
                            "split": split,
                            "worker_index": worker_index,
                            "num_workers": num_workers,
                            "uploaded": uploaded,
                            "skipped_existing": skipped_existing,
                            "assistant_text": nemotron_first_assistant_text(row.get("messages")),
                        },
                    )
                )
    return records


def nemotron_indexed_subsets(files: Iterable[str]) -> set[str]:
    return {
        path.split("/", 1)[0]
        for path in files
        if path.endswith("/media/.nv-meta/index.sqlite") and "/" in path
    }


def nemotron_subset_for_path(path: str) -> str | None:
    parts = PurePosixPath(path).parts
    return parts[0] if parts else None


def nemotron_first_media_ref(row: dict[str, Any]) -> NemotronMediaRef | None:
    for message in row.get("messages") or ():
        if not isinstance(message, dict):
            continue
        for item in message.get("content") or ():
            if not isinstance(item, dict):
                continue
            kind = item.get("type")
            if kind == "video" and item.get("video"):
                return NemotronMediaRef(kind="video", ref=str(item["video"]), metadata={k: v for k, v in item.items() if k not in {"type", "video"}})
            if kind == "image" and item.get("image"):
                return NemotronMediaRef(kind="image", ref=str(item["image"]), metadata={k: v for k, v in item.items() if k not in {"type", "image"}})
    return None


def nemotron_prompt_from_messages(messages: Any) -> str:
    blocks: list[str] = []
    for message in messages or ():
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user")
        if role == "assistant":
            break
        texts = [str(item.get("text", "")).strip() for item in message.get("content") or () if isinstance(item, dict) and item.get("type") == "text"]
        text = "\n".join(part for part in texts if part)
        if not text:
            continue
        blocks.append(f"[{role}]\n{text}" if role != "user" else text)
    return "\n\n".join(blocks).strip()


def nemotron_first_assistant_text(messages: Any, *, max_chars: int = 2000) -> str | None:
    for message in messages or ():
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        texts = [str(item.get("text", "")).strip() for item in message.get("content") or () if isinstance(item, dict) and item.get("type") == "text"]
        text = "\n".join(part for part in texts if part).strip()
        return text[:max_chars] if text else None
    return None


def lookup_nemotron_media_part(index_path: Path, media_ref: str) -> NemotronIndexedPart | None:
    candidates = nemotron_sample_key_candidates(media_ref)
    with sqlite3.connect(index_path) as conn:
        for sample_key in candidates:
            row = conn.execute(
                """
                SELECT s.tar_file_id, s.sample_index, p.part_name, p.content_byte_offset, p.content_byte_size
                FROM samples s
                JOIN sample_parts p
                  ON s.tar_file_id = p.tar_file_id AND s.sample_index = p.sample_index
                WHERE s.sample_key = ?
                ORDER BY p.part_name
                LIMIT 1
                """,
                (sample_key,),
            ).fetchone()
            if row is None:
                continue
            return NemotronIndexedPart(
                sample_key=sample_key,
                tar_file_id=int(row[0]),
                sample_index=int(row[1]),
                part_name=str(row[2]),
                content_byte_offset=int(row[3]),
                content_byte_size=int(row[4]),
            )
    return None


def nemotron_sample_key_candidates(media_ref: str) -> list[str]:
    path = PurePosixPath(media_ref.replace("\\", "/"))
    candidates = [path.stem, str(path.with_suffix(""))]
    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            unique.append(candidate)
            seen.add(candidate)
    return unique


def nemotron_media_output_name(media_ref: str, part: NemotronIndexedPart) -> str:
    path = PurePosixPath(media_ref.replace("\\", "/"))
    suffix = path.suffix
    if suffix:
        return str(path)
    return part.filename


def s3_key_for_nemotron_media(prefix: str, *, repo_slug: str, subset: str, media_name: str) -> str:
    parts = [
        "media",
        repo_slug,
        safe_s3_path_part(subset),
        *safe_s3_member_parts(media_name),
    ]
    suffix = "/".join(part for part in parts if part)
    return f"{prefix.rstrip('/')}/{suffix}" if prefix else suffix


def s3_object_exists_with_size(client: Any, bucket: str, key: str, expected_size: int) -> bool:
    try:
        response = client.head_object(Bucket=bucket, Key=key)
    except Exception as exc:
        code = getattr(exc, "response", {}).get("Error", {}).get("Code")
        if str(code) in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise
    size = response.get("ContentLength")
    return size is None or int(size) == expected_size


def build_from_hf_tar_range(
    *,
    repo_id: str,
    shard_globs: tuple[str, ...],
    member_globs: tuple[str, ...],
    prompt: str,
    tags: tuple[str, ...],
    max_records: int,
    max_shards: int | None,
    max_shard_gb: float | None,
    seed: int,
    split_ratios: tuple[tuple[str, float], ...],
    media_type: str,
    shard_list_uri: str | None = None,
    worker_index: int = 0,
    num_workers: int = 1,
    sidecar_suffix: str | None = None,
    sidecar_caption_field: str = "caption",
    sidecar_metadata_mode: str = "compact",
    sidecar_shard_strategy: str | None = None,
    sidecar_prompt_source: str = "robotsim_sidecar",
    sidecar_metadata_key: str = "robotsim_sidecar",
) -> list[dict[str, Any]]:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("HF tar range manifests require huggingface_hub.") from exc
    api = HfApi()
    max_shard_bytes = int(max_shard_gb * 1_000_000_000) if max_shard_gb and max_shard_gb > 0 else None
    shards = (
        list_hf_tar_shards_from_uri(shard_list_uri, shard_globs=shard_globs, max_shard_bytes=max_shard_bytes)
        if shard_list_uri
        else list_hf_tar_shards(api, repo_id=repo_id, shard_globs=shard_globs, max_shard_bytes=max_shard_bytes)
    )
    repo_files = set(api.list_repo_files(repo_id, repo_type="dataset")) if sidecar_shard_strategy == "phyxsim_caption_tar" else None
    if sidecar_shard_strategy == "phyxsim_caption_tar":
        shards = filter_phyxsim_captioned_video_shards(repo_files or set(), shards=shards)
    rng = random.Random(seed)
    rng.shuffle(shards)
    shards = partition_shards_for_worker(shards, worker_index=worker_index, num_workers=num_workers)
    if max_shards is not None and max_shards > 0:
        shards = shards[:max_shards]

    records: list[dict[str, Any]] = []
    tmp_parent = os.environ.get("COSMOS_SAE_TAR_TMPDIR")
    if tmp_parent:
        Path(tmp_parent).mkdir(parents=True, exist_ok=True)
    for shard_idx, shard in enumerate(shards):
        if record_limit_reached(len(records), max_records):
            break
        with tempfile.TemporaryDirectory(prefix="sae_hf_tar_range_", dir=tmp_parent) as download_dir:
            resolved_shard, shard_path = download_hf_tar_shard(hf_hub_download, repo_id=repo_id, shard=shard, local_dir=Path(download_dir))
            with tarfile.open(shard_path, mode="r:*") as tar:
                members = [
                    member
                    for member in tar.getmembers()
                    if member.isfile() and match_any(member.name, member_globs) and infer_media_type(member.name, media_type) != "text"
                ]
                rng.shuffle(members)
                sidecars = load_range_tar_sidecars(
                    hf_hub_download,
                    repo_id=repo_id,
                    shard=resolved_shard,
                    media_tar=tar,
                    media_members=members,
                    sidecar_suffix=sidecar_suffix,
                    sidecar_shard_strategy=sidecar_shard_strategy,
                    sidecar_candidate_shards=(
                        phyxsim_caption_shards_for_video_shard(repo_files or set(), resolved_shard)
                        if sidecar_shard_strategy == "phyxsim_caption_tar"
                        else None
                    ),
                    local_dir=Path(download_dir),
                )
                for member in members:
                    if record_limit_reached(len(records), max_records):
                        break
                    record = hf_tar_range_record(
                        repo_id=repo_id,
                        shard=resolved_shard,
                        member=member,
                        prompt=prompt,
                        tags=tags,
                        seed=seed,
                        split_ratios=split_ratios,
                        media_type=media_type,
                        shard_index=shard_idx,
                        worker_index=worker_index,
                        num_workers=num_workers,
                    )
                    records.append(
                        attach_tar_sidecar(
                            record,
                            repo_id=repo_id,
                            shard=resolved_shard,
                            sidecars=sidecars,
                            sidecar_suffix=sidecar_suffix,
                            sidecar_caption_field=sidecar_caption_field,
                            sidecar_metadata_mode=sidecar_metadata_mode,
                            sidecar_prompt_source=sidecar_prompt_source,
                            sidecar_metadata_key=sidecar_metadata_key,
                            sidecar_s3_uri=None,
                            sidecar_uploader=None,
                        )
                    )
    return records


def hf_tar_range_record(
    *,
    repo_id: str,
    shard: str,
    member: tarfile.TarInfo,
    prompt: str,
    tags: tuple[str, ...],
    seed: int,
    split_ratios: tuple[tuple[str, float], ...],
    media_type: str,
    shard_index: int,
    worker_index: int,
    num_workers: int,
) -> dict[str, Any]:
    record_id = f"hf-tar-range:{repo_id}:{shard}:{member.name}"
    split = split_for_record_id(record_id, seed, split_ratios)
    media_path = hf_tar_range_uri(repo_id=repo_id, shard=shard, member_name=member.name, offset=member.offset_data, size=member.size)
    record_tags = sorted(set(tags + (split,) + tags_from_path(shard) + tags_from_path(member.name)))
    return make_manifest_dict(
        record_id=record_id,
        media_type=infer_media_type(member.name, media_type),
        prompt=render_template(prompt, path=member.name, stem=Path(member.name).stem, repo_id=repo_id, shard=shard),
        media_path=media_path,
        tags=record_tags,
        metadata={
            "source": "hf-tar-range",
            "repo_id": repo_id,
            "shard": shard,
            "member": member.name,
            "source_uri": f"hf://dataset/{repo_id}/{shard}#{member.name}",
            "media_range_uri": media_path,
            "content_byte_offset": int(member.offset_data),
            "content_byte_size": int(member.size),
            "split": split,
            "shard_index": shard_index,
            "worker_index": worker_index,
            "num_workers": num_workers,
        },
    )


def hf_tar_range_uri(*, repo_id: str, shard: str, member_name: str, offset: int, size: int) -> str:
    query = urlencode({"offset": int(offset), "size": int(size), "name": member_name})
    return f"hf-tar-range://dataset/{repo_id}/{shard}?{query}"


def build_from_hf_tar_s3(
    *,
    repo_id: str,
    shard_globs: tuple[str, ...],
    member_globs: tuple[str, ...],
    prompt: str,
    tags: tuple[str, ...],
    s3_uri: str,
    max_records: int,
    max_shards: int | None,
    max_shard_gb: float | None,
    seed: int,
    split_ratios: tuple[tuple[str, float], ...],
    media_type: str,
    shard_list_uri: str | None = None,
    worker_index: int = 0,
    num_workers: int = 1,
    resume: bool = False,
    stream_tars: bool = False,
    sidecar_suffix: str | None = None,
    sidecar_caption_field: str = "caption",
    sidecar_metadata_mode: str = "compact",
    sidecar_prompt_source: str = "robotsim_sidecar",
    sidecar_metadata_key: str = "robotsim_sidecar",
    sidecar_s3_uri: str | None = None,
) -> list[dict[str, Any]]:
    try:
        from huggingface_hub import HfApi, hf_hub_download, hf_hub_url
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("HF tar materialization requires huggingface_hub.") from exc
    bucket, prefix = parse_s3_uri(s3_uri)
    s3 = s3_client()
    sidecar_uploader = s3_json_uploader(s3) if sidecar_s3_uri else None
    api = HfApi()
    max_shard_bytes = int(max_shard_gb * 1_000_000_000) if max_shard_gb and max_shard_gb > 0 else None
    shards = (
        list_hf_tar_shards_from_uri(shard_list_uri, shard_globs=shard_globs, max_shard_bytes=max_shard_bytes)
        if shard_list_uri
        else list_hf_tar_shards(api, repo_id=repo_id, shard_globs=shard_globs, max_shard_bytes=max_shard_bytes)
    )
    rng = random.Random(seed)
    rng.shuffle(shards)
    shards = partition_shards_for_worker(shards, worker_index=worker_index, num_workers=num_workers)
    if max_shards is not None and max_shards > 0:
        shards = shards[:max_shards]

    records: list[dict[str, Any]] = []
    repo_slug = repo_id.replace("/", "--")
    for shard_idx, shard in enumerate(shards):
        if record_limit_reached(len(records), max_records):
            break
        if stream_tars:
            shard_records = materialize_hf_tar_stream(
                hf_hub_url=hf_hub_url,
                repo_id=repo_id,
                shard=shard,
                member_globs=member_globs,
                prompt=prompt,
                tags=tags,
                bucket=bucket,
                prefix=prefix,
                repo_slug=repo_slug,
                s3=s3,
                max_records=max_records - len(records) if max_records > 0 else 0,
                seed=seed,
                split_ratios=split_ratios,
                media_type=media_type,
                resume=resume,
                shard_index=shard_idx,
                worker_index=worker_index,
                num_workers=num_workers,
                sidecar_suffix=sidecar_suffix,
                sidecar_caption_field=sidecar_caption_field,
                sidecar_metadata_mode=sidecar_metadata_mode,
                sidecar_prompt_source=sidecar_prompt_source,
                sidecar_metadata_key=sidecar_metadata_key,
                sidecar_s3_uri=sidecar_s3_uri,
            )
            records.extend(shard_records)
            continue
        tmp_parent = os.environ.get("COSMOS_SAE_TAR_TMPDIR")
        if tmp_parent:
            Path(tmp_parent).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="sae_hf_tar_", dir=tmp_parent) as download_dir:
            shard, shard_path = download_hf_tar_shard(hf_hub_download, repo_id=repo_id, shard=shard, local_dir=Path(download_dir))
            with tarfile.open(shard_path, mode="r:*") as tar:
                members = [
                    member
                    for member in tar.getmembers()
                    if member.isfile() and match_any(member.name, member_globs) and infer_media_type(member.name, media_type) != "text"
                ]
                rng.shuffle(members)
                sidecars = load_tar_json_sidecars(tar, members, sidecar_suffix=sidecar_suffix)
                for member in members:
                    if record_limit_reached(len(records), max_records):
                        break
                    extracted = tar.extractfile(member)
                    if extracted is None:
                        continue
                    record = materialize_tar_member(
                        extracted=extracted,
                        repo_id=repo_id,
                        shard=shard,
                        member_name=member.name,
                        prompt=prompt,
                        tags=tags,
                        bucket=bucket,
                        prefix=prefix,
                        repo_slug=repo_slug,
                        s3=s3,
                        seed=seed,
                        split_ratios=split_ratios,
                        media_type=media_type,
                        resume=resume,
                        shard_index=shard_idx,
                        worker_index=worker_index,
                        num_workers=num_workers,
                    )
                    records.append(
                        attach_tar_sidecar(
                            record,
                            repo_id=repo_id,
                            shard=shard,
                            sidecars=sidecars,
                            sidecar_suffix=sidecar_suffix,
                            sidecar_caption_field=sidecar_caption_field,
                            sidecar_metadata_mode=sidecar_metadata_mode,
                            sidecar_prompt_source=sidecar_prompt_source,
                            sidecar_metadata_key=sidecar_metadata_key,
                            sidecar_s3_uri=sidecar_s3_uri,
                            sidecar_uploader=sidecar_uploader,
                        )
                    )
    return records


def materialize_hf_tar_stream(
    *,
    hf_hub_url: Any,
    repo_id: str,
    shard: str,
    member_globs: tuple[str, ...],
    prompt: str,
    tags: tuple[str, ...],
    bucket: str,
    prefix: str,
    repo_slug: str,
    s3: Any,
    max_records: int,
    seed: int,
    split_ratios: tuple[tuple[str, float], ...],
    media_type: str,
    resume: bool,
    shard_index: int,
    worker_index: int,
    num_workers: int,
    sidecar_suffix: str | None = None,
    sidecar_caption_field: str = "caption",
    sidecar_metadata_mode: str = "compact",
    sidecar_prompt_source: str = "robotsim_sidecar",
    sidecar_metadata_key: str = "robotsim_sidecar",
    sidecar_s3_uri: str | None = None,
) -> list[dict[str, Any]]:
    try:
        import requests
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Streaming HF tar materialization requires requests.") from exc

    response, resolved_shard = open_hf_tar_response(requests, hf_hub_url=hf_hub_url, repo_id=repo_id, shard=shard)
    records: list[dict[str, Any]] = []
    sidecars: dict[str, dict[str, Any]] = {}
    needed_sidecars: set[str] = set()
    with response:
        response.raw.decode_content = True
        with tarfile.open(fileobj=response.raw, mode="r|*") as tar:
            for member in tar:
                if record_limit_reached(len(records), max_records) and (not sidecar_suffix or not needed_sidecars):
                    break
                if not member.isfile():
                    continue
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                if sidecar_suffix and member.name.lower().endswith(sidecar_suffix.lower()):
                    with extracted:
                        sidecars[member.name] = json.loads(extracted.read().decode("utf-8"))
                    needed_sidecars.discard(member.name)
                    continue
                if record_limit_reached(len(records), max_records):
                    extracted.close()
                    continue
                if not match_any(member.name, member_globs) or infer_media_type(member.name, media_type) == "text":
                    extracted.close()
                    continue
                sidecar_member = sidecar_member_for_media_member(member.name, sidecar_suffix=sidecar_suffix)
                if sidecar_member and sidecar_member not in sidecars:
                    needed_sidecars.add(sidecar_member)
                records.append(
                    materialize_tar_member(
                        extracted=extracted,
                        repo_id=repo_id,
                        shard=resolved_shard,
                        member_name=member.name,
                        prompt=prompt,
                        tags=tags,
                        bucket=bucket,
                        prefix=prefix,
                        repo_slug=repo_slug,
                        s3=s3,
                        seed=seed,
                        split_ratios=split_ratios,
                        media_type=media_type,
                        resume=resume,
                        shard_index=shard_index,
                        worker_index=worker_index,
                        num_workers=num_workers,
                    )
                )
    sidecar_uploader = s3_json_uploader(s3) if sidecar_s3_uri else None
    return [
        attach_tar_sidecar(
            record,
            repo_id=repo_id,
            shard=resolved_shard,
            sidecars=sidecars,
            sidecar_suffix=sidecar_suffix,
            sidecar_caption_field=sidecar_caption_field,
            sidecar_metadata_mode=sidecar_metadata_mode,
            sidecar_prompt_source=sidecar_prompt_source,
            sidecar_metadata_key=sidecar_metadata_key,
            sidecar_s3_uri=sidecar_s3_uri,
            sidecar_uploader=sidecar_uploader,
        )
        for record in records
    ]


def materialize_tar_member(
    *,
    extracted: Any,
    repo_id: str,
    shard: str,
    member_name: str,
    prompt: str,
    tags: tuple[str, ...],
    bucket: str,
    prefix: str,
    repo_slug: str,
    s3: Any,
    seed: int,
    split_ratios: tuple[tuple[str, float], ...],
    media_type: str,
    resume: bool,
    shard_index: int,
    worker_index: int,
    num_workers: int,
) -> dict[str, Any]:
    record_id = f"hf-tar:{repo_id}:{shard}:{member_name}"
    split = split_for_record_id(record_id, seed, split_ratios)
    key = s3_key_for_tar_member(prefix, repo_slug, shard, member_name)
    uploaded = False
    skipped_existing = False
    if resume and s3_object_exists(s3, bucket, key):
        skipped_existing = True
        extracted.close()
    else:
        with extracted:
            if is_seekable_fileobj(extracted):
                s3.upload_fileobj(
                    extracted,
                    bucket,
                    key,
                    ExtraArgs={"ContentType": content_type_for_path(member_name)},
                )
            else:
                upload_spooled_fileobj(
                    s3,
                    extracted,
                    bucket=bucket,
                    key=key,
                    content_type=content_type_for_path(member_name),
                )
        uploaded = True
    media_path = f"s3://{bucket}/{key}"
    record_tags = sorted(set(tags + (split,) + tags_from_path(shard) + tags_from_path(member_name)))
    return make_manifest_dict(
        record_id=record_id,
        media_type=infer_media_type(member_name, media_type),
        prompt=render_template(prompt, path=member_name, stem=Path(member_name).stem, repo_id=repo_id, shard=shard),
        media_path=media_path,
        tags=record_tags,
        metadata={
            "source": "hf-tar-s3",
            "repo_id": repo_id,
            "shard": shard,
            "member": member_name,
            "source_uri": f"hf://dataset/{repo_id}/{shard}#{member_name}",
            "bucket": bucket,
            "key": key,
            "split": split,
            "shard_index": shard_index,
            "worker_index": worker_index,
            "num_workers": num_workers,
            "uploaded": uploaded,
            "skipped_existing": skipped_existing,
        },
    )


def load_tar_json_sidecars(tar: tarfile.TarFile, members: list[tarfile.TarInfo], *, sidecar_suffix: str | None) -> dict[str, dict[str, Any]]:
    if not sidecar_suffix:
        return {}
    needed = {
        sidecar_member
        for member in members
        if (sidecar_member := sidecar_member_for_media_member(member.name, sidecar_suffix=sidecar_suffix))
    }
    sidecars: dict[str, dict[str, Any]] = {}
    for member in tar.getmembers():
        if member.name not in needed:
            continue
        extracted = tar.extractfile(member)
        if extracted is None:
            continue
        with extracted:
            sidecars[member.name] = json.loads(extracted.read().decode("utf-8"))
    return sidecars


def load_range_tar_sidecars(
    hf_hub_download: Any,
    *,
    repo_id: str,
    shard: str,
    media_tar: tarfile.TarFile,
    media_members: list[tarfile.TarInfo],
    sidecar_suffix: str | None,
    sidecar_shard_strategy: str | None,
    local_dir: Path,
    sidecar_candidate_shards: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    if not sidecar_suffix:
        return {}
    if sidecar_shard_strategy is None:
        return load_tar_json_sidecars(media_tar, media_members, sidecar_suffix=sidecar_suffix)
    if sidecar_shard_strategy == "phyxsim_caption_tar":
        caption_shards = sidecar_candidate_shards
        if caption_shards is None:
            caption_shard = phyxsim_caption_shard_for_video_shard(shard)
            caption_shards = [caption_shard] if caption_shard else []
        return load_phyxsim_range_caption_sidecars(
            hf_hub_download,
            repo_id=repo_id,
            caption_shards=caption_shards,
            media_members=media_members,
            sidecar_suffix=sidecar_suffix,
            local_dir=local_dir,
        )
    raise ValueError(f"unsupported tar sidecar shard strategy: {sidecar_shard_strategy!r}")


def load_phyxsim_range_caption_sidecars(
    hf_hub_download: Any,
    *,
    repo_id: str,
    caption_shards: list[str],
    media_members: list[tarfile.TarInfo],
    sidecar_suffix: str,
    local_dir: Path,
) -> dict[str, dict[str, Any]]:
    needed = {
        sidecar_member
        for member in media_members
        if (sidecar_member := sidecar_member_for_media_member(member.name, sidecar_suffix=sidecar_suffix))
    }
    found: dict[str, dict[str, Any]] = {}
    for caption_shard in caption_shards:
        missing = needed - set(found)
        if not missing:
            break
        try:
            _resolved_sidecar_shard, sidecar_path = download_hf_tar_shard(
                hf_hub_download,
                repo_id=repo_id,
                shard=caption_shard,
                local_dir=local_dir,
            )
        except Exception:
            continue
        with tarfile.open(sidecar_path, mode="r:*") as sidecar_tar:
            found.update(load_tar_json_sidecars_by_name(sidecar_tar, missing))
    return found


def load_tar_json_sidecars_by_name(tar: tarfile.TarFile, needed: set[str]) -> dict[str, dict[str, Any]]:
    sidecars: dict[str, dict[str, Any]] = {}
    for member in tar.getmembers():
        if member.name not in needed:
            continue
        extracted = tar.extractfile(member)
        if extracted is None:
            continue
        with extracted:
            sidecars[member.name] = json.loads(extracted.read().decode("utf-8"))
    return sidecars


def phyxsim_caption_shard_for_video_shard(shard: str) -> str | None:
    parts = PurePosixPath(shard).parts
    if len(parts) != 3 or parts[0] != "videos":
        return None
    category = parts[1]
    filename = parts[2]
    if not filename.startswith(f"videos-{category}-") or not filename.endswith(".tar"):
        return None
    suffix = filename[len(f"videos-{category}-") :]
    return str(PurePosixPath("captions") / category / f"captions-{category}-{suffix}")


def phyxsim_caption_shards_for_video_shard(files: set[str], shard: str) -> list[str]:
    parts = PurePosixPath(shard).parts
    if len(parts) != 3 or parts[0] != "videos":
        return []
    category = parts[1]
    prefix = str(PurePosixPath("captions") / category / f"captions-{category}-")
    return sorted(path for path in files if path.startswith(prefix) and path.endswith(".tar"))


def filter_phyxsim_captioned_video_shards(files: set[str], *, shards: list[str]) -> list[str]:
    return [shard for shard in shards if phyxsim_caption_shards_for_video_shard(files, shard)]


def attach_tar_sidecar(
    record: dict[str, Any],
    *,
    repo_id: str,
    shard: str,
    sidecars: dict[str, dict[str, Any]],
    sidecar_suffix: str | None,
    sidecar_caption_field: str,
    sidecar_metadata_mode: str,
    sidecar_prompt_source: str,
    sidecar_metadata_key: str,
    sidecar_s3_uri: str | None,
    sidecar_uploader: Any | None = None,
) -> dict[str, Any]:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    member_name = metadata.get("member")
    sidecar_member = sidecar_member_for_media_member(member_name, sidecar_suffix=sidecar_suffix) if isinstance(member_name, str) else None
    sidecar = sidecars.get(sidecar_member) if sidecar_member else None
    if sidecar is None:
        return record
    sidecar_path = None
    if sidecar_s3_uri and sidecar_member:
        sidecar_path = upload_robotsim_sidecar(
            sidecar_s3_uri,
            repo_id=repo_id,
            shard=shard,
            sidecar_member=sidecar_member,
            sidecar=sidecar,
            uploader=sidecar_uploader,
        )
    next_record = merge_sidecar(
        record,
        sidecar,
        sidecar_member=sidecar_member,
        sidecar_path=sidecar_path,
        metadata_mode=sidecar_metadata_mode,
        prompt_source=sidecar_prompt_source,
        metadata_key=sidecar_metadata_key,
    )
    caption = extract_json_text(sidecar, sidecar_caption_field)
    if caption:
        next_record["prompt"] = caption
    return next_record


def sidecar_member_for_media_member(member: str, *, sidecar_suffix: str | None = ".json") -> str | None:
    if not sidecar_suffix:
        return None
    lower = member.lower()
    if lower.endswith(".mp4"):
        return member[:-4] + sidecar_suffix
    return None


def merge_sidecar(
    record: dict[str, Any],
    sidecar: dict[str, Any],
    *,
    sidecar_member: str | None,
    sidecar_path: str | None = None,
    metadata_mode: str,
    prompt_source: str,
    metadata_key: str,
) -> dict[str, Any]:
    next_record = dict(record)
    metadata = dict(next_record.get("metadata") or {})
    metadata["prompt_source"] = prompt_source
    if sidecar_member:
        metadata["sidecar_member"] = sidecar_member
    if sidecar_path:
        metadata["sidecar_path"] = sidecar_path
    if metadata_mode == "compact":
        compact = compact_sidecar_metadata(sidecar)
        if compact:
            metadata[metadata_key] = compact
    elif metadata_mode == "full":
        metadata[metadata_key] = sidecar
    elif metadata_mode != "none":
        raise ValueError(f"unsupported sidecar metadata mode: {metadata_mode!r}")
    next_record["metadata"] = metadata
    return next_record


def compact_sidecar_metadata(sidecar: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in sidecar.items():
        if key == "caption":
            continue
        if value is None or isinstance(value, (str, int, float, bool)):
            compact[key] = value
    return compact


def upload_robotsim_sidecar(
    sidecar_s3_uri: str,
    *,
    repo_id: str,
    shard: str,
    sidecar_member: str,
    sidecar: dict[str, Any],
    uploader: Any | None = None,
) -> str:
    bucket, prefix = parse_s3_uri(sidecar_s3_uri)
    key = sidecar_s3_key(prefix, repo_id=repo_id, shard=shard, sidecar_member=sidecar_member)
    body = json.dumps(sidecar, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
    if uploader is None:
        s3_client().put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType="application/json; charset=utf-8",
        )
    else:
        uploader(bucket, key, body)
    return f"s3://{bucket}/{key}"


def s3_json_uploader(client: Any):
    def upload(bucket: str, key: str, body: bytes) -> None:
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType="application/json; charset=utf-8",
        )

    return upload


def sidecar_s3_key(prefix: str, *, repo_id: str, shard: str, sidecar_member: str) -> str:
    repo_slug = repo_id.replace("/", "--")
    shard_stem = Path(shard).stem
    parts = [
        prefix.rstrip("/"),
        repo_slug,
        safe_s3_path_part(shard_stem),
        *safe_s3_member_parts(sidecar_member),
    ]
    return "/".join(part for part in parts if part)


def source_shard_and_member(record: dict[str, Any]) -> tuple[str | None, str | None]:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    shard = metadata.get("shard")
    member = metadata.get("member")
    if isinstance(shard, str) and isinstance(member, str):
        return shard, member
    source_uri = metadata.get("source_uri")
    if isinstance(source_uri, str) and source_uri.startswith("hf://dataset/") and "#" in source_uri:
        before, member = source_uri.split("#", 1)
        parts = before.removeprefix("hf://dataset/").split("/", 2)
        if len(parts) == 3:
            return parts[2], member
    return None, None


def enrich_records(
    records: list[dict[str, Any]],
    *,
    repo_id: str,
    caption_field: str,
    metadata_mode: str,
    missing: str,
    sidecar_s3_uri: str | None = None,
    worker_index: int = 0,
    num_workers: int = 1,
    sidecar_lookup: dict[tuple[str, str], dict[str, Any]] | None = None,
    sidecar_uploader: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    validate_worker_partition(worker_index=worker_index, num_workers=num_workers)
    selected_records = [
        record for idx, record in enumerate(records) if record_assigned(idx, worker_index=worker_index, num_workers=num_workers)
    ]
    by_shard: dict[str, set[str]] = defaultdict(set)
    record_sidecars: list[tuple[dict[str, Any], str | None, str | None]] = []
    for record in selected_records:
        shard, member = source_shard_and_member(record)
        sidecar_member = sidecar_member_for_media_member(member) if member else None
        if shard and sidecar_member:
            by_shard[shard].add(sidecar_member)
        record_sidecars.append((record, shard, sidecar_member))

    fetched: dict[tuple[str, str], dict[str, Any]] = dict(sidecar_lookup or {})
    if sidecar_lookup is None:
        for shard, sidecar_members in sorted(by_shard.items()):
            for member, sidecar in fetch_json_sidecars_from_hf_tar(repo_id, shard, sidecar_members).items():
                fetched[(shard, member)] = sidecar

    enriched: list[dict[str, Any]] = []
    stats = {
        "input_records": len(records),
        "selected_records": len(selected_records),
        "output_records": 0,
        "enriched_records": 0,
        "missing_sidecars": 0,
        "missing_captions": 0,
        "uploaded_sidecars": 0,
    }
    for record, shard, sidecar_member in record_sidecars:
        sidecar = fetched.get((shard, sidecar_member)) if shard and sidecar_member else None
        if sidecar is None:
            stats["missing_sidecars"] += 1
            if missing == "drop":
                continue
            if missing == "error":
                raise ValueError(f"missing RobotSim sidecar for record {record.get('id')!r}")
            enriched.append(record)
            continue

        sidecar_path = None
        if sidecar_s3_uri and shard and sidecar_member:
            sidecar_path = upload_robotsim_sidecar(
                sidecar_s3_uri,
                repo_id=repo_id,
                shard=shard,
                sidecar_member=sidecar_member,
                sidecar=sidecar,
                uploader=sidecar_uploader,
            )
            stats["uploaded_sidecars"] += 1

        caption = extract_json_text(sidecar, caption_field)
        next_record = merge_sidecar(
            record,
            sidecar,
            sidecar_member=sidecar_member,
            sidecar_path=sidecar_path,
            metadata_mode=metadata_mode,
            prompt_source="robotsim_sidecar",
            metadata_key="robotsim_sidecar",
        )
        if not caption:
            stats["missing_captions"] += 1
            if missing == "drop":
                continue
            if missing == "error":
                raise ValueError(f"missing {caption_field!r} in RobotSim sidecar for record {record.get('id')!r}")
        else:
            next_record["prompt"] = caption
            stats["enriched_records"] += 1
        enriched.append(next_record)

    stats["output_records"] = len(enriched)
    return enriched, stats


def fetch_json_sidecars_from_hf_tar(repo_id: str, shard: str, sidecar_members: Iterable[str]) -> dict[str, dict[str, Any]]:
    try:
        import requests
        from huggingface_hub import hf_hub_url
    except Exception as exc:  # pragma: no cover - optional dependencies
        raise RuntimeError("RobotSim JSON sidecar loading requires requests and huggingface_hub.") from exc

    needed = set(sidecar_members)
    if not needed:
        return {}

    session = requests.Session()
    url = hf_hub_url(repo_id, shard, repo_type="dataset")
    found: dict[str, dict[str, Any]] = {}
    offset = 0
    pax_attrs: dict[str, str] = {}
    gnu_long_name: str | None = None
    while needed:
        header = http_range(session, url, offset, offset + 511)
        parsed = parse_tar_header(header)
        if parsed is None:
            break
        raw_name, size, typeflag = parsed
        data_offset = offset + 512
        member_name = pax_attrs.pop("path", None) or gnu_long_name or raw_name
        gnu_long_name = None

        if typeflag == "x":
            text = http_range(session, url, data_offset, data_offset + size - 1).decode("utf-8", "replace") if size else ""
            pax_attrs.update(parse_pax_headers(text))
        elif typeflag == "L":
            text = http_range(session, url, data_offset, data_offset + size - 1).decode("utf-8", "replace") if size else ""
            gnu_long_name = text.split("\0", 1)[0]
        elif member_name in needed:
            data = http_range(session, url, data_offset, data_offset + size - 1) if size else b"{}"
            found[member_name] = json.loads(data.decode("utf-8"))
            needed.remove(member_name)

        offset = data_offset + tar_data_span(size)
    return found


def http_range(session: Any, url: str, start: int, end: int) -> bytes:
    response = session.get(url, headers={"Range": f"bytes={start}-{end}", **hf_auth_headers()}, timeout=60)
    response.raise_for_status()
    return response.content


def parse_tar_header(block: bytes) -> tuple[str, int, str] | None:
    if len(block) < 512 or all(byte == 0 for byte in block):
        return None
    name = block[0:100].split(b"\0", 1)[0].decode("utf-8", "replace")
    prefix = block[345:500].split(b"\0", 1)[0].decode("utf-8", "replace")
    if prefix:
        name = f"{prefix}/{name}"
    size_raw = block[124:136].split(b"\0", 1)[0].strip() or b"0"
    try:
        size = int(size_raw, 8)
    except ValueError:
        size = 0
    typeflag = block[156:157].decode("ascii", "ignore") or "0"
    return name, size, typeflag


def parse_pax_headers(text: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    index = 0
    while index < len(text):
        space = text.find(" ", index)
        if space < 0:
            break
        try:
            length = int(text[index:space])
        except ValueError:
            break
        record = text[space + 1 : index + length]
        if "=" in record:
            key, value = record.rstrip("\n").split("=", 1)
            attrs[key] = value
        index += length
    return attrs


def tar_data_span(size: int) -> int:
    return ((size + 511) // 512) * 512


def record_assigned(index: int, *, worker_index: int, num_workers: int) -> bool:
    return index % num_workers == worker_index


def is_seekable_fileobj(fileobj: Any) -> bool:
    try:
        return bool(fileobj.seekable())
    except Exception:
        return False


def upload_spooled_fileobj(s3: Any, fileobj: Any, *, bucket: str, key: str, content_type: str) -> None:
    tmp_parent = os.environ.get("COSMOS_SAE_TAR_TMPDIR")
    if tmp_parent:
        Path(tmp_parent).mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="sae_hf_member_", dir=tmp_parent) as tmp:
        shutil.copyfileobj(fileobj, tmp)
        tmp.flush()
        tmp.seek(0)
        s3.upload_fileobj(tmp, bucket, key, ExtraArgs={"ContentType": content_type})


def open_hf_tar_response(requests_module: Any, *, hf_hub_url: Any, repo_id: str, shard: str) -> tuple[Any, str]:
    for candidate in hf_tar_shard_candidates(shard):
        url = hf_hub_url(repo_id=repo_id, filename=candidate, repo_type="dataset")
        headers = hf_auth_headers()
        response = requests_module.get(url, headers=headers, stream=True, timeout=(30, 300))
        if response.status_code == 404 and candidate != hf_tar_shard_candidates(shard)[-1]:
            response.close()
            continue
        try:
            response.raise_for_status()
        except Exception:
            response.close()
            raise
        return response, candidate
    raise RuntimeError(f"could not resolve HF tar shard: {shard}")


def hf_tar_shard_candidates(shard: str) -> list[str]:
    if shard.startswith("data/"):
        return [shard]
    return [shard, f"data/{shard}"]


def hf_auth_headers() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def download_hf_tar_shard(hf_hub_download: Any, *, repo_id: str, shard: str, local_dir: Path | None = None) -> tuple[str, str]:
    kwargs: dict[str, Any] = {"repo_id": repo_id, "repo_type": "dataset", "filename": shard}
    if local_dir is not None:
        kwargs["local_dir"] = str(local_dir)
    try:
        return shard, hf_hub_download(**kwargs)
    except Exception as original_exc:
        if shard.startswith("data/"):
            raise
        prefixed_shard = f"data/{shard}"
        kwargs["filename"] = prefixed_shard
        try:
            return prefixed_shard, hf_hub_download(**kwargs)
        except Exception:
            raise original_exc


def record_limit_reached(count: int, max_records: int) -> bool:
    return max_records > 0 and count >= max_records


def partition_shards_for_worker(shards: list[str], *, worker_index: int, num_workers: int) -> list[str]:
    validate_worker_partition(worker_index=worker_index, num_workers=num_workers)
    if num_workers == 1:
        return shards
    return [shard for shard in shards if shard_worker_index(shard, num_workers=num_workers) == worker_index]


def validate_worker_partition(*, worker_index: int, num_workers: int) -> None:
    if num_workers < 1:
        raise ValueError("--num-workers must be at least 1")
    if worker_index < 0 or worker_index >= num_workers:
        raise ValueError("--worker-index must be in [0, --num-workers)")


def shard_worker_index(shard: str, *, num_workers: int) -> int:
    digest = hashlib.sha256(shard.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % num_workers


def list_hf_tar_shards_from_uri(uri: str, *, shard_globs: tuple[str, ...], max_shard_bytes: int | None) -> list[str]:
    shards: list[str] = []
    for line_no, line in enumerate(read_text_uri(uri).splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if isinstance(obj, str):
            path = obj
            size = None
        elif isinstance(obj, dict):
            path = str(obj.get("shard_path") or obj.get("path") or obj.get("shard") or obj.get("filename") or "")
            size = obj.get("tar_bytes") or obj.get("size") or obj.get("bytes")
        else:
            raise ValueError(f"{uri}:{line_no}: expected object or string JSONL row")
        if not path:
            raise ValueError(f"{uri}:{line_no}: shard row is missing a path")
        if not path.endswith(".tar") or not match_any(path, shard_globs):
            continue
        if max_shard_bytes is not None and size is not None and int(size) > max_shard_bytes:
            continue
        shards.append(path)
    return shards


def s3_object_exists(client: Any, bucket: str, key: str) -> bool:
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception as exc:
        code = getattr(exc, "response", {}).get("Error", {}).get("Code")
        if str(code) in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def list_hf_tar_shards(api: Any, *, repo_id: str, shard_globs: tuple[str, ...], max_shard_bytes: int | None) -> list[str]:
    try:
        items = api.list_repo_tree(repo_id, repo_type="dataset", recursive=True, expand=True)
    except Exception:
        files = api.list_repo_files(repo_id, repo_type="dataset")
        return [path for path in files if match_any(path, shard_globs) and path.endswith(".tar")]
    shards: list[str] = []
    for item in items:
        path = getattr(item, "path", None)
        size = getattr(item, "size", None)
        if not path or not path.endswith(".tar") or not match_any(path, shard_globs):
            continue
        if max_shard_bytes is not None and size is not None and size > max_shard_bytes:
            continue
        shards.append(path)
    return shards


def build_from_s3_prefix(config: BuildCorpusConfig) -> list[dict[str, Any]]:
    s3_uri = required(config.s3_uri, "--s3-uri")
    bucket, prefix = parse_s3_uri(s3_uri)
    client = s3_client()
    rng = random.Random(config.seed)
    candidates: list[str] = []
    seen = 0
    token = None
    while True:
        request: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if token:
            request["ContinuationToken"] = token
        response = client.list_objects_v2(**request)
        for obj in response.get("Contents", []):
            key = obj["Key"]
            if not match_any(key, config.include_globs) or infer_media_type(key, config.media_type) == "text":
                continue
            seen += 1
            if len(candidates) < config.max_records:
                candidates.append(key)
            else:
                replacement = rng.randrange(seen)
                if replacement < config.max_records:
                    candidates[replacement] = key
        if not response.get("IsTruncated"):
            break
        token = response.get("NextContinuationToken")
    rng.shuffle(candidates)
    prompt = required(config.prompt, "--prompt")
    records: list[dict[str, Any]] = []
    for idx, key in enumerate(candidates):
        record_id = f"s3:{bucket}:{key}"
        split = split_for_record_id(record_id, config.seed, config.split_ratios)
        records.append(
            make_manifest_dict(
                record_id=record_id,
                media_type=infer_media_type(key, config.media_type),
                prompt=render_template(prompt, path=key, stem=Path(key).stem, repo_id=bucket),
                media_path=f"s3://{bucket}/{key}",
                tags=sorted({"s3", "external", split} | set(tags_from_path(key))),
                metadata={"source": "s3-prefix", "source_uri": f"s3://{bucket}/{key}", "bucket": bucket, "key": key, "split": split},
            )
        )
    return records


def s3_key_for_tar_member(prefix: str, repo_slug: str, shard: str, member: str) -> str:
    shard_stem = Path(shard).stem
    parts = [
        "media",
        repo_slug,
        safe_s3_path_part(shard_stem),
        *safe_s3_member_parts(member),
    ]
    suffix = "/".join(part for part in parts if part)
    return f"{prefix.rstrip('/')}/{suffix}" if prefix else suffix


def s3_child_uri(parent: str, child: str) -> str:
    if not parent.startswith("s3://"):
        raise ValueError(f"not an S3 URI: {parent}")
    return f"{parent.rstrip('/')}/{child.lstrip('/')}"


def safe_s3_member_parts(path: str) -> list[str]:
    parts: list[str] = []
    for part in PurePosixPath(path.replace("\\", "/")).parts:
        if part in {"", ".", "..", "/"}:
            continue
        parts.append(safe_s3_path_part(part))
    return parts


def safe_s3_path_part(value: str) -> str:
    return quote(value, safe="._-")


def content_type_for_path(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".mp4", ".m4v"}:
        return "video/mp4"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    return "application/octet-stream"


def build_from_jsonl_uri(uri: str, config: BuildCorpusConfig) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, obj in enumerate(iter_jsonl_uri(uri)):
        if len(records) >= config.max_records:
            break
        obj = dict(obj)
        obj.setdefault("id", f"jsonl:{idx}")
        metadata = dict(obj.get("metadata") or {})
        metadata.setdefault("source", "jsonl")
        metadata.setdefault("source_uri", uri)
        metadata.setdefault("row_index", idx)
        metadata.setdefault("split", split_for_record_id(str(obj["id"]), config.seed, config.split_ratios))
        obj["metadata"] = metadata
        record = ManifestRecord.from_json(obj)
        records.append(record.to_json())
    return records


def iter_jsonl_uri(uri: str) -> Iterable[dict[str, Any]]:
    parsed = urlparse(uri)
    if parsed.scheme == "":
        stream: Iterable[str] = Path(uri).open("r", encoding="utf-8")
    elif parsed.scheme in {"http", "https"}:
        stream = (line.decode("utf-8") for line in urlopen(uri))
    elif parsed.scheme == "s3":
        bucket, key = parse_s3_uri(uri)
        body = s3_client().get_object(Bucket=bucket, Key=key)["Body"]
        stream = (line.decode("utf-8") for line in body.iter_lines())
    else:
        raise ValueError(f"unsupported JSONL URI scheme for {uri!r}")
    for line_no, line in enumerate(stream, start=1):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise ValueError(f"{uri}:{line_no}: expected object JSONL row")
        yield obj


def make_manifest_dict(
    *,
    record_id: str,
    media_type: MediaType,
    prompt: str,
    media_path: str | None,
    tags: list[str],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    record = ManifestRecord(
        id=record_id,
        media_type=media_type,
        prompt=prompt,
        media_path=media_path,
        tags=tuple(tags),
        metadata=metadata,
    )
    return record.to_json()


def prompt_from_row(row: dict[str, Any], config: BuildCorpusConfig) -> str | None:
    if config.prompt_field and row.get(config.prompt_field) is not None:
        return str(row[config.prompt_field])
    if config.prompt:
        return render_template(config.prompt, **{key: stringify_value(value) for key, value in row.items()})
    if config.text_field and row.get(config.text_field) is not None:
        return str(row[config.text_field])
    return None


def value_as_path(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("path", "url", "uri", "filename"):
            if value.get(key):
                return str(value[key])
    path = getattr(value, "filename", None)
    return str(path) if path else None


def infer_media_type(path: str, requested: str) -> MediaType:
    if requested in {"text", "image", "video"}:
        return requested  # type: ignore[return-value]
    suffix = Path(urlparse(path).path).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    return "text"


def match_any(path: str, patterns: tuple[str, ...]) -> bool:
    return not patterns or any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def split_for_index(index: int, total: int, ratios: tuple[tuple[str, float], ...]) -> str:
    if not ratios:
        return "sae_train"
    if total <= 0:
        return ratios[0][0]
    position = (index + 0.5) / total
    acc = 0.0
    for name, ratio in ratios:
        acc += ratio
        if position <= acc:
            return name
    return ratios[-1][0]


def split_for_record_id(record_id: str, seed: int, ratios: tuple[tuple[str, float], ...]) -> str:
    if not ratios:
        return "sae_train"
    digest = hashlib.sha256(f"{seed}:{record_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(1 << 64)
    acc = 0.0
    for name, ratio in ratios:
        acc += ratio
        if value < acc:
            return name
    return ratios[-1][0]


def parse_split_ratios(raw: str) -> tuple[tuple[str, float], ...]:
    pairs: list[tuple[str, float]] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        name, sep, value = part.partition("=")
        if not sep or not name:
            raise ValueError(f"invalid split ratio {part!r}; expected name=float")
        pairs.append((name, float(value)))
    total = sum(value for _name, value in pairs)
    if total <= 0:
        raise ValueError("split ratios must sum to a positive value")
    return tuple((name, value / total) for name, value in pairs)


def tags_from_path(path: str) -> tuple[str, ...]:
    parts = [part for part in Path(path).parts if part and part not in {".", "/"}]
    useful = []
    for part in parts[:4]:
        clean = part.lower().replace("_", "-")
        if clean and not clean.endswith((".jpg", ".jpeg", ".png", ".mp4", ".json")):
            useful.append(clean)
    return tuple(useful)


def hf_template_values(*, path: str, repo_id: str) -> dict[str, str]:
    parts = PurePosixPath(path).parts
    hf_split = ""
    if len(parts) >= 3 and parts[0] == "sft_dataset_bridge":
        hf_split = parts[1]
    return {
        "path": path,
        "stem": PurePosixPath(path).stem,
        "repo_id": repo_id,
        "hf_split": hf_split,
    }


def render_template(template: str, **values: str) -> str:
    safe_values = {key: values.get(key, "") for _literal, key, _fmt, _conv in Formatter().parse(template) if key}
    safe_values.update(values)
    return template.format_map(_SafeFormatDict(safe_values))


def stringify_value(value: Any) -> str:
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=True, sort_keys=True)[:1000]


def required(value: str | None, name: str) -> str:
    if not value:
        raise ValueError(f"{name} is required")
    return value


class _SafeFormatDict(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return ""
