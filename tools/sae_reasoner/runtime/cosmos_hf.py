from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, replace
from typing import Any, Iterator, Literal

import torch

from ..manifest import ManifestRecord
from ..media import materialize_media_path

PromptFormat = Literal["chat"]


class RuntimeLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class HookPoint:
    layer: int
    name: str


class CosmosReasonerRuntime:
    """Direct-Python Cosmos3 Reasoner runtime with residual stream hooks.

    This adapter deliberately avoids vLLM/NIM. If the installed Transformers or
    Cosmos packages cannot instantiate the reasoner model, callers receive a
    clear RuntimeLoadError rather than a no-hook serving fallback.
    """

    def __init__(
        self,
        model_id: str,
        *,
        device: str | None = None,
        dtype: str = "bfloat16",
        init_mode: str = "pretrained",
        trust_remote_code: bool = True,
    ):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = _resolve_dtype(dtype)
        if init_mode not in {"pretrained", "random", "meta"}:
            raise ValueError("init_mode must be one of: pretrained, random, meta")
        self.init_mode = init_mode
        self.trust_remote_code = trust_remote_code
        self.processor: Any | None = None
        self.model: torch.nn.Module | None = None

    def load(self) -> "CosmosReasonerRuntime":
        try:
            from transformers import AutoConfig, AutoProcessor
        except Exception as exc:  # pragma: no cover - depends on pod env
            raise RuntimeLoadError(
                "Missing transformers. Install a Cosmos-compatible Transformers build before loading the model."
            ) from exc

        model_cls = _resolve_model_class()
        try:
            if self.init_mode != "meta":
                self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=self.trust_remote_code)
            if self.init_mode == "pretrained":
                self.model = model_cls.from_pretrained(
                    self.model_id,
                    torch_dtype=self.dtype,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=self.trust_remote_code,
                )
            else:
                config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=self.trust_remote_code)
                if self.init_mode == "meta":
                    try:
                        from accelerate import init_empty_weights
                    except Exception as exc:  # pragma: no cover - depends on pod env
                        raise RuntimeLoadError("init_mode=meta requires accelerate.") from exc
                    with init_empty_weights():
                        self.model = model_cls.from_config(config, trust_remote_code=self.trust_remote_code)
                else:
                    with _default_dtype(self.dtype):
                        self.model = model_cls.from_config(config, trust_remote_code=self.trust_remote_code)
                    self.model.to(device=self.device, dtype=self.dtype)
            if self.init_mode not in {"meta", "random"} and self.device != "cuda":
                self.model.to(self.device)
            if self.init_mode != "meta":
                self.model.eval()
        except Exception as exc:  # pragma: no cover - depends on pod env
            raise RuntimeLoadError(
                "Could not load Cosmos3 Reasoner with direct Python hooks. "
                "Install the required Transformers/cosmos-framework runtime, verify HF access, "
                "and avoid vLLM/NIM for SAE collection."
            ) from exc
        return self

    def require_loaded(self) -> tuple[Any | None, torch.nn.Module]:
        if self.model is None:
            raise RuntimeLoadError("runtime is not loaded; call load() first")
        return self.processor, self.model

    def require_processor_and_model(self) -> tuple[Any, torch.nn.Module]:
        processor, model = self.require_loaded()
        if processor is None:
            raise RuntimeLoadError("processor is not loaded for this init mode")
        return processor, model

    def hook_points(self) -> list[HookPoint]:
        _, model = self.require_loaded()
        blocks = find_transformer_blocks(model)
        return [HookPoint(layer=i, name=name) for i, (name, _) in enumerate(blocks)]

    def describe(self) -> dict[str, Any]:
        _, model = self.require_loaded()
        blocks = self.hook_points()
        hidden_size = None
        config = getattr(model, "config", None)
        for candidate in (
            getattr(config, "hidden_size", None),
            getattr(getattr(config, "text_config", None), "hidden_size", None),
        ):
            if candidate is not None:
                hidden_size = int(candidate)
                break
        return {
            "model_id": self.model_id,
            "device": self.device,
            "dtype": str(self.dtype),
            "init_mode": self.init_mode,
            "num_hook_points": len(blocks),
            "hidden_size": hidden_size,
            "num_parameters": int(sum(p.numel() for p in model.parameters())),
            "hook_points": [point.__dict__ for point in blocks],
        }

    def inputs_for_record(
        self,
        record: ManifestRecord,
        *,
        prompt_format: PromptFormat = "chat",
        system_prompt: str | None = None,
    ) -> tuple[dict[str, torch.Tensor], str]:
        if self.init_mode == "meta":
            raise RuntimeLoadError("init_mode=meta can inspect architecture but cannot run inputs or forward passes")
        processor, _ = self.require_processor_and_model()
        materialized_media_path = None
        render_record = record
        if record.media_type != "text":
            materialized_media_path = materialize_media_path(record.media_path)
            render_record = replace(record, media_path=materialized_media_path)
        text = render_record_prompt(
            processor,
            render_record,
            prompt_format=prompt_format,
            system_prompt=system_prompt,
        )
        kwargs: dict[str, Any] = {"text": [text], "return_tensors": "pt"}
        if record.media_type == "image":
            from PIL import Image

            kwargs["images"] = [Image.open(materialized_media_path).convert("RGB")]
        elif record.media_type == "video":
            video_frames, video_metadata = load_video_frames_with_metadata(materialized_media_path)
            kwargs["videos"] = [video_frames]
            kwargs["video_metadata"] = [video_metadata]
        try:
            batch = processor(**kwargs)
        except Exception as exc:  # pragma: no cover - processor-specific
            raise RuntimeLoadError(f"processor could not build inputs for record {record.id!r}") from exc
        return {k: v.to(self.device) if hasattr(v, "to") else v for k, v in batch.items()}, text

    @torch.no_grad()
    def collect_prefill(
        self,
        record: ManifestRecord,
        *,
        layer: int,
        prompt_format: PromptFormat = "chat",
        system_prompt: str | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if self.init_mode == "meta":
            raise RuntimeLoadError("init_mode=meta can inspect architecture but cannot collect activations")
        _, model = self.require_loaded()
        captures: list[torch.Tensor] = []

        def capture(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            hidden = _extract_hidden(output)
            captures.append(hidden.detach().float().cpu())

        inputs, rendered_prompt = self.inputs_for_record(
            record,
            prompt_format=prompt_format,
            system_prompt=system_prompt,
        )
        token_map, token_meta = build_token_map(
            processor=getattr(self, "processor", None),
            batch=inputs,
            media_type=record.media_type,
        )
        with self._hook(layer, capture):
            outputs = model(**inputs, use_cache=False)
        if not captures:
            raise RuntimeLoadError(f"no activation captured at layer {layer}")
        hidden = captures[-1].squeeze(0)
        meta = {
            "id": record.id,
            "media_type": record.media_type,
            "prompt": record.prompt,
            "prompt_format": prompt_format,
            "system_prompt": system_prompt,
            "media_path": record.media_path,
            "tags": list(record.tags),
            "metadata": record.metadata or {},
            "rendered_prompt": rendered_prompt,
            "num_tokens": int(hidden.shape[0]),
            "hidden_dim": int(hidden.shape[-1]),
            "model_output_type": type(outputs).__name__,
            "input_summary": summarize_batch(inputs),
            "token_kind_counts": count_token_kinds(token_map),
            "visual_grid": token_meta.get("visual_grid"),
            "token_map": token_map,
        }
        return hidden, meta

    @torch.no_grad()
    def generate(
        self,
        *,
        prompt: str,
        layer: int | None = None,
        edit_fn: Any | None = None,
        max_new_tokens: int = 128,
        prompt_format: PromptFormat = "chat",
        system_prompt: str | None = None,
    ) -> str:
        if self.init_mode == "meta":
            raise RuntimeLoadError("init_mode=meta can inspect architecture but cannot generate")
        processor, model = self.require_processor_and_model()
        text = render_text_prompt(
            processor,
            prompt,
            prompt_format=prompt_format,
            system_prompt=system_prompt,
        )
        inputs = processor(text=[text], return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        manager = self._hook(layer, edit_fn) if layer is not None and edit_fn is not None else contextlib.nullcontext()
        with manager:
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
        prompt_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
        if prompt_len and generated.ndim == 2 and generated.shape[-1] > prompt_len:
            generated = generated[:, prompt_len:]
        if hasattr(processor, "batch_decode"):
            return processor.batch_decode(generated, skip_special_tokens=True)[0]
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None:
            return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        return str(generated)

    @contextlib.contextmanager
    def _hook(self, layer: int | None, fn: Any) -> Iterator[None]:
        if layer is None:
            yield
            return
        _, model = self.require_loaded()
        blocks = find_transformer_blocks(model)
        if layer < 0 or layer >= len(blocks):
            raise RuntimeLoadError(f"layer {layer} is outside available hook range [0, {len(blocks)})")
        _name, module = blocks[layer]
        handle = module.register_forward_hook(fn)
        try:
            yield
        finally:
            handle.remove()


def _resolve_dtype(dtype: str) -> torch.dtype:
    if dtype in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype in {"fp16", "float16"}:
        return torch.float16
    if dtype in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype={dtype!r}")


@contextlib.contextmanager
def _default_dtype(dtype: torch.dtype) -> Iterator[None]:
    previous = torch.get_default_dtype()
    if dtype.is_floating_point:
        torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _resolve_model_class() -> Any:
    try:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText
    except Exception:
        pass
    try:
        from transformers import AutoModelForVision2Seq

        return AutoModelForVision2Seq
    except Exception:
        pass
    try:
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM
    except Exception as exc:
        raise RuntimeLoadError(
            "No usable Transformers auto model class found. Need ImageTextToText, Vision2Seq, or CausalLM support."
        ) from exc


def render_record_prompt(
    processor: Any,
    record: ManifestRecord,
    *,
    prompt_format: PromptFormat = "chat",
    system_prompt: str | None = None,
) -> str:
    if prompt_format != "chat":
        raise ValueError("prompt_format must be chat")
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    messages.append({"role": "user", "content": _message_content(record)})
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def render_text_prompt(
    processor: Any,
    prompt: str,
    *,
    prompt_format: PromptFormat = "chat",
    system_prompt: str | None = None,
) -> str:
    if prompt_format != "chat":
        raise ValueError("prompt_format must be chat")
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_token_map(
    *,
    processor: Any | None,
    batch: dict[str, Any],
    media_type: str,
    context_radius: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    input_ids = _first_row(batch.get("input_ids"))
    mm_token_type_ids = _first_row(batch.get("mm_token_type_ids"))
    tokenizer = getattr(processor, "tokenizer", None)
    visual_grid = _visual_grid_metadata(processor, batch, media_type, mm_token_type_ids)
    visual_ordinal = 0
    tokens: list[dict[str, Any]] = []
    for idx, token_id in enumerate(input_ids):
        token_text = _decode_token(tokenizer, token_id)
        is_visual = idx < len(mm_token_type_ids) and int(mm_token_type_ids[idx]) != 0
        kind = media_type if is_visual and media_type in {"image", "video"} else _token_kind(tokenizer, token_id, token_text)
        entry: dict[str, Any] = {
            "index": idx,
            "kind": kind,
            "token_id": int(token_id),
            "token_text": token_text,
        }
        if kind == "text":
            entry["text_context"] = _decode_window(tokenizer, input_ids, idx, context_radius)
        if is_visual:
            entry["visual_ordinal"] = visual_ordinal
            visual_position = _visual_position(visual_grid, visual_ordinal)
            if visual_position:
                entry["visual_position"] = visual_position
            visual_ordinal += 1
        tokens.append(entry)
    return tokens, {"visual_grid": visual_grid}


def summarize_batch(batch: dict[str, Any]) -> dict[str, Any]:
    summary = {}
    for key, value in batch.items():
        if hasattr(value, "shape"):
            summary[key] = {
                "shape": [int(dim) for dim in value.shape],
                "dtype": str(getattr(value, "dtype", "")),
            }
        else:
            summary[key] = {"type": type(value).__name__}
    return summary


def count_token_kinds(token_map: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in token_map:
        kind = str(token.get("kind", "unknown"))
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def load_video_frames(path: str, *, max_frames: int | None = None) -> Any:
    frames, _metadata = load_video_frames_with_metadata(path, max_frames=max_frames)
    return frames


def load_video_frames_with_metadata(path: str, *, max_frames: int | None = None) -> tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
        from transformers.video_utils import VideoMetadata
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeLoadError("video records require opencv-python-headless for frame decoding") from exc

    frame_limit = max_frames or int(os.environ.get("COSMOS_SAE_VIDEO_FRAMES", "16"))
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeLoadError(f"could not open video file: {path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or None
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None
        sampled_indices: list[int] = []
        if frame_count > 0:
            if frame_count <= frame_limit:
                indices = list(range(frame_count))
            else:
                indices = np.linspace(0, frame_count - 1, frame_limit).round().astype(int).tolist()
            sampled_indices = [int(index) for index in indices]
            frames = []
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if ok:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            frames = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if len(frames) > frame_limit:
                indices = np.linspace(0, len(frames) - 1, frame_limit).round().astype(int).tolist()
                sampled_indices = [int(index) for index in indices]
                frames = [frames[index] for index in indices]
            else:
                sampled_indices = list(range(len(frames)))
    finally:
        capture.release()
    if not frames:
        raise RuntimeLoadError(f"could not decode frames from video file: {path}")
    duration = float(frame_count / fps) if frame_count and fps else None
    metadata = VideoMetadata(
        total_num_frames=len(frames),
        fps=fps,
        width=width,
        height=height,
        duration=duration,
        video_backend="opencv",
        frames_indices=sampled_indices,
    )
    return np.stack(frames, axis=0), metadata


def _first_row(value: Any) -> list[int]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(item) for item in value]


def _decode_token(tokenizer: Any | None, token_id: int) -> str:
    if tokenizer is None:
        return str(token_id)
    try:
        return tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except TypeError:
        return tokenizer.decode([int(token_id)])


def _decode_window(tokenizer: Any | None, input_ids: list[int], token_index: int, radius: int) -> str:
    if tokenizer is None:
        return ""
    start = max(0, token_index - radius)
    end = min(len(input_ids), token_index + radius + 1)
    try:
        return tokenizer.decode(input_ids[start:end], skip_special_tokens=False)
    except TypeError:
        return tokenizer.decode(input_ids[start:end])


def _token_kind(tokenizer: Any | None, token_id: int, token_text: str) -> str:
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    if int(token_id) in special_ids or (token_text.startswith("<|") and token_text.endswith("|>")):
        return "special"
    return "text"


def _visual_grid_metadata(
    processor: Any | None,
    batch: dict[str, Any],
    media_type: str,
    mm_token_type_ids: list[int],
) -> dict[str, Any] | None:
    grid_key = "video_grid_thw" if media_type == "video" else "image_grid_thw"
    grid_rows = _grid_rows(batch.get(grid_key))
    if not grid_rows:
        return None
    t, h, w = grid_rows[0]
    visual_tokens = sum(1 for value in mm_token_type_ids if int(value) != 0)
    merge_size = _merge_size(processor, t, h, w, visual_tokens)
    merged_h = max(1, h // merge_size)
    merged_w = max(1, w // merge_size)
    return {
        "media_type": media_type,
        "grid_key": grid_key,
        "grid_thw": [t, h, w],
        "merge_size": merge_size,
        "merged_grid_thw": [t, merged_h, merged_w],
        "visual_tokens": visual_tokens,
    }


def _grid_rows(value: Any) -> list[list[int]]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [[int(item) for item in row] for row in value]


def _merge_size(processor: Any | None, t: int, h: int, w: int, visual_tokens: int) -> int:
    image_processor = getattr(processor, "image_processor", None)
    configured = getattr(image_processor, "merge_size", None)
    if isinstance(configured, int) and configured > 0:
        return configured
    if visual_tokens <= 0:
        return 1
    full_units = max(1, t * h * w)
    ratio = max(1, round((full_units / visual_tokens) ** 0.5))
    return ratio if h % ratio == 0 and w % ratio == 0 else 1


def _visual_position(visual_grid: dict[str, Any] | None, visual_ordinal: int) -> dict[str, Any] | None:
    if not visual_grid:
        return None
    _t, merged_h, merged_w = visual_grid["merged_grid_thw"]
    merge_size = int(visual_grid["merge_size"])
    tokens_per_frame = max(1, int(merged_h) * int(merged_w))
    frame = visual_ordinal // tokens_per_frame
    offset = visual_ordinal % tokens_per_frame
    patch_y = offset // int(merged_w)
    patch_x = offset % int(merged_w)
    return {
        "frame": int(frame),
        "patch_y": int(patch_y),
        "patch_x": int(patch_x),
        "patch_y_range": [int(patch_y * merge_size), int((patch_y + 1) * merge_size)],
        "patch_x_range": [int(patch_x * merge_size), int((patch_x + 1) * merge_size)],
    }


def _message_content(record: ManifestRecord) -> list[dict[str, Any]]:
    if record.media_type == "text":
        return [{"type": "text", "text": record.prompt}]
    if record.media_type == "image":
        return [
            {"type": "image", "image": str(record.media_path)},
            {"type": "text", "text": record.prompt},
        ]
    if record.media_type == "video":
        return [
            {"type": "video", "video": str(record.media_path)},
            {"type": "text", "text": record.prompt},
        ]
    raise ValueError(f"unsupported media_type={record.media_type!r}")


def _extract_hidden(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        output = output[0]
    if hasattr(output, "last_hidden_state"):
        output = output.last_hidden_state
    if not isinstance(output, torch.Tensor):
        raise RuntimeLoadError(f"hook output did not contain tensor hidden states: {type(output).__name__}")
    if output.ndim == 2:
        output = output.unsqueeze(0)
    if output.ndim != 3:
        raise RuntimeLoadError(f"expected hidden states [B,T,D], got shape {tuple(output.shape)}")
    return output


def find_transformer_blocks(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    candidates = [
        "model.layers",
        "language_model.model.layers",
        "language_model.layers",
        "text_model.layers",
        "model.language_model.layers",
    ]
    for dotted in candidates:
        obj: Any = model
        for part in dotted.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if isinstance(obj, torch.nn.ModuleList) and len(obj) > 0:
            return [(f"{dotted}.{i}", module) for i, module in enumerate(obj)]
    # Last-resort heuristic for unfamiliar wrappers.
    module_lists: list[tuple[str, torch.nn.ModuleList]] = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and "layer" in name.lower() and len(module) > 0:
            module_lists.append((name, module))
    if module_lists:
        name, module_list = max(module_lists, key=lambda item: len(item[1]))
        return [(f"{name}.{i}", module) for i, module in enumerate(module_list)]
    raise RuntimeLoadError("could not locate transformer block ModuleList for residual-stream hooks")
