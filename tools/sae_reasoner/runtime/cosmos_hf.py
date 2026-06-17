from __future__ import annotations

import contextlib
import os
import subprocess
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Literal

import torch

from ..manifest import ManifestRecord
from ..media import materialize_media_path

PromptFormat = Literal["chat"]
ActivationPhase = Literal["prefill", "decode", "both"]
ActivationSaveDType = Literal["auto", "float32", "bfloat16", "float16"]


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

    def inputs_for_records(
        self,
        records: list[ManifestRecord],
        *,
        prompt_format: PromptFormat = "chat",
        system_prompt: str | None = None,
    ) -> tuple[dict[str, torch.Tensor], list[str]]:
        if not records:
            raise ValueError("records must not be empty")
        if self.init_mode == "meta":
            raise RuntimeLoadError("init_mode=meta can inspect architecture but cannot run inputs or forward passes")
        media_types = {record.media_type for record in records}
        if len(media_types) != 1:
            raise RuntimeLoadError("batched activation collection requires records with the same media_type")
        media_type = records[0].media_type
        processor, _ = self.require_processor_and_model()
        rendered_prompts: list[str] = []
        materialized_paths: list[str] = []
        for record in records:
            render_record = record
            if record.media_type != "text":
                materialized = materialize_media_path(record.media_path)
                materialized_paths.append(materialized)
                render_record = replace(record, media_path=materialized)
            rendered_prompts.append(
                render_record_prompt(
                    processor,
                    render_record,
                    prompt_format=prompt_format,
                    system_prompt=system_prompt,
                )
            )
        kwargs: dict[str, Any] = {"text": rendered_prompts, "return_tensors": "pt", "padding": True}
        if media_type == "image":
            from PIL import Image

            kwargs["images"] = [Image.open(path).convert("RGB") for path in materialized_paths]
        elif media_type == "video":
            videos = []
            video_metadata = []
            for path in materialized_paths:
                video_frames, one_video_metadata = load_video_frames_with_metadata(path)
                videos.append(video_frames)
                video_metadata.append(one_video_metadata)
            kwargs["videos"] = videos
            kwargs["video_metadata"] = video_metadata
        try:
            batch = processor(**kwargs)
        except Exception as exc:  # pragma: no cover - processor-specific
            raise RuntimeLoadError(f"processor could not build batched inputs for {len(records)} records") from exc
        return {k: v.to(self.device) if hasattr(v, "to") else v for k, v in batch.items()}, rendered_prompts

    @torch.no_grad()
    def collect_prefill(
        self,
        record: ManifestRecord,
        *,
        layer: int,
        prompt_format: PromptFormat = "chat",
        system_prompt: str | None = None,
        activation_dtype: ActivationSaveDType = "bfloat16",
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if self.init_mode == "meta":
            raise RuntimeLoadError("init_mode=meta can inspect architecture but cannot collect activations")
        _, model = self.require_loaded()
        captures: list[torch.Tensor] = []
        save_dtype = _resolve_activation_save_dtype(activation_dtype, self.dtype)

        def capture(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            hidden = _extract_hidden(output)
            captures.append(_capture_activation(hidden, save_dtype))

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
            "activation_dtype": str(hidden.dtype).replace("torch.", ""),
            "model_output_type": type(outputs).__name__,
            "input_summary": summarize_batch(inputs),
            "token_kind_counts": count_token_kinds(token_map),
            "visual_grid": token_meta.get("visual_grid"),
            "token_map": token_map,
        }
        return hidden, meta

    @torch.no_grad()
    def collect_prefill_batch(
        self,
        records: list[ManifestRecord],
        *,
        layer: int,
        prompt_format: PromptFormat = "chat",
        system_prompt: str | None = None,
        activation_dtype: ActivationSaveDType = "bfloat16",
    ) -> list[tuple[torch.Tensor, dict[str, Any]]]:
        if len(records) == 1:
            return [
                self.collect_prefill(
                    records[0],
                    layer=layer,
                    prompt_format=prompt_format,
                    system_prompt=system_prompt,
                    activation_dtype=activation_dtype,
                )
            ]
        if self.init_mode == "meta":
            raise RuntimeLoadError("init_mode=meta can inspect architecture but cannot collect activations")
        _, model = self.require_loaded()
        captures: list[torch.Tensor] = []
        save_dtype = _resolve_activation_save_dtype(activation_dtype, self.dtype)

        def capture(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            hidden = _extract_hidden(output)
            captures.append(_capture_activation(hidden, save_dtype))

        inputs, rendered_prompts = self.inputs_for_records(
            records,
            prompt_format=prompt_format,
            system_prompt=system_prompt,
        )
        with self._hook(layer, capture):
            outputs = model(**inputs, use_cache=False)
        if not captures:
            raise RuntimeLoadError(f"no activation captured at layer {layer}")
        hidden_batch = captures[-1]
        out: list[tuple[torch.Tensor, dict[str, Any]]] = []
        for row_index, record in enumerate(records):
            token_map, token_meta = build_token_map(
                processor=getattr(self, "processor", None),
                batch=inputs,
                media_type=record.media_type,
                row_index=row_index,
            )
            hidden = hidden_batch[row_index]
            keep_positions = _active_positions(inputs.get("attention_mask"), row_index=row_index)
            if keep_positions and len(keep_positions) <= int(hidden.shape[0]):
                hidden = hidden[torch.tensor(keep_positions, dtype=torch.long)]
            elif len(token_map) < int(hidden.shape[0]):
                hidden = hidden[: len(token_map)]
            if len(token_map) != int(hidden.shape[0]):
                token_map = token_map[: int(hidden.shape[0])]
            meta = {
                "id": record.id,
                "media_type": record.media_type,
                "prompt": record.prompt,
                "prompt_format": prompt_format,
                "system_prompt": system_prompt,
                "media_path": record.media_path,
                "tags": list(record.tags),
                "metadata": record.metadata or {},
                "rendered_prompt": rendered_prompts[row_index],
                "num_tokens": int(hidden.shape[0]),
                "hidden_dim": int(hidden.shape[-1]),
                "activation_dtype": str(hidden.dtype).replace("torch.", ""),
                "model_output_type": type(outputs).__name__,
                "input_summary": summarize_batch(inputs),
                "token_kind_counts": count_token_kinds(token_map),
                "visual_grid": token_meta.get("visual_grid"),
                "token_map": token_map,
                "phase": "prefill",
                "token_phase_counts": count_token_phases(token_map),
            }
            out.append((hidden, meta))
        return out

    @torch.no_grad()
    def collect_activations(
        self,
        record: ManifestRecord,
        *,
        layer: int,
        phase: ActivationPhase = "both",
        max_new_tokens: int = 128,
        prompt_format: PromptFormat = "chat",
        system_prompt: str | None = None,
        activation_dtype: ActivationSaveDType = "bfloat16",
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if phase not in {"prefill", "decode", "both"}:
            raise ValueError("phase must be one of: prefill, decode, both")
        if phase == "prefill":
            hidden, meta = self.collect_prefill(
                record,
                layer=layer,
                prompt_format=prompt_format,
                system_prompt=system_prompt,
                activation_dtype=activation_dtype,
            )
            meta["phase"] = "prefill"
            meta["token_kind_counts"] = count_token_kinds(meta.get("token_map") or [])
            meta["token_phase_counts"] = count_token_phases(meta.get("token_map") or [])
            return hidden, meta
        if self.init_mode == "meta":
            raise RuntimeLoadError("init_mode=meta can inspect architecture but cannot collect activations")

        processor, model = self.require_processor_and_model()
        captures: list[tuple[str, torch.Tensor]] = []
        save_dtype = _resolve_activation_save_dtype(activation_dtype, self.dtype)

        def capture(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            hidden = _capture_activation(_extract_hidden(output), save_dtype)
            capture_phase = "prefill" if not captures else "decode"
            captures.append((capture_phase, hidden))

        inputs, rendered_prompt = self.inputs_for_record(
            record,
            prompt_format=prompt_format,
            system_prompt=system_prompt,
        )
        prefill_token_map, token_meta = build_token_map(
            processor=processor,
            batch=inputs,
            media_type=record.media_type,
        )
        with self._hook(layer, capture):
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
        if not captures:
            raise RuntimeLoadError(f"no activation captured at layer {layer}")

        prompt_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
        generated_ids = _first_row(generated)
        new_token_ids = generated_ids[prompt_len:] if prompt_len and len(generated_ids) >= prompt_len else generated_ids
        generated_text = decode_tokens(processor, new_token_ids)

        selected_activations: list[torch.Tensor] = []
        selected_token_map: list[dict[str, Any]] = []
        for capture_phase, capture_hidden in captures:
            hidden = capture_hidden.squeeze(0)
            if capture_phase == "prefill":
                if phase in {"prefill", "both"}:
                    selected_activations.append(hidden)
                    selected_token_map.extend(mark_token_phase(prefill_token_map, phase="prefill"))
                continue
            decode_hidden = hidden.reshape(-1, hidden.shape[-1])
            decode_token_map = build_decode_token_map(
                processor=processor,
                token_ids=new_token_ids,
                start=len(selected_token_map),
                offset=sum(1 for token in selected_token_map if token.get("phase") == "decode"),
                count=decode_hidden.shape[0],
            )
            if phase in {"decode", "both"} and decode_token_map:
                keep = min(decode_hidden.shape[0], len(decode_token_map))
                selected_activations.append(decode_hidden[:keep])
                selected_token_map.extend(decode_token_map[:keep])

        if not selected_activations:
            raise RuntimeLoadError(
                f"no {phase} activations captured at layer {layer}; "
                "decode-only collection may need max_new_tokens greater than 1"
            )
        hidden = torch.cat(selected_activations, dim=0)
        meta = {
            "id": record.id,
            "media_type": record.media_type,
            "prompt": record.prompt,
            "prompt_format": prompt_format,
            "system_prompt": system_prompt,
            "phase": phase,
            "media_path": record.media_path,
            "tags": list(record.tags),
            "metadata": record.metadata or {},
            "rendered_prompt": rendered_prompt,
            "generated_text": generated_text,
            "generated_token_ids": new_token_ids,
            "num_tokens": int(hidden.shape[0]),
            "hidden_dim": int(hidden.shape[-1]),
            "activation_dtype": str(hidden.dtype).replace("torch.", ""),
            "model_output_type": type(generated).__name__,
            "input_summary": summarize_batch(inputs),
            "token_kind_counts": count_token_kinds(selected_token_map),
            "token_phase_counts": count_token_phases(selected_token_map),
            "visual_grid": token_meta.get("visual_grid"),
            "token_map": selected_token_map,
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

    @torch.no_grad()
    def generate_for_record(
        self,
        record: ManifestRecord,
        *,
        layer: int | None = None,
        edit_fn: Any | None = None,
        max_new_tokens: int = 128,
        prompt_format: PromptFormat = "chat",
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        if self.init_mode == "meta":
            raise RuntimeLoadError("init_mode=meta can inspect architecture but cannot generate")
        processor, model = self.require_processor_and_model()
        inputs, rendered_prompt = self.inputs_for_record(
            record,
            prompt_format=prompt_format,
            system_prompt=system_prompt,
        )
        token_map, token_meta = build_token_map(
            processor=processor,
            batch=inputs,
            media_type=record.media_type,
        )
        if edit_fn is not None and hasattr(edit_fn, "token_map") and getattr(edit_fn, "token_map") is None:
            edit_fn.token_map = token_map
        manager = self._hook(layer, edit_fn) if layer is not None and edit_fn is not None else contextlib.nullcontext()
        with manager:
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
        prompt_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
        generated_ids = _first_row(generated)
        new_token_ids = generated_ids[prompt_len:] if prompt_len and len(generated_ids) >= prompt_len else generated_ids
        return {
            "text": decode_tokens(processor, new_token_ids),
            "rendered_prompt": rendered_prompt,
            "generated_token_ids": new_token_ids,
            "token_kind_counts": count_token_kinds(token_map),
            "token_phase_counts": count_token_phases(token_map),
            "visual_grid": token_meta.get("visual_grid"),
        }

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


def _resolve_activation_save_dtype(dtype: ActivationSaveDType, model_dtype: torch.dtype) -> torch.dtype:
    if dtype == "auto":
        return model_dtype if model_dtype in {torch.bfloat16, torch.float16, torch.float32} else torch.float32
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    if dtype == "float32":
        return torch.float32
    raise ValueError("activation_dtype must be one of: auto, float32, bfloat16, float16")


def _capture_activation(hidden: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    captured = hidden.detach()
    if captured.dtype != dtype:
        captured = captured.to(dtype=dtype)
    return captured.cpu()


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
    row_index: int = 0,
    context_radius: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    input_ids = _row_values(batch.get("input_ids"), row_index=row_index)
    mm_token_type_ids = _row_values(batch.get("mm_token_type_ids"), row_index=row_index)
    attention_mask = _row_values(batch.get("attention_mask"), row_index=row_index)
    tokenizer = getattr(processor, "tokenizer", None)
    visual_grid = _visual_grid_metadata(processor, batch, media_type, mm_token_type_ids, row_index=row_index)
    visual_ordinal = 0
    tokens: list[dict[str, Any]] = []
    role_state = ChatRoleState()
    for raw_idx, token_id in enumerate(input_ids):
        if raw_idx < len(attention_mask) and int(attention_mask[raw_idx]) == 0:
            continue
        idx = len(tokens)
        token_text = _decode_token(tokenizer, token_id)
        is_visual = raw_idx < len(mm_token_type_ids) and int(mm_token_type_ids[raw_idx]) != 0
        kind = media_type if is_visual and media_type in {"image", "video"} else _token_kind(tokenizer, token_id, token_text)
        role = role_state.update(token_text)
        entry: dict[str, Any] = {
            "index": idx,
            "kind": kind,
            "phase": "prefill",
            "role": role,
            # Residual stream this token is routed through. Cosmos 3 is a Mixture-of-Transformers:
            # AR-routed and diffusion-routed params share attention but write separate residual
            # streams. The understanding forward only exercises the AR stream, so this is "ar";
            # a future generation collector would tag diffusion-subsequence tokens "diffusion".
            "stream": "ar",
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


def count_token_phases(token_map: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in token_map:
        phase = str(token.get("phase", "unknown"))
        counts[phase] = counts.get(phase, 0) + 1
    return counts


class ChatRoleState:
    def __init__(self) -> None:
        self.current: str | None = None
        self.expect_role = False

    def update(self, token_text: str) -> str | None:
        role = self.current
        if "<|im_start|>" in token_text:
            self.expect_role = True
            self.current = None
            return None
        stripped = token_text.strip()
        if self.expect_role and stripped in {"system", "user", "assistant"}:
            self.current = stripped
            self.expect_role = False
            return self.current
        if "<|im_end|>" in token_text:
            role = self.current
            self.current = None
            self.expect_role = False
            return role
        return self.current


def mark_token_phase(token_map: list[dict[str, Any]], *, phase: str) -> list[dict[str, Any]]:
    return [{**token, "index": idx, "phase": phase} for idx, token in enumerate(token_map)]


def build_decode_token_map(
    *,
    processor: Any | None,
    token_ids: list[int],
    start: int,
    offset: int,
    count: int,
    context_radius: int = 8,
) -> list[dict[str, Any]]:
    tokenizer = getattr(processor, "tokenizer", None)
    selected = token_ids[offset : offset + count]
    out: list[dict[str, Any]] = []
    for local_idx, token_id in enumerate(selected):
        token_text = _decode_token(tokenizer, token_id)
        generated_index = offset + local_idx
        entry = {
            "index": start + local_idx,
            "kind": _token_kind(tokenizer, token_id, token_text),
            "phase": "decode",
            "role": "assistant",
            "stream": "ar",  # decode tokens are autoregressive; see prefill note on Mixture-of-Transformers streams
            "token_id": int(token_id),
            "token_text": token_text,
            "generated_index": int(generated_index),
        }
        if entry["kind"] == "text":
            entry["text_context"] = _decode_window(tokenizer, token_ids, generated_index, context_radius)
        out.append(entry)
    return out


def decode_tokens(processor: Any | None, token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    if processor is not None and hasattr(processor, "batch_decode"):
        return processor.batch_decode([token_ids], skip_special_tokens=True)[0]
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        return tokenizer.batch_decode([token_ids], skip_special_tokens=True)[0]
    return " ".join(str(token_id) for token_id in token_ids)


def load_video_frames(path: str, *, max_frames: int | None = None) -> Any:
    frames, _metadata = load_video_frames_with_metadata(path, max_frames=max_frames)
    return frames


def load_video_frames_with_metadata(path: str, *, max_frames: int | None = None) -> tuple[Any, Any]:
    frame_limit = max_frames or int(os.environ.get("COSMOS_SAE_VIDEO_FRAMES", "16"))
    if _video_codec_name(path) == "av1":
        try:
            return _load_video_frames_with_ffmpeg(path, frame_limit=frame_limit, codec="av1")
        except RuntimeLoadError as ffmpeg_exc:
            try:
                return _load_video_frames_with_pyav(path, frame_limit=frame_limit)
            except RuntimeLoadError as pyav_exc:
                raise RuntimeLoadError(
                    f"could not decode AV1 video file with FFmpeg/libdav1d or PyAV: {path}; "
                    f"ffmpeg={ffmpeg_exc}; pyav={pyav_exc}"
                ) from pyav_exc
    try:
        return _load_video_frames_with_opencv(path, frame_limit=frame_limit)
    except RuntimeLoadError as opencv_exc:
        try:
            return _load_video_frames_with_pyav(path, frame_limit=frame_limit)
        except RuntimeLoadError as pyav_exc:
            raise RuntimeLoadError(
                f"could not decode video file with OpenCV or PyAV: {path}; "
                f"opencv={opencv_exc}; pyav={pyav_exc}"
            ) from pyav_exc


def _video_codec_name(path: str) -> str | None:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "default=nw=1:nk=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().splitlines()[0].lower()
    except Exception:
        pass
    try:
        import av
    except Exception:
        return None
    try:
        with av.open(str(path), mode="r") as container:
            stream = next((candidate for candidate in container.streams if candidate.type == "video"), None)
            if stream is None:
                return None
            codec_context = getattr(stream, "codec_context", None)
            name = getattr(codec_context, "name", None) or getattr(stream, "codec", None)
            return str(name).lower() if name else None
    except Exception:
        return None


def _load_video_frames_with_ffmpeg(path: str, *, frame_limit: int, codec: str | None = None) -> tuple[Any, Any]:
    try:
        import numpy as np
        from PIL import Image
        from transformers.video_utils import VideoMetadata
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeLoadError("FFmpeg video fallback requires numpy, pillow, and transformers") from exc

    probe = _video_probe(path)
    total_frames = probe.get("frames") or 0
    step = max(1, int(total_frames // frame_limit)) if total_frames > frame_limit else 1
    select_filter = f"select=not(mod(n\\,{step}))"
    input_args = ["-fflags", "+discardcorrupt", "-err_detect", "ignore_err", "-hwaccel", "none"]
    if codec == "av1":
        input_args += ["-c:v", "libdav1d"]
    with tempfile.TemporaryDirectory(prefix="cosmos_sae_frames_") as tmp:
        output_pattern = str(Path(tmp) / "frame_%06d.png")
        cmd = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            *input_args,
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            select_filter,
            "-frames:v",
            str(frame_limit),
            "-vsync",
            "vfr",
            output_pattern,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        frame_paths = sorted(Path(tmp).glob("frame_*.png"))
        if result.returncode != 0 and not frame_paths:
            stderr = (result.stderr or "").strip().splitlines()
            raise RuntimeLoadError(f"FFmpeg could not decode frames from video file: {path}: {' | '.join(stderr[:3])}")
        frames = [np.asarray(Image.open(frame_path).convert("RGB")) for frame_path in frame_paths[:frame_limit]]
    if not frames:
        raise RuntimeLoadError(f"FFmpeg decoded zero frames from video file: {path}")
    sampled_indices = [int(index * step) for index in range(len(frames))]
    height, width = int(frames[0].shape[0]), int(frames[0].shape[1])
    metadata = VideoMetadata(
        total_num_frames=len(frames),
        fps=probe.get("fps"),
        width=probe.get("width") or width,
        height=probe.get("height") or height,
        duration=probe.get("duration"),
        video_backend="ffmpeg",
        frames_indices=sampled_indices,
    )
    return np.stack(frames, axis=0), metadata


def _video_probe(path: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,avg_frame_rate,nb_frames,duration",
                "-of",
                "default=nw=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return result
    if probe.returncode != 0:
        return result
    for line in probe.stdout.splitlines():
        key, sep, value = line.partition("=")
        if not sep or value in {"", "N/A"}:
            continue
        if key in {"width", "height", "nb_frames"}:
            try:
                result["frames" if key == "nb_frames" else key] = int(value)
            except ValueError:
                pass
        elif key == "duration":
            try:
                result["duration"] = float(value)
            except ValueError:
                pass
        elif key == "avg_frame_rate":
            result["fps"] = _parse_frame_rate(value)
    return result


def _parse_frame_rate(value: str) -> float | None:
    numerator, sep, denominator = value.partition("/")
    try:
        if sep:
            denom = float(denominator)
            return float(numerator) / denom if denom else None
        return float(value)
    except ValueError:
        return None


def _load_video_frames_with_opencv(path: str, *, frame_limit: int) -> tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
        from transformers.video_utils import VideoMetadata
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeLoadError("video records require opencv-python-headless for frame decoding") from exc

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


def _load_video_frames_with_pyav(path: str, *, frame_limit: int) -> tuple[Any, Any]:
    try:
        import av
        import numpy as np
        from transformers.video_utils import VideoMetadata
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeLoadError("PyAV video fallback requires av, numpy, and transformers") from exc

    frames = []
    try:
        with av.open(str(path), mode="r") as container:
            stream = next((candidate for candidate in container.streams if candidate.type == "video"), None)
            if stream is None:
                raise RuntimeLoadError(f"video file has no video stream: {path}")
            decoded = [frame.to_rgb().to_ndarray() for frame in container.decode(stream)]
            total_decoded = len(decoded)
            if not decoded:
                raise RuntimeLoadError(f"PyAV decoded zero frames from video file: {path}")
            if total_decoded > frame_limit:
                indices = np.linspace(0, total_decoded - 1, frame_limit).round().astype(int).tolist()
                frames = [decoded[index] for index in indices]
            else:
                indices = list(range(total_decoded))
                frames = decoded
            fps = float(stream.average_rate) if stream.average_rate else None
            width = int(stream.codec_context.width or stream.width or frames[0].shape[1])
            height = int(stream.codec_context.height or stream.height or frames[0].shape[0])
            duration = None
            if stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
    except RuntimeLoadError:
        raise
    except Exception as exc:
        raise RuntimeLoadError(f"PyAV could not decode video file: {path}") from exc

    metadata = VideoMetadata(
        total_num_frames=len(frames),
        fps=fps,
        width=width,
        height=height,
        duration=duration,
        video_backend="pyav",
        frames_indices=[int(index) for index in indices],
    )
    return np.stack(frames, axis=0), metadata


def _first_row(value: Any) -> list[int]:
    return _row_values(value, row_index=0)


def _row_values(value: Any, *, row_index: int) -> list[int]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if row_index >= len(value):
            return []
        value = value[row_index]
    return [int(item) for item in value]


def _active_positions(attention_mask: Any, *, row_index: int) -> list[int]:
    row = _row_values(attention_mask, row_index=row_index)
    if not row:
        return []
    return [idx for idx, value in enumerate(row) if int(value) != 0]


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
    row_index: int = 0,
) -> dict[str, Any] | None:
    grid_key = "video_grid_thw" if media_type == "video" else "image_grid_thw"
    grid_rows = _grid_rows(batch.get(grid_key))
    if not grid_rows:
        return None
    t, h, w = grid_rows[min(row_index, len(grid_rows) - 1)]
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
