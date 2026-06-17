from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

from tools.sae_reasoner.manifest import ManifestRecord, iter_manifest
from tools.sae_reasoner.runtime import CosmosReasonerRuntime, RuntimeLoadError

DEFAULT_MODEL_ID = os.environ.get("COSMOS_SAE_MODEL_ID", "nvidia/Cosmos3-Nano")
DEFAULT_MANIFEST = os.environ.get(
    "COSMOS_SAE_MANIFEST",
    "outputs/sae_reasoner/manifests/physical_ai_instruct_10m_with_synhuman_20260616_2018.jsonl",
)
UPLOAD_DIR = Path(os.environ.get("COSMOS_SAE_STREAMLIT_UPLOAD_DIR", "/tmp/cosmos_sae_reasoner_streamlit_uploads"))


st.set_page_config(page_title="Cosmos Reasoner", layout="wide")


@st.cache_resource(show_spinner="Loading Cosmos Reasoner runtime")
def load_runtime(model_id: str, device: str, dtype: str, init_mode: str) -> CosmosReasonerRuntime:
    return CosmosReasonerRuntime(
        model_id,
        device=device or None,
        dtype=dtype,
        init_mode=init_mode,
    ).load()


@st.cache_data(show_spinner=False)
def load_manifest_records(path: str, limit: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, record in enumerate(iter_manifest(path)):
        if limit > 0 and idx >= limit:
            break
        records.append({"index": idx, "record": record.to_json()})
    return records


def record_from_json(obj: dict[str, Any]) -> ManifestRecord:
    return ManifestRecord.from_json(obj)


def save_upload(uploaded_file: Any) -> str:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name or "").suffix
    with tempfile.NamedTemporaryFile(prefix="media_", suffix=suffix, dir=UPLOAD_DIR, delete=False) as handle:
        handle.write(uploaded_file.getbuffer())
        return handle.name


def sidebar_config() -> dict[str, Any]:
    with st.sidebar:
        st.header("Runtime")
        model_id = st.text_input("Model ID", value=DEFAULT_MODEL_ID)
        device = st.selectbox("Device", ["cuda", "cpu"], index=0)
        dtype = st.selectbox("Dtype", ["bfloat16", "float16", "float32"], index=0)
        init_mode = st.selectbox("Init mode", ["pretrained", "random"], index=0)
        max_new_tokens = st.slider("Max new tokens", min_value=1, max_value=512, value=128, step=1)
        system_prompt = st.text_area("System prompt", value="", height=100)
        show_rendered_prompt = st.checkbox("Show rendered prompt", value=False)
    return {
        "model_id": model_id,
        "device": device,
        "dtype": dtype,
        "init_mode": init_mode,
        "max_new_tokens": int(max_new_tokens),
        "system_prompt": system_prompt or None,
        "show_rendered_prompt": show_rendered_prompt,
    }


def select_record() -> ManifestRecord | None:
    mode = st.radio("Input", ["Text", "Manifest", "Upload"], horizontal=True)

    if mode == "Text":
        prompt = st.text_area(
            "Prompt",
            value="Describe the physical situation and what should happen next.",
            height=180,
        )
        if prompt.strip():
            return ManifestRecord(id="streamlit_text", media_type="text", prompt=prompt.strip(), tags=("streamlit",))

    if mode == "Manifest":
        manifest = st.text_input("Manifest path or URI", value=DEFAULT_MANIFEST)
        limit = st.number_input("Records to index", min_value=1, max_value=10000, value=500, step=100)
        try:
            records = load_manifest_records(manifest, int(limit))
        except Exception as exc:
            st.error(f"Could not load manifest: {exc}")
            records = []
        if records:
            labels = [
                f"{item['index']:06d} | {item['record'].get('media_type')} | {item['record'].get('metadata', {}).get('source_dataset', '')} | {item['record'].get('id')}"
                for item in records
            ]
            selected = st.selectbox("Record", labels)
            selected_idx = labels.index(selected)
            record = record_from_json(records[selected_idx]["record"])
            st.caption(record.id)
            st.text_area("Prompt from manifest", value=record.prompt, height=180, disabled=True)
            return record

    if mode == "Upload":
        media_type = st.radio("Media type", ["image", "video"], horizontal=True)
        uploaded = st.file_uploader("Media", type=["png", "jpg", "jpeg", "webp", "mp4", "mov", "avi", "mkv"])
        prompt = st.text_area(
            "Prompt for media",
            value="Answer using only visible evidence from the media. Describe the physical state and likely next action.",
            height=160,
        )
        if uploaded is not None and prompt.strip():
            path = save_upload(uploaded)
            if media_type == "image":
                st.image(path, caption=uploaded.name)
            else:
                st.video(path)
            return ManifestRecord(
                id=f"streamlit_upload:{uploaded.name}",
                media_type=media_type,
                media_path=path,
                prompt=prompt.strip(),
                tags=("streamlit", "upload", media_type),
            )

    return None


def render_record_details(record: ManifestRecord) -> None:
    with st.expander("Record JSON"):
        st.json(record.to_json())


def main() -> None:
    config = sidebar_config()
    st.title("Cosmos Reasoner")
    st.caption("Uses `CosmosReasonerRuntime.generate_for_record()` from the SAE activation-collection runtime.")

    record = select_record()
    if record is None:
        st.info("Choose an input to generate.")
        return
    render_record_details(record)

    if st.button("Generate", type="primary"):
        try:
            runtime = load_runtime(
                config["model_id"],
                config["device"],
                config["dtype"],
                config["init_mode"],
            )
            with st.spinner("Generating"):
                result = runtime.generate_for_record(
                    record,
                    max_new_tokens=config["max_new_tokens"],
                    prompt_format="chat",
                    system_prompt=config["system_prompt"],
                )
        except RuntimeLoadError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.exception(exc)
            return

        st.subheader("Response")
        st.write(result["text"])
        cols = st.columns(3)
        cols[0].metric("Generated tokens", len(result.get("generated_token_ids") or []))
        cols[1].json(result.get("token_kind_counts") or {})
        cols[2].json(result.get("token_phase_counts") or {})
        if config["show_rendered_prompt"]:
            st.subheader("Rendered Prompt")
            st.text_area("Rendered chat template", value=result.get("rendered_prompt") or "", height=300)


if __name__ == "__main__":
    main()
