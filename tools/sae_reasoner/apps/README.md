# Cosmos Reasoner Streamlit App

This app is a thin UI over the same direct-Python runtime used by activation
collection. Generation calls go through:

```text
CosmosReasonerRuntime.generate_for_record()
```

That means manifest records, uploaded media, prompt rendering, media
materialization, processor inputs, model loading, dtype/device selection, and
generation all share the `tools.sae_reasoner.runtime` code path.

Deploy and start on a new RunPod RTX 3090 from the repo root:

```bash
tools/sae_reasoner/apps/deploy_streamlit_runpod.sh
```

The script creates or starts a RunPod pod, syncs the app code and manifest,
runs `tools/sae_reasoner/runpod_setup_uv.sh`, starts Streamlit in tmux, and
prints the app URL. By default it creates a 3090 in `EU-CZ-1` with a 24-hour
termination guard.

Useful overrides:

```bash
COSMOS_SAE_STREAMLIT_POD_ID=<existing-pod-id> tools/sae_reasoner/apps/deploy_streamlit_runpod.sh
COSMOS_SAE_STREAMLIT_DATA_CENTER_IDS=EU-CZ-1 tools/sae_reasoner/apps/deploy_streamlit_runpod.sh
COSMOS_SAE_STREAMLIT_GPU_ID="NVIDIA GeForce RTX 4090" tools/sae_reasoner/apps/deploy_streamlit_runpod.sh
COSMOS_SAE_STREAMLIT_ENV_FILE=.env tools/sae_reasoner/apps/deploy_streamlit_runpod.sh
```

`COSMOS_SAE_STREAMLIT_ENV_FILE` is optional and intentionally explicit because
it copies credentials/secrets to the pod.

Manual run on a GPU pod after setup:

```bash
cd /workspace/cosmos-sae-reasoner
set -a; [ -f .env ] && source .env; set +a
export HF_HOME=/workspace/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub
export COSMOS_SAE_MEDIA_CACHE=/workspace/.cache/sae_reasoner/media
.venv/bin/streamlit run tools/sae_reasoner/apps/reasoner_streamlit.py \
  --server.address 0.0.0.0 \
  --server.port 8501
```

Optional defaults:

```bash
export COSMOS_SAE_MODEL_ID=nvidia/Cosmos3-Nano
export COSMOS_SAE_MANIFEST=outputs/sae_reasoner/manifests/physical_ai_instruct_10m_with_synhuman_20260616_2018.jsonl
```
