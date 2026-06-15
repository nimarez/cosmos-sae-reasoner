# Cosmos3 Reasoner SAE Harness

This is a lightweight research harness for collecting Cosmos3 Reasoner
activations, training a sparse autoencoder, and testing feature steering.

The code is intentionally direct-Python only. vLLM and NIM are useful serving
paths, but they do not expose the residual-stream hooks needed for SAE training
and feature multiplication.

## Quick Local Smoke

These commands do not require model weights:

```bash
uv run --no-project python -m tools.sae_reasoner make-sample-jsonl \
  --output outputs/sae_reasoner/sample_manifest.jsonl

uv run --no-project --with pytest --with torch python -m pytest tools/sae_reasoner/tests
```

## Corpus Manifests

The real SAE corpus should not be the tiny smoke manifest. Use neutral,
pretraining-like records and keep media remote until activation collection.
The builder writes JSONL manifests with `hf://...` or `s3://...` media paths;
`collect-activations` downloads only the current record into
`COSMOS_SAE_MEDIA_CACHE` or `.cache/sae_reasoner/media`.

First-party PhysicalAI recipes:

```bash
python -m tools.sae_reasoner build-corpus-manifest \
  --source recipe \
  --recipe physicalai-vantage \
  --max-records 5000 \
  --output outputs/sae_reasoner/manifests/physicalai_vantage.jsonl

python -m tools.sae_reasoner build-corpus-manifest \
  --source recipe \
  --recipe physicalai-driving \
  --max-records 5000 \
  --output outputs/sae_reasoner/manifests/physicalai_driving.jsonl
```

Generic HF file repos:

```bash
python -m tools.sae_reasoner build-corpus-manifest \
  --source hf-files \
  --hf-repo-id nvidia/PhysicalAI-VANTAGE-Bench \
  --include-glob 'data/*/sequence_*/images/*.jpg' \
  --prompt 'Describe the scene, spatial relations, and physical affordances.' \
  --max-records 1000 \
  --output outputs/sae_reasoner/manifests/custom_hf.jsonl
```

S3 media prefixes:

```bash
python -m tools.sae_reasoner build-corpus-manifest \
  --source s3-prefix \
  --s3-uri s3://my-bucket/cosmos/media/ \
  --include-glob '*.mp4' \
  --prompt 'Describe the key physical events and likely next state.' \
  --max-records 1000 \
  --output outputs/sae_reasoner/manifests/custom_s3.jsonl
```

HF streaming text/row datasets are also supported when a dataset exposes rows:

```bash
python -m tools.sae_reasoner build-corpus-manifest \
  --source hf-dataset \
  --hf-repo-id some-org/some-dataset \
  --hf-split train \
  --text-field text \
  --max-records 1000 \
  --output outputs/sae_reasoner/manifests/text_stream.jsonl
```

## RunPod Flow

On a GPU pod with a compatible Cosmos3 Python runtime installed:

```bash
PYTHON_VERSION=3.12 UV_TORCH_BACKEND=auto bash tools/sae_reasoner/runpod_setup_uv.sh

python -m tools.sae_reasoner prepare-snapshot \
  --model-id nvidia/Cosmos3-Nano \
  --include-weights

python -m tools.sae_reasoner inspect-model \
  --model-id nvidia/Cosmos3-Nano

# No weights: full-size architecture only, no forward pass.
python -m tools.sae_reasoner inspect-model \
  --model-id nvidia/Cosmos3-Nano \
  --init-mode meta

# No checkpoint weights: full-size random weights. This still needs full model
# memory, so use it on RunPod rather than a laptop.
python -m tools.sae_reasoner inspect-model \
  --model-id nvidia/Cosmos3-Nano \
  --init-mode random \
  --dtype bfloat16

python -m tools.sae_reasoner collect-activations \
  --model-id nvidia/Cosmos3-Nano \
  --manifest outputs/sae_reasoner/sample_manifest.jsonl \
  --layer 18 \
  --output-dir outputs/sae_reasoner/activations/sample_l18 \
  --max-examples 8

# Save activation shards directly to S3. Credentials are read from the
# environment by boto3: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, optional
# AWS_SESSION_TOKEN, and AWS_DEFAULT_REGION.
python -m tools.sae_reasoner collect-activations \
  --model-id nvidia/Cosmos3-Nano \
  --manifest outputs/sae_reasoner/sample_manifest.jsonl \
  --layer 18 \
  --output-dir s3://my-bucket/cosmos/sae_reasoner/activations/sample_l18 \
  --max-examples 8

python -m tools.sae_reasoner train-sae \
  --activation-dir outputs/sae_reasoner/activations/sample_l18 \
  --output outputs/sae_reasoner/saes/l18.pt \
  --recon-loss mse \
  --feature-l1-coeff 0.0 \
  --steps 500

python -m tools.sae_reasoner find-features \
  --activation-dir outputs/sae_reasoner/activations/sample_l18 \
  --sae outputs/sae_reasoner/saes/l18.pt \
  --output outputs/sae_reasoner/reports/l18_features.jsonl

python -m tools.sae_reasoner render-feature-report \
  --features outputs/sae_reasoner/reports/l18_features.jsonl \
  --output outputs/sae_reasoner/reports/l18_features.html

python -m tools.sae_reasoner steer \
  --model-id nvidia/Cosmos3-Nano \
  --sae outputs/sae_reasoner/saes/l18.pt \
  --layer 18 \
  --feature-id 0 \
  --multiplier 5 \
  --scope decode \
  --prompt "What is the robot likely to do next?"
```

## Manifest Format

The activation manifest is JSONL. Keep it neutral: no intended feature labels
or expected concepts are used during collection.

```json
{"id":"robot_caption","media_type":"image","media_path":"cookbooks/cosmos3/reasoner/assets/robot_153.jpg","prompt":"Caption the image in detail.","tags":["robotics"],"metadata":{"split":"sae_train"}}
```

Supported `media_type` values are `text`, `image`, and `video`. Video support
depends on the installed processor; if the current Cosmos runtime cannot ingest
video directly, the loader fails with a clear error.

Remote `media_path` values can use `hf://dataset/<namespace>/<repo>/<path>`,
`s3://bucket/key`, or HTTPS URLs. S3 credentials are read from the standard AWS
environment variables.

## Visualization

`render-feature-report` creates a standalone HTML browser for the
`find-features` JSONL output. The layout is intentionally close to the
Anthropic feature-browser workflow: feature list on the left, top activating
examples on the right, activation bars, token positions, prompts, tags, and
source media paths. It is static HTML, so it can be uploaded as a RunPod
artifact or opened locally.
