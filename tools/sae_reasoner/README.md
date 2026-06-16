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

First-party robotics recipes:

```bash
python -m tools.sae_reasoner build-corpus-manifest \
  --source recipe \
  --recipe robotics-bridge-captions \
  --max-records 5000 \
  --output outputs/sae_reasoner/manifests/robotics_bridge_captions.jsonl

python -m tools.sae_reasoner build-corpus-manifest \
  --source recipe \
  --recipe robotics-bridge \
  --max-records 5000 \
  --output outputs/sae_reasoner/manifests/robotics_bridge.jsonl

python -m tools.sae_reasoner build-corpus-manifest \
  --source recipe \
  --recipe robotics-libero \
  --max-records 1000 \
  --output outputs/sae_reasoner/manifests/robotics_libero.jsonl
```

`physicalai-driving` and `physicalai-vantage` remain available, but the default
SAE notebook is robotics-focused.

The `robotics-bridge-captions` recipe pairs each video with the repo's
per-clip `caption.txt` sidecar when available. Passing `--prompt` overrides
that and falls back to a fixed template.

Cosmos RobotSim SDG is tar-sharded on Hugging Face, so use a bounded
materialization step to extract a small sample of MP4s to S3 and write a normal
manifest:

```bash
set -a
source .env
set +a

python -m tools.sae_reasoner build-corpus-manifest \
  --source recipe \
  --recipe physicalai-robotsim \
  --s3-uri s3://my-bucket/cosmos/robotsim/run_001 \
  --manifest-s3-uri s3://my-bucket/cosmos/robotsim/run_001/manifest.jsonl \
  --max-records 100 \
  --max-shards 1 \
  --max-shard-gb 1.0 \
  --output outputs/sae_reasoner/manifests/robotsim_run_001.jsonl
```

`--max-shards` is intentionally important: each RobotSim shard can be large, so
start with `--max-records 10 --max-shards 1 --max-shard-gb 1.0` before scaling
up. Use `--max-shard-gb 0` only when you are willing to download large shards.
The generated manifest points at `s3://...` media and can be used directly by
`collect-activations`; uploaded S3 manifests can also be used as `--manifest
s3://bucket/key.jsonl`.

For a full RobotSim materialization, fan out the existing extractor across
cheap CPU pods. Each worker owns a stable hash partition of tar shards, uploads
its MP4s to the shared media prefix, and writes a per-worker manifest:

```bash
python -m tools.sae_reasoner.scripts.plan_robotsim_fanout \
  --s3-uri s3://my-bucket/cosmos/robotsim/full_v1 \
  --num-workers 8 \
  --output-root outputs/sae_reasoner/robotsim_fanout/full_v1 \
  --max-records-per-worker 0 \
  --max-shards 0 \
  --max-shard-gb 0
```

Copy one generated `scripts/run_worker_*.sh` command to each CPU pod, or run
`outputs/sae_reasoner/robotsim_fanout/full_v1/run_all_local_parallel.sh` for a
local smoke. Workers use `--resume`, so retrying skips already-uploaded media.
After all worker manifests are uploaded, combine them into one sorted manifest:

```bash
python -m tools.sae_reasoner.scripts.combine_manifest_shards \
  --input-prefix s3://my-bucket/cosmos/robotsim/full_v1/manifests/ \
  --output outputs/sae_reasoner/manifests/robotsim_full_v1.jsonl \
  --manifest-s3-uri s3://my-bucket/cosmos/robotsim/full_v1/final_manifest.jsonl
```

To use RobotSim's generated per-clip captions without re-uploading MP4s, enrich
the combined manifest from the paired JSON sidecars. This preserves each
`media_path`, replaces the generic prompt with the sidecar `caption`, and stores
compact non-caption sidecar fields under `metadata.robotsim_sidecar`:

```bash
python -m tools.sae_reasoner.scripts.enrich_robotsim_manifest_sidecars \
  --manifest s3://my-bucket/cosmos/robotsim/full_v1/final_manifest.jsonl \
  --output outputs/sae_reasoner/manifests/robotsim_full_v1_captioned.jsonl \
  --sidecar-s3-uri s3://my-bucket/cosmos/robotsim/full_v1/sidecars \
  --manifest-s3-uri s3://my-bucket/cosmos/robotsim/full_v1/final_manifest_captioned.jsonl
```

For faster enrichment, run the same command across workers with unique local
outputs and unique `--manifest-s3-uri` values, adding `--worker-index N
--num-workers K`; then combine those captioned worker manifests with
`combine_manifest_shards`.

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
  --phase prefill \
  --max-new-tokens 128 \
  --activation-dtype bfloat16 \
  --max-examples 8

# Real runs should save activation shards directly to S3. Credentials are read
# from the environment by boto3: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
# optional AWS_SESSION_TOKEN, and AWS_DEFAULT_REGION.
export COSMOS_SAE_ACTIVATION_S3_URI="${COSMOS_SAE_ACTIVATION_S3_URI:-s3://cosmos-interpretability/sae_reasoner/activations}"
ACTIVATION_URI="${COSMOS_SAE_ACTIVATION_S3_URI%/}/sample_l18"

python -m tools.sae_reasoner collect-activations \
  --model-id nvidia/Cosmos3-Nano \
  --manifest outputs/sae_reasoner/sample_manifest.jsonl \
  --layer 18 \
  --output-dir "$ACTIVATION_URI" \
  --phase prefill \
  --max-new-tokens 128 \
  --activation-dtype bfloat16 \
  --resume \
  --wandb-project "${WANDB_PROJECT:-cosmos-sae-reasoner}" \
  --wandb-run-name sample_l18_collect \
  --wandb-tags collection,cosmos3,sae \
  --max-examples 8

# Large manifests can be collected across multiple GPU pods. The output prefix
# is shared; each worker owns records by global manifest index modulo
# --num-workers, writes stable per-record shards, and writes
# metadata/worker_*.jsonl instead of clobbering metadata.jsonl.
python -m tools.sae_reasoner.scripts.plan_activation_fanout \
  --manifest s3://cosmos-interpretability/cosmos/robotsim/full_v1_stream_20260616/final_manifest_captioned.jsonl \
  --output-dir s3://cosmos-interpretability/sae_reasoner/activations/robotsim_l18_full_prefill \
  --output-root outputs/sae_reasoner/activation_fanout/robotsim_l18_full_prefill \
  --num-workers 8 \
  --layer 18 \
  --phase prefill \
  --activation-dtype bfloat16 \
  --resume \
  --wandb-project "${WANDB_PROJECT:-cosmos-sae-reasoner}" \
  --run-prefix robotsim_l18_full_prefill

python -m tools.sae_reasoner train-sae \
  --activation-dir "$ACTIVATION_URI" \
  --output outputs/sae_reasoner/saes/l18.pt \
  --topk-activation relu_topk \
  --activation-norm sqrt_d \
  --init-method kaiming \
  --recon-loss mse \
  --feature-l1-coeff 0.0 \
  --warmup-steps 0 \
  --lr-schedule constant \
  --max-grad-norm 0 \
  --train-splits sae_train \
  --val-splits sae_val \
  --log-every 10 \
  --steps 500

python -m tools.sae_reasoner find-features \
  --activation-dir "$ACTIVATION_URI" \
  --sae outputs/sae_reasoner/saes/l18.pt \
  --output outputs/sae_reasoner/reports/l18_features.jsonl

python -m tools.sae_reasoner render-feature-report \
  --features outputs/sae_reasoner/reports/l18_features.jsonl \
  --output outputs/sae_reasoner/reports/l18_features.html

python -m tools.sae_reasoner find-neighbors \
  --activation-dir "$ACTIVATION_URI" \
  --output outputs/sae_reasoner/reports/l18_neighbors.jsonl \
  --max-tokens 5000 \
  --num-queries 40 \
  --neighbors 8

python -m tools.sae_reasoner render-neighbor-report \
  --neighbors outputs/sae_reasoner/reports/l18_neighbors.jsonl \
  --output outputs/sae_reasoner/reports/l18_neighbors.html

python -m tools.sae_reasoner steer \
  --model-id nvidia/Cosmos3-Nano \
  --sae outputs/sae_reasoner/saes/l18.pt \
  --layer 18 \
  --feature-id 0 \
  --multiplier 5 \
  --scope prefill \
  --manifest outputs/sae_reasoner/manifests/robotics_l18_5000.jsonl \
  --record-id physical_plausibility \
  --steer-token-kinds video \
  --steer-roles user
```

`collect-activations` defaults to `--phase prefill`, so shards include the
prompt/media tokens the model reads, including any pre-existing caption text in
the manifest prompt. Use `--phase both` only when you explicitly want generated
assistant decode tokens included as training rows. Saved shards default to
`--activation-dtype bfloat16` to avoid doubling GPU-to-CPU transfer and S3
storage; `train-sae` keeps loaded activations in their stored dtype and casts
sampled mini-batches to float32 for optimization.
When W&B is enabled, collection logs live progress metrics including collected
examples, total tokens, per-record seconds, examples/minute, tokens/second,
estimated activation GB, token-kind counts, and token-phase counts.
Use `--resume` with a frozen manifest to skip records whose deterministic shard
and `metadata/<shard>.json` sidecar already exist. This makes interrupted runs
and later extensions idempotent: build a larger manifest with the same source
and seed, point at the same activation prefix, increase `--max-examples`, and
existing rows are skipped by shard name. Manifest splits are assigned by a
stable hash of `record_id` plus seed, so train/validation labels do not change
when a manifest is extended.
For an already-running S3 collection that was launched without W&B, use
`python -m tools.sae_reasoner.scripts.monitor_s3_collection --activation-dir s3://bucket/prefix --target-examples 5000 --wandb-project cosmos-sae-reasoner`
to log S3 shard-count progress from a sidecar process.
`train-sae` and `find-neighbors` default to all token kinds and phases. Use
optional filters such as `--token-kinds video,image`, `--phases decode`,
`--query-kinds image,video`, or `find-features --token-kinds video,special` for
control runs and media/special-token feature browsing. If you explicitly train
with raw signed TopK via `--topk-activation topk`, `find-features` can rank by
`--feature-rank absolute` and still records the signed activation value.

`train-sae` streams JSON metric rows during training and writes the same metrics
to `<output>.metrics.jsonl`. Metrics include reconstruction loss, MSE,
explained variance, L0, batch-local feature firing/dead/usage summaries,
train-vs-validation gaps, decoder-norm summaries, approximate decoder duplicate
cosine summaries, gradient norm, learning rate, tokens seen, elapsed seconds,
and grouped reconstruction metrics for token classes such as `kind:video`,
`kind:special`, `phase_kind:prefill:video`, and `role:user`. Set
`WANDB_API_KEY` and pass `--wandb-project`, or set `WANDB_PROJECT` in the
environment, to log the same metrics to W&B. W&B runs also receive live
histograms for batch-local feature fire rates/counts, token L0, active feature
magnitudes, decoder norms, approximate nearest decoder cosine, and validation
feature fire rates when validation is enabled. Use
`--no-wandb-diagnostic-histograms` to disable those histogram payloads.
When W&B is enabled, `train-sae` also runs a post-train top-activating-example
pass on `feature_labeling` by default and attaches the results to the same run
as a `top_activating_examples` table plus JSONL/HTML artifacts. Control it with
`--feature-report-splits`, `--feature-report-max-features`,
`--feature-report-top-n`, `--feature-report-rank`, or disable it with
`--no-wandb-top-feature-table`.

After training, run a full-dataset feature activity pass before selecting a
checkpoint:

```bash
python -m tools.sae_reasoner analyze-sae \
  --activation-dir "$ACTIVATION_URI" \
  --sae outputs/sae_reasoner/saes/l18.pt \
  --output outputs/sae_reasoner/reports/l18_feature_activity.json \
  --splits sae_train,sae_val \
  --batch-size 4096
```

This writes a summary JSON plus `<output>.features.jsonl` with per-feature
firing counts, firing rates, sign counts, and activation magnitudes across the
selected tokens. If W&B is enabled for `analyze-sae`, it logs full-dataset
histograms for feature fire rate, fire count, and mean absolute active
activation. Use these full-dataset dead/rare-feature metrics alongside W&B
training curves; batch-local `dead_feature_frac_batch` is only a dynamics signal.

For the BridgeData synthetic-caption prefill dataset, start with the compact
batch-size/LR sweep at 8x expansion rather than 16x:

```bash
python -m tools.sae_reasoner.scripts.plan_training_sweep \
  --activation-dir "$ACTIVATION_URI" \
  --output-root outputs/sae_reasoner/sweeps/bridgecaps_prefill \
  --stage lr_batch \
  --wandb-project "${WANDB_PROJECT:-cosmos-sae-reasoner}"
```

The generated script runs the 9 combinations of `batch_size={512,1024,2048}` and
`lr={1e-4,3e-4,1e-3}` with `expansion_factor=8`, `top_k=32`, ReLU+TopK,
Kaiming/parallel init, constant LR, and no gradient clipping. Each run is
followed by `analyze-sae`. After picking the best batch/LR, generate later
stages with `--stage capacity` for `expansion_factor={4,8,16}`, then
`--stage topk` for `top_k={16,32,64}`, and only then `--stage ablations`.

SAE training defaults to ReLU+TopK activations and random parallel
initialization: `W_enc` is Kaiming-initialized, `W_dec` starts as the transpose,
and decoder columns are normalized. Data-point blended initialization remains
available as an ablation with `--init-method data --init-blend 0.8`. Raw signed
TopK remains available as an ablation with `--topk-activation topk`. BatchTopK
is available as an opt-in experiment with `--topk-activation batch_topk`; do not
use it for the first baseline unless you want variable per-example sparsity at
inference via the learned threshold.

Matryoshka SAE training is also available as an opt-in experiment with
`--matryoshka-prefixes`. Pass comma-separated nested dictionary cutoffs as either
absolute feature counts or fractions of the full dictionary, for example
`--matryoshka-prefixes 0.03125,0.0625,0.125,0.25,0.5`. The full-dictionary
reconstruction loss remains the main `recon_loss`; each smaller prefix adds an
extra reconstruction loss scaled by `--matryoshka-loss-coeff`. Leave
`--matryoshka-prefixes` empty for the first baseline.

Training also defaults to `--activation-norm sqrt_d`, which scales raw residual
activations inside the SAE so their average L2 norm is `sqrt(hidden_dim)`. The
saved SAE stores that scale and unscales reconstruction deltas, so steering hooks
still edit the model's raw residual stream.

By default, manifests use separate splits for SAE optimization, reconstruction
validation, feature browsing, and final steering checks:
`sae_train=0.85,sae_val=0.10,feature_labeling=0.025,steering_eval=0.025`.
`train-sae` trains only on `sae_train` and computes held-out validation metrics
only on `sae_val`. `feature_labeling` and `steering_eval` are reserved for
downstream interpretation and steering demos, not for overfitting checks. Shards
without split metadata are treated as `sae_train` for backward compatibility.

## Manifest Format

The activation manifest is JSONL. Keep it neutral: no intended feature labels
or expected concepts are used during collection.

```json
{"id":"robot_caption","media_type":"image","media_path":"cookbooks/cosmos3/reasoner/assets/robot_153.jpg","prompt":"Caption the image in detail.","tags":["robotics"],"metadata":{"split":"sae_train"}}
```

Supported `media_type` values are `text`, `image`, and `video`. Video records
are decoded to RGB frames with OpenCV before they are passed to the Cosmos
processor, avoiding runtime-specific `torchvision`/`torchcodec` video IO. The
collector samples up to `COSMOS_SAE_VIDEO_FRAMES` frames per video, defaulting
to `16`.

Remote `media_path` values can use `hf://dataset/<namespace>/<repo>/<path>`,
`s3://bucket/key`, or HTTPS URLs. S3 credentials are read from the standard AWS
environment variables.

## Token Maps

Activation shards include a `token_map` in their saved metadata. Each token
entry stores the token index, token id, decoded token text, token phase
(`prefill` or `decode`), approximate chat role, and token kind: `text`,
`special`, `image`, or `video`. For image/video tokens, the collector also saves
processor grid metadata such as `image_grid_thw`/`video_grid_thw` and
approximate frame/patch coordinates when the active processor exposes them.

`metadata.jsonl` stays compact and stores only record-level metadata plus token
kind counts and grid summaries. `find-features` reads the full shard token map
and attaches the relevant token entry to each top activating feature example.

`find-neighbors` uses the same token maps to build a pre-SAE nearest-neighbor
browser over raw activation vectors. This is useful for checking whether local
activation neighborhoods already group similar visual/text tokens before SAE
training.

Feature steering can run either from a text-only `--prompt` or from a manifest
record via `--manifest` and `--record-id`. Manifest steering uses the same
Cosmos chat-template/media path as activation collection. `--steer-token-kinds`
and `--steer-roles` restrict prefill edits to specific token classes, which is
useful for media-token or special-token steering sweeps.

## Visualization

`render-feature-report` creates a standalone HTML browser for the
`find-features` JSONL output. The layout is intentionally close to the
Anthropic feature-browser workflow: feature list on the left, top activating
examples on the right, activation bars, token positions, prompts, tags, and
source media paths. When token maps are available, the report also shows token
kind, decoded text context, or approximate visual frame/patch coordinates. It
is static HTML, so it can be uploaded as a RunPod artifact or opened locally.
