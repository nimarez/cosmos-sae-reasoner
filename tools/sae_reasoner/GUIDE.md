# SAE Reasoner — Practical Guides

Two end-to-end walkthroughs:

1. [Running the linear-probe baselines](#guide-1--running-the-linear-probe-baselines) — measure whether a
   concept is linearly decodable, and whether SAE features beat a raw-activation probe.
2. [Discovering and naming features](#guide-2--discovering-and-naming-features) — triage an SAE's latents,
   inspect what they fire on, name them, and validate the names.

Both assume you have already collected activations and (for SAE steps) trained an SAE. See the
[README](README.md) for `build-corpus-manifest` → `collect-activations` → `train-sae`. Throughout,
these shell variables are used:

```bash
export ACT="$ACTIVATION_URI"                 # collect-activations output dir or s3:// prefix
export MANIFEST="$MANIFEST_URI"              # the SAME manifest used for collection (carries tags/metadata)
export SAE=outputs/sae_reasoner/saes/l18.pt # a trained SAE checkpoint
export OUT=outputs/sae_reasoner             # where reports/probes land
```

The manifest matters for probing: labels come from each record's `tags` / `metadata`, so point
`--manifest` at the manifest you collected activations from (record ids must match the shards).

---

## Guide 1 — Running the linear-probe baselines

**Why.** Per Kantamneni et al., *"Are Sparse Autoencoders Useful?"* (arXiv:2502.16681), SAE features
rarely beat a plain logistic-regression probe on the raw residual stream. So before believing a
feature "represents concept X," confirm a cheap linear probe doesn't already decode X at least as
well. `train-linear-probe` runs that baseline; `compare-sae-probe` runs the head-to-head.

### Step 1 — Choose a label source

A probe needs a labeled task. Pick exactly one of:

| Flag | Meaning | When to use |
|------|---------|-------------|
| `--label-field media_type` | label = a record field (`media_type`, …) | smoke test (see Step 2) |
| `--label-tag physics` | binary: 1 if the tag is in `record.tags`, else 0 | concept probes off existing tags, **zero relabeling** |
| `--label-key plausible` | label = `record.metadata["plausible"]` (binary or categorical) | explicit labels you added to the manifest |

### Step 2 — Smoke test first (don't skip)

Probe `media_type`. A probe *should* score AUC ≈ 1.0 — the three modalities are trivially separable.
If it doesn't, your activation↔label join is broken; fix that before trusting any real result.

```bash
python -m tools.sae_reasoner train-linear-probe \
  --activation-dir "$ACT" --manifest "$MANIFEST" \
  --label-field media_type --aggregation last \
  --output "$OUT/probes/media_type.json"
```

### Step 3 — Run a real concept probe

```bash
python -m tools.sae_reasoner train-linear-probe \
  --activation-dir "$ACT" --manifest "$MANIFEST" \
  --label-tag physics --aggregation mean \
  --output "$OUT/probes/physics.json"
```

`--aggregation` reduces a record's tokens to one vector: `last` (paper default), `mean`
(good for whole-record concept tags), or `max`. Restrict which tokens are loaded with
`--token-kinds` (e.g. `image,video`), `--phases` (`prefill`/`decode`), `--splits`, and `--stream`
(a load spanning >1 residual stream is rejected — pick one, e.g. `--stream ar`).

### Step 4 — Read the result JSON

```jsonc
{
  "label_source": "tag:physics",
  "n_labeled_records": 2000, "n_probed_records": 2000,
  "result": {
    "test_auc": 0.78,          // headline: held-out AUC
    "best_c": 1.0,             // chosen inverse-regularization
    "n_features": 4096,        // hidden_dim probed
    "class_counts": {"0": 1700, "1": 300},   // overall balance
    "test_class_counts": {"0": 340, "1": 60},
    "cv_scheme": "80/20 holdout"
  }
}
```

### Step 5 — Compare SAE features vs the baseline

This is the rigorous test. It trains the raw baseline **and** an SAE-feature probe (top-k latents
selected by mean-absolute class difference, L1) on the *same records and split*, per k:

```bash
python -m tools.sae_reasoner compare-sae-probe \
  --activation-dir "$ACT" --manifest "$MANIFEST" --sae "$SAE" \
  --label-tag physics --aggregation mean --k-values 16,128 \
  --output "$OUT/probes/physics_compare.json"
```

Read the top level: `baseline.test_auc`, each `sae_probes[].{k, auc_delta, sae_wins,
result.selected_feature_ids}`, and `sae_beats_baseline`. **`sae_beats_baseline: true` is the bar a
"useful feature" claim must clear** — and per the paper it usually won't.

### Sanity checks & troubleshooting

- **Random-label control.** A meaningless task should land near AUC 0.5. If a *real* probe also
  sits at ~0.5, the concept isn't linearly decodable here (or the labels are noise).
- **`ValueError: test split has no examples of class(es) [...]`.** The task is too imbalanced for
  the test split (common with rare tags). Add more rare-class examples, or raise `--test-frac`.
  (This is deliberately a loud error — the alternative would be a silent `nan` AUC.)
- **Tiny corpora.** The sample manifest's tags are small; for a trustworthy AUC point `--manifest`
  / `--activation-dir` at a larger labeled collection.

---

## Guide 2 — Discovering and naming features

The loop: **triage → inspect → name → validate.** A name read off max-activating examples is a
hypothesis; Step 4 is what turns it into evidence.

### Step 1 — Triage with full-dataset firing stats

```bash
python -m tools.sae_reasoner analyze-sae \
  --activation-dir "$ACT" --sae "$SAE" \
  --splits sae_train,sae_val \
  --output "$OUT/reports/l18_activity.json"
```

Writes a summary JSON + `<output>.features.jsonl` with per-feature fire counts, fire rates, sign
counts, and activation magnitudes. Use it to drop **dead** features (never fire) and **ultra-rare /
ultra-frequent** ones (too few examples to interpret, or vague), and to pick interesting feature ids.

### Step 2 — Inspect what features fire on (max-activating examples)

```bash
python -m tools.sae_reasoner find-features \
  --activation-dir "$ACT" --sae "$SAE" \
  --feature-ids 12,87,1043 --top-n 20 \
  --feature-rank absolute \
  --output "$OUT/reports/l18_top_features.jsonl"
```

Leave `--feature-ids` empty to rank across all features. `--feature-rank positive` ranks by signed
activation (vs `absolute`). `--token-kinds` / `--phases` scope which tokens count. Each row carries
the activation, the token, and the source record/prompt/media — the raw material for a label.

### Step 3 — Browse and name (HTML report)

```bash
python -m tools.sae_reasoner render-feature-report \
  --features "$OUT/reports/l18_top_features.jsonl" \
  --output "$OUT/reports/features.html" \
  --title "Layer 18 SAE features"
```

Read the top examples per feature and write a one-line name. Sample across activation strengths, not
just the very top — a monosemantic feature stays coherent across its range; a polysemantic one
fractures (flag it rather than forcing a name).

### Step 4 — Validate a name (the rigor step)

A label is only trustworthy when input evidence, a causal check, and a baseline comparison agree:

- **Causal (steering).** Clamp the feature and see if generation shifts toward the named concept:
  ```bash
  python -m tools.sae_reasoner steer \
    --sae "$SAE" --layer 18 --feature-id 1043 --multiplier 8 \
    --prompt "Describe what is happening." \
    --output "$OUT/reports/steer_1043.json"
  # multimodal: --manifest "$MANIFEST" --record-id <id> --scope prefill \
  #             --steer-token-kinds video --steer-roles user
  ```
  A feature you can *steer with* is far better evidenced than one you only *read off*.
- **Beats a baseline?** Turn the name into a labeled task (`--label-tag` / `--label-key`) and run
  `compare-sae-probe` (Guide 1, Step 5). If a raw-activation probe matches the SAE feature, the
  feature isn't adding decodable signal — temper the claim.
- **Random-init control.** SAEs trained on a *randomly initialized* model also yield
  "interpretable-looking" features (Heap et al. 2025). If a finding must be airtight, confirm it
  doesn't reproduce on a random-weights control.

### Step 5 — Optional: nearest-neighbor context

To understand the activation geometry around tokens (independent of the SAE):

```bash
python -m tools.sae_reasoner find-neighbors \
  --activation-dir "$ACT" --num-queries 40 --neighbors 8 \
  --query-kinds image,video --output "$OUT/reports/neighbors.json"
python -m tools.sae_reasoner render-neighbor-report \
  --neighbors "$OUT/reports/neighbors.json" \
  --output "$OUT/reports/neighbors.html"
```

---

### Rigor checklist for a feature claim

A feature is "named with confidence" only when:

- [ ] Input evidence: coherent max-activating examples across the activation range (`find-features`).
- [ ] Causal: steering toward the concept works, ablation degrades it (`steer`).
- [ ] Not trivially redundant: a raw-activation probe does **not** match it (`compare-sae-probe`).
- [ ] (For airtight claims) does not reproduce under a random-init control.
