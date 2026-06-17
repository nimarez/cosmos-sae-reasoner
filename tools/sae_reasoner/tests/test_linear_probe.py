import json
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.sae_reasoner.cli import build_parser
from tools.sae_reasoner.linear_probe import (
    LinearProbeConfig,
    _adaptive_cv,
    assemble_probe_matrix,
    assemble_sae_feature_matrix,
    build_label_map,
    fit_probe,
    select_top_k_features,
)
from tools.sae_reasoner.manifest import ManifestRecord
from tools.sae_reasoner.sae import SAEConfig, TopKSAE, save_sae


def _record(rid, *, media_type="text", tags=(), metadata=None, media_path=None):
    return ManifestRecord(
        id=rid,
        media_type=media_type,
        prompt="p",
        media_path=media_path,
        tags=tuple(tags),
        metadata=metadata,
    )


def test_build_label_map_requires_exactly_one_selector():
    records = [_record("a")]
    for kwargs in ({}, {"label_tag": "x", "label_key": "y"}):
        try:
            build_label_map(records, **kwargs)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"expected ValueError for {kwargs}")


def test_build_label_map_tag_is_binary_presence():
    records = [_record("a", tags=("physics",)), _record("b", tags=("spatial",))]
    assert build_label_map(records, label_tag="physics") == {"a": 1, "b": 0}


def test_build_label_map_metadata_key_drops_missing():
    records = [
        _record("a", metadata={"plausible": 1}),
        _record("b", metadata={"plausible": 0}),
        _record("c", metadata={"other": 1}),
        _record("d", metadata=None),
    ]
    assert build_label_map(records, label_key="plausible") == {"a": 1, "b": 0}


def test_build_label_map_field():
    records = [_record("a", media_type="text"), _record("b", media_type="image", media_path="s3://x/y.png")]
    assert build_label_map(records, label_field="media_type") == {"a": "text", "b": "image"}


def _dataset(rows):
    """rows: list of (record_id, token_index, vector)."""
    acts = torch.tensor([vec for _, _, vec in rows], dtype=torch.float32)
    coords = [{"record": rid, "index": idx} for rid, idx, _ in rows]
    return types.SimpleNamespace(activations=acts, token_coords=coords)


def test_assemble_matrix_last_uses_highest_token_index():
    data = _dataset([("a", 0, [1.0, 0.0]), ("a", 1, [9.0, 9.0]), ("b", 0, [2.0, 2.0])])
    cfg = LinearProbeConfig(aggregation="last")
    X, y, ids = assemble_probe_matrix(data, {"a": 1, "b": 0}, cfg)
    order = {rid: row for row, rid in enumerate(ids)}
    assert np.allclose(X[order["a"]], [9.0, 9.0])  # token index 1, not 0
    assert np.allclose(X[order["b"]], [2.0, 2.0])
    assert sorted(zip(ids, y)) == [("a", 1), ("b", 0)]


def test_assemble_matrix_mean_and_max():
    data = _dataset([("a", 0, [0.0, 4.0]), ("a", 1, [2.0, 0.0])])
    mean_X, _, _ = assemble_probe_matrix(data, {"a": 1}, LinearProbeConfig(aggregation="mean"))
    max_X, _, _ = assemble_probe_matrix(data, {"a": 1}, LinearProbeConfig(aggregation="max"))
    assert np.allclose(mean_X[0], [1.0, 2.0])
    assert np.allclose(max_X[0], [2.0, 4.0])


def test_assemble_matrix_skips_unlabeled_records():
    data = _dataset([("a", 0, [1.0, 1.0]), ("unlabeled", 0, [5.0, 5.0])])
    X, y, ids = assemble_probe_matrix(data, {"a": 1}, LinearProbeConfig())
    assert ids == ["a"]
    assert X.shape == (1, 2)


def _blob(center, n, rng):
    return rng.normal(loc=center, scale=0.3, size=(n, 4)).astype(np.float32)


def test_fit_probe_separable_scores_high_auc():
    rng = np.random.default_rng(0)
    X = np.concatenate([_blob([5, 0, 0, 0], 100, rng), _blob([0, 5, 0, 0], 100, rng)])
    y = [0] * 100 + [1] * 100
    result = fit_probe(X, y, LinearProbeConfig(seed=0))
    assert result.test_auc > 0.99
    assert result.n_classes == 2
    assert result.n_features == 4


def test_fit_probe_random_labels_near_chance():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 4)).astype(np.float32)
    y = ([0, 1] * 100)
    result = fit_probe(X, y, LinearProbeConfig(seed=0))
    assert 0.3 < result.test_auc < 0.7  # no real signal -> chance, not leaking


def test_fit_probe_raises_on_imbalanced_test_split():
    # 2 positives / 200 negatives: the stratified test split gets 0 positives -> AUC undefined.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(202, 4)).astype(np.float32)
    y = [1, 1] + [0] * 200
    with pytest.raises(ValueError, match="test split has no examples of class"):
        fit_probe(X, y, LinearProbeConfig(seed=0))


def test_fit_probe_max_train_boundary_does_not_crash():
    # total=1282 -> first split train=1025 == max_train(1024)+1, complement=1 < n_classes: must not crash.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1282, 4)).astype(np.float32)
    y = ([0, 1] * 641)
    result = fit_probe(X, y, LinearProbeConfig(seed=0, max_train=1024))
    assert result.n_train <= 1025  # cap skipped near the boundary rather than raising


def test_adaptive_cv_size_boundaries():
    cfg = LinearProbeConfig()
    assert _adaptive_cv(10, minority=5, config=cfg)[1].startswith("leave-two-out")
    assert _adaptive_cv(100, minority=50, config=cfg)[1] == "6-fold"
    assert _adaptive_cv(500, minority=250, config=cfg)[1] == "80/20 holdout"
    cv, scheme = _adaptive_cv(100, minority=1, config=cfg)
    assert cv is None and scheme.startswith("none")


def test_select_top_k_features_picks_discriminative_columns():
    # column 0 separates classes; columns 1-2 are noise/constant.
    X = np.array([[0.0, 1.0, 9.0], [0.0, 1.0, 9.0], [5.0, 1.0, 9.0], [5.0, 1.0, 9.0]], dtype=np.float32)
    y = np.array([0, 0, 1, 1])
    sel = select_top_k_features(X, y, k=1)
    assert list(sel) == [0]
    assert len(select_top_k_features(X, y, k=99)) == 3  # k capped at column count


def _identity_sae(input_dim=4, expansion_factor=2):
    sae = TopKSAE(SAEConfig(input_dim=input_dim, expansion_factor=expansion_factor, top_k=input_dim * expansion_factor, topk_activation="topk"))
    with torch.no_grad():
        sae.encoder.weight.zero_()
        sae.encoder.bias.zero_()
        sae.pre_bias.zero_()
        for i in range(input_dim):
            sae.encoder.weight[i, i] = 1.0  # feature i mirrors input dim i
    sae.eval()
    return sae


def test_assemble_sae_feature_matrix_encodes_then_aggregates():
    data = _dataset([("a", 0, [4.0, 0.0, 0.0, 0.0]), ("b", 0, [0.0, 4.0, 0.0, 0.0])])
    sae = _identity_sae()
    X, y, ids = assemble_sae_feature_matrix(data, sae, {"a": 1, "b": 0}, LinearProbeConfig(aggregation="last"))
    order = {rid: row for row, rid in enumerate(ids)}
    assert X.shape == (2, 8)  # feature_dim = 4 * 2
    assert X[order["a"], 0] == 4.0 and X[order["a"], 1] == 0.0  # feature 0 fires for record a
    assert X[order["b"], 1] == 4.0 and X[order["b"], 0] == 0.0  # feature 1 fires for record b


def test_fit_probe_select_top_k_records_selection():
    rng = np.random.default_rng(2)
    signal = np.concatenate([_blob([5, 0, 0, 0], 100, rng), _blob([0, 5, 0, 0], 100, rng)])
    noise = rng.normal(size=(200, 6)).astype(np.float32)
    X = np.concatenate([signal, noise], axis=1)  # 10 features; only first two are informative
    y = [0] * 100 + [1] * 100
    result = fit_probe(X, y, LinearProbeConfig(penalty="l1", seed=0), select_top_k=2)
    assert result.n_features == 2
    assert result.selected_feature_ids is not None and len(result.selected_feature_ids) == 2
    assert set(result.selected_feature_ids) <= {0, 1}  # picked the informative columns
    assert result.test_auc > 0.99


def _write_shard(path: Path, rid: str, vector: list[float], kind: str):
    torch.save(
        {
            "activations": torch.tensor([vector, vector], dtype=torch.float32),
            "meta": {
                "id": rid,
                "prompt": "p",
                "media_type": "text",
                "metadata": {"split": "sae_train"},
                "token_map": [
                    {"index": 0, "kind": kind, "phase": "prefill", "role": "user"},
                    {"index": 1, "kind": kind, "phase": "prefill", "role": "user"},
                ],
            },
        },
        path,
    )


def test_cli_train_linear_probe_media_type_smoke(tmp_path: Path):
    act_dir = tmp_path / "acts"
    act_dir.mkdir()
    lines = []
    for i in range(10):
        _write_shard(act_dir / f"{i:06d}_text.pt", f"text-{i}", [4.0, 0.0, 0.0, 0.0], "text")
        lines.append(json.dumps({"id": f"text-{i}", "media_type": "text", "prompt": "p", "tags": []}))
    for i in range(10):
        rid = f"img-{i}"
        _write_shard(act_dir / f"{i:06d}_img.pt", rid, [0.0, 4.0, 0.0, 0.0], "image")
        lines.append(json.dumps({"id": rid, "media_type": "image", "prompt": "p", "tags": [], "media_path": f"s3://b/{rid}.png"}))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out = tmp_path / "probe.json"

    parser = build_parser()
    args = parser.parse_args(
        [
            "train-linear-probe",
            "--activation-dir", str(act_dir),
            "--manifest", str(manifest),
            "--label-field", "media_type",
            "--output", str(out),
        ]
    )
    assert args.func(args) == 0

    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary["label_source"] == "field:media_type"
    assert summary["n_probed_records"] == 20
    assert summary["result"]["test_auc"] > 0.99  # activations perfectly separate the two classes


def test_cli_compare_sae_probe_smoke(tmp_path: Path):
    act_dir = tmp_path / "acts"
    act_dir.mkdir()
    lines = []
    for i in range(10):
        _write_shard(act_dir / f"{i:06d}_text.pt", f"text-{i}", [4.0, 0.0, 0.0, 0.0], "text")
        lines.append(json.dumps({"id": f"text-{i}", "media_type": "text", "prompt": "p", "tags": []}))
    for i in range(10):
        rid = f"img-{i}"
        _write_shard(act_dir / f"{i:06d}_img.pt", rid, [0.0, 4.0, 0.0, 0.0], "image")
        lines.append(json.dumps({"id": rid, "media_type": "image", "prompt": "p", "tags": [], "media_path": f"s3://b/{rid}.png"}))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    sae_path = tmp_path / "sae.pt"
    save_sae(str(sae_path), _identity_sae())
    out = tmp_path / "compare.json"

    parser = build_parser()
    args = parser.parse_args(
        [
            "compare-sae-probe",
            "--activation-dir", str(act_dir),
            "--manifest", str(manifest),
            "--sae", str(sae_path),
            "--label-field", "media_type",
            "--k-values", "2,4",
            "--output", str(out),
        ]
    )
    assert args.func(args) == 0

    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary["feature_dim"] == 8
    assert summary["baseline"]["test_auc"] > 0.99
    assert [r["k"] for r in summary["sae_probes"]] == [2, 4]
    for r in summary["sae_probes"]:
        assert r["result"]["test_auc"] > 0.99
        assert len(r["result"]["selected_feature_ids"]) == r["k"]
    assert "sae_beats_baseline" in summary
