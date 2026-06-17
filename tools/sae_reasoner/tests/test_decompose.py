import numpy as np
import pytest
import torch

from tools.sae_reasoner.cli import build_parser, collect_top_feature_records
from tools.sae_reasoner.decompose import (
    DecompositionConfig,
    LinearBasis,
    fit_pca,
    load_basis,
    save_basis,
)


def _anisotropic(n=4000, seed=0):
    """Gaussian cloud whose variance is concentrated on the first axis, then second, etc."""
    g = torch.Generator().manual_seed(seed)
    scales = torch.tensor([10.0, 3.0, 1.0, 0.3])
    return torch.randn(n, 4, generator=g) * scales + torch.tensor([5.0, -2.0, 0.0, 1.0])


def test_fit_pca_orders_components_by_variance():
    acts = _anisotropic()
    basis, stats = fit_pca(acts, n_components=4)
    ratios = stats["explained_variance_ratio"]
    # Strictly decreasing explained variance, summing to ~1 over the full rank.
    assert ratios == sorted(ratios, reverse=True)
    assert stats["cumulative_variance_ratio"] == pytest.approx(1.0, abs=1e-4)
    # Top component aligns with the planted high-variance (first) axis.
    top = basis.components[0].abs()
    assert int(torch.argmax(top)) == 0
    assert top[0] > 0.95


def test_fit_pca_matches_numpy_svd():
    acts = _anisotropic(n=500)
    basis, _ = fit_pca(acts, n_components=4, batch_size=64)  # batched covariance path
    X = acts.numpy().astype(np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    for i in range(4):
        cos = abs(float(np.dot(basis.components[i].numpy(), vt[i])))
        assert cos == pytest.approx(1.0, abs=1e-4)  # same direction up to sign


def test_linear_basis_encode_shape_and_projection():
    acts = _anisotropic(n=100)
    basis, _ = fit_pca(acts, n_components=3)
    proj = basis.encode(acts.float())
    assert proj.shape == (100, 3)
    # Centered projection has ~zero mean per component.
    assert torch.allclose(proj.mean(dim=0), torch.zeros(3), atol=1e-3)


def test_save_load_basis_round_trip(tmp_path):
    acts = _anisotropic(n=200)
    basis, stats = fit_pca(acts, n_components=4, whiten=True)
    path = tmp_path / "pca.pt"
    save_basis(str(path), basis, stats=stats)
    loaded = load_basis(str(path))
    assert loaded.config.method == "pca"
    assert loaded.config.feature_dim == 4
    assert loaded.config.whiten is True
    assert torch.allclose(loaded.components, basis.components)
    assert torch.allclose(loaded.encode(acts.float()), basis.encode(acts.float()), atol=1e-5)


def test_basis_is_drop_in_for_feature_browser(tmp_path):
    """A LinearBasis stands in for the SAE in collect_top_feature_records (find-components path)."""
    shard = tmp_path / "000000_demo.pt"
    torch.save(
        {
            "activations": torch.tensor([[1.0, 0.0], [3.0, 2.0], [2.0, 1.0]]),
            "meta": {
                "id": "rec-1",
                "prompt": "demo",
                "media_type": "text",
                "media_path": None,
                "tags": ["demo"],
                "metadata": {"split": "sae_train"},
                "token_map": [
                    {"kind": "text", "phase": "prefill", "role": "user"},
                    {"kind": "text", "phase": "prefill", "role": "user"},
                    {"kind": "text", "phase": "decode", "role": "assistant"},
                ],
            },
        },
        shard,
    )
    # Identity-ish basis: two axis-aligned components, zero mean.
    config = DecompositionConfig(method="pca", input_dim=2, n_components=2)
    basis = LinearBasis(config, torch.eye(2), torch.zeros(2))
    records = collect_top_feature_records(
        activation_dir=tmp_path,
        sae=basis,
        feature_ids=[0, 1],
        top_n=2,
        feature_rank="absolute",
        token_kinds=set(),
        phases=set(),
        splits=set(),
    )
    assert len(records) == 4
    # find-features-compatible schema so render-feature-report works unchanged.
    for row in records:
        assert set(row) >= {"feature_id", "activation", "token_index", "token_info", "record_id"}
        assert row["record_id"] == "rec-1"


def test_decompose_and_find_components_parsers():
    parser = build_parser()
    a = parser.parse_args(
        ["decompose-activations", "--activation-dir", "d", "--output", "b.pt", "--method", "ica", "--n-components", "32"]
    )
    assert a.method == "ica" and a.n_components == 32
    b = parser.parse_args(["find-components", "--activation-dir", "d", "--basis", "b.pt", "--output", "o.jsonl"])
    assert b.feature_rank == "absolute"
    c = parser.parse_args(
        ["compare-sae-probe", "--activation-dir", "d", "--manifest", "m", "--label-tag", "physics", "--sae", "s.pt", "--decomp-basis", "b.pt", "--output", "o.json"]
    )
    assert str(c.decomp_basis) == "b.pt"
