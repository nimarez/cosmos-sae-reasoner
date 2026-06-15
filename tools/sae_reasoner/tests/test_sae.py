import torch

from tools.sae_reasoner.sae import SAEConfig, TopKSAE, reconstruction_loss, train_sae_from_tensor


def test_topk_sae_shapes_and_sparsity():
    sae = TopKSAE(SAEConfig(input_dim=8, expansion_factor=4, top_k=3))
    x = torch.randn(5, 8)
    recon, features = sae(x)
    assert recon.shape == x.shape
    assert features.shape == (5, 32)
    assert torch.all((features > 0).sum(dim=-1) <= 3)


def test_feature_delta_changes_input_shape():
    sae = TopKSAE(SAEConfig(input_dim=8, expansion_factor=2, top_k=2))
    x = torch.randn(4, 8)
    delta = sae.feature_delta(x, feature_id=0, multiplier=5.0)
    assert delta.shape == x.shape


def test_train_sae_from_tensor_smoke():
    acts = torch.randn(64, 8)
    sae, metrics = train_sae_from_tensor(acts, expansion_factor=2, top_k=2, steps=3, batch_size=16, device="cpu")
    assert sae.config.input_dim == 8
    assert metrics[-1]["step"] == 3.0
    assert "recon_loss" in metrics[-1]


def test_train_sae_supports_l1_reconstruction_loss():
    acts = torch.randn(64, 8)
    sae, metrics = train_sae_from_tensor(
        acts,
        expansion_factor=2,
        top_k=2,
        recon_loss="l1",
        feature_l1_coeff=0.01,
        steps=3,
        batch_size=16,
        device="cpu",
    )
    assert sae.config.input_dim == 8
    assert metrics[-1]["loss"] >= metrics[-1]["recon_loss"]


def test_reconstruction_loss_variants():
    x = torch.tensor([[1.0, 3.0]])
    y = torch.tensor([[2.0, 1.0]])
    assert torch.isclose(reconstruction_loss(x, y, "mse"), torch.tensor(2.5))
    assert torch.isclose(reconstruction_loss(x, y, "l1"), torch.tensor(1.5))
    assert reconstruction_loss(x, y, "smooth_l1") > 0
