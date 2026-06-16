import torch

from tools.sae_reasoner.sae import (
    SAEConfig,
    TopKSAE,
    activation_l2_scale,
    load_sae,
    lr_for_step,
    matryoshka_reconstruction_loss,
    reconstruction_loss,
    resolve_matryoshka_prefixes,
    save_sae,
    train_sae_from_tensor,
)


def test_topk_sae_shapes_and_sparsity():
    sae = TopKSAE(SAEConfig(input_dim=8, expansion_factor=4, top_k=3))
    x = torch.randn(5, 8)
    recon, features = sae(x)
    assert recon.shape == x.shape
    assert features.shape == (5, 32)
    assert torch.all((features != 0).sum(dim=-1) <= 3)


def test_batch_topk_sae_uses_batch_budget_and_eval_threshold():
    sae = TopKSAE(SAEConfig(input_dim=8, expansion_factor=4, top_k=3, topk_activation="batch_topk"))
    sae.train()
    x = torch.randn(5, 8)
    _recon, features = sae(x)
    assert int((features != 0).sum().item()) <= 15
    assert float(sae.batch_topk_threshold.item()) != 0.0
    sae.eval()
    _recon_eval, eval_features = sae(x)
    assert eval_features.shape == features.shape


def test_topk_sae_kaiming_initialization_sets_decoder_transpose():
    sae = TopKSAE(SAEConfig(input_dim=8, expansion_factor=2, top_k=3, init_method="kaiming"))

    expected_decoder = sae.encoder.weight.T
    expected_decoder = expected_decoder / expected_decoder.norm(dim=0, keepdim=True).clamp_min(1e-6)
    assert torch.allclose(sae.decoder.weight, expected_decoder, atol=1e-5)
    assert torch.allclose(sae.decoder.weight.norm(dim=0), torch.ones(16), atol=1e-5)


def test_topk_sae_data_initialization_sets_decoder_transpose():
    sae = TopKSAE(SAEConfig(input_dim=8, expansion_factor=2, top_k=3))
    data = torch.randn(32, 8)

    sae.initialize_from_data(data, blend=0.8)

    assert sae.encoder.weight.shape == (16, 8)
    assert sae.decoder.weight.shape == (8, 16)
    expected_decoder = sae.encoder.weight.T
    expected_decoder = expected_decoder / expected_decoder.norm(dim=0, keepdim=True).clamp_min(1e-6)
    assert torch.allclose(sae.decoder.weight, expected_decoder, atol=1e-5)
    assert torch.allclose(sae.decoder.weight.norm(dim=0), torch.ones(16), atol=1e-5)
    assert not torch.allclose(sae.encoder.bias, torch.zeros_like(sae.encoder.bias))


def test_training_uses_plain_adam_without_weight_decay(monkeypatch):
    created = []
    original_adam = torch.optim.Adam

    class TrackingAdam(original_adam):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    def forbidden_adamw(*_args, **_kwargs):
        raise AssertionError("train_sae_from_tensor should use Adam, not AdamW")

    monkeypatch.setattr(torch.optim, "Adam", TrackingAdam)
    monkeypatch.setattr(torch.optim, "AdamW", forbidden_adamw)

    train_sae_from_tensor(torch.randn(16, 8), expansion_factor=2, top_k=2, steps=1, batch_size=4, device="cpu")

    assert created
    assert all(group["weight_decay"] == 0 for group in created[0].param_groups)


def test_load_sae_tolerates_missing_batch_topk_threshold(tmp_path):
    path = tmp_path / "sae.pt"
    sae = TopKSAE(SAEConfig(input_dim=8, expansion_factor=2, top_k=3))
    save_sae(str(path), sae)
    payload = torch.load(path)
    payload["state_dict"].pop("batch_topk_threshold")
    torch.save(payload, path)

    loaded = load_sae(str(path))

    assert loaded.config.input_dim == 8


def test_load_legacy_sae_without_topk_activation_uses_relu_topk(tmp_path):
    path = tmp_path / "legacy.pt"
    sae = TopKSAE(SAEConfig(input_dim=8, expansion_factor=2, top_k=3, topk_activation="relu_topk"))
    save_sae(str(path), sae)
    payload = torch.load(path)
    payload["config"].pop("topk_activation")
    payload["state_dict"].pop("batch_topk_threshold")
    torch.save(payload, path)

    loaded = load_sae(str(path))

    assert loaded.config.topk_activation == "relu_topk"


def test_feature_delta_changes_input_shape():
    sae = TopKSAE(SAEConfig(input_dim=8, expansion_factor=2, top_k=2, input_scale=2.0))
    x = torch.randn(4, 8)
    delta = sae.feature_delta(x, feature_id=0, multiplier=5.0)
    assert delta.shape == x.shape


def test_feature_delta_uses_decoder_vector_and_unscales():
    sae = TopKSAE(SAEConfig(input_dim=4, expansion_factor=2, top_k=8, input_scale=2.0))
    x = torch.randn(3, 4)
    with torch.no_grad():
        features = sae.encode(x)
        old = features[:, 1].clone()
        expected = old.unsqueeze(-1) * (3.0 - 1.0) * sae.decoder.weight[:, 1] / sae.config.input_scale

    delta = sae.feature_delta(x, feature_id=1, multiplier=3.0)

    assert torch.allclose(delta, expected, atol=1e-6)


def test_activation_l2_scale_targets_sqrt_hidden_dim():
    acts = torch.randn(128, 8) * 5
    scale = activation_l2_scale(acts, mode="sqrt_d")
    scaled_mean_norm = (acts * scale).norm(dim=-1).mean()
    assert torch.isclose(scaled_mean_norm, torch.tensor(8**0.5), atol=1e-5)


def test_train_sae_from_tensor_smoke():
    acts = torch.randn(64, 8)
    val = torch.randn(16, 8)
    seen = []
    sae, metrics = train_sae_from_tensor(
        acts,
        validation_activations=val,
        token_groups=[("all", "kind:text", "phase_kind:prefill:text")] * 64,
        validation_token_groups=[("all", "kind:video", "phase_kind:prefill:video")] * 16,
        expansion_factor=2,
        top_k=2,
        init_method="data",
        topk_activation="topk",
        steps=3,
        batch_size=16,
        warmup_steps=2,
        lr_schedule="cosine",
        max_grad_norm=1.0,
        device="cpu",
        log_every=1,
        progress_callback=seen.append,
    )
    assert sae.config.input_dim == 8
    assert metrics[-1]["step"] == 3.0
    assert "recon_loss" in metrics[-1]
    assert "normalized_mse" in metrics[-1]
    assert "explained_variance" in metrics[-1]
    assert "dead_feature_frac_batch" in metrics[-1]
    assert "grad_norm" in metrics[-1]
    assert "grad_clipped" in metrics[-1]
    assert "val_recon_loss" in metrics[-1]
    assert "train_mse_by_group/kind_text" in metrics[-1]
    assert "val_mse_by_group/kind_video" in metrics[-1]
    assert metrics[-1]["val_tokens"] == 16.0
    assert sae.config.input_scale != 1.0
    assert [row["step"] for row in seen] == [1.0, 2.0, 3.0]
    assert metrics[0]["lr"] < 3e-4


def test_train_sae_accepts_bfloat16_activations_without_persistent_fp32_input():
    acts = torch.randn(32, 8).bfloat16()

    sae, metrics = train_sae_from_tensor(acts, expansion_factor=2, top_k=2, steps=2, batch_size=8, device="cpu")

    assert sae.config.input_dim == 8
    assert metrics[-1]["step"] == 2.0


def test_resolve_matryoshka_prefixes_accepts_counts_and_fractions():
    assert resolve_matryoshka_prefixes("0.25,8,1.0", feature_dim=16) == (4, 8, 16)


def test_matryoshka_reconstruction_loss_uses_decoder_prefixes():
    sae = TopKSAE(SAEConfig(input_dim=2, expansion_factor=2, top_k=4))
    with torch.no_grad():
        sae.decoder.weight.zero_()
        sae.decoder.weight[:, 0] = torch.tensor([1.0, 0.0])
        sae.decoder.weight[:, 1] = torch.tensor([0.0, 1.0])
        sae.decoder.weight[:, 2] = torch.tensor([1.0, 1.0])
        sae.post_bias.zero_()
    features = torch.tensor([[2.0, 3.0, 5.0, 7.0]])
    target = torch.tensor([[2.0, 3.0]])

    loss, by_prefix = matryoshka_reconstruction_loss(sae, features, target, prefixes=(1, 2, 4), recon_loss="mse")

    assert torch.isclose(by_prefix[1], torch.tensor(4.5))
    assert torch.isclose(by_prefix[2], torch.tensor(0.0))
    assert 4 not in by_prefix
    assert torch.isclose(loss, torch.tensor(4.5))


def test_train_sae_supports_opt_in_matryoshka_loss():
    acts = torch.randn(64, 8)

    sae, metrics = train_sae_from_tensor(
        acts,
        expansion_factor=2,
        top_k=2,
        matryoshka_prefixes=(4, 8),
        matryoshka_loss_coeff=0.5,
        steps=2,
        batch_size=16,
        device="cpu",
        log_every=1,
    )

    assert sae.config.matryoshka_prefixes == (4, 8)
    assert metrics[-1]["matryoshka_num_prefixes"] == 2.0
    assert metrics[-1]["matryoshka_loss_coeff"] == 0.5
    assert "matryoshka_recon_loss_prefix/4" in metrics[-1]
    assert metrics[-1]["loss"] >= metrics[-1]["recon_loss"]


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


def test_lr_for_step_supports_warmup_and_cosine_decay():
    assert lr_for_step(1.0, step=1, total_steps=10, warmup_steps=2, schedule="constant") == 0.5
    assert lr_for_step(1.0, step=3, total_steps=10, warmup_steps=2, schedule="constant") == 1.0
    assert lr_for_step(1.0, step=10, total_steps=10, warmup_steps=2, schedule="cosine") == 0.0
