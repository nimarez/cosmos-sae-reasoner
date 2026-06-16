import torch

from tools.sae_reasoner.sae import (
    SAEConfig,
    ShuffledEpochSampler,
    TopKSAE,
    activation_l2_scale,
    checkpoint_path,
    find_latest_checkpoint,
    load_sae,
    load_training_checkpoint,
    lr_for_step,
    matryoshka_reconstruction_loss,
    reconstruction_loss,
    resolve_matryoshka_prefixes,
    save_sae,
    train_sae_from_tensor,
)


def _convert_payload_to_legacy_post_bias_format(payload):
    # Legacy checkpoints used parameter order:
    # pre_bias, post_bias, encoder.weight, encoder.bias, decoder.weight.
    payload["state_dict"]["post_bias"] = payload["state_dict"]["pre_bias"].clone()
    optimizer_state_dict = payload.get("optimizer_state_dict")
    if optimizer_state_dict is not None:
        remapped_state = {}
        for param_id, state in optimizer_state_dict.get("state", {}).items():
            param_id = int(param_id)
            remapped_state[param_id if param_id == 0 else param_id + 1] = state
        remapped_state[1] = {}
        optimizer_state_dict["state"] = remapped_state
        for group in optimizer_state_dict.get("param_groups", []):
            shifted = [int(param_id) if int(param_id) == 0 else int(param_id) + 1 for param_id in group.get("params", [])]
            group["params"] = [shifted[0], 1, *shifted[1:]] if shifted else [1]
    return payload


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


def test_load_legacy_sae_with_post_bias_still_works(tmp_path):
    path = tmp_path / "legacy_post_bias.pt"
    sae = TopKSAE(SAEConfig(input_dim=8, expansion_factor=2, top_k=3))
    save_sae(str(path), sae)
    payload = torch.load(path)
    payload = _convert_payload_to_legacy_post_bias_format(payload)
    torch.save(payload, path)

    loaded = load_sae(str(path))

    assert torch.allclose(loaded.pre_bias, sae.pre_bias)


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
        log_diagnostic_histograms=True,
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
    assert "train_val_gap/recon_loss" in metrics[-1]
    assert "feature_live_count_batch" in metrics[-1]
    assert "feature_used_ge_1pct_count_batch" in metrics[-1]
    assert "decoder_norm_mean" in metrics[-1]
    assert "decoder_nearest_abs_cosine_sample_max" in metrics[-1]
    assert "train_mse_by_group/kind_text" in metrics[-1]
    assert "val_mse_by_group/kind_video" in metrics[-1]
    assert metrics[-1]["val_tokens"] == 16.0
    assert sae.config.input_scale != 1.0
    assert [row["step"] for row in seen] == [1.0, 2.0, 3.0]
    assert "_wandb_histograms" not in metrics[-1]
    assert "hist/feature_fire_rate" in seen[-1]["_wandb_histograms"]
    assert "hist/val_feature_fire_rate" in seen[-1]["_wandb_histograms"]
    assert "hist/decoder_norm" in seen[-1]["_wandb_histograms"]
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
        sae.pre_bias.zero_()
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


def _train_kwargs(**overrides):
    base = dict(expansion_factor=2, top_k=2, batch_size=16, device="cpu", log_every=10)
    base.update(overrides)
    return base


def test_train_sae_writes_periodic_checkpoints(tmp_path):
    acts = torch.randn(64, 8)
    ckpt_dir = tmp_path / "ckpts"
    train_sae_from_tensor(acts, steps=4, checkpoint_dir=str(ckpt_dir), checkpoint_every=2, **_train_kwargs())

    files = sorted(p.name for p in ckpt_dir.iterdir())
    assert files == ["checkpoint-000002.pt", "checkpoint-000004.pt"]
    assert find_latest_checkpoint(str(ckpt_dir)) == str(ckpt_dir / "checkpoint-000004.pt")

    loaded = load_training_checkpoint(checkpoint_path(str(ckpt_dir), 2))
    assert loaded["step"] == 2
    assert loaded["optimizer_state_dict"] is not None
    assert isinstance(loaded["sae"], TopKSAE)


def test_train_sae_resume_matches_uninterrupted_run(tmp_path):
    acts = torch.randn(64, 8)

    torch.manual_seed(0)
    full_sae, _ = train_sae_from_tensor(acts, steps=4, **_train_kwargs())

    ckpt_dir = tmp_path / "ckpts"
    torch.manual_seed(0)
    train_sae_from_tensor(acts, steps=2, checkpoint_dir=str(ckpt_dir), checkpoint_every=2, **_train_kwargs())
    resumed_sae, _ = train_sae_from_tensor(acts, steps=4, resume_from=str(ckpt_dir), **_train_kwargs())

    assert torch.allclose(full_sae.encoder.weight, resumed_sae.encoder.weight, atol=1e-6)
    assert torch.allclose(full_sae.decoder.weight, resumed_sae.decoder.weight, atol=1e-6)
    assert torch.allclose(full_sae.pre_bias, resumed_sae.pre_bias, atol=1e-6)


def test_train_sae_can_resume_legacy_post_bias_checkpoint(tmp_path):
    acts = torch.randn(64, 8)

    torch.manual_seed(0)
    full_sae, _ = train_sae_from_tensor(acts, steps=4, **_train_kwargs())

    ckpt_dir = tmp_path / "ckpts"
    torch.manual_seed(0)
    train_sae_from_tensor(acts, steps=2, checkpoint_dir=str(ckpt_dir), checkpoint_every=2, **_train_kwargs())
    ckpt_path = checkpoint_path(str(ckpt_dir), 2)
    payload = torch.load(ckpt_path)
    payload = _convert_payload_to_legacy_post_bias_format(payload)
    torch.save(payload, ckpt_path)

    resumed_sae, _ = train_sae_from_tensor(acts, steps=4, resume_from=ckpt_path, **_train_kwargs())

    assert torch.allclose(full_sae.encoder.weight, resumed_sae.encoder.weight, atol=1e-6)
    assert torch.allclose(full_sae.decoder.weight, resumed_sae.decoder.weight, atol=1e-6)
    assert torch.allclose(full_sae.pre_bias, resumed_sae.pre_bias, atol=1e-6)


def test_resume_rejects_mismatched_config(tmp_path):
    acts = torch.randn(64, 8)
    ckpt_dir = tmp_path / "ckpts"
    train_sae_from_tensor(acts, steps=2, checkpoint_dir=str(ckpt_dir), checkpoint_every=2, **_train_kwargs())

    try:
        train_sae_from_tensor(acts, steps=4, resume_from=str(ckpt_dir), **_train_kwargs(top_k=4))
    except ValueError as exc:
        assert "config" in str(exc)
    else:
        raise AssertionError("expected ValueError on mismatched resume config")


def test_resume_rejects_non_training_checkpoint(tmp_path):
    # A save_sae model file ends in .pt but is not a resumable training checkpoint.
    model_path = tmp_path / "model.pt"
    sae = TopKSAE(SAEConfig(input_dim=8, expansion_factor=2, top_k=2))
    save_sae(str(model_path), sae)

    acts = torch.randn(64, 8)
    try:
        train_sae_from_tensor(acts, steps=4, resume_from=str(model_path), **_train_kwargs())
    except ValueError as exc:
        assert "training checkpoint" in str(exc)
    else:
        raise AssertionError("expected ValueError resuming from a save_sae model file")


def test_resume_rejects_mismatched_batch_size(tmp_path):
    acts = torch.randn(64, 8)
    ckpt_dir = tmp_path / "ckpts"
    train_sae_from_tensor(acts, steps=2, checkpoint_dir=str(ckpt_dir), checkpoint_every=2, **_train_kwargs(batch_size=16))

    try:
        train_sae_from_tensor(acts, steps=4, resume_from=str(ckpt_dir), **_train_kwargs(batch_size=8))
    except ValueError as exc:
        assert "geometry" in str(exc)
    else:
        raise AssertionError("expected ValueError resuming with a different batch_size")


def test_resume_rejects_already_finished_run(tmp_path):
    acts = torch.randn(64, 8)
    ckpt_dir = tmp_path / "ckpts"
    train_sae_from_tensor(acts, steps=4, checkpoint_dir=str(ckpt_dir), checkpoint_every=2, **_train_kwargs())

    try:
        train_sae_from_tensor(acts, steps=4, resume_from=str(ckpt_dir), **_train_kwargs())
    except ValueError as exc:
        assert "steps" in str(exc)
    else:
        raise AssertionError("expected ValueError resuming a run already at the requested steps")


def test_shuffled_epoch_sampler_covers_each_index_once_per_epoch():
    sampler = ShuffledEpochSampler(10, batch_size=5, seed=0, device=torch.device("cpu"))
    assert sampler.steps_per_epoch() == 2
    epoch0 = torch.cat([sampler.next_indices() for _ in range(2)])
    assert sorted(epoch0.tolist()) == list(range(10))  # without replacement: every index once
    assert sampler.epoch == 0 and sampler.position == 2

    epoch1 = torch.cat([sampler.next_indices() for _ in range(2)])
    assert sampler.epoch == 1
    assert sorted(epoch1.tolist()) == list(range(10))
    assert not torch.equal(epoch0, epoch1)  # fresh shuffle each epoch


def test_shuffled_epoch_sampler_drops_remainder():
    sampler = ShuffledEpochSampler(10, batch_size=4, seed=1, device=torch.device("cpu"))
    assert sampler.steps_per_epoch() == 2  # 10 // 4, trailing 2 dropped
    drawn = torch.cat([sampler.next_indices() for _ in range(2)])
    assert drawn.numel() == 8


def test_shuffled_epoch_sampler_resume_is_deterministic():
    full = ShuffledEpochSampler(10, batch_size=5, seed=7, device=torch.device("cpu"))
    full_batches = [full.next_indices() for _ in range(3)]

    a = ShuffledEpochSampler(10, batch_size=5, seed=7, device=torch.device("cpu"))
    _ = a.next_indices()
    b = ShuffledEpochSampler(10, batch_size=5, seed=7, device=torch.device("cpu"))
    b.load_state_dict(a.state_dict())
    resumed_batches = [b.next_indices() for _ in range(2)]

    assert torch.equal(full_batches[1], resumed_batches[0])
    assert torch.equal(full_batches[2], resumed_batches[1])


def test_train_logs_coverage_and_epoch_metrics():
    acts = torch.randn(64, 8)
    _, metrics = train_sae_from_tensor(acts, steps=8, **_train_kwargs(batch_size=16, log_every=1))
    last = metrics[-1]
    assert "new_tokens_seen" in last and "data_coverage" in last and "epoch" in last
    # 8 steps * batch 16 = 128 draws over 64 rows without replacement -> full coverage by epoch 2
    assert last["new_tokens_seen"] == 64.0
    assert last["data_coverage"] == 1.0
    assert last["epoch"] == 2.0
    assert last["new_tokens_seen"] <= last["tokens_seen"]


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
