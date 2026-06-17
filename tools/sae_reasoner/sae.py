from __future__ import annotations

import time
from dataclasses import dataclass
from math import sqrt
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .storage import join_uri, list_uri_names, load_torch_uri, save_torch_uri, uri_exists


@dataclass(frozen=True)
class SAEConfig:
    input_dim: int
    expansion_factor: int = 16
    top_k: int = 32
    normalize_decoder: bool = True
    topk_activation: str = "relu_topk"
    init_method: str = "kaiming"
    init_blend: float = 0.8
    input_scale: float = 1.0
    batch_topk_momentum: float = 0.01
    matryoshka_prefixes: tuple[int, ...] = ()

    @property
    def feature_dim(self) -> int:
        return self.input_dim * self.expansion_factor


class TopKSAE(nn.Module):
    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        self.encoder = nn.Linear(config.input_dim, config.feature_dim)
        self.decoder = nn.Linear(config.feature_dim, config.input_dim, bias=False)
        self.pre_bias = nn.Parameter(torch.zeros(config.input_dim))
        self.register_buffer("batch_topk_threshold", torch.tensor(0.0))
        nn.init.kaiming_uniform_(self.encoder.weight, a=5**0.5)
        self._init_encoder_bias()
        self._init_decoder_from_encoder()

    def scale_input(self, x: torch.Tensor) -> torch.Tensor:
        return x * float(self.config.input_scale)

    def unscale_output(self, x: torch.Tensor) -> torch.Tensor:
        scale = max(float(self.config.input_scale), 1e-12)
        return x / scale

    @torch.no_grad()
    def _init_encoder_bias(self) -> None:
        if self.encoder.bias is None:
            return
        bound = 1.0 / sqrt(self.config.input_dim) if self.config.input_dim > 0 else 0.0
        nn.init.uniform_(self.encoder.bias, -bound, bound)

    @torch.no_grad()
    def _renorm_decoder(self) -> None:
        if not self.config.normalize_decoder:
            return
        weight = self.decoder.weight
        norms = weight.norm(dim=0, keepdim=True).clamp_min(1e-6)
        weight.div_(norms)

    @torch.no_grad()
    def _init_decoder_from_encoder(self) -> None:
        self.decoder.weight.copy_(self.encoder.weight.T)
        self._renorm_decoder()

    @torch.no_grad()
    def initialize_from_data(self, data: torch.Tensor, *, blend: float | None = None) -> None:
        if data.ndim != 2 or data.shape[-1] != self.config.input_dim:
            raise ValueError(f"data init expected [N, {self.config.input_dim}], got {tuple(data.shape)}")
        if data.shape[0] == 0:
            raise ValueError("data init requires at least one activation row")
        blend = self.config.init_blend if blend is None else blend
        blend = float(max(0.0, min(1.0, blend)))
        feature_dim = self.config.feature_dim
        sample_idx = torch.randint(0, data.shape[0], (feature_dim,), device=data.device)
        sampled = self.scale_input(data.index_select(0, sample_idx).float())
        centered = sampled - self.scale_input(activation_mean(data)).to(device=sampled.device)
        random_weight = torch.empty_like(centered)
        nn.init.kaiming_uniform_(random_weight, a=5**0.5)
        weight = blend * centered + (1.0 - blend) * random_weight
        self.encoder.weight.copy_(weight.to(device=self.encoder.weight.device, dtype=self.encoder.weight.dtype))
        self.decoder.weight.copy_(self.encoder.weight.T)
        self._init_encoder_bias()
        nn.init.zeros_(self.pre_bias)
        self._renorm_decoder()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        acts = self.encoder(self.scale_input(x) - self.pre_bias)
        if self.config.topk_activation == "relu_topk":
            acts = torch.relu(acts)
            return self._apply_row_topk(acts)
        if self.config.topk_activation == "topk":
            return self._apply_row_topk(acts)
        if self.config.topk_activation == "batch_topk":
            return self._apply_batch_topk(acts)
        raise ValueError("topk_activation must be one of: topk, relu_topk, batch_topk")

    def _apply_row_topk(self, acts: torch.Tensor) -> torch.Tensor:
        if self.config.top_k <= 0 or self.config.top_k >= acts.shape[-1]:
            return acts
        values, indexes = torch.topk(acts, k=self.config.top_k, dim=-1)
        sparse = torch.zeros_like(acts)
        sparse.scatter_(-1, indexes, values)
        return sparse

    def _apply_batch_topk(self, acts: torch.Tensor) -> torch.Tensor:
        if self.config.top_k <= 0:
            return torch.zeros_like(acts)
        if not self.training:
            threshold = self.batch_topk_threshold.to(device=acts.device, dtype=acts.dtype)
            if float(threshold.detach().cpu()) == 0.0:
                return self._apply_row_topk(acts)
            return torch.where(acts > threshold, acts, torch.zeros_like(acts))
        flat = acts.reshape(-1)
        keep = min(int(self.config.top_k) * max(1, acts.shape[0]), flat.numel())
        if keep >= flat.numel():
            selected = acts
        else:
            values, indexes = torch.topk(flat, k=keep, sorted=False)
            selected_flat = torch.zeros_like(flat)
            selected_flat.scatter_(0, indexes, values)
            selected = selected_flat.reshape_as(acts)
        with torch.no_grad():
            selected_values = selected[selected != 0]
            if selected_values.numel():
                threshold = selected_values.min().to(device=self.batch_topk_threshold.device, dtype=self.batch_topk_threshold.dtype)
                momentum = float(max(0.0, min(1.0, self.config.batch_topk_momentum)))
                self.batch_topk_threshold.mul_(1.0 - momentum).add_(threshold * momentum)
        return selected

    def decode(self, features: torch.Tensor, *, unscale: bool = True) -> torch.Tensor:
        recon = self.decoder(features) + self.pre_bias
        return self.unscale_output(recon) if unscale else recon

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(x)
        recon = self.decode(features)
        return recon, features

    def feature_delta(
        self,
        x: torch.Tensor,
        *,
        feature_id: int,
        multiplier: float,
    ) -> torch.Tensor:
        if feature_id < 0 or feature_id >= self.config.feature_dim:
            raise IndexError(f"feature_id={feature_id} outside [0, {self.config.feature_dim})")
        features = self.encode(x)
        original = features[..., feature_id].clone()
        delta_coeff = original * (float(multiplier) - 1.0)
        decoder_vector = self.decoder.weight[:, feature_id].to(device=x.device, dtype=x.dtype)
        return self.unscale_output(delta_coeff.unsqueeze(-1) * decoder_vector)


def save_sae(path: str, sae: TopKSAE, *, metadata: dict | None = None) -> str:
    return save_torch_uri(
        path,
        {
            "config": sae.config.__dict__,
            "state_dict": sae.state_dict(),
            "metadata": metadata or {},
        },
    )


def _sae_from_payload(payload: dict) -> TopKSAE:
    raw_config = dict(payload["config"])
    if "topk_activation" not in raw_config:
        raw_config["topk_activation"] = "relu_topk"
    config = SAEConfig(**raw_config)
    sae = TopKSAE(config)
    state_dict = _compatible_sae_state_dict(payload.get("state_dict", {}))
    load_result = sae.load_state_dict(state_dict, strict=False)
    tolerated_missing = {"batch_topk_threshold"}
    missing = set(load_result.missing_keys) - tolerated_missing
    if missing or load_result.unexpected_keys:
        raise RuntimeError(
            f"invalid SAE checkpoint state_dict; missing={sorted(missing)} unexpected={sorted(load_result.unexpected_keys)}"
        )
    return sae


def _compatible_sae_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Map legacy SAE checkpoints with a separate post_bias onto the tied-bias model."""
    compatible = dict(state_dict)
    if "post_bias" in compatible:
        compatible.pop("post_bias")
    return compatible


def _compatible_optimizer_state_dict(optimizer_state_dict: dict[str, Any] | None, *, state_dict: dict[str, Any]) -> dict[str, Any] | None:
    """Map legacy optimizer ids from the two-bias model onto the tied-bias model.

    Legacy checkpoints used parameter order:
    pre_bias, post_bias, encoder.weight, encoder.bias, decoder.weight

    Current checkpoints use:
    pre_bias, encoder.weight, encoder.bias, decoder.weight
    """
    if optimizer_state_dict is None or "post_bias" not in state_dict:
        return optimizer_state_dict
    groups = optimizer_state_dict.get("param_groups") or []
    compatible_state = {
        (int(param_id) - 1 if int(param_id) > 1 else int(param_id)): value
        for param_id, value in (optimizer_state_dict.get("state") or {}).items()
        if int(param_id) != 1
    }
    remapped_groups = []
    for group in groups:
        remapped_group = dict(group)
        remapped_group["params"] = [
            (int(param_id) - 1 if int(param_id) > 1 else int(param_id))
            for param_id in group.get("params", [])
            if int(param_id) != 1
        ]
        remapped_groups.append(remapped_group)
    return {
        **optimizer_state_dict,
        "state": compatible_state,
        "param_groups": remapped_groups,
    }


def load_sae(path: str, map_location: str | torch.device = "cpu") -> TopKSAE:
    payload = load_torch_uri(path, map_location=str(map_location))
    sae = _sae_from_payload(payload)
    sae.eval()
    return sae


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    cpu_state = state.get("cpu")
    if cpu_state is not None:
        torch.set_rng_state(cpu_state.cpu())
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in cuda_state])


_CHECKPOINT_PREFIX = "checkpoint-"
_CHECKPOINT_SUFFIX = ".pt"


def checkpoint_path(checkpoint_dir: str, step: int) -> str:
    """Local path or S3 URI for a given training step's checkpoint."""
    return join_uri(checkpoint_dir, f"{_CHECKPOINT_PREFIX}{int(step):06d}{_CHECKPOINT_SUFFIX}")


def save_training_checkpoint(
    path: str,
    *,
    sae: TopKSAE,
    optimizer: torch.optim.Optimizer,
    step: int,
    metrics: list[dict[str, float]],
    rng_state: dict[str, Any] | None = None,
    sampler_state: dict[str, Any] | None = None,
    metadata: dict | None = None,
) -> str:
    """Write a resumable training checkpoint (model + optimizer + step + RNG + sampler) locally or to S3."""
    payload = {
        "format": "sae_training_checkpoint",
        "config": sae.config.__dict__,
        "state_dict": sae.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": int(step),
        "metrics": metrics,
        "rng_state": rng_state if rng_state is not None else _capture_rng_state(),
        "sampler_state": sampler_state or {},
        "metadata": metadata or {},
    }
    return save_torch_uri(path, payload)


def load_training_checkpoint(path: str, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a training checkpoint, returning the rebuilt SAE plus optimizer/step/metrics/RNG state."""
    payload = load_torch_uri(path, map_location=str(map_location))
    if payload.get("format") != "sae_training_checkpoint":
        raise ValueError(
            f"{path!r} is not a resumable training checkpoint "
            "(missing format='sae_training_checkpoint'); a save_sae model file cannot be resumed from"
        )
    sae = _sae_from_payload(payload)
    return {
        "sae": sae,
        "config": sae.config,
        "optimizer_state_dict": _compatible_optimizer_state_dict(
            payload.get("optimizer_state_dict"),
            state_dict=payload.get("state_dict", {}),
        ),
        "step": int(payload.get("step", 0)),
        "metrics": list(payload.get("metrics", [])),
        "rng_state": payload.get("rng_state", {}),
        "sampler_state": payload.get("sampler_state", {}),
        "metadata": payload.get("metadata", {}),
    }


def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    """Return the highest-step checkpoint URI in a local dir or S3 prefix, or None if there are none."""
    best_step = -1
    best_name: str | None = None
    for name in list_uri_names(checkpoint_dir, prefix=_CHECKPOINT_PREFIX, suffix=_CHECKPOINT_SUFFIX):
        stem = name[len(_CHECKPOINT_PREFIX) : -len(_CHECKPOINT_SUFFIX)]
        if not stem.isdigit():
            continue
        step = int(stem)
        if step > best_step:
            best_step = step
            best_name = name
    return join_uri(checkpoint_dir, best_name) if best_name is not None else None


def _resolve_resume_checkpoint(resume_from: str) -> str:
    # A path/URI ending in the checkpoint suffix is a single checkpoint file; anything
    # else is treated as a directory/prefix to scan for the latest checkpoint.
    if resume_from.endswith(_CHECKPOINT_SUFFIX):
        if not uri_exists(resume_from):
            raise FileNotFoundError(f"resume checkpoint {resume_from!r} does not exist")
        return resume_from
    latest = find_latest_checkpoint(resume_from)
    if latest is None:
        raise FileNotFoundError(f"no {_CHECKPOINT_PREFIX}*{_CHECKPOINT_SUFFIX} found under resume target {resume_from!r}")
    return latest


class ShuffledEpochSampler:
    """Without-replacement mini-batch sampler: each epoch draws a fresh shuffled permutation.

    Batch order is fully determined by ``seed`` and the integer epoch, so resuming only needs
    ``(seed, epoch, position)`` — no RNG blob. The trailing ``n % batch_size`` indices of each
    epoch's permutation are dropped (drop_last) to keep a constant batch size; a different tail
    is dropped each epoch because the permutation is reshuffled.
    """

    def __init__(self, n: int, batch_size: int, *, seed: int, device: torch.device):
        if n <= 0:
            raise ValueError("ShuffledEpochSampler requires n > 0")
        self.n = int(n)
        self.batch_size = max(1, min(int(batch_size), self.n))
        self.seed = int(seed)
        self.device = device
        self.epoch = 0
        self.position = 0
        self._perm = self._make_perm(self.epoch)

    def steps_per_epoch(self) -> int:
        return max(1, self.n // self.batch_size)

    def _make_perm(self, epoch: int) -> torch.Tensor:
        gen = torch.Generator(device=self.device)
        gen.manual_seed(self.seed + int(epoch))
        return torch.randperm(self.n, generator=gen, device=self.device)

    def next_indices(self) -> torch.Tensor:
        if self.position >= self.steps_per_epoch():
            self.epoch += 1
            self.position = 0
            self._perm = self._make_perm(self.epoch)
        start = self.position * self.batch_size
        idx = self._perm[start : start + self.batch_size]
        self.position += 1
        return idx

    def fractional_epoch(self) -> float:
        return self.epoch + self.position / self.steps_per_epoch()

    def state_dict(self) -> dict[str, int]:
        return {"seed": self.seed, "epoch": self.epoch, "position": self.position, "n": self.n, "batch_size": self.batch_size}

    def load_state_dict(self, state: dict[str, int]) -> None:
        # Batch ordering is only reproducible when data/batch geometry matches; refuse to resume
        # silently against a different shape, which would desync coverage and re-traverse data.
        saved_n = int(state.get("n", -1))
        saved_batch_size = int(state.get("batch_size", -1))
        if saved_n != self.n or saved_batch_size != self.batch_size:
            raise ValueError(
                "cannot resume sampler: checkpoint geometry "
                f"(n={saved_n}, batch_size={saved_batch_size}) does not match current "
                f"(n={self.n}, batch_size={self.batch_size})"
            )
        self.seed = int(state["seed"])
        self.epoch = int(state["epoch"])
        self.position = int(state["position"])
        self._perm = self._make_perm(self.epoch)


def train_sae_from_tensor(
    activations: torch.Tensor,
    *,
    validation_activations: torch.Tensor | None = None,
    token_groups: Sequence[Sequence[str]] | None = None,
    validation_token_groups: Sequence[Sequence[str]] | None = None,
    expansion_factor: int = 16,
    top_k: int = 32,
    topk_activation: str = "relu_topk",
    init_method: str = "kaiming",
    init_blend: float = 0.8,
    activation_norm: str = "sqrt_d",
    batch_topk_momentum: float = 0.01,
    matryoshka_prefixes: Sequence[int | float | str] | str | None = None,
    matryoshka_loss_coeff: float = 1.0,
    recon_loss: str = "mse",
    feature_l1_coeff: float = 0.0,
    steps: int = 1000,
    batch_size: int = 1024,
    shuffle_seed: int = 0,
    lr: float = 3e-4,
    warmup_steps: int = 0,
    lr_schedule: str = "constant",
    max_grad_norm: float | None = None,
    val_batch_size: int | None = None,
    device: str | torch.device | None = None,
    log_every: int | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    log_diagnostic_histograms: bool = False,
    decoder_similarity_sample_size: int = 256,
    checkpoint_dir: str | None = None,
    checkpoint_every: int | None = None,
    resume_from: str | None = None,
    checkpoint_metadata: dict | None = None,
) -> tuple[TopKSAE, list[dict[str, float]]]:
    if activations.ndim != 2:
        raise ValueError(f"activations must be rank-2 [N, D], got shape {tuple(activations.shape)}")
    if lr_schedule not in {"constant", "cosine"}:
        raise ValueError("lr_schedule must be one of: constant, cosine")
    if topk_activation not in {"topk", "relu_topk", "batch_topk"}:
        raise ValueError("topk_activation must be one of: topk, relu_topk, batch_topk")
    if init_method not in {"data", "kaiming"}:
        raise ValueError("init_method must be one of: data, kaiming")
    if activation_norm not in {"sqrt_d", "none"}:
        raise ValueError("activation_norm must be one of: sqrt_d, none")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    acts = activations.detach().contiguous()
    val_acts = validation_activations.detach().contiguous() if validation_activations is not None and validation_activations.numel() else None
    train_group_masks = build_group_masks(token_groups, len(acts), device=acts.device, prefix="train")
    val_group_masks = build_group_masks(
        validation_token_groups,
        len(val_acts) if val_acts is not None else 0,
        device=val_acts.device if val_acts is not None else acts.device,
        prefix="validation",
    )
    if val_acts is not None and val_acts.shape[-1] != acts.shape[-1]:
        raise ValueError(f"validation activations hidden dim {val_acts.shape[-1]} does not match training dim {acts.shape[-1]}")
    input_scale = activation_l2_scale(acts, mode=activation_norm)
    feature_dim = int(acts.shape[-1]) * int(expansion_factor)
    resolved_matryoshka_prefixes = resolve_matryoshka_prefixes(matryoshka_prefixes, feature_dim=feature_dim)
    sae = TopKSAE(
        SAEConfig(
            input_dim=acts.shape[-1],
            expansion_factor=expansion_factor,
            top_k=top_k,
            topk_activation=topk_activation,
            init_method=init_method,
            init_blend=init_blend,
            input_scale=input_scale,
            batch_topk_momentum=batch_topk_momentum,
            matryoshka_prefixes=resolved_matryoshka_prefixes,
        )
    ).to(device)
    if init_method == "data":
        sae.initialize_from_data(acts, blend=init_blend)
    start_step = 0
    metrics: list[dict[str, float]] = []
    resumed: dict[str, Any] | None = None
    if resume_from is not None:
        resumed = load_training_checkpoint(_resolve_resume_checkpoint(resume_from), map_location=device)
        if resumed["config"] != sae.config:
            raise ValueError(
                "resume checkpoint config does not match the requested training config; "
                f"checkpoint={resumed['config']} requested={sae.config}"
            )
        sae = resumed["sae"].to(device)
        start_step = resumed["step"]
        if start_step >= steps:
            raise ValueError(
                f"resume checkpoint is at step {start_step}, which is >= requested steps {steps}; "
                "increase --steps to continue training"
            )
        metrics = list(resumed["metrics"])
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    if resumed is not None:
        if resumed["optimizer_state_dict"] is not None:
            opt.load_state_dict(resumed["optimizer_state_dict"])
        _restore_rng_state(resumed["rng_state"])
    n = acts.shape[0]
    sampler = ShuffledEpochSampler(n, batch_size, seed=shuffle_seed, device=acts.device)
    seen_mask = torch.zeros(n, dtype=torch.bool, device=acts.device)
    if resumed is not None:
        sampler_state = resumed["sampler_state"]
        if sampler_state.get("sampler"):
            sampler.load_state_dict(sampler_state["sampler"])
        resumed_seen = sampler_state.get("seen_mask")
        if resumed_seen is not None and resumed_seen.numel() == n:
            seen_mask = resumed_seen.to(device=acts.device, dtype=torch.bool)
    log_interval = max(1, int(log_every or max(1, steps // 10)))
    start = time.time()
    for step in range(start_step + 1, steps + 1):
        step_lr = lr_for_step(lr, step=step, total_steps=steps, warmup_steps=warmup_steps, schedule=lr_schedule)
        for group in opt.param_groups:
            group["lr"] = step_lr
        idx = sampler.next_indices()
        seen_mask[idx] = True
        batch = acts.index_select(0, idx).to(device=device, dtype=torch.float32)
        batch_group_masks = {label: mask.index_select(0, idx).to(device=device) for label, mask in train_group_masks.items()}
        recon, features, recon_norm, target_norm = sae_training_outputs(sae, batch)
        recon_objective = reconstruction_loss(recon_norm, target_norm, recon_loss)
        matryoshka_loss, matryoshka_losses_by_prefix = matryoshka_reconstruction_loss(
            sae,
            features,
            target_norm,
            prefixes=resolved_matryoshka_prefixes,
            recon_loss=recon_loss,
        )
        feature_l1 = features.abs().mean()
        loss = recon_objective + matryoshka_loss_coeff * matryoshka_loss + feature_l1_coeff * feature_l1
        opt.zero_grad(set_to_none=True)
        loss.backward()
        remove_decoder_parallel_grad(sae)
        grad_norm = gradient_norm(sae.parameters())
        did_clip = bool(max_grad_norm and max_grad_norm > 0 and grad_norm > max_grad_norm)
        if max_grad_norm and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(sae.parameters(), max_grad_norm)
        opt.step()
        sae._renorm_decoder()
        if step == 1 or step == steps or step % log_interval == 0:
            with torch.no_grad():
                usage_metrics, histogram_payload = feature_usage_diagnostics(
                    features,
                    include_histograms=log_diagnostic_histograms,
                )
                decoder_metrics, decoder_histograms = decoder_diagnostics(
                    sae,
                    include_histograms=log_diagnostic_histograms,
                    similarity_sample_size=decoder_similarity_sample_size,
                )
                histogram_payload.update(decoder_histograms)
                mse = F.mse_loss(recon, batch).item()
                residual_ss = (recon - batch).pow(2).sum()
                centered_ss = (batch - batch.mean(dim=0, keepdim=True)).pow(2).sum().clamp_min(1e-12)
                explained_variance = 1.0 - float((residual_ss / centered_ss).item())
                metric = {
                    "step": float(step),
                    "loss": float(loss.item()),
                    "recon_loss": float(recon_objective.item()),
                    "matryoshka_loss": float(matryoshka_loss.item()),
                    "matryoshka_loss_coeff": float(matryoshka_loss_coeff),
                    "matryoshka_num_prefixes": float(len(resolved_matryoshka_prefixes)),
                    "mse": float(mse),
                    "normalized_mse": float(F.mse_loss(recon_norm, target_norm).item()),
                    "explained_variance": float(explained_variance),
                    "feature_l1": float(feature_l1.item()),
                    "grad_norm": float(grad_norm),
                    "grad_clipped": float(did_clip),
                    "lr": float(step_lr),
                    "activation_norm": 1.0 if activation_norm == "sqrt_d" else 0.0,
                    "input_scale": float(input_scale),
                    # tokens_seen = cumulative batch rows processed (re-counts rows across epochs);
                    # new_tokens_seen = distinct dataset rows drawn at least once (see data_coverage).
                    "tokens_seen": float(step * sampler.batch_size),
                    "epoch": float(sampler.fractional_epoch()),
                    "new_tokens_seen": float(int(seen_mask.sum().item())),
                    "data_coverage": float(seen_mask.float().mean().item()) if n else 0.0,
                    "elapsed_seconds": float(time.time() - start),
                }
                metric.update(usage_metrics)
                metric.update(decoder_metrics)
                for prefix, prefix_loss in matryoshka_losses_by_prefix.items():
                    metric[f"matryoshka_recon_loss_prefix/{prefix}"] = float(prefix_loss.item())
                metric.update(group_reconstruction_metrics(recon, batch, batch_group_masks, prefix="train"))
                if val_acts is not None:
                    val_metric = validation_metrics(
                        sae,
                        val_acts,
                        group_masks=val_group_masks,
                        recon_loss=recon_loss,
                        matryoshka_prefixes=resolved_matryoshka_prefixes,
                        batch_size=val_batch_size or batch_size,
                        include_histograms=log_diagnostic_histograms,
                    )
                    histogram_payload.update(val_metric.pop("_wandb_histograms", {}))
                    metric.update(val_metric)
                    metric.update(train_val_gap_metrics(metric))
                metrics.append(metric)
                if progress_callback is not None:
                    callback_metric: dict[str, Any] = dict(metric)
                    if histogram_payload:
                        callback_metric["_wandb_histograms"] = histogram_payload
                    progress_callback(callback_metric)
        if checkpoint_dir and checkpoint_every and (step % checkpoint_every == 0 or step == steps):
            save_training_checkpoint(
                checkpoint_path(checkpoint_dir, step),
                sae=sae,
                optimizer=opt,
                step=step,
                metrics=metrics,
                sampler_state={"seen_mask": seen_mask.detach().cpu(), "sampler": sampler.state_dict()},
                metadata=checkpoint_metadata,
            )
    return sae.cpu(), metrics


def activation_l2_scale(activations: torch.Tensor, *, mode: str, chunk_size: int = 65536) -> float:
    if mode == "none":
        return 1.0
    if mode != "sqrt_d":
        raise ValueError("activation_norm must be one of: sqrt_d, none")
    if activations.numel() == 0:
        return 1.0
    total_norm = 0.0
    total_rows = 0
    for start in range(0, activations.shape[0], max(1, chunk_size)):
        chunk = activations[start : start + chunk_size].float()
        total_norm += float(chunk.norm(dim=-1).sum().item())
        total_rows += int(chunk.shape[0])
    mean_norm = max(total_norm / max(1, total_rows), 1e-12)
    return float(sqrt(float(activations.shape[-1])) / mean_norm)


def activation_mean(activations: torch.Tensor, *, chunk_size: int = 65536) -> torch.Tensor:
    if activations.ndim != 2:
        raise ValueError(f"activations must be rank-2 [N, D], got shape {tuple(activations.shape)}")
    total = torch.zeros(activations.shape[-1], dtype=torch.float32, device=activations.device)
    total_rows = 0
    for start in range(0, activations.shape[0], max(1, chunk_size)):
        chunk = activations[start : start + chunk_size].float()
        total.add_(chunk.sum(dim=0))
        total_rows += int(chunk.shape[0])
    return total / max(1, total_rows)


def sae_training_outputs(sae: TopKSAE, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    features = sae.encode(batch)
    recon_norm = sae.decode(features, unscale=False)
    target_norm = sae.scale_input(batch)
    recon = sae.unscale_output(recon_norm)
    return recon, features, recon_norm, target_norm


def resolve_matryoshka_prefixes(prefixes: Sequence[int | float | str] | str | None, *, feature_dim: int) -> tuple[int, ...]:
    if prefixes is None or prefixes == "":
        return ()
    if feature_dim <= 0:
        raise ValueError("feature_dim must be positive")
    if isinstance(prefixes, str):
        raw_values: Sequence[int | float | str] = [part.strip() for part in prefixes.split(",") if part.strip()]
    else:
        raw_values = prefixes
    resolved: list[int] = []
    for raw in raw_values:
        if isinstance(raw, str):
            value = float(raw)
            is_fraction = ("." in raw or "e" in raw.lower()) and 0.0 < value <= 1.0
        else:
            value = float(raw)
            is_fraction = isinstance(raw, float) and 0.0 < value <= 1.0
        if value <= 0:
            raise ValueError("matryoshka prefixes must be positive")
        prefix = int(round(value * feature_dim)) if is_fraction else int(value)
        if prefix <= 0 or prefix > feature_dim:
            raise ValueError(f"matryoshka prefix {raw!r} resolves to {prefix}, outside [1, {feature_dim}]")
        resolved.append(prefix)
    return tuple(sorted(set(resolved)))


def matryoshka_reconstruction_loss(
    sae: TopKSAE,
    features: torch.Tensor,
    target_norm: torch.Tensor,
    *,
    prefixes: Sequence[int],
    recon_loss: str,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    losses: dict[int, torch.Tensor] = {}
    feature_dim = int(sae.config.feature_dim)
    for prefix in prefixes:
        if prefix >= feature_dim:
            continue
        prefix_recon = F.linear(features[:, :prefix], sae.decoder.weight[:, :prefix], sae.pre_bias)
        losses[int(prefix)] = reconstruction_loss(prefix_recon, target_norm, recon_loss)
    if not losses:
        return target_norm.new_zeros(()), {}
    return sum(losses.values(), target_norm.new_zeros(())), losses


@torch.no_grad()
def feature_usage_diagnostics(
    features: torch.Tensor,
    *,
    prefix: str = "",
    include_histograms: bool = False,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    active = features != 0
    batch_size = max(1, int(features.shape[0]))
    feature_dim = max(1, int(features.shape[-1]))
    fire_counts = active.sum(dim=0).to(dtype=torch.float32)
    fire_rates = fire_counts / float(batch_size)
    live = fire_counts > 0
    active_values = features[active].detach().abs().float()
    token_l0 = active.sum(dim=-1).to(dtype=torch.float32)
    often_threshold = max(2.0, 0.01 * float(batch_size))
    name = metric_name(prefix)
    metrics = {
        name("l0"): float(token_l0.mean().item()) if token_l0.numel() else 0.0,
        name("positive_feature_frac"): float((features > 0).float().mean().item()) if features.numel() else 0.0,
        name("feature_density"): float(active.float().mean().item()) if active.numel() else 0.0,
        name("dead_feature_frac_batch"): float((~live).float().mean().item()) if fire_counts.numel() else 0.0,
        name("feature_live_count_batch"): float(live.sum().item()),
        name("feature_live_frac_batch"): float(live.float().mean().item()) if live.numel() else 0.0,
        name("feature_dead_count_batch"): float(feature_dim - int(live.sum().item())),
        name("feature_used_ge_2_count_batch"): float((fire_counts >= 2).sum().item()),
        name("feature_used_ge_1pct_count_batch"): float((fire_counts >= often_threshold).sum().item()),
        name("feature_fire_count_max_batch"): float(fire_counts.max().item()) if fire_counts.numel() else 0.0,
        name("feature_fire_rate_p50_batch"): tensor_quantile(fire_rates, 0.50),
        name("feature_fire_rate_p90_batch"): tensor_quantile(fire_rates, 0.90),
        name("feature_fire_rate_p99_batch"): tensor_quantile(fire_rates, 0.99),
        name("feature_fire_rate_max_batch"): float(fire_rates.max().item()) if fire_rates.numel() else 0.0,
        name("feature_abs_activation_active_mean"): tensor_mean(active_values),
        name("feature_abs_activation_active_p99"): tensor_quantile(active_values, 0.99),
        name("feature_abs_activation_active_max"): tensor_max(active_values),
    }
    histograms: dict[str, torch.Tensor] = {}
    if include_histograms:
        histograms[histogram_name(prefix, "feature_fire_rate")] = fire_rates.detach().float().cpu()
        histograms[histogram_name(prefix, "feature_fire_count")] = fire_counts.detach().float().cpu()
        histograms[histogram_name(prefix, "token_l0")] = token_l0.detach().float().cpu()
        if active_values.numel():
            histograms[histogram_name(prefix, "feature_abs_activation_active")] = active_values.detach().float().cpu()
    return metrics, histograms


@torch.no_grad()
def decoder_diagnostics(
    sae: TopKSAE,
    *,
    include_histograms: bool = False,
    similarity_sample_size: int = 256,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    weight = sae.decoder.weight.detach().float()
    norms = weight.norm(dim=0)
    metrics = {
        "decoder_norm_mean": tensor_mean(norms),
        "decoder_norm_std": tensor_std(norms),
        "decoder_norm_min": tensor_min(norms),
        "decoder_norm_p50": tensor_quantile(norms, 0.50),
        "decoder_norm_p90": tensor_quantile(norms, 0.90),
        "decoder_norm_p99": tensor_quantile(norms, 0.99),
        "decoder_norm_max": tensor_max(norms),
    }
    histograms: dict[str, torch.Tensor] = {}
    if include_histograms:
        histograms[histogram_name("", "decoder_norm")] = norms.detach().float().cpu()
    similarity_metrics, similarity_histograms = decoder_similarity_diagnostics(
        weight,
        include_histograms=include_histograms,
        sample_size=similarity_sample_size,
    )
    metrics.update(similarity_metrics)
    histograms.update(similarity_histograms)
    return metrics, histograms


@torch.no_grad()
def decoder_similarity_diagnostics(
    decoder_weight: torch.Tensor,
    *,
    include_histograms: bool = False,
    sample_size: int = 256,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    feature_dim = int(decoder_weight.shape[1])
    sample_n = min(max(0, int(sample_size)), feature_dim)
    if sample_n <= 1:
        return {}, {}
    if sample_n == feature_dim:
        indexes = torch.arange(feature_dim, device=decoder_weight.device)
    else:
        indexes = torch.linspace(0, feature_dim - 1, steps=sample_n, device=decoder_weight.device).round().long()
    columns = decoder_weight.index_select(1, indexes).T
    columns = F.normalize(columns, dim=-1, eps=1e-12)
    abs_cosine = (columns @ columns.T).abs()
    abs_cosine.fill_diagonal_(0.0)
    nearest = abs_cosine.max(dim=-1).values
    metrics = {
        "decoder_nearest_abs_cosine_sample_mean": tensor_mean(nearest),
        "decoder_nearest_abs_cosine_sample_p95": tensor_quantile(nearest, 0.95),
        "decoder_nearest_abs_cosine_sample_p99": tensor_quantile(nearest, 0.99),
        "decoder_nearest_abs_cosine_sample_max": tensor_max(nearest),
        "decoder_duplicate_frac_cos_gt_0_90_sample": float((nearest > 0.90).float().mean().item()),
        "decoder_duplicate_frac_cos_gt_0_95_sample": float((nearest > 0.95).float().mean().item()),
        "decoder_duplicate_frac_cos_gt_0_99_sample": float((nearest > 0.99).float().mean().item()),
        "decoder_similarity_sample_size": float(sample_n),
    }
    histograms = {}
    if include_histograms:
        histograms[histogram_name("", "decoder_nearest_abs_cosine_sample")] = nearest.detach().float().cpu()
    return metrics, histograms


def train_val_gap_metrics(metric: dict[str, float]) -> dict[str, float]:
    gaps: dict[str, float] = {}
    for key in ["recon_loss", "mse", "normalized_mse", "explained_variance", "l0", "feature_density", "dead_feature_frac_batch"]:
        val_key = f"val_{key}"
        if key in metric and val_key in metric:
            gaps[f"train_val_gap/{key}"] = float(metric[val_key] - metric[key])
    return gaps


def metric_name(prefix: str) -> Callable[[str], str]:
    clean = prefix.strip("_")
    if not clean:
        return lambda name: name
    return lambda name: f"{clean}_{name}"


def histogram_name(prefix: str, name: str) -> str:
    clean = prefix.strip("_")
    return f"hist/{clean}_{name}" if clean else f"hist/{name}"


def tensor_mean(values: torch.Tensor) -> float:
    values = values.detach().float().reshape(-1)
    return float(values.mean().item()) if values.numel() else 0.0


def tensor_std(values: torch.Tensor) -> float:
    values = values.detach().float().reshape(-1)
    return float(values.std(unbiased=False).item()) if values.numel() else 0.0


def tensor_min(values: torch.Tensor) -> float:
    values = values.detach().float().reshape(-1)
    return float(values.min().item()) if values.numel() else 0.0


def tensor_max(values: torch.Tensor) -> float:
    values = values.detach().float().reshape(-1)
    return float(values.max().item()) if values.numel() else 0.0


def tensor_quantile(values: torch.Tensor, q: float) -> float:
    values = values.detach().float().reshape(-1)
    if not values.numel():
        return 0.0
    return float(torch.quantile(values, float(q)).item())


@torch.no_grad()
def validation_metrics(
    sae: TopKSAE,
    validation_activations: torch.Tensor,
    *,
    group_masks: dict[str, torch.Tensor] | None = None,
    recon_loss: str,
    batch_size: int,
    matryoshka_prefixes: Sequence[int] = (),
    include_histograms: bool = False,
) -> dict[str, Any]:
    n = validation_activations.shape[0]
    if n <= 0:
        return {}
    sample_n = min(max(1, batch_size), n)
    device = next(sae.parameters()).device
    if sample_n < n:
        # Validation is a one-shot metric estimate, so with-replacement sampling is fine here;
        # only training uses the without-replacement ShuffledEpochSampler.
        idx = torch.randint(0, n, (sample_n,), device=validation_activations.device)
        batch = validation_activations.index_select(0, idx).to(device=device, dtype=torch.float32)
        sampled_group_masks = {label: mask.index_select(0, idx).to(device=device) for label, mask in (group_masks or {}).items()}
    else:
        batch = validation_activations.to(device=device, dtype=torch.float32)
        sampled_group_masks = {label: mask.to(device=device) for label, mask in (group_masks or {}).items()}
    recon, features, recon_norm, target_norm = sae_training_outputs(sae, batch)
    val_recon = reconstruction_loss(recon_norm, target_norm, recon_loss)
    val_matryoshka, val_matryoshka_by_prefix = matryoshka_reconstruction_loss(
        sae,
        features,
        target_norm,
        prefixes=matryoshka_prefixes,
        recon_loss=recon_loss,
    )
    val_mse = F.mse_loss(recon, batch)
    residual_ss = (recon - batch).pow(2).sum()
    centered_ss = (batch - batch.mean(dim=0, keepdim=True)).pow(2).sum().clamp_min(1e-12)
    usage_metrics, histogram_payload = feature_usage_diagnostics(
        features,
        prefix="val",
        include_histograms=include_histograms,
    )
    metrics = {
        "val_recon_loss": float(val_recon.item()),
        "val_matryoshka_loss": float(val_matryoshka.item()),
        "val_mse": float(val_mse.item()),
        "val_normalized_mse": float(F.mse_loss(recon_norm, target_norm).item()),
        "val_explained_variance": float(1.0 - (residual_ss / centered_ss).item()),
        "val_tokens": float(n),
        "val_sample_tokens": float(sample_n),
    }
    metrics.update(usage_metrics)
    for prefix, prefix_loss in val_matryoshka_by_prefix.items():
        metrics[f"val_matryoshka_recon_loss_prefix/{prefix}"] = float(prefix_loss.item())
    metrics.update(group_reconstruction_metrics(recon, batch, sampled_group_masks, prefix="val"))
    if histogram_payload:
        metrics["_wandb_histograms"] = histogram_payload
    return metrics


def build_group_masks(
    token_groups: Sequence[Sequence[str]] | None,
    expected_len: int,
    *,
    device: torch.device,
    prefix: str,
) -> dict[str, torch.Tensor]:
    if token_groups is None:
        return {}
    if len(token_groups) != expected_len:
        raise ValueError(f"{prefix} token_groups length {len(token_groups)} does not match activations length {expected_len}")
    labels = sorted({label for groups in token_groups for label in groups})
    return {
        label: torch.tensor([label in groups for groups in token_groups], dtype=torch.bool, device=device)
        for label in labels
    }


@torch.no_grad()
def group_reconstruction_metrics(
    recon: torch.Tensor,
    target: torch.Tensor,
    group_masks: dict[str, torch.Tensor],
    *,
    prefix: str,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for label, mask in sorted(group_masks.items()):
        count = int(mask.sum().item())
        if count <= 0:
            continue
        group_recon = recon[mask]
        group_target = target[mask]
        residual_ss = (group_recon - group_target).pow(2).sum()
        centered_ss = (group_target - group_target.mean(dim=0, keepdim=True)).pow(2).sum()
        safe_label = sanitize_metric_label(label)
        metrics[f"{prefix}_mse_by_group/{safe_label}"] = float(F.mse_loss(group_recon, group_target).item())
        metrics[f"{prefix}_explained_variance_by_group/{safe_label}"] = (
            float(1.0 - (residual_ss / centered_ss.clamp_min(1e-12)).item()) if count > 1 else float("nan")
        )
        metrics[f"{prefix}_tokens_by_group/{safe_label}"] = float(count)
    return metrics


def sanitize_metric_label(label: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in label).strip("_") or "unknown"


def gradient_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        param_norm = parameter.grad.detach().float().norm(2).item()
        total += param_norm * param_norm
    return total**0.5


def remove_decoder_parallel_grad(sae: TopKSAE) -> None:
    if sae.decoder.weight.grad is None:
        return
    with torch.no_grad():
        weight = sae.decoder.weight
        grad = sae.decoder.weight.grad
        parallel = (grad * weight).sum(dim=0, keepdim=True)
        norm_sq = (weight * weight).sum(dim=0, keepdim=True).clamp_min(1e-12)
        grad.sub_(parallel / norm_sq * weight)


def lr_for_step(base_lr: float, *, step: int, total_steps: int, warmup_steps: int, schedule: str) -> float:
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * step / warmup_steps
    if schedule == "constant":
        return base_lr
    if schedule == "cosine":
        import math

        decay_steps = max(1, total_steps - max(0, warmup_steps))
        progress = min(1.0, max(0.0, (step - max(0, warmup_steps)) / decay_steps))
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    raise ValueError("lr_schedule must be one of: constant, cosine")


def reconstruction_loss(recon: torch.Tensor, target: torch.Tensor, loss_type: str) -> torch.Tensor:
    if loss_type == "mse":
        return F.mse_loss(recon, target)
    if loss_type == "l1":
        return F.l1_loss(recon, target)
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(recon, target)
    raise ValueError("recon_loss must be one of: mse, l1, smooth_l1")
