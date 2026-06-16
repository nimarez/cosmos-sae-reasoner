from __future__ import annotations

import time
from dataclasses import dataclass
from math import sqrt
from typing import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class SAEConfig:
    input_dim: int
    expansion_factor: int = 16
    top_k: int = 32
    normalize_decoder: bool = True
    topk_activation: str = "relu_topk"
    init_method: str = "data"
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
        self.post_bias = nn.Parameter(torch.zeros(config.input_dim))
        self.register_buffer("batch_topk_threshold", torch.tensor(0.0))
        nn.init.kaiming_uniform_(self.encoder.weight, a=5**0.5)
        self._init_encoder_bias()
        nn.init.kaiming_uniform_(self.decoder.weight, a=5**0.5)
        self._renorm_decoder()

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
        nn.init.zeros_(self.post_bias)
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
        recon = self.decoder(features) + self.post_bias
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


def save_sae(path: str, sae: TopKSAE, *, metadata: dict | None = None) -> None:
    torch.save(
        {
            "config": sae.config.__dict__,
            "state_dict": sae.state_dict(),
            "metadata": metadata or {},
        },
        path,
    )


def load_sae(path: str, map_location: str | torch.device = "cpu") -> TopKSAE:
    payload = torch.load(path, map_location=map_location)
    raw_config = dict(payload["config"])
    if "topk_activation" not in raw_config:
        raw_config["topk_activation"] = "relu_topk"
    config = SAEConfig(**raw_config)
    sae = TopKSAE(config)
    load_result = sae.load_state_dict(payload["state_dict"], strict=False)
    tolerated_missing = {"batch_topk_threshold"}
    missing = set(load_result.missing_keys) - tolerated_missing
    if missing or load_result.unexpected_keys:
        raise RuntimeError(
            f"invalid SAE checkpoint state_dict; missing={sorted(missing)} unexpected={sorted(load_result.unexpected_keys)}"
        )
    sae.eval()
    return sae


def train_sae_from_tensor(
    activations: torch.Tensor,
    *,
    validation_activations: torch.Tensor | None = None,
    token_groups: Sequence[Sequence[str]] | None = None,
    validation_token_groups: Sequence[Sequence[str]] | None = None,
    expansion_factor: int = 16,
    top_k: int = 32,
    topk_activation: str = "relu_topk",
    init_method: str = "data",
    init_blend: float = 0.8,
    activation_norm: str = "sqrt_d",
    batch_topk_momentum: float = 0.01,
    matryoshka_prefixes: Sequence[int | float | str] | str | None = None,
    matryoshka_loss_coeff: float = 1.0,
    recon_loss: str = "mse",
    feature_l1_coeff: float = 0.0,
    steps: int = 1000,
    batch_size: int = 1024,
    lr: float = 3e-4,
    warmup_steps: int = 0,
    lr_schedule: str = "constant",
    max_grad_norm: float | None = None,
    val_batch_size: int | None = None,
    device: str | torch.device | None = None,
    log_every: int | None = None,
    progress_callback: Callable[[dict[str, float]], None] | None = None,
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
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    metrics: list[dict[str, float]] = []
    n = acts.shape[0]
    log_interval = max(1, int(log_every or max(1, steps // 10)))
    start = time.time()
    for step in range(1, steps + 1):
        step_lr = lr_for_step(lr, step=step, total_steps=steps, warmup_steps=warmup_steps, schedule=lr_schedule)
        for group in opt.param_groups:
            group["lr"] = step_lr
        idx = torch.randint(0, n, (min(batch_size, n),), device=acts.device)
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
                active = features != 0
                l0 = active.float().sum(dim=-1).mean().item()
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
                    "positive_feature_frac": float((features > 0).float().mean().item()),
                    "l0": float(l0),
                    "feature_density": float(active.float().mean().item()),
                    "dead_feature_frac_batch": float((active.sum(dim=0) == 0).float().mean().item()),
                    "grad_norm": float(grad_norm),
                    "grad_clipped": float(did_clip),
                    "lr": float(step_lr),
                    "activation_norm": 1.0 if activation_norm == "sqrt_d" else 0.0,
                    "input_scale": float(input_scale),
                    "tokens_seen": float(step * min(batch_size, n)),
                    "elapsed_seconds": float(time.time() - start),
                }
                for prefix, prefix_loss in matryoshka_losses_by_prefix.items():
                    metric[f"matryoshka_recon_loss_prefix/{prefix}"] = float(prefix_loss.item())
                metric.update(group_reconstruction_metrics(recon, batch, batch_group_masks, prefix="train"))
                if val_acts is not None:
                    metric.update(
                        validation_metrics(
                            sae,
                            val_acts,
                            group_masks=val_group_masks,
                            recon_loss=recon_loss,
                            matryoshka_prefixes=resolved_matryoshka_prefixes,
                            batch_size=val_batch_size or batch_size,
                        )
                    )
                metrics.append(metric)
                if progress_callback is not None:
                    progress_callback(metric)
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
        prefix_recon = F.linear(features[:, :prefix], sae.decoder.weight[:, :prefix], sae.post_bias)
        losses[int(prefix)] = reconstruction_loss(prefix_recon, target_norm, recon_loss)
    if not losses:
        return target_norm.new_zeros(()), {}
    return sum(losses.values(), target_norm.new_zeros(())), losses


@torch.no_grad()
def validation_metrics(
    sae: TopKSAE,
    validation_activations: torch.Tensor,
    *,
    group_masks: dict[str, torch.Tensor] | None = None,
    recon_loss: str,
    batch_size: int,
    matryoshka_prefixes: Sequence[int] = (),
) -> dict[str, float]:
    n = validation_activations.shape[0]
    if n <= 0:
        return {}
    sample_n = min(max(1, batch_size), n)
    device = next(sae.parameters()).device
    if sample_n < n:
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
    active = features != 0
    metrics = {
        "val_recon_loss": float(val_recon.item()),
        "val_matryoshka_loss": float(val_matryoshka.item()),
        "val_mse": float(val_mse.item()),
        "val_normalized_mse": float(F.mse_loss(recon_norm, target_norm).item()),
        "val_explained_variance": float(1.0 - (residual_ss / centered_ss).item()),
        "val_l0": float(active.float().sum(dim=-1).mean().item()),
        "val_positive_feature_frac": float((features > 0).float().mean().item()),
        "val_dead_feature_frac_batch": float((active.sum(dim=0) == 0).float().mean().item()),
        "val_tokens": float(n),
        "val_sample_tokens": float(sample_n),
    }
    for prefix, prefix_loss in val_matryoshka_by_prefix.items():
        metrics[f"val_matryoshka_recon_loss_prefix/{prefix}"] = float(prefix_loss.item())
    metrics.update(group_reconstruction_metrics(recon, batch, sampled_group_masks, prefix="val"))
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
