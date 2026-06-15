from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class SAEConfig:
    input_dim: int
    expansion_factor: int = 16
    top_k: int = 32
    normalize_decoder: bool = True

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
        nn.init.kaiming_uniform_(self.encoder.weight, a=5**0.5)
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.decoder.weight, a=5**0.5)
        self._renorm_decoder()

    @torch.no_grad()
    def _renorm_decoder(self) -> None:
        if not self.config.normalize_decoder:
            return
        weight = self.decoder.weight
        norms = weight.norm(dim=0, keepdim=True).clamp_min(1e-6)
        weight.div_(norms)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        acts = torch.relu(self.encoder(x - self.pre_bias))
        if self.config.top_k <= 0 or self.config.top_k >= acts.shape[-1]:
            return acts
        values, indexes = torch.topk(acts, k=self.config.top_k, dim=-1)
        sparse = torch.zeros_like(acts)
        sparse.scatter_(-1, indexes, values)
        return sparse

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(features) + self.post_bias

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
        mode: str = "multiply",
    ) -> torch.Tensor:
        if feature_id < 0 or feature_id >= self.config.feature_dim:
            raise IndexError(f"feature_id={feature_id} outside [0, {self.config.feature_dim})")
        features = self.encode(x)
        original = features[..., feature_id].clone()
        if mode == "multiply":
            features[..., feature_id] = original * multiplier
        elif mode == "clamp":
            features[..., feature_id] = float(multiplier)
        else:
            raise ValueError(f"unsupported steering mode {mode!r}; expected multiply or clamp")
        edited = self.decode(features)
        baseline = self.decode(self.encode(x))
        return edited - baseline


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
    config = SAEConfig(**payload["config"])
    sae = TopKSAE(config)
    sae.load_state_dict(payload["state_dict"])
    sae.eval()
    return sae


def train_sae_from_tensor(
    activations: torch.Tensor,
    *,
    expansion_factor: int = 16,
    top_k: int = 32,
    recon_loss: str = "mse",
    feature_l1_coeff: float = 0.0,
    steps: int = 1000,
    batch_size: int = 1024,
    lr: float = 3e-4,
    device: str | torch.device | None = None,
) -> tuple[TopKSAE, list[dict[str, float]]]:
    if activations.ndim != 2:
        raise ValueError(f"activations must be rank-2 [N, D], got shape {tuple(activations.shape)}")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    acts = activations.float().to(device)
    sae = TopKSAE(SAEConfig(input_dim=acts.shape[-1], expansion_factor=expansion_factor, top_k=top_k)).to(device)
    opt = torch.optim.AdamW(sae.parameters(), lr=lr)
    metrics: list[dict[str, float]] = []
    n = acts.shape[0]
    for step in range(1, steps + 1):
        idx = torch.randint(0, n, (min(batch_size, n),), device=device)
        batch = acts[idx]
        recon, features = sae(batch)
        recon_objective = reconstruction_loss(recon, batch, recon_loss)
        feature_l1 = features.abs().mean()
        loss = recon_objective + feature_l1_coeff * feature_l1
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sae._renorm_decoder()
        if step == 1 or step == steps or step % max(1, steps // 10) == 0:
            with torch.no_grad():
                l0 = (features > 0).float().sum(dim=-1).mean().item()
                metrics.append(
                    {
                        "step": float(step),
                        "loss": float(loss.item()),
                        "recon_loss": float(recon_objective.item()),
                        "feature_l1": float(feature_l1.item()),
                        "l0": float(l0),
                    }
                )
    return sae.cpu(), metrics


def reconstruction_loss(recon: torch.Tensor, target: torch.Tensor, loss_type: str) -> torch.Tensor:
    if loss_type == "mse":
        return F.mse_loss(recon, target)
    if loss_type == "l1":
        return F.l1_loss(recon, target)
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(recon, target)
    raise ValueError("recon_loss must be one of: mse, l1, smooth_l1")
