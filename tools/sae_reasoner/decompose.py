"""Unsupervised linear decompositions (PCA / ICA) of the residual stream.

A complement to the SAE: where the SAE learns an overcomplete, sparse, hopefully-monosemantic
basis, PCA and ICA give label-free *linear* bases.

- **PCA** finds the orthogonal directions of maximum variance. Cheap and exact: we never form an
  ``N x H`` SVD, only stream the ``H x H`` covariance and eigendecompose it (``H`` is a few
  thousand). Answers "how much structure is just a handful of top-variance directions?".
- **ICA** (FastICA on a subsample) finds statistically-independent, non-Gaussian directions, which
  are historically closer to interpretable "features" than PCA's variance axes.

The result is a :class:`LinearBasis` whose ``encode(x) = (x - mean) @ components.T`` (the signed
projection onto each direction) is the analogue of an SAE feature activation. Because it exposes
``encode`` / ``parameters`` / ``eval`` / ``config.feature_dim`` exactly like the SAE, it is a
drop-in for the existing feature browser (``collect_top_feature_records``) and the probe comparison
(``assemble_sae_feature_matrix``) with no downstream changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from .storage import load_torch_uri, save_torch_uri


@dataclass(frozen=True)
class DecompositionConfig:
    method: str  # "pca" | "ica"
    input_dim: int
    n_components: int
    whiten: bool = False  # divide each projection by the component's std (unit-variance sources)
    seed: int = 0
    max_samples: int = 50000  # ICA subsample cap (ICA is not cheaply streamable)

    @property
    def feature_dim(self) -> int:
        return self.n_components


class LinearBasis(nn.Module):
    """A fitted linear basis with an SAE-compatible ``encode`` surface.

    ``components`` are stored as a non-trainable parameter (so ``parameters()`` is non-empty and
    ``.to(device)`` moves the basis, matching how callers detect the SAE's device).
    """

    def __init__(
        self,
        config: DecompositionConfig,
        components: torch.Tensor,
        mean: torch.Tensor,
        *,
        component_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.components = nn.Parameter(components.to(torch.float32), requires_grad=False)
        self.mean = nn.Parameter(mean.to(torch.float32), requires_grad=False)
        std = component_std if component_std is not None else torch.ones(components.shape[0])
        self.component_std = nn.Parameter(std.to(torch.float32), requires_grad=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        centered = x.to(self.components.dtype) - self.mean
        proj = centered @ self.components.t()
        if self.config.whiten:
            proj = proj / self.component_std.clamp_min(1e-8)
        return proj

    # Mirror the small slice of the SAE surface that decode-side helpers may touch.
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        if self.config.whiten:
            features = features * self.component_std
        return features @ self.components + self.mean


def _iter_batches(acts: torch.Tensor, batch_size: int):
    for start in range(0, acts.shape[0], batch_size):
        yield acts[start : start + batch_size].to(dtype=torch.float32)


def fit_pca(acts: torch.Tensor, *, n_components: int, batch_size: int = 4096, whiten: bool = False) -> tuple[LinearBasis, dict[str, Any]]:
    """Streaming, exact PCA via the covariance eigendecomposition.

    Two passes over ``acts``: accumulate the mean, then the ``H x H`` covariance, then
    ``eigh``. Returns the basis plus a stats dict (``explained_variance_ratio`` over the kept
    components, cumulative variance, and the participation ratio over the full spectrum).
    """
    num_tokens, input_dim = int(acts.shape[0]), int(acts.shape[-1])
    if num_tokens < 2:
        raise ValueError(f"PCA needs at least 2 tokens, found {num_tokens}")
    n_components = min(n_components, input_dim, num_tokens)

    mean = torch.zeros(input_dim, dtype=torch.float64)
    for batch in _iter_batches(acts, batch_size):
        mean += batch.to(torch.float64).sum(dim=0)
    mean /= num_tokens

    cov = torch.zeros(input_dim, input_dim, dtype=torch.float64)
    for batch in _iter_batches(acts, batch_size):
        centered = batch.to(torch.float64) - mean
        cov += centered.t() @ centered
    cov /= num_tokens - 1

    # eigh returns ascending eigenvalues; flip to descending and clamp tiny negatives from roundoff.
    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals = eigvals.flip(0).clamp_min(0.0)
    eigvecs = eigvecs.flip(1)
    total_variance = float(eigvals.sum().item())

    components = eigvecs[:, :n_components].t().contiguous()  # [n_components, input_dim]
    component_std = eigvals[:n_components].clamp_min(0.0).sqrt()
    kept_ratio = (eigvals[:n_components] / max(total_variance, 1e-12)).to(torch.float64)

    config = DecompositionConfig(method="pca", input_dim=input_dim, n_components=n_components, whiten=whiten)
    basis = LinearBasis(config, components.to(torch.float32), mean.to(torch.float32), component_std=component_std.to(torch.float32))
    stats = {
        "method": "pca",
        "n_components": n_components,
        "input_dim": input_dim,
        "n_tokens": num_tokens,
        "total_variance": total_variance,
        "explained_variance_ratio": [float(v) for v in kept_ratio.tolist()],
        "cumulative_variance_ratio": float(kept_ratio.sum().item()),
        "participation_ratio": _participation_ratio(eigvals),
    }
    return basis, stats


def fit_ica(acts: torch.Tensor, *, n_components: int, seed: int = 0, max_samples: int = 50000, batch_size: int = 4096) -> tuple[LinearBasis, dict[str, Any]]:
    """FastICA on a random subsample (ICA is not cheaply streamable).

    sklearn's FastICA internally PCA-whitens to ``n_components`` first, then finds the unmixing
    directions. ``components_`` and ``mean_`` give ``sources = (X - mean) @ components_.T``, so we
    wrap them directly in a non-whitening :class:`LinearBasis`.
    """
    import numpy as np
    from sklearn.decomposition import FastICA

    num_tokens, input_dim = int(acts.shape[0]), int(acts.shape[-1])
    n_components = min(n_components, input_dim, num_tokens)

    generator = torch.Generator().manual_seed(seed)
    if num_tokens > max_samples:
        idx = torch.randperm(num_tokens, generator=generator)[:max_samples]
        sample = acts.index_select(0, idx)
    else:
        sample = acts
    # Single conversion to the dtype FastICA works in; .cpu() keeps it robust to GPU-resident inputs.
    sample_np = sample.detach().cpu().to(torch.float64).numpy()

    ica = FastICA(n_components=n_components, random_state=seed, whiten="unit-variance", max_iter=1000)
    ica.fit(sample_np)

    components = torch.from_numpy(np.asarray(ica.components_, dtype=np.float32))
    mean = torch.from_numpy(np.asarray(ica.mean_, dtype=np.float32))
    config = DecompositionConfig(method="ica", input_dim=input_dim, n_components=n_components, whiten=False, seed=seed, max_samples=max_samples)
    basis = LinearBasis(config, components, mean)
    stats = {
        "method": "ica",
        "n_components": n_components,
        "input_dim": input_dim,
        "n_tokens": num_tokens,
        "n_samples_used": int(sample.shape[0]),
        "n_iter": int(getattr(ica, "n_iter_", 0) or 0),
        "explained_variance_ratio": None,  # not defined for ICA
    }
    return basis, stats


def _participation_ratio(eigvals: torch.Tensor) -> float:
    """Effective number of dimensions: ``(sum λ)^2 / sum(λ^2)``. ~1 if one direction dominates."""
    s1 = float(eigvals.sum().item())
    s2 = float((eigvals * eigvals).sum().item())
    if s2 <= 0:
        return 0.0
    return s1 * s1 / s2


def save_basis(path: str, basis: LinearBasis, *, stats: dict[str, Any] | None = None, metadata: dict | None = None) -> str:
    return save_torch_uri(
        path,
        {
            "config": basis.config.__dict__,
            "components": basis.components.detach().cpu(),
            "mean": basis.mean.detach().cpu(),
            "component_std": basis.component_std.detach().cpu(),
            "stats": stats or {},
            "metadata": metadata or {},
        },
    )


def load_basis(path: str, map_location: str | torch.device = "cpu") -> LinearBasis:
    payload = load_torch_uri(path, map_location=str(map_location))
    config = DecompositionConfig(**dict(payload["config"]))
    basis = LinearBasis(
        config,
        payload["components"],
        payload["mean"],
        component_std=payload.get("component_std"),
    )
    basis.eval()
    return basis
