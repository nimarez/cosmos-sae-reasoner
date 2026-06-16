from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import torch

from ..sae import TopKSAE

Scope = Literal["prefill", "decode", "both"]


@dataclass
class FeatureSteeringHook:
    sae: TopKSAE
    feature_id: int
    multiplier: float
    scope: Scope = "decode"
    token_map: Sequence[dict[str, Any]] | None = None
    token_kinds: frozenset[str] = frozenset()
    roles: frozenset[str] = frozenset()
    call_index: int = 0

    def __call__(self, _module: torch.nn.Module, _inputs: tuple, output):
        hidden = _extract_hidden(output)
        phase = "prefill" if hidden.shape[1] > 1 and self.call_index == 0 else "decode"
        self.call_index += 1
        if self.scope != "both" and self.scope != phase:
            return output
        flat = hidden.reshape(-1, hidden.shape[-1]).float()
        edit_mask = self._edit_mask(phase, flat.device, flat.shape[0])
        if edit_mask is not None and not bool(edit_mask.any()):
            return output
        sae = self.sae.to(device=flat.device)
        if edit_mask is None:
            delta = sae.feature_delta(
                flat,
                feature_id=self.feature_id,
                multiplier=self.multiplier,
            ).to(dtype=hidden.dtype)
        else:
            delta = torch.zeros_like(flat, dtype=hidden.dtype)
            delta[edit_mask] = sae.feature_delta(
                flat[edit_mask],
                feature_id=self.feature_id,
                multiplier=self.multiplier,
            ).to(dtype=hidden.dtype)
        edited = hidden + delta.reshape_as(hidden)
        return _replace_hidden(output, edited)

    def _edit_mask(self, phase: str, device: torch.device, length: int) -> torch.Tensor | None:
        if not self.token_kinds and not self.roles:
            return None
        if phase != "prefill" or not self.token_map:
            return torch.zeros(length, dtype=torch.bool, device=device)
        values = []
        for token in list(self.token_map)[:length]:
            kind_ok = not self.token_kinds or str(token.get("kind") or "") in self.token_kinds
            role_ok = not self.roles or str(token.get("role") or "") in self.roles
            values.append(kind_ok and role_ok)
        values.extend([False] * max(0, length - len(values)))
        return torch.tensor(values[:length], dtype=torch.bool, device=device)


def _extract_hidden(output) -> torch.Tensor:
    if isinstance(output, tuple):
        hidden = output[0]
    elif hasattr(output, "last_hidden_state"):
        hidden = output.last_hidden_state
    else:
        hidden = output
    if not isinstance(hidden, torch.Tensor):
        raise TypeError(f"unsupported hook output type: {type(output).__name__}")
    if hidden.ndim == 2:
        hidden = hidden.unsqueeze(0)
    return hidden


def _replace_hidden(output, hidden: torch.Tensor):
    if isinstance(output, tuple):
        return (hidden,) + output[1:]
    if hasattr(output, "last_hidden_state"):
        try:
            output.last_hidden_state = hidden
            return output
        except Exception:
            return hidden
    return hidden
