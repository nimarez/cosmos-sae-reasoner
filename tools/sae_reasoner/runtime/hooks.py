from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from ..sae import TopKSAE

Scope = Literal["prefill", "decode", "both"]
Mode = Literal["multiply", "clamp"]


@dataclass
class FeatureSteeringHook:
    sae: TopKSAE
    feature_id: int
    multiplier: float
    scope: Scope = "decode"
    mode: Mode = "multiply"
    call_index: int = 0

    def __call__(self, _module: torch.nn.Module, _inputs: tuple, output):
        hidden = _extract_hidden(output)
        phase = "prefill" if hidden.shape[1] > 1 and self.call_index == 0 else "decode"
        self.call_index += 1
        if self.scope != "both" and self.scope != phase:
            return output
        flat = hidden.reshape(-1, hidden.shape[-1]).float()
        sae = self.sae.to(device=flat.device)
        delta = sae.feature_delta(
            flat,
            feature_id=self.feature_id,
            multiplier=self.multiplier,
            mode=self.mode,
        ).to(dtype=hidden.dtype)
        edited = hidden + delta.reshape_as(hidden)
        return _replace_hidden(output, edited)


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

