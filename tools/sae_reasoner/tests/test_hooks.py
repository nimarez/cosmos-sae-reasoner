import torch

from tools.sae_reasoner.runtime.hooks import FeatureSteeringHook
from tools.sae_reasoner.sae import SAEConfig, TopKSAE


def test_feature_steering_hook_tuple_output_shape():
    sae = TopKSAE(SAEConfig(input_dim=6, expansion_factor=2, top_k=2))
    hook = FeatureSteeringHook(sae=sae, feature_id=0, multiplier=3.0, scope="both")
    hidden = torch.randn(1, 4, 6)
    output = hook(torch.nn.Identity(), (), (hidden,))
    assert isinstance(output, tuple)
    assert output[0].shape == hidden.shape


def test_feature_steering_hook_can_target_prefill_token_kind():
    sae = TopKSAE(SAEConfig(input_dim=6, expansion_factor=2, top_k=12))
    hook = FeatureSteeringHook(
        sae=sae,
        feature_id=0,
        multiplier=3.0,
        scope="prefill",
        token_map=[
            {"kind": "video", "role": "user"},
            {"kind": "text", "role": "user"},
            {"kind": "video", "role": "assistant"},
        ],
        token_kinds=frozenset({"video"}),
        roles=frozenset({"user"}),
    )
    hidden = torch.randn(1, 3, 6)
    output = hook(torch.nn.Identity(), (), (hidden,))

    delta = output[0] - hidden
    assert torch.count_nonzero(delta[0, 0]).item() > 0
    assert torch.count_nonzero(delta[0, 1]).item() == 0
    assert torch.count_nonzero(delta[0, 2]).item() == 0
