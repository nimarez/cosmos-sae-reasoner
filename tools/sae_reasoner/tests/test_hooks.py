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

