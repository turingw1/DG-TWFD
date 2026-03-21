import torch

from dgfm.models import build_velocity_model


def test_velocity_model_forward_shape() -> None:
    cfg = {
        "dataset": {"channels": 3},
        "model": {"hidden_channels": 32, "time_embed_dim": 32, "num_res_blocks": 2},
    }
    model = build_velocity_model(cfg)
    x = torch.randn(4, 3, 32, 32)
    t = torch.rand(4)
    y = model(x, t)
    assert y.shape == x.shape
