import torch

from dgfm.models import build_velocity_model


def test_velocity_model_forward_shape() -> None:
    cfg = {
        "dataset": {"channels": 3},
        "model": {"family": "legacy_conv", "hidden_channels": 32, "time_embed_dim": 32, "num_res_blocks": 2},
    }
    model = build_velocity_model(cfg)
    x = torch.randn(4, 3, 32, 32)
    t = torch.rand(4)
    y = model(x, t)
    assert y.shape == x.shape


def test_official_velocity_unet_forward_shape() -> None:
    cfg = {
        "dataset": {"channels": 3},
        "model": {
            "family": "official_fm_unet",
            "hidden_channels": 32,
            "num_res_blocks": 1,
            "attention_resolutions": [2],
            "dropout": 0.1,
            "channel_mult": [1, 1],
            "conv_resample": False,
            "dims": 2,
            "num_classes": None,
            "use_checkpoint": False,
            "num_heads": 1,
            "num_head_channels": -1,
            "num_heads_upsample": -1,
            "use_scale_shift_norm": True,
            "resblock_updown": False,
            "use_new_attention_order": True,
            "with_fourier_features": False,
        },
    }
    model = build_velocity_model(cfg)
    x = torch.randn(2, 3, 32, 32)
    t = torch.rand(2)
    y = model(x, t, extra={})
    assert y.shape == x.shape
