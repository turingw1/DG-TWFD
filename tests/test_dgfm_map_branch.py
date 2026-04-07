import torch

from dgfm.evaluators.common import objective_mode, sample_from_model, sample_from_model_batched, solver_nfe
from dgfm.models import build_map_model
from dgfm.paths import build_path
from dgfm.targets import build_target_builder


def _map_config() -> dict:
    return {
        "experiment": {"seed": 42},
        "runtime": {"device": "cpu", "amp": False},
        "train": {"objective": "explicit_map", "skewed_timesteps": False},
        "dataset": {"channels": 3, "image_size": 32},
        "model": {
            "family": "official_map_unet",
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
            "map_conditioning_channels": 2,
            "prediction_type": "residual",
            "residual_scale_by_delta": True,
            "residual_tanh_scale": 1.0,
            "use_preconditioning": False,
            "sigma_data": 0.5,
            "sigma_min": 1.0e-3,
            "time_embed_mode": "time",
            "inner_parametrization": "no",
            "outer_parametrization": "euler",
        },
        "path": {"name": "condot"},
        "target": {"builder": "analytic_path", "loss_type": "mse", "min_time": 1e-4, "max_time": 0.999, "min_time_gap": 0.05},
        "loss": {"pixel_weight": 1.0, "perceptual_weight": 0.0, "endpoint_weight": 0.0},
    }


def test_map_model_forward_shape() -> None:
    cfg = _map_config()
    model = build_map_model(cfg)
    x_t = torch.randn(2, 3, 32, 32)
    t = torch.tensor([0.1, 0.2])
    s = torch.tensor([0.6, 0.8])
    out = model(x_t, t, s, extra={})
    assert out.shape == x_t.shape


def test_map_model_forward_shape_with_preconditioning() -> None:
    cfg = _map_config()
    cfg["model"]["use_preconditioning"] = True
    cfg["model"]["time_embed_mode"] = "log_noise"
    cfg["model"]["inner_parametrization"] = "edm"
    cfg["model"]["outer_parametrization"] = "euler"
    model = build_map_model(cfg)
    x_t = torch.randn(2, 3, 32, 32)
    t = torch.tensor([0.05, 0.25])
    s = torch.tensor([0.6, 0.8])
    out = model(x_t, t, s, extra={})
    assert out.shape == x_t.shape


def test_analytic_target_builder_outputs_ordered_times() -> None:
    cfg = _map_config()
    builder = build_target_builder(cfg)
    path = build_path(cfg)
    x_0 = torch.randn(4, 3, 32, 32)
    x_1 = torch.randn(4, 3, 32, 32)
    batch = builder.build(x_0=x_0, x_1=x_1, path=path)
    assert batch.x_t.shape == x_0.shape
    assert batch.x_s_target.shape == x_0.shape
    assert torch.all(batch.s > batch.t)
    assert torch.all(batch.s - batch.t >= 0.049)


class _ToyMap(torch.nn.Module):
    def forward(self, x_t, t, s, extra=None):
        return x_t + (s - t).view(-1, 1, 1, 1)


def test_sample_from_model_dispatches_to_map_rollout() -> None:
    cfg = _map_config()
    x_init = torch.zeros(2, 3, 4, 4)
    out = sample_from_model(config=cfg, model=_ToyMap(), x_init=x_init, step_count=4, method="map_rollout")
    assert out.shape == x_init.shape
    assert torch.allclose(out, torch.ones_like(x_init), atol=1e-6)
    assert objective_mode(cfg) == "explicit_map"
    assert solver_nfe(step_count=4, mode="explicit_map") == 4


def test_map_rollout_remains_differentiable() -> None:
    x_init = torch.zeros(1, 1, 2, 2, requires_grad=True)
    model = _ToyMap()
    out = sample_from_model(config=_map_config(), model=model, x_init=x_init, step_count=2, method="map_rollout")
    out.sum().backward()
    assert x_init.grad is not None


def test_sample_from_model_batched_matches_full_rollout() -> None:
    cfg = _map_config()
    x_init = torch.zeros(5, 1, 2, 2)
    model = _ToyMap()
    full = sample_from_model(config=cfg, model=model, x_init=x_init, step_count=4, method="map_rollout")
    chunked = sample_from_model_batched(
        config=cfg,
        model=model,
        x_init=x_init,
        step_count=4,
        method="map_rollout",
        max_batch_size=2,
        move_to_cpu=True,
    )
    assert chunked.device.type == "cpu"
    assert torch.allclose(full.cpu(), chunked, atol=1e-6)
