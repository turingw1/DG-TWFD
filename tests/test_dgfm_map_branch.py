import pytest
import torch

from dgfm.evaluators.common import (
    load_timewarp_from_checkpoint,
    objective_mode,
    sample_from_model,
    sample_from_model_batched,
    solver_nfe,
)
from dgfm.models import build_map_model
from dgfm.paths import build_path
from dgfm.schedulers import TimeWarpMonotone
from dgfm.targets import build_target_builder
from dgtd.sample_dgtd import load_time_grid_from_schedule, save_schedule, search_oss_time_grid


def _map_config() -> dict:
    return {
        "experiment": {"seed": 42},
        "runtime": {"device": "cpu", "amp": False},
        "train": {"objective": "explicit_map", "skewed_timesteps": False},
        "dataset": {"channels": 3, "image_size": 32},
        "model": {
            "family": "local_map_resnet",
            "hidden_channels": 32,
            "num_res_blocks": 1,
            "time_embed_dim": 32,
            "cond_dim": 32,
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
    try:
        path = build_path(cfg)
    except ModuleNotFoundError as exc:
        pytest.skip(f"official flow_matching path module is not vendored: {exc}")
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


class _ToyMapUsesS(torch.nn.Module):
    def forward(self, x_t, t, s, extra=None):
        return x_t + s.view(-1, 1, 1, 1)


def test_sample_from_model_dispatches_to_map_rollout() -> None:
    cfg = _map_config()
    x_init = torch.zeros(2, 3, 4, 4)
    out = sample_from_model(config=cfg, model=_ToyMap(), x_init=x_init, step_count=4, method="map_rollout")
    assert out.shape == x_init.shape
    assert torch.allclose(out, torch.ones_like(x_init), atol=1e-6)
    assert objective_mode(cfg) == "explicit_map"
    assert solver_nfe(step_count=4, mode="explicit_map") == 4


def test_sample_from_model_uses_config_timewarp_when_enabled() -> None:
    cfg = _map_config()
    cfg["scheduler"] = {"timewarp": {"enabled": True, "type": "data_dense_power@2.0"}}
    x_init = torch.zeros(1, 1, 1, 1)
    out = sample_from_model(config=cfg, model=_ToyMapUsesS(), x_init=x_init, step_count=2, method="map_rollout")
    assert torch.allclose(out, torch.full_like(x_init, 1.75), atol=1e-6)


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


def test_load_timewarp_from_checkpoint_restores_learned_state(tmp_path) -> None:
    cfg = _map_config()
    cfg["scheduler"] = {"timewarp": {"enabled": True, "type": "learnable_monotone", "num_bins": 8}}
    checkpoint = tmp_path / "ckpt.pt"
    timewarp = TimeWarpMonotone(num_bins=8, init_bias=0.0)
    timewarp.logits.data.copy_(torch.linspace(-2.0, 2.0, steps=8))
    torch.save({"timewarp": timewarp.state_dict()}, checkpoint)
    restored = load_timewarp_from_checkpoint(cfg, checkpoint, device=torch.device("cpu"))
    assert restored is not None
    linear = torch.linspace(0.0, 1.0, steps=5)
    warped = restored(linear)
    assert not torch.allclose(warped, linear)


def test_oss_schedule_search_exports_valid_grid(tmp_path) -> None:
    x_search = torch.zeros(3, 1, 2, 2)
    time_grid, payload = search_oss_time_grid(
        model=_ToyMap(),
        x_search=x_search,
        step_count=2,
        reference_steps=4,
        cost_batch_size=2,
    )
    assert payload["status"] == "ok"
    assert payload["step_count"] == 2
    assert time_grid.shape == (3,)
    assert torch.all(time_grid[1:] >= time_grid[:-1])
    assert time_grid[0].item() == pytest.approx(0.0)
    assert time_grid[-1].item() == pytest.approx(1.0)

    schedule_path = tmp_path / "oss_schedule_steps2.json"
    save_schedule(payload, schedule_path)
    restored, restored_payload = load_time_grid_from_schedule(
        schedule_path,
        step_count=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert restored_payload["mode"] == "oss_schedule_search"
    assert torch.allclose(restored, time_grid.cpu(), atol=1e-6)
