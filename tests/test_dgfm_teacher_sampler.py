import torch

from dgfm.targets import build_target_builder
from dgfm.targets.pair_sampling import sample_target_pair_indices, sample_target_triplet_indices


class _BridgeMap(torch.nn.Module):
    def forward(self, x_t, t, s, extra=None):
        del extra
        return x_t + (s - t).view(-1, 1, 1, 1)


def _teacher_sampler_config() -> dict:
    return {
        "experiment": {"seed": 42},
        "runtime": {"device": "cpu", "amp": False},
        "train": {"objective": "explicit_map", "batch_size": 2, "num_workers": 0},
        "dataset": {"channels": 3, "image_size": 32, "name": "cifar10", "data_root": "/tmp"},
        "teacher": {
            "type": "sampler",
            "backend": "dummy",
            "x0_std": 1.0,
        },
        "target": {
            "builder": "teacher_sampler",
            "sampling_mode": "ctm_discrete",
            "start_scales": 9,
            "teacher_sampler_sub_batch": 1,
            "num_heun_step": 3,
            "num_heun_step_random": False,
            "heun_step_strategy": "uniform",
            "sample_s_strategy": "uniform",
            "bridge_source": "ema_model_rollout",
            "bridge_steps": 1,
            "bridge_stop_grad": True,
            "loss_type": "mse",
        },
        "loss": {"pixel_weight": 1.0, "perceptual_weight": 0.0, "endpoint_weight": 0.0},
    }


def test_teacher_sampler_target_builder_returns_teacher_pairs() -> None:
    cfg = _teacher_sampler_config()
    builder = build_target_builder(cfg)
    assert builder.uses_dataset_images is False
    batch = (torch.zeros(2, 3, 32, 32), torch.zeros(2, dtype=torch.long))
    target = builder.build_from_batch(batch, device=torch.device("cpu"), path=None)
    assert target.x_t.shape == target.x_s_target.shape == (2, 3, 32, 32)
    assert target.x_0.shape == target.x_1.shape == (2, 3, 32, 32)
    assert target.x_t_dt is not None
    assert target.t_dt is not None
    assert target.target_construction == "ctm_consistency"
    assert target.target_source == "ema_model"
    assert target.bridge_source == "teacher"
    assert torch.all(target.s > target.t)
    assert torch.all(target.t_dt > target.t)
    assert torch.all(target.s >= target.t_dt)


def test_ctm_discrete_pair_sampler_respects_fixed_num_heun_step() -> None:
    cfg = _teacher_sampler_config()
    t_indices, s_indices = sample_target_pair_indices(
        num_points=9,
        target_cfg=cfg["target"],
        batch_size=128,
        device=torch.device("cpu"),
    )
    assert torch.all(s_indices > t_indices)
    assert torch.all(s_indices - t_indices >= 3)
    assert int((s_indices - t_indices).min().item()) == 3


def test_ctm_discrete_pair_sampler_supports_smallest_s_strategy() -> None:
    cfg = _teacher_sampler_config()
    cfg["target"]["sample_s_strategy"] = "smallest"
    t_indices, s_indices = sample_target_pair_indices(
        num_points=9,
        target_cfg=cfg["target"],
        batch_size=32,
        device=torch.device("cpu"),
    )
    assert torch.all(s_indices == 8)
    assert torch.all(t_indices <= 5)


def test_ctm_discrete_triplet_sampler_respects_t_dt_bridge() -> None:
    cfg = _teacher_sampler_config()
    t_indices, t_dt_indices, s_indices = sample_target_triplet_indices(
        num_points=9,
        target_cfg=cfg["target"],
        batch_size=128,
        device=torch.device("cpu"),
    )
    assert torch.all(t_dt_indices > t_indices)
    assert torch.all(s_indices >= t_dt_indices)
    assert int((t_dt_indices - t_indices).min().item()) == 3


def test_teacher_sampler_builder_can_use_warped_time_grid() -> None:
    cfg = _teacher_sampler_config()
    cfg["scheduler"] = {"timewarp": {"enabled": True, "type": "data_dense_power@2.0"}}
    builder = build_target_builder(cfg)
    grid = builder.current_u_grid(device=torch.device("cpu"), dtype=torch.float32)
    uniform = torch.linspace(0.0, 1.0, steps=9)
    assert not torch.allclose(grid, uniform)


def test_teacher_sampler_builder_can_use_model_rollout_bridge() -> None:
    cfg = _teacher_sampler_config()
    builder = build_target_builder(cfg)
    batch = (torch.zeros(2, 3, 32, 32), torch.zeros(2, dtype=torch.long))
    target = builder.build_from_batch(
        batch,
        device=torch.device("cpu"),
        path=None,
        model=_BridgeMap(),
        target_model=_BridgeMap(),
    )
    assert target.bridge_source == "ema_model_rollout"
    assert target.x_t_dt is not None
    expected = target.x_t + (target.t_dt - target.t).view(-1, 1, 1, 1)
    assert torch.allclose(target.x_t_dt, expected, atol=1e-6)
