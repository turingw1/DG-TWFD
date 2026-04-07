import torch

from dgfm.targets import build_target_builder
from dgfm.targets.pair_sampling import sample_target_pair_indices


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
            "loss_type": "mse",
        },
        "loss": {"pixel_weight": 1.0, "perceptual_weight": 0.0, "endpoint_weight": 0.0},
    }


def test_teacher_sampler_target_builder_returns_teacher_pairs() -> None:
    cfg = _teacher_sampler_config()
    builder = build_target_builder(cfg)
    batch = (torch.zeros(2, 3, 32, 32), torch.zeros(2, dtype=torch.long))
    target = builder.build_from_batch(batch, device=torch.device("cpu"), path=None)
    assert target.x_t.shape == target.x_s_target.shape == (2, 3, 32, 32)
    assert target.x_0.shape == target.x_1.shape == (2, 3, 32, 32)
    assert torch.all(target.s > target.t)


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


def test_teacher_sampler_builder_can_use_warped_time_grid() -> None:
    cfg = _teacher_sampler_config()
    cfg["scheduler"] = {"timewarp": {"enabled": True, "type": "data_dense_power@2.0"}}
    builder = build_target_builder(cfg)
    uniform = torch.linspace(0.0, 1.0, steps=9)
    assert not torch.allclose(builder.u_grid, uniform)
