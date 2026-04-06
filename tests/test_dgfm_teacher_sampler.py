import torch

from dgfm.targets import build_target_builder


def _teacher_sampler_config() -> dict:
    return {
        "experiment": {"seed": 42},
        "runtime": {"device": "cpu", "amp": False},
        "train": {"objective": "explicit_map", "batch_size": 2, "num_workers": 0},
        "dataset": {"channels": 3, "image_size": 32, "name": "cifar10", "data_root": "/tmp"},
        "teacher": {
            "type": "sampler",
            "backend": "dummy",
            "retain_num_points": 9,
            "x0_std": 1.0,
        },
        "target": {
            "builder": "teacher_sampler",
            "teacher_sampler_sub_batch": 1,
            "pair_short_max": 2,
            "pair_mid_max": 4,
            "pair_long_max": 8,
            "pair_short_weight": 1.0,
            "pair_mid_weight": 0.0,
            "pair_long_weight": 0.0,
            "pair_endpoint_weight": 0.0,
            "high_noise_t_weight": 1.0,
            "high_noise_t_fraction": 0.5,
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
