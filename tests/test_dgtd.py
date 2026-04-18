from pathlib import Path

import torch

from dgfm.config import load_experiment_config, resolve_run_roots
from dgfm.distributed import DistributedContext
from dgfm.evaluators import build_evaluator
from dgfm.evaluators.common import load_timewarp_from_checkpoint, objective_mode
from dgfm.trainers import build_trainer
from dgtd import (
    MonotoneDensityWarp,
    TrajectoryCacheDataset,
    build_mode_a_time_grid,
    build_sigma_schedule,
    interpolate_curvature,
    interpolate_state,
)
from dgtd.defect import build_target_density
from dgtd.teacher import CONTINUATION_SOURCE_CACHED_AFFINE, build_teacher_adapter
from dgtd.train_dgtd import DGTDTrainer


def _dgtd_config(tmp_path: Path) -> dict:
    return {
        "experiment": {"name": "dgtd_test", "seed": 42},
        "runtime": {"device": "cpu", "amp": False},
        "train": {
            "objective": "dgtd_map",
            "batch_size": 1,
            "num_workers": 0,
            "pin_memory": False,
        },
        "dataset": {"channels": 1, "image_size": 2, "name": "toy", "data_root": str(tmp_path)},
        "model": {
            "family": "official_map_unet",
            "hidden_channels": 32,
            "num_res_blocks": 1,
            "attention_resolutions": [2],
            "dropout": 0.0,
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
        "scheduler": {
            "timewarp": {
                "enabled": True,
                "type": "dgtd_density",
                "num_bins": 8,
                "init": "uniform",
                "eps": 1.0e-6,
            }
        },
        "target": {
            "shard_root": str(tmp_path / "traj"),
            "trajectory_file_glob": "*.pt",
            "cache_limit": 1,
        },
        "teacher": {"type": "none"},
        "eval": {"metrics": ["fid"]},
        "dgtd": {
            "disable_online_teacher": True,
            "symmetric_residual": True,
            "teacher_continuation_mode": "affine_fallback",
            "sigma_mode": "linear_1mt",
            "qd_use_hf_bar": False,
            "qd_hf_weight": 0.0,
            "qd_ratio_cap": 0.0,
        },
    }


def test_dgtd_config_loads() -> None:
    cfg = load_experiment_config("configs/experiment/dgtd_cifar10_v3.yaml")
    assert cfg["train"]["objective"] == "dgtd_map"
    assert cfg["scheduler"]["timewarp"]["type"] == "dgtd_density"
    assert cfg["target"]["builder"] == "trajectory_shard"
    assert cfg["dgtd"]["disable_online_teacher"] is True
    assert cfg["dgtd"]["symmetric_residual"] is True
    assert cfg["dgtd"]["teacher_continuation_mode"] == "affine_fallback"
    assert cfg["dgtd"]["sigma_mode"] == "linear_1mt"
    assert cfg["dgtd"]["qd_use_hf_bar"] is False

    no_warp = load_experiment_config("configs/experiment/dgtd_cifar10_v3_ablation_no_warp.yaml")
    assert no_warp["dgtd"]["disable_warp"] is True
    assert no_warp["dgtd"]["uniform_time"] is True

    no_hf = load_experiment_config("configs/experiment/dgtd_cifar10_v3_ablation_warp_no_hf.yaml")
    assert no_hf["dgtd"]["disable_hf_metric"] is True
    assert no_hf["dgtd"]["lambda_hf_max"] == 0.0


def test_dgtd_dispatch_uses_map_mode_and_trainer(tmp_path: Path) -> None:
    cfg = _dgtd_config(tmp_path)
    roots = resolve_run_roots(tmp_path / "run")
    trainer = build_trainer(cfg, roots=roots, dist_ctx=DistributedContext(enabled=False))
    evaluator = build_evaluator(cfg, checkpoint=tmp_path / "ckpt.pt", eval_root=tmp_path / "eval")
    assert objective_mode(cfg) == "explicit_map"
    assert isinstance(trainer, DGTDTrainer)
    assert evaluator.__class__.__name__ == "MapEvaluationRunner"


def test_dgtd_warp_roundtrip_and_triplets() -> None:
    warp = MonotoneDensityWarp(num_bins=16, init="logit_normal")
    t = torch.linspace(0.0, 1.0, steps=17)
    r = warp.t_to_r(t)
    restored = warp.r_to_t(r)
    assert torch.all(r[1:] >= r[:-1])
    assert torch.allclose(restored, t, atol=5e-2)

    r_t, r_s, r_u = warp.sample_triplets(batch_size=128, device=torch.device("cpu"))
    assert torch.all(r_t < r_s)
    assert torch.all(r_s < r_u)


def test_dgtd_timewarp_checkpoint_restore(tmp_path: Path) -> None:
    cfg = _dgtd_config(tmp_path)
    checkpoint = tmp_path / "ckpt.pt"
    warp = MonotoneDensityWarp(num_bins=8, init="uniform")
    warp.density_raw.data.copy_(torch.linspace(-1.0, 1.0, steps=8))
    torch.save({"timewarp": warp.state_dict()}, checkpoint)
    restored = load_timewarp_from_checkpoint(cfg, checkpoint, device=torch.device("cpu"))
    assert restored is not None
    grid = build_mode_a_time_grid(warp=restored, step_count=4, device=torch.device("cpu"), dtype=torch.float32)
    assert torch.all(grid[1:] >= grid[:-1])
    assert not torch.allclose(grid, torch.linspace(0.0, 1.0, steps=5))


def test_trajectory_cache_dataset_and_interpolation(tmp_path: Path) -> None:
    root = tmp_path / "traj"
    root.mkdir(parents=True)
    sample = {
        "t_grid": torch.tensor([0.0, 0.5, 1.0]),
        "x_grid": torch.tensor(
            [
                [[[0.0, 0.0], [0.0, 0.0]]],
                [[[1.0, 1.0], [1.0, 1.0]]],
                [[[2.0, 2.0], [2.0, 2.0]]],
            ]
        ),
        "label": 3,
    }
    torch.save([sample], root / "toy.pt")
    cfg = _dgtd_config(tmp_path)
    dataset = TrajectoryCacheDataset(cfg, split="train")
    item = dataset[0]
    x_mid = interpolate_state(item, torch.tensor([0.25]))
    k_mid = interpolate_curvature(item, torch.tensor([0.5]))
    assert item["cond"].item() == 3
    assert item["states"].shape == (3, 1, 2, 2)
    assert torch.allclose(x_mid, torch.full((1, 1, 2, 2), 0.5))
    assert k_mid.shape == (1,)


def test_sigma_schedule_and_affine_teacher_fallback(tmp_path: Path) -> None:
    root = tmp_path / "traj"
    root.mkdir(parents=True)
    sample = {
        "t_grid": torch.tensor([0.0, 0.5, 1.0]),
        "x_grid": torch.tensor(
            [
                [[[0.0, 0.0], [0.0, 0.0]]],
                [[[1.0, 1.0], [1.0, 1.0]]],
                [[[2.0, 2.0], [2.0, 2.0]]],
            ]
        ),
    }
    torch.save([sample], root / "toy.pt")
    cfg = _dgtd_config(tmp_path)
    schedule = build_sigma_schedule(cfg)
    t = torch.tensor([0.25, 0.75])
    sigma = schedule.sigma(t)
    assert torch.allclose(sigma, torch.tensor([0.75, 0.25]), atol=1.0e-6)

    adapter = build_teacher_adapter(cfg)
    dataset = TrajectoryCacheDataset(cfg, split="train")
    item = dataset[0]
    s = torch.tensor([0.5])
    u = torch.tensor([1.0])
    z = torch.full((1, 1, 2, 2), 1.4, requires_grad=True)
    teacher_info = adapter.local_flow(item, s, u, z)
    assert teacher_info.continuation is not None
    assert int(teacher_info.source_ids.item()) == CONTINUATION_SOURCE_CACHED_AFFINE
    teacher_info.continuation.sum().backward()
    assert z.grad is not None
    assert torch.all(z.grad > 0)


def test_target_density_can_disable_hf_bar_and_cap_ratio() -> None:
    D_bar = torch.tensor([1.0, 3.0, 2.0, 1.0])
    K_bar = torch.tensor([1.0, 1.0, 1.0, 1.0])
    HF_bar = torch.tensor([100.0, 0.0, 0.0, 0.0])
    q_base = torch.full((4,), 0.25)
    q_no_hf = build_target_density(
        D_bar,
        K_bar,
        HF_bar,
        q_base,
        beta=1.0,
        eps=1.0e-6,
        curvature_weight=0.25,
        use_hf_bar=False,
        hf_weight=0.0,
        ratio_cap=0.0,
    )
    q_with_cap = build_target_density(
        D_bar,
        K_bar,
        HF_bar,
        q_base,
        beta=1.0,
        eps=1.0e-6,
        curvature_weight=0.25,
        use_hf_bar=False,
        hf_weight=0.0,
        ratio_cap=1.25,
    )
    assert torch.isclose(q_no_hf.sum(), torch.tensor(1.0))
    assert torch.isclose(q_with_cap.sum(), torch.tensor(1.0))
    assert float(torch.max(q_with_cap / q_base).item()) <= 1.25 + 1.0e-5
