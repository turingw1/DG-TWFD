from __future__ import annotations

import resource

import torch

from dg_twfd.config import load_config
from dg_twfd.data.dataloader import build_dataloader
from dg_twfd.data.teacher import DummyTeacherTrajectory
from dg_twfd.utils.seed import seed_everything


def _report_memory() -> None:
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"CUDA max_memory_allocated: {peak:.2f} MiB")
    else:
        usage_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(f"Approx RAM usage: {usage_kb / 1024:.2f} MiB")


def test_debug_profile_data_pipeline() -> None:
    cfg = load_config("debug_4060")
    seed_everything(cfg.experiment.seed)
    teacher = DummyTeacherTrajectory(cfg)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        x0 = teacher.sample_x0(cfg.data.batch_size, "cuda")
        t = torch.full((cfg.data.batch_size,), 0.8, device="cuda")
        s = torch.full((cfg.data.batch_size,), 0.2, device="cuda")
        with torch.autocast(device_type="cuda", enabled=cfg.runtime.amp):
            x_s = teacher.forward_map(x0, t, s)
        assert x_s.device.type == "cuda"
        assert x_s.shape == x0.shape
        assert torch.cuda.max_memory_allocated() > 0

    dataloader = build_dataloader(cfg, teacher, split="train")

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    batches = []
    autocast_enabled = cfg.runtime.amp and device_type == "cuda"
    with torch.autocast(device_type=device_type, enabled=autocast_enabled):
        for idx, batch in enumerate(dataloader):
            batches.append(batch)
            assert batch["x_t"].shape == batch["x_s"].shape
            assert batch["x_t"].shape == (
                cfg.data.batch_size,
                cfg.data.channels,
                cfg.data.image_size,
                cfg.data.image_size,
            )
            assert batch["t"].shape == batch["s"].shape == (cfg.data.batch_size,)
            assert torch.all(batch["t"] > batch["s"])
            assert batch["x_t"].dtype == torch.float32
            if idx == 1:
                break

    assert len(batches) == 2
    _report_memory()
