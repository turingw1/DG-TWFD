from __future__ import annotations

import resource

import torch

from dg_twfd.config import load_config
from dg_twfd.data.dataloader import build_dataloader
from dg_twfd.data.dataset import TrajectoryPairDataset
from dg_twfd.data.teacher import DummyTeacherTrajectory
from dg_twfd.losses import BoundaryLoss, MatchLoss, SemigroupDefectLoss, WarpLoss
from dg_twfd.models import BoundaryCorrector, FlowStudent, TimeWarpMonotone
from dg_twfd.schedule import DefectAdaptiveScheduler
from dg_twfd.utils.seed import seed_everything


def _report_memory() -> None:
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"CUDA peak memory: {peak:.2f} MiB")
    else:
        usage_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(f"Approx RAM usage: {usage_kb / 1024:.2f} MiB")


def test_phase3_losses_backward() -> None:
    cfg = load_config("debug_4060")
    seed_everything(cfg.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher = DummyTeacherTrajectory(cfg)
    dataloader = build_dataloader(cfg, teacher, split="train")
    dataset = TrajectoryPairDataset(cfg, teacher, split="train", cached=False, device=device)
    batch = next(iter(dataloader))

    x_t = batch["x_t"].to(device)
    x_s = batch["x_s"].to(device)
    t = batch["t"].to(device)
    s = batch["s"].to(device)

    timewarp = TimeWarpMonotone(
        num_bins=cfg.model.timewarp_num_bins,
        init_bias=cfg.model.timewarp_init_bias,
    ).to(device)
    student = FlowStudent(
        channels=cfg.data.channels,
        hidden_channels=cfg.model.hidden_channels,
        time_embed_dim=cfg.model.time_embed_dim,
        cond_dim=cfg.model.cond_dim,
        num_blocks=cfg.model.student_num_blocks,
        predict_residual=cfg.model.predict_residual,
    ).to(device)
    boundary = BoundaryCorrector(
        channels=cfg.data.channels,
        hidden_channels=cfg.model.boundary_hidden_channels,
        num_blocks=cfg.model.boundary_num_blocks,
    ).to(device)

    match_loss_fn = MatchLoss(cfg.loss.match_loss_type, cfg.loss.huber_delta)
    defect_loss_fn = SemigroupDefectLoss(cfg.loss.per_pixel_mean)
    warp_loss_fn = WarpLoss(cfg.loss.per_pixel_mean)
    boundary_loss_fn = BoundaryLoss(cfg.loss.match_loss_type, cfg.loss.huber_delta)
    defect_scheduler = DefectAdaptiveScheduler(
        num_bins=cfg.schedule.num_bins,
        ema_decay=cfg.schedule.ema_decay,
        eta=cfg.schedule.eta,
        eps=cfg.schedule.eps,
        seed=cfg.schedule.seed,
    )

    params = list(student.parameters()) + list(timewarp.parameters()) + list(boundary.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)
    optimizer.zero_grad(set_to_none=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    autocast_enabled = cfg.runtime.amp and device.type == "cuda"
    with torch.autocast(device_type=device.type, enabled=autocast_enabled):
        x_s_pred = student(x_t, t, s)
        match_loss = match_loss_fn(x_s_pred, x_s)

        defect_loss, u, defect_per_sample = defect_loss_fn(
            student=student,
            x_t=x_t,
            t=t,
            s=s,
            scheduler=defect_scheduler,
        )
        defect_scheduler.update(t, defect_per_sample.detach())

        triplet_batch = warp_loss_fn.sample_triplet_batch(
            dataset=dataset,
            batch_size=cfg.data.batch_size,
            device=device,
        )
        warp_loss, warp_stats = warp_loss_fn(timewarp, triplet_batch)

        x_boundary = teacher.sample_x0(cfg.data.batch_size, device)
        t_max = torch.ones(cfg.data.batch_size, device=device)
        t_prev = torch.full((cfg.data.batch_size,), 0.9, device=device)
        x_boundary_target = teacher.forward_map(x_boundary, t_max, t_prev)
        boundary_loss, _ = boundary_loss_fn(
            boundary_model=boundary,
            x_boundary=x_boundary,
            target=x_boundary_target,
            gate_weight=cfg.boundary.gate_weight,
            enabled=True,
        )

        total_loss = (
            match_loss
            + cfg.loss.defect_weight * defect_loss
            + cfg.loss.warp_weight * warp_loss
            + cfg.loss.boundary_weight * boundary_loss
        )

    total_loss.backward()

    student_has_grad = any(param.grad is not None for param in student.parameters())
    timewarp_has_grad = any(param.grad is not None for param in timewarp.parameters())
    assert student_has_grad
    assert timewarp_has_grad
    assert torch.isfinite(total_loss.detach())
    assert torch.isfinite(warp_stats["base_loss"])
    assert torch.isfinite(defect_loss.detach())
    assert torch.all(u <= s)

    for module in (student, timewarp, boundary):
        for param in module.parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all()

    optimizer.step()
    print(f"total_loss: {total_loss.item():.6f}")
    print(f"warp balance: {warp_stats['balance'].item():.6f}")
    print("warp loss uses teacher finite differences and model composition only; no JVP.")
    _report_memory()
