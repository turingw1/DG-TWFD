from __future__ import annotations

import argparse
import copy
import json
import signal
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.edm_map_lib import (
    build_cifar10_loader,
    clone_student_from_teacher,
    init_warp,
    load_config,
    load_edm_network,
    make_labels,
    sample_labels,
    sample_triplet_u,
    set_seed,
    sigma_from_u,
    student_transition,
    teacher_rollout_transition,
    teacher_transition,
    update_warp_from_defect,
    write_json,
)
from dgfm.losses.perceptual import MultiScaleL1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an isolated EDM-first continuous map student.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def _device(cfg: dict) -> torch.device:
    requested = str(cfg.get("runtime", {}).get("device", "cuda"))
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _flat_mse_per_sample(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x - y).flatten(1).square().mean(dim=1)


def _safe_mean_normalized(numer: torch.Tensor, denom: torch.Tensor) -> torch.Tensor:
    return (numer / denom.clamp_min(1.0e-6)).mean()


def _copy_state_to_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def _update_ema_model(ema_model: torch.nn.Module, student: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        student_state = student.state_dict()
        ema_state = ema_model.state_dict()
        for key, ema_value in ema_state.items():
            source = student_state[key].detach().to(device=ema_value.device)
            if torch.is_floating_point(ema_value):
                ema_value.mul_(decay).add_(source.to(dtype=ema_value.dtype), alpha=1.0 - decay)
            else:
                ema_value.copy_(source)


def _adaptive_loss_weight(
    *,
    base_weight: float,
    reference_loss: torch.Tensor,
    aux_loss: torch.Tensor,
    min_scale: float,
    max_scale: float,
) -> torch.Tensor:
    if base_weight <= 0.0:
        return aux_loss.new_tensor(0.0)
    scale = (reference_loss.detach() / aux_loss.detach().clamp_min(1.0e-6)).clamp(
        min=float(min_scale),
        max=float(max_scale),
    )
    return aux_loss.new_tensor(float(base_weight)) * scale


def _next_cifar_batch(data_iter, loader):
    try:
        return next(data_iter), data_iter
    except StopIteration:
        data_iter = iter(loader)
        return next(data_iter), data_iter


def _warp_snapshot_payload(*, step: int, warp, q_base, q_target, warp_stats) -> dict:
    payload = {"step": int(step)}
    if warp is not None:
        density = warp.density().detach().float().cpu()
        payload["density"] = [float(x) for x in density.tolist()]
        payload["entropy"] = float((-(density * torch.log(torch.clamp(density, min=1.0e-6))).sum()).item())
        if q_base is not None:
            base = q_base.detach().float().cpu()
            payload["q_base"] = [float(x) for x in base.tolist()]
            payload["max_qphi_over_qbase"] = float((density / torch.clamp(base, min=1.0e-6)).max().item())
    if q_target is not None:
        payload["q_target"] = [float(x) for x in q_target.detach().float().cpu().tolist()]
    if warp_stats is not None:
        payload["D_bar"] = [float(x) for x in warp_stats["D_bar"].detach().float().cpu().tolist()]
        payload["D_count"] = [float(x) for x in warp_stats["D_count"].detach().float().cpu().tolist()]
    return payload


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_root = Path(args.run_root)
    log_dir = run_root / "logs"
    ckpt_dir = run_root / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, log_dir / "config.yaml")
    write_json(log_dir / "config_resolved.json", cfg)

    seed = int(cfg.get("runtime", {}).get("seed", 42))
    set_seed(seed)
    device = _device(cfg)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    train_cfg = cfg.get("train", {})
    teacher_cfg = cfg.get("teacher", {})
    runtime_cfg = cfg.get("runtime", {})
    max_steps = int(train_cfg.get("max_steps", 1000))
    max_seconds = float(train_cfg.get("max_seconds", 0.0) or 0.0)
    log_every = max(1, int(train_cfg.get("log_every", 50)))
    save_every = max(1, int(train_cfg.get("save_every", 500)))
    sigma_min = float(train_cfg.get("sigma_min", 0.002))
    sigma_max = float(train_cfg.get("sigma_max", 80.0))
    rho = float(train_cfg.get("rho", 7.0))
    match_weight = float(train_cfg.get("match_weight", 1.0))
    defect_weight = float(train_cfg.get("defect_weight", 0.05))
    anchor_weight = float(train_cfg.get("anchor_weight", 0.25))
    endpoint_weight = float(train_cfg.get("endpoint_weight", match_weight))
    bridge_weight = float(train_cfg.get("bridge_weight", defect_weight))
    objective = str(train_cfg.get("objective", "triplet"))
    defect_grad_mode = str(train_cfg.get("defect_grad_mode", "both")).lower()
    perceptual_weight = float(train_cfg.get("perceptual_weight", cfg.get("loss", {}).get("perceptual_weight", 0.0)))
    preserve_mid_u_values = [float(value) for value in train_cfg.get("preserve_mid_u_values", [])]
    preserve_bridge_weight = float(train_cfg.get("preserve_bridge_weight", 0.0))
    preserve_match_weight = float(train_cfg.get("preserve_match_weight", 0.0))
    preserve_defect_weight = float(train_cfg.get("preserve_defect_weight", 0.0))
    preserve_perceptual_weight = float(train_cfg.get("preserve_perceptual_weight", 0.0))
    data_denoise_weight = float(train_cfg.get("data_denoise_weight", 0.0))
    data_denoise_u_low = float(train_cfg.get("data_denoise_u_low", 0.35))
    data_denoise_u_high = float(train_cfg.get("data_denoise_u_high", 0.95))
    data_denoise_normalize = bool(train_cfg.get("data_denoise_normalize", False))
    data_denoise_adaptive = bool(train_cfg.get("data_denoise_adaptive", False))
    data_transition_weight = float(train_cfg.get("data_transition_weight", 0.0))
    data_transition_u_low = float(train_cfg.get("data_transition_u_low", 0.02))
    data_transition_u_high = float(train_cfg.get("data_transition_u_high", 0.92))
    data_transition_min_delta_u = float(train_cfg.get("data_transition_min_delta_u", 0.05))
    data_transition_endpoint_prob = float(train_cfg.get("data_transition_endpoint_prob", 0.50))
    data_transition_normalize = bool(train_cfg.get("data_transition_normalize", True))
    data_transition_adaptive = bool(train_cfg.get("data_transition_adaptive", False))
    data_adaptive_min_scale = float(train_cfg.get("data_adaptive_min_scale", 0.25))
    data_adaptive_max_scale = float(train_cfg.get("data_adaptive_max_scale", 4.0))
    amp_enabled = bool(runtime_cfg.get("amp", True)) and device.type == "cuda"
    ema_enabled = bool(train_cfg.get("ema_enabled", False))
    ema_decay = float(train_cfg.get("ema_decay", 0.9995))
    ema_update_every = max(1, int(train_cfg.get("ema_update_every", 1)))
    ema_device_name = str(train_cfg.get("ema_device", "cpu"))
    target_ema_enabled = bool(train_cfg.get("target_ema_enabled", False))
    target_ema_decay = float(train_cfg.get("target_ema_decay", 0.999))
    target_ema_update_every = max(1, int(train_cfg.get("target_ema_update_every", 1)))
    target_ema_device_name = str(train_cfg.get("target_ema_device", str(device)))
    ctm_target_enabled = bool(train_cfg.get("ctm_target_enabled", False))
    ctm_target_weight = float(train_cfg.get("ctm_target_weight", 0.0))
    ctm_target_bridge_weight = float(train_cfg.get("ctm_target_bridge_weight", 0.5))
    ctm_target_adaptive = bool(train_cfg.get("ctm_target_adaptive", False))
    ctm_target_adaptive_min_scale = float(train_cfg.get("ctm_target_adaptive_min_scale", data_adaptive_min_scale))
    ctm_target_adaptive_max_scale = float(train_cfg.get("ctm_target_adaptive_max_scale", data_adaptive_max_scale))
    ctm_target_loss_norm = str(train_cfg.get("ctm_target_loss_norm", "endpoint")).lower()
    ctm_target_perceptual_weight = float(train_cfg.get("ctm_target_perceptual_weight", 0.0))
    ctm_target_source = str(train_cfg.get("ctm_target_source", "teacher_dt_target")).lower()
    ctm_teacher_dt_u = float(train_cfg.get("ctm_teacher_dt_u", data_transition_min_delta_u))
    data_transition_target_mode = str(train_cfg.get("data_transition_target_mode", "teacher")).lower()

    teacher = load_edm_network(cfg["paths"]["network"], device=device, use_fp16=bool(teacher_cfg.get("use_fp16", True)))
    teacher.requires_grad_(False)
    student = clone_student_from_teacher(teacher, cfg=cfg).to(device)
    student_ema = None
    if ema_enabled:
        ema_device = torch.device(ema_device_name)
        student_ema = copy.deepcopy(student).to(ema_device)
        student_ema.eval().requires_grad_(False)
    target_model = None
    if target_ema_enabled or ctm_target_enabled or data_transition_target_mode in {"ctm", "ctm_stopgrad", "target", "target_model"}:
        target_device = torch.device(target_ema_device_name)
        target_model = copy.deepcopy(student).to(target_device)
        target_model.eval().requires_grad_(False)
    base_lr = float(train_cfg.get("lr", 1.0e-6))
    adapter_lr = train_cfg.get("transition_adapter_lr")
    if adapter_lr is not None and hasattr(student, "base"):
        base_params = []
        adapter_params = []
        for name, param in student.named_parameters():
            if name.startswith("base."):
                base_params.append(param)
            else:
                adapter_params.append(param)
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": base_params,
                    "lr": base_lr,
                    "weight_decay": float(train_cfg.get("weight_decay", 0.0)),
                },
                {
                    "params": adapter_params,
                    "lr": float(adapter_lr),
                    "weight_decay": float(train_cfg.get("transition_adapter_weight_decay", 0.0)),
                },
            ]
        )
    else:
        optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=base_lr,
            weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        )
    scaler = torch.amp.GradScaler(device="cuda", enabled=amp_enabled)
    perceptual_metric = None
    if perceptual_weight > 0.0:
        perceptual_metric = MultiScaleL1(levels=int(train_cfg.get("perceptual_levels", cfg.get("loss", {}).get("perceptual_fallback_levels", 3)))).to(device)
        perceptual_metric.eval().requires_grad_(False)

    warp, q_base, q_target = init_warp(cfg, device=device)
    warp_optimizer = None
    warp_stats = None
    if warp is not None:
        warp_optimizer = torch.optim.AdamW(
            warp.parameters(),
            lr=float(cfg.get("timewarp", {}).get("lr", 1.0e-3)),
            weight_decay=0.0,
        )
        warp_stats = {
            "D_bar": torch.zeros(warp.num_bins, device=device),
            "D_count": torch.zeros(warp.num_bins, device=device),
        }

    start_step = 0
    best_loss = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        resume_student_key = str(train_cfg.get("resume_student_key", "student"))
        if resume_student_key not in ckpt or ckpt.get(resume_student_key) is None:
            raise KeyError(f"resume_student_key={resume_student_key!r} not found in checkpoint")
        student.load_state_dict(ckpt[resume_student_key])
        if student_ema is not None:
            ema_state = ckpt.get("student_ema") if bool(train_cfg.get("resume_ema", True)) else None
            if ema_state is not None:
                student_ema.load_state_dict(ema_state)
            else:
                student_ema.load_state_dict(student.state_dict())
        if target_model is not None:
            target_state = ckpt.get("target_model") or ckpt.get("ctm_target_model")
            if target_state is None:
                target_state = ckpt.get("student_ema") if ckpt.get("student_ema") is not None else ckpt[resume_student_key]
            target_model.load_state_dict(target_state)
        resume_optimizer = bool(train_cfg.get("resume_optimizer", True))
        resume_step = bool(train_cfg.get("resume_step", resume_optimizer))
        resume_warp = bool(train_cfg.get("resume_warp", True))
        ignore_incompatible_warp = bool(train_cfg.get("ignore_incompatible_warp_resume", False))
        if resume_optimizer:
            optimizer.load_state_dict(ckpt["optimizer"])
        if resume_warp and warp is not None and ckpt.get("warp") is not None:
            try:
                warp.load_state_dict(ckpt["warp"])
            except RuntimeError:
                if not ignore_incompatible_warp:
                    raise
                print("Skipping incompatible warp checkpoint state; using freshly initialized warp.", flush=True)
        if resume_warp and resume_optimizer and warp_optimizer is not None and ckpt.get("warp_optimizer") is not None:
            try:
                warp_optimizer.load_state_dict(ckpt["warp_optimizer"])
            except ValueError:
                if not ignore_incompatible_warp:
                    raise
                print("Skipping incompatible warp optimizer state; using freshly initialized optimizer.", flush=True)
        if resume_warp and q_target is not None and ckpt.get("q_target") is not None:
            loaded_q_target = ckpt["q_target"].to(device)
            if loaded_q_target.shape == q_target.shape:
                q_target.copy_(loaded_q_target)
            elif not ignore_incompatible_warp:
                raise RuntimeError(f"q_target shape mismatch: expected {tuple(q_target.shape)}, got {tuple(loaded_q_target.shape)}")
        if resume_warp and warp_stats is not None and ckpt.get("warp_stats") is not None:
            for key, value in ckpt["warp_stats"].items():
                if key in warp_stats and value.shape == warp_stats[key].shape:
                    warp_stats[key].copy_(value.to(device))
                elif not ignore_incompatible_warp:
                    raise RuntimeError(f"warp_stats[{key}] shape mismatch")
        if resume_step:
            start_step = int(ckpt.get("step", 0))
            best_loss = float(ckpt.get("best_loss", best_loss))

    loader = build_cifar10_loader(cfg) if objective == "triplet" or data_denoise_weight > 0.0 or data_transition_weight > 0.0 else None
    data_iter = iter(loader) if loader is not None else None
    history_path = log_dir / "train.jsonl"
    start_time = time.time()
    last_log = {}
    last_saved_step = 0
    stop_requested = {"value": False}

    def _request_stop(_signum, _frame) -> None:
        stop_requested["value"] = True

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    for step in range(start_step + 1, max_steps + 1):
        if stop_requested["value"]:
            break
        if max_seconds > 0.0 and time.time() - start_time >= max_seconds:
            break
        batch_size = int(train_cfg.get("batch_size", 64))
        preserve_loss = torch.zeros((), device=device)
        preserve_bridge_loss = torch.zeros((), device=device)
        preserve_match_loss = torch.zeros((), device=device)
        preserve_defect_loss = torch.zeros((), device=device)
        preserve_perceptual_loss = torch.zeros((), device=device)
        data_denoise_loss = torch.zeros((), device=device)
        data_transition_loss = torch.zeros((), device=device)
        data_denoise_effective_weight = torch.zeros((), device=device)
        data_transition_effective_weight = torch.zeros((), device=device)
        ctm_target_loss = torch.zeros((), device=device)
        ctm_target_direct_loss = torch.zeros((), device=device)
        ctm_target_bridge_loss = torch.zeros((), device=device)
        ctm_target_perceptual_loss = torch.zeros((), device=device)
        ctm_target_effective_weight = torch.zeros((), device=device)
        ctm_target_mid_loss = torch.zeros((), device=device)
        if objective in {"prior_endpoint", "prior_fullstack"}:
            labels = sample_labels(teacher, batch_size, device=device)
            sigma_t = sigma_from_u(torch.zeros(batch_size, device=device), sigma_min=sigma_min, sigma_max=sigma_max, rho=rho, net=teacher)
            sigma_u = torch.zeros_like(sigma_t)
            mid_u_low = float(train_cfg.get("prior_mid_u_low", 0.25))
            mid_u_high = float(train_cfg.get("prior_mid_u_high", 0.85))
            r_mid = torch.empty(batch_size, device=device).uniform_(mid_u_low, mid_u_high)
            u_mid = warp.r_to_t(r_mid) if warp is not None else r_mid
            u_u = u_mid
            sigma_s = sigma_from_u(u_mid, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho, net=teacher)
            x_t = torch.randn(
                batch_size,
                int(getattr(teacher, "img_channels", 3)),
                int(getattr(teacher, "img_resolution", 32)),
                int(getattr(teacher, "img_resolution", 32)),
                device=device,
            ) * sigma_t.view(-1, 1, 1, 1)
            with torch.no_grad():
                x_u_ref = teacher_rollout_transition(
                    teacher,
                    x_t,
                    sigma_t[0],
                    0.0,
                    labels,
                    num_steps=int(teacher_cfg.get("rollout_steps", 18)),
                    rho=rho,
                )
                x_s_ref = (
                    teacher_transition(teacher, x_t, sigma_t, sigma_s, labels)
                    if objective == "prior_fullstack"
                    else x_u_ref
                )

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                if objective == "prior_fullstack":
                    x_s = student_transition(student, x_t, sigma_t, sigma_s, labels)
                    x_u_bridge = student_transition(student, x_s, sigma_s, sigma_u, labels)
                    x_u_direct = student_transition(student, x_t, sigma_t, sigma_u, labels)
                    match_per_sample = _flat_mse_per_sample(x_s, x_s_ref)
                    anchor_per_sample = _flat_mse_per_sample(x_u_direct, x_u_ref)
                    bridge_per_sample = _flat_mse_per_sample(x_u_bridge, x_u_ref)
                    defect_raw = _flat_mse_per_sample(x_u_direct, x_u_bridge)
                    if defect_grad_mode in {"bridge_to_direct", "bridge", "bridge_only"}:
                        defect_per_sample = _flat_mse_per_sample(x_u_bridge, x_u_direct.detach())
                    elif defect_grad_mode in {"direct_to_bridge", "direct", "direct_only"}:
                        defect_per_sample = _flat_mse_per_sample(x_u_direct, x_u_bridge.detach())
                    elif defect_grad_mode in {"none", "off", "disabled"}:
                        defect_per_sample = torch.zeros_like(defect_raw)
                    else:
                        defect_per_sample = defect_raw
                    match_denom = _flat_mse_per_sample(x_s_ref, x_t).clamp_min(1.0e-6)
                    endpoint_denom = _flat_mse_per_sample(x_u_ref, x_t).clamp_min(1.0e-6)
                    match_loss = _safe_mean_normalized(match_per_sample, match_denom)
                    anchor_loss = anchor_per_sample.mean()
                    bridge_loss = bridge_per_sample.mean()
                    defect_loss = defect_per_sample.mean()
                    perceptual_direct = (
                        perceptual_metric(x_u_direct.float(), x_u_ref.float())
                        if perceptual_metric is not None
                        else torch.zeros((), device=device, dtype=anchor_loss.dtype)
                    )
                    perceptual_bridge = (
                        perceptual_metric(x_u_bridge.float(), x_u_ref.float())
                        if perceptual_metric is not None and float(train_cfg.get("bridge_perceptual_weight", 0.0)) > 0.0
                        else torch.zeros((), device=device, dtype=anchor_loss.dtype)
                    )
                    perceptual_loss = perceptual_direct + float(train_cfg.get("bridge_perceptual_weight", 0.0)) * perceptual_bridge
                    loss = (
                        endpoint_weight * anchor_loss
                        + match_weight * match_loss
                        + bridge_weight * bridge_loss
                        + defect_weight * defect_loss
                        + perceptual_weight * perceptual_loss
                    )
                    if ctm_target_enabled and ctm_target_weight > 0.0:
                        if target_model is None:
                            raise RuntimeError("ctm_target_enabled requires target_model")
                        with torch.no_grad():
                            target_labels = labels.to(target_ema_device_name) if labels is not None and target_ema_device_name != str(device) else labels
                            x_t_target = x_t.detach().to(target_ema_device_name)
                            sigma_t_target = sigma_t.detach().to(target_ema_device_name)
                            sigma_s_target = sigma_s.detach().to(target_ema_device_name)
                            sigma_u_target = sigma_u.detach().to(target_ema_device_name)
                            if ctm_target_source in {"target", "target_model", "target_model_sg"}:
                                x_s_target_seed = student_transition(
                                    target_model,
                                    x_t_target,
                                    sigma_t_target,
                                    sigma_s_target,
                                    target_labels,
                                )
                            elif ctm_target_source in {"teacher_dt_target", "ctm", "ctm_stopgrad"}:
                                u_dt_target = torch.minimum(
                                    u_mid.detach(),
                                    torch.full_like(u_mid.detach(), max(ctm_teacher_dt_u, 1.0e-6)),
                                )
                                sigma_dt_target = sigma_from_u(
                                    u_dt_target,
                                    sigma_min=sigma_min,
                                    sigma_max=sigma_max,
                                    rho=rho,
                                    net=teacher,
                                )
                                x_dt_target = teacher_transition(
                                    teacher,
                                    x_t,
                                    sigma_t,
                                    sigma_dt_target,
                                    labels,
                                ).to(target_ema_device_name)
                                x_s_target_seed = student_transition(
                                    target_model,
                                    x_dt_target,
                                    sigma_dt_target.to(target_ema_device_name),
                                    sigma_s_target,
                                    target_labels,
                                )
                            else:
                                x_s_target_seed = teacher_transition(
                                    teacher,
                                    x_t,
                                    sigma_t,
                                    sigma_s,
                                    labels,
                                ).to(target_ema_device_name)
                            x_u_ctm_target = student_transition(
                                target_model,
                                x_s_target_seed,
                                sigma_s_target,
                                sigma_u_target,
                                target_labels,
                            ).to(device=device, dtype=x_u_direct.dtype)
                        ctm_target_mid_weight = float(train_cfg.get("ctm_target_mid_weight", 0.0))
                        if ctm_target_mid_weight > 0.0:
                            x_s_target_for_loss = x_s_target_seed.to(device=device, dtype=x_s.dtype)
                            ctm_target_mid_per_sample = _flat_mse_per_sample(x_s, x_s_target_for_loss)
                            ctm_target_mid_loss = _safe_mean_normalized(ctm_target_mid_per_sample, match_denom.detach())
                        ctm_target_direct_per_sample = _flat_mse_per_sample(x_u_direct, x_u_ctm_target)
                        ctm_target_bridge_per_sample = _flat_mse_per_sample(x_u_bridge, x_u_ctm_target)
                        if ctm_target_loss_norm in {"endpoint", "trajectory", "edm"}:
                            ctm_target_denom = endpoint_denom.detach()
                        elif ctm_target_loss_norm in {"target_delta", "target"}:
                            ctm_target_denom = _flat_mse_per_sample(x_u_ctm_target.detach(), x_t.detach())
                        elif ctm_target_loss_norm in {"none", "raw", "mse"}:
                            ctm_target_denom = torch.ones_like(ctm_target_direct_per_sample)
                        else:
                            raise RuntimeError(f"unknown ctm_target_loss_norm: {ctm_target_loss_norm}")
                        ctm_target_direct_loss = _safe_mean_normalized(ctm_target_direct_per_sample, ctm_target_denom)
                        ctm_target_bridge_loss = _safe_mean_normalized(ctm_target_bridge_per_sample, ctm_target_denom)
                        ctm_target_loss = (
                            ctm_target_mid_weight * ctm_target_mid_loss
                            + ctm_target_direct_loss
                            + ctm_target_bridge_weight * ctm_target_bridge_loss
                        )
                        if ctm_target_perceptual_weight > 0.0 and perceptual_metric is not None:
                            ctm_target_perceptual_loss = perceptual_metric(
                                x_u_direct.float(),
                                x_u_ctm_target.float(),
                            )
                            ctm_target_loss = ctm_target_loss + ctm_target_perceptual_weight * ctm_target_perceptual_loss
                        if ctm_target_adaptive:
                            ctm_target_effective_weight = _adaptive_loss_weight(
                                base_weight=ctm_target_weight,
                                reference_loss=loss.detach(),
                                aux_loss=ctm_target_loss,
                                min_scale=ctm_target_adaptive_min_scale,
                                max_scale=ctm_target_adaptive_max_scale,
                            )
                        else:
                            ctm_target_effective_weight = ctm_target_loss.new_tensor(ctm_target_weight)
                        loss = loss + ctm_target_effective_weight * ctm_target_loss
                    if preserve_mid_u_values and (
                        preserve_bridge_weight > 0.0
                        or preserve_match_weight > 0.0
                        or preserve_defect_weight > 0.0
                        or preserve_perceptual_weight > 0.0
                    ):
                        preserve_bridge_terms = []
                        preserve_match_terms = []
                        preserve_defect_terms = []
                        preserve_perceptual_terms = []
                        for preserve_u_value in preserve_mid_u_values:
                            preserve_u = torch.full(
                                (batch_size,),
                                preserve_u_value,
                                device=device,
                                dtype=u_mid.dtype,
                            )
                            preserve_sigma = sigma_from_u(
                                preserve_u,
                                sigma_min=sigma_min,
                                sigma_max=sigma_max,
                                rho=rho,
                                net=teacher,
                            )
                            x_s_preserve = student_transition(student, x_t, sigma_t, preserve_sigma, labels)
                            x_u_preserve = student_transition(student, x_s_preserve, preserve_sigma, sigma_u, labels)
                            preserve_bridge_terms.append(_flat_mse_per_sample(x_u_preserve, x_u_ref).mean())
                            preserve_defect_terms.append(_flat_mse_per_sample(x_u_preserve, x_u_direct.detach()).mean())
                            if preserve_match_weight > 0.0:
                                with torch.no_grad():
                                    x_s_preserve_ref = teacher_transition(teacher, x_t, sigma_t, preserve_sigma, labels)
                                preserve_match_per_sample = _flat_mse_per_sample(x_s_preserve, x_s_preserve_ref)
                                preserve_match_denom = _flat_mse_per_sample(x_s_preserve_ref, x_t).clamp_min(1.0e-6)
                                preserve_match_terms.append(_safe_mean_normalized(preserve_match_per_sample, preserve_match_denom))
                            if preserve_perceptual_weight > 0.0 and perceptual_metric is not None:
                                preserve_perceptual_terms.append(
                                    perceptual_metric(x_u_preserve.float(), x_u_ref.float())
                                )
                        preserve_bridge_loss = torch.stack(preserve_bridge_terms).mean()
                        preserve_defect_loss = torch.stack(preserve_defect_terms).mean()
                        if preserve_match_terms:
                            preserve_match_loss = torch.stack(preserve_match_terms).mean()
                        if preserve_perceptual_terms:
                            preserve_perceptual_loss = torch.stack(preserve_perceptual_terms).mean()
                        preserve_loss = (
                            preserve_bridge_weight * preserve_bridge_loss
                            + preserve_match_weight * preserve_match_loss
                            + preserve_defect_weight * preserve_defect_loss
                            + preserve_perceptual_weight * preserve_perceptual_loss
                        )
                        loss = loss + preserve_loss
                else:
                    x_u_direct = student_transition(student, x_t, sigma_t, sigma_u, labels)
                    with torch.no_grad():
                        x_s = student_transition(student, x_t, sigma_t, sigma_s, labels)
                        x_u_bridge = student_transition(student, x_s, sigma_s, sigma_u, labels)
                    match_per_sample = _flat_mse_per_sample(x_u_direct, x_u_ref)
                    anchor_per_sample = match_per_sample
                    bridge_per_sample = torch.zeros_like(match_per_sample)
                    match_loss = match_per_sample.mean()
                    anchor_loss = match_loss
                    bridge_loss = torch.zeros((), device=device, dtype=match_loss.dtype)
                    defect_raw = _flat_mse_per_sample(x_u_direct, x_u_bridge.detach())
                    endpoint_denom = _flat_mse_per_sample(x_u_ref, x_t).clamp_min(1.0e-6)
                    defect_loss = _safe_mean_normalized(defect_raw, endpoint_denom)
                    perceptual_loss = (
                        perceptual_metric(x_u_direct.float(), x_u_ref.float())
                        if perceptual_metric is not None
                        else torch.zeros((), device=device, dtype=match_loss.dtype)
                    )
                    loss = match_weight * match_loss + defect_weight * defect_loss + perceptual_weight * perceptual_loss

                consistency_loss_for_balance = loss.detach()
                if data_denoise_weight > 0.0 or data_transition_weight > 0.0:
                    if data_iter is None or loader is None:
                        raise RuntimeError("real-data auxiliary losses require a CIFAR-10 loader")
                    (real_images, real_labels_int), data_iter = _next_cifar_batch(data_iter, loader)
                    real_images = real_images.to(device=device, non_blocking=True) * 2.0 - 1.0
                    real_labels = make_labels(
                        real_labels_int,
                        label_dim=int(getattr(teacher, "label_dim", 0) or 0),
                        device=device,
                    )
                    real_batch = int(real_images.shape[0])
                if data_denoise_weight > 0.0:
                    data_u = torch.empty(real_batch, device=device).uniform_(data_denoise_u_low, data_denoise_u_high)
                    data_sigma = sigma_from_u(data_u, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho, net=teacher)
                    noisy_images = real_images + torch.randn_like(real_images) * data_sigma.view(-1, 1, 1, 1)
                    denoised_images = student(noisy_images, data_sigma, real_labels).to(real_images.dtype)
                    data_denoise_per_sample = _flat_mse_per_sample(denoised_images, real_images)
                    if data_denoise_normalize:
                        data_denoise_denom = _flat_mse_per_sample(noisy_images, real_images).clamp_min(1.0e-6)
                        data_denoise_loss = _safe_mean_normalized(data_denoise_per_sample, data_denoise_denom)
                    else:
                        data_denoise_loss = data_denoise_per_sample.mean()
                    if data_denoise_adaptive:
                        data_denoise_effective_weight = _adaptive_loss_weight(
                            base_weight=data_denoise_weight,
                            reference_loss=consistency_loss_for_balance,
                            aux_loss=data_denoise_loss,
                            min_scale=data_adaptive_min_scale,
                            max_scale=data_adaptive_max_scale,
                        )
                    else:
                        data_denoise_effective_weight = data_denoise_loss.new_tensor(data_denoise_weight)
                    loss = loss + data_denoise_effective_weight * data_denoise_loss
                if data_transition_weight > 0.0:
                    trans_u_t = torch.empty(real_batch, device=device).uniform_(data_transition_u_low, data_transition_u_high)
                    room = torch.clamp(1.0 - trans_u_t - data_transition_min_delta_u, min=0.0)
                    trans_u_s = trans_u_t + data_transition_min_delta_u + room * torch.rand_like(trans_u_t)
                    endpoint_mask = torch.rand(real_batch, device=device) < data_transition_endpoint_prob
                    trans_u_s = torch.where(endpoint_mask, torch.ones_like(trans_u_s), trans_u_s.clamp(max=1.0))
                    trans_sigma_t = sigma_from_u(trans_u_t, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho, net=teacher)
                    trans_sigma_s = sigma_from_u(trans_u_s, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho, net=teacher)
                    trans_sigma_s = torch.where(endpoint_mask, torch.zeros_like(trans_sigma_s), trans_sigma_s)
                    trans_noisy = real_images + torch.randn_like(real_images) * trans_sigma_t.view(-1, 1, 1, 1)
                    with torch.no_grad():
                        if data_transition_target_mode in {"ctm", "ctm_stopgrad", "target", "target_model"}:
                            if target_model is None:
                                raise RuntimeError("CTM data transition target requires target_model")
                            dt_room = torch.clamp(trans_u_s - trans_u_t, min=0.0)
                            trans_u_dt = trans_u_t + torch.minimum(
                                dt_room,
                                torch.full_like(dt_room, max(ctm_teacher_dt_u, 1.0e-6)),
                            )
                            trans_sigma_dt = sigma_from_u(
                                trans_u_dt,
                                sigma_min=sigma_min,
                                sigma_max=sigma_max,
                                rho=rho,
                                net=teacher,
                            )
                            trans_sigma_dt = torch.where(trans_u_dt >= 1.0, torch.zeros_like(trans_sigma_dt), trans_sigma_dt)
                            trans_dt = teacher_transition(
                                teacher,
                                trans_noisy,
                                trans_sigma_t,
                                trans_sigma_dt,
                                real_labels,
                            )
                            target_labels = (
                                real_labels.to(target_ema_device_name)
                                if real_labels is not None and target_ema_device_name != str(device)
                                else real_labels
                            )
                            trans_target = student_transition(
                                target_model,
                                trans_dt.to(target_ema_device_name),
                                trans_sigma_dt.to(target_ema_device_name),
                                trans_sigma_s.to(target_ema_device_name),
                                target_labels,
                            ).to(device=device, dtype=trans_noisy.dtype)
                        else:
                            trans_target = real_images.clone()
                            non_endpoint = ~endpoint_mask
                            if bool(non_endpoint.any()):
                                non_labels = real_labels[non_endpoint] if real_labels is not None else None
                                trans_target[non_endpoint] = teacher_transition(
                                    teacher,
                                    trans_noisy[non_endpoint],
                                    trans_sigma_t[non_endpoint],
                                    trans_sigma_s[non_endpoint],
                                    non_labels,
                                )
                    trans_estimate = student_transition(student, trans_noisy, trans_sigma_t, trans_sigma_s, real_labels)
                    data_transition_per_sample = _flat_mse_per_sample(trans_estimate, trans_target)
                    if data_transition_normalize:
                        data_transition_denom = _flat_mse_per_sample(trans_noisy, trans_target).clamp_min(1.0e-6)
                        data_transition_loss = _safe_mean_normalized(data_transition_per_sample, data_transition_denom)
                    else:
                        data_transition_loss = data_transition_per_sample.mean()
                    if data_transition_adaptive:
                        data_transition_effective_weight = _adaptive_loss_weight(
                            base_weight=data_transition_weight,
                            reference_loss=consistency_loss_for_balance,
                            aux_loss=data_transition_loss,
                            min_scale=data_adaptive_min_scale,
                            max_scale=data_adaptive_max_scale,
                        )
                    else:
                        data_transition_effective_weight = data_transition_loss.new_tensor(data_transition_weight)
                    loss = loss + data_transition_effective_weight * data_transition_loss
        else:
            assert data_iter is not None and loader is not None
            try:
                images, labels_int = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                images, labels_int = next(data_iter)

            images = images.to(device=device, non_blocking=True) * 2.0 - 1.0
            labels = make_labels(labels_int, label_dim=int(getattr(teacher, "label_dim", 0) or 0), device=device)
            batch_size = int(images.shape[0])
            u_t, u_s, u_u = sample_triplet_u(warp, batch_size, device=device)
            sigma_t = sigma_from_u(u_t, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho, net=teacher)
            sigma_s = sigma_from_u(u_s, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho, net=teacher)
            sigma_u = sigma_from_u(u_u, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho, net=teacher)
            noise = torch.randn_like(images)
            x_t = images + noise * sigma_t.view(-1, 1, 1, 1)

            with torch.no_grad():
                x_s_ref = teacher_transition(teacher, x_t, sigma_t, sigma_s, labels)
                x_u_ref = teacher_transition(teacher, x_t, sigma_t, sigma_u, labels)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                x_s = student_transition(student, x_t, sigma_t, sigma_s, labels)
                x_u_direct = student_transition(student, x_t, sigma_t, sigma_u, labels)
                x_u_bridge = student_transition(student, x_s, sigma_s, sigma_u, labels)
                match_per_sample = (x_s - x_s_ref).flatten(1).square().mean(dim=1)
                anchor_per_sample = (x_u_direct - x_u_ref).flatten(1).square().mean(dim=1)
                bridge_per_sample = torch.zeros_like(anchor_per_sample)
                match_denom = (x_s_ref - x_t).flatten(1).square().mean(dim=1).clamp_min(1.0e-6)
                anchor_denom = (x_u_ref - x_t).flatten(1).square().mean(dim=1).clamp_min(1.0e-6)
                match_loss = (match_per_sample / match_denom).mean()
                anchor_loss = (anchor_per_sample / anchor_denom).mean()
                defect_raw = (x_u_direct - x_u_bridge.detach()).flatten(1).square().mean(dim=1)
                defect_denom = (x_u_ref - x_t).flatten(1).square().mean(dim=1).clamp_min(1.0e-6)
                defect_loss = (defect_raw / defect_denom).mean()
                loss = match_weight * match_loss + anchor_weight * anchor_loss + defect_weight * defect_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if student_ema is not None and step % ema_update_every == 0:
            _update_ema_model(student_ema, student, ema_decay)
        if target_model is not None and target_ema_enabled and step % target_ema_update_every == 0:
            _update_ema_model(target_model, student, target_ema_decay)

        denom = endpoint_denom.detach().float() if "endpoint_denom" in locals() else _flat_mse_per_sample(x_u_ref, x_t).clamp_min(1.0e-6).detach().float()
        raw_defect_per_sample = defect_raw.detach().float()
        warp_signal_mode = str(train_cfg.get("warp_defect_signal", "raw" if objective == "prior_fullstack" else "normalized")).lower()
        if objective == "prior_fullstack" and warp_signal_mode in {"raw", "direct_bridge"}:
            norm_defect_per_sample = raw_defect_per_sample
        elif objective == "prior_fullstack" and warp_signal_mode in {"bridge_error", "bridge"}:
            norm_defect_per_sample = bridge_per_sample.detach().float()
        elif objective == "prior_fullstack" and warp_signal_mode in {"composition_gap", "gap"}:
            gap = torch.clamp(bridge_per_sample.detach().float() - anchor_per_sample.detach().float(), min=0.0)
            norm_defect_per_sample = (
                raw_defect_per_sample
                + float(train_cfg.get("warp_composition_gap_weight", 1.0)) * gap
                + float(train_cfg.get("warp_bridge_error_weight", 0.0)) * bridge_per_sample.detach().float()
            )
        elif objective == "prior_fullstack" and warp_signal_mode in {"teacher_blend", "teacher"}:
            norm_defect_per_sample = raw_defect_per_sample + float(train_cfg.get("warp_bridge_error_weight", 0.25)) * bridge_per_sample.detach().float()
        else:
            norm_defect_per_sample = raw_defect_per_sample / denom
        norm_defect = norm_defect_per_sample.mean()
        warp_loss_value = None
        if (
            warp is not None
            and warp_optimizer is not None
            and q_base is not None
            and q_target is not None
            and warp_stats is not None
            and step % max(1, int(cfg.get("timewarp", {}).get("update_every", 25))) == 0
        ):
            warp_loss = update_warp_from_defect(
                warp=warp,
                warp_optimizer=warp_optimizer,
                q_base=q_base,
                q_target=q_target,
                stats=warp_stats,
                u=u_u,
                defect=norm_defect_per_sample.detach(),
                cfg=cfg,
            )
            warp_loss_value = float(warp_loss.item())

        elapsed = time.time() - start_time
        record = {
            "step": step,
            "objective": objective,
            "loss": float(loss.detach().float().item()),
            "match_loss": float(match_loss.detach().float().item()),
            "anchor_loss": float(anchor_loss.detach().float().item()),
            "bridge_loss": float(bridge_loss.detach().float().item()) if "bridge_loss" in locals() else 0.0,
            "defect_loss": float(defect_loss.detach().float().item()),
            "preserve_loss": float(preserve_loss.detach().float().item()),
            "preserve_bridge_loss": float(preserve_bridge_loss.detach().float().item()),
            "preserve_match_loss": float(preserve_match_loss.detach().float().item()),
            "preserve_defect_loss": float(preserve_defect_loss.detach().float().item()),
            "preserve_perceptual_loss": float(preserve_perceptual_loss.detach().float().item()),
            "data_denoise_loss": float(data_denoise_loss.detach().float().item()),
            "data_denoise_weight": data_denoise_weight,
            "data_denoise_effective_weight": float(data_denoise_effective_weight.detach().float().item()),
            "data_transition_loss": float(data_transition_loss.detach().float().item()),
            "data_transition_weight": data_transition_weight,
            "data_transition_effective_weight": float(data_transition_effective_weight.detach().float().item()),
            "data_transition_target_mode": data_transition_target_mode,
            "ctm_target_enabled": bool(ctm_target_enabled),
            "ctm_target_source": ctm_target_source,
            "ctm_target_loss": float(ctm_target_loss.detach().float().item()),
            "ctm_target_mid_loss": float(ctm_target_mid_loss.detach().float().item()),
            "ctm_target_direct_loss": float(ctm_target_direct_loss.detach().float().item()),
            "ctm_target_bridge_loss": float(ctm_target_bridge_loss.detach().float().item()),
            "ctm_target_perceptual_loss": float(ctm_target_perceptual_loss.detach().float().item()),
            "ctm_target_loss_norm": ctm_target_loss_norm,
            "ctm_target_weight": ctm_target_weight,
            "ctm_target_effective_weight": float(ctm_target_effective_weight.detach().float().item()),
            "transition_adapter_enabled": bool(cfg.get("transition_adapter", {}).get("enabled", False)),
            "ema_enabled": ema_enabled,
            "ema_decay": ema_decay if ema_enabled else None,
            "target_ema_enabled": bool(target_ema_enabled),
            "target_ema_decay": target_ema_decay if target_ema_enabled else None,
            "preserve_mid_u_values": preserve_mid_u_values,
            "defect_grad_mode": defect_grad_mode,
            "perceptual_loss": float(perceptual_loss.detach().float().item()) if "perceptual_loss" in locals() else 0.0,
            "raw_match_mse": float(match_per_sample.detach().float().mean().item()),
            "raw_anchor_mse": float(anchor_per_sample.detach().float().mean().item()),
            "raw_bridge_mse": float(bridge_per_sample.detach().float().mean().item()) if "bridge_per_sample" in locals() else 0.0,
            "norm_defect": float(norm_defect.detach().float().item()),
            "warp_defect_signal": warp_signal_mode,
            "sigma_t_mean": float(sigma_t.detach().float().mean().item()),
            "sigma_s_mean": float(sigma_s.detach().float().mean().item()),
            "sigma_u_mean": float(sigma_u.detach().float().mean().item()),
            "elapsed_sec": elapsed,
            "samples_seen": step * batch_size,
            "warp_loss": warp_loss_value,
            "entropy_q_phi": float((-(warp.density() * torch.log(torch.clamp(warp.density(), min=1.0e-6))).sum()).item()) if warp is not None else None,
            "max_qphi_over_qbase": float((warp.density() / torch.clamp(q_base, min=1.0e-6)).max().item()) if warp is not None and q_base is not None else None,
        }
        last_log = record
        if step % log_every == 0 or step == 1:
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(
                f"step={step}/{max_steps} loss={record['loss']:.6f} "
                f"match={record['match_loss']:.6f} defect={record['norm_defect']:.6f} "
                f"qmax={record['max_qphi_over_qbase']} elapsed={elapsed:.1f}s",
                flush=True,
            )

        should_save = step % save_every == 0 or step == max_steps or stop_requested["value"]
        if should_save:
            ckpt = {
                "step": step,
                "student": student.state_dict(),
                "student_ema": _copy_state_to_cpu(student_ema) if student_ema is not None else None,
                "target_model": _copy_state_to_cpu(target_model) if target_model is not None else None,
                "optimizer": optimizer.state_dict(),
                "warp": warp.state_dict() if warp is not None else None,
                "warp_optimizer": warp_optimizer.state_dict() if warp_optimizer is not None else None,
                "q_target": q_target.detach().cpu() if q_target is not None else None,
                "warp_stats": {key: value.detach().cpu() for key, value in (warp_stats or {}).items()} if warp_stats is not None else None,
                "config": cfg,
                "best_loss": best_loss,
                "last_log": record,
            }
            _save_checkpoint(ckpt_dir / "last.pt", ckpt)
            _save_checkpoint(ckpt_dir / f"step{step}.pt", ckpt)
            last_saved_step = step
            if warp is not None:
                warp_payload = _warp_snapshot_payload(step=step, warp=warp, q_base=q_base, q_target=q_target, warp_stats=warp_stats)
                write_json(log_dir / "warp_latest.json", warp_payload)
                write_json(log_dir / f"warp_step{step}.json", warp_payload)
            if float(record["loss"]) <= best_loss:
                best_loss = float(record["loss"])
                ckpt["best_loss"] = best_loss
                _save_checkpoint(ckpt_dir / "best.pt", ckpt)

    if last_log and int(last_log.get("step", 0) or 0) > last_saved_step:
        step = int(last_log["step"])
        ckpt = {
            "step": step,
            "student": student.state_dict(),
            "student_ema": _copy_state_to_cpu(student_ema) if student_ema is not None else None,
            "target_model": _copy_state_to_cpu(target_model) if target_model is not None else None,
            "optimizer": optimizer.state_dict(),
            "warp": warp.state_dict() if warp is not None else None,
            "warp_optimizer": warp_optimizer.state_dict() if warp_optimizer is not None else None,
            "q_target": q_target.detach().cpu() if q_target is not None else None,
            "warp_stats": {key: value.detach().cpu() for key, value in (warp_stats or {}).items()} if warp_stats is not None else None,
            "config": cfg,
            "best_loss": best_loss,
            "last_log": last_log,
        }
        if float(last_log["loss"]) <= best_loss:
            best_loss = float(last_log["loss"])
            ckpt["best_loss"] = best_loss
            _save_checkpoint(ckpt_dir / "best.pt", ckpt)
        _save_checkpoint(ckpt_dir / "last.pt", ckpt)
        _save_checkpoint(ckpt_dir / f"step{step}.pt", ckpt)
        if warp is not None:
            warp_payload = _warp_snapshot_payload(step=step, warp=warp, q_base=q_base, q_target=q_target, warp_stats=warp_stats)
            write_json(log_dir / "warp_latest.json", warp_payload)
            write_json(log_dir / f"warp_step{step}.json", warp_payload)

    write_json(run_root / "train_summary.json", {"last": last_log, "best_loss": best_loss})
    print(f"edm-first training completed: {run_root}")


if __name__ == "__main__":
    main()
