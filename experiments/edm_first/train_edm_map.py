from __future__ import annotations

import argparse
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
    perceptual_weight = float(train_cfg.get("perceptual_weight", cfg.get("loss", {}).get("perceptual_weight", 0.0)))
    amp_enabled = bool(runtime_cfg.get("amp", True)) and device.type == "cuda"

    teacher = load_edm_network(cfg["paths"]["network"], device=device, use_fp16=bool(teacher_cfg.get("use_fp16", True)))
    teacher.requires_grad_(False)
    student = clone_student_from_teacher(teacher).to(device)
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=float(train_cfg.get("lr", 1.0e-6)),
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
        student.load_state_dict(ckpt["student"])
        resume_optimizer = bool(train_cfg.get("resume_optimizer", True))
        resume_step = bool(train_cfg.get("resume_step", resume_optimizer))
        if resume_optimizer:
            optimizer.load_state_dict(ckpt["optimizer"])
        if warp is not None and ckpt.get("warp") is not None:
            warp.load_state_dict(ckpt["warp"])
        if resume_optimizer and warp_optimizer is not None and ckpt.get("warp_optimizer") is not None:
            warp_optimizer.load_state_dict(ckpt["warp_optimizer"])
        if q_target is not None and ckpt.get("q_target") is not None:
            q_target.copy_(ckpt["q_target"].to(device))
        if warp_stats is not None and ckpt.get("warp_stats") is not None:
            for key, value in ckpt["warp_stats"].items():
                warp_stats[key].copy_(value.to(device))
        if resume_step:
            start_step = int(ckpt.get("step", 0))
            best_loss = float(ckpt.get("best_loss", best_loss))

    loader = build_cifar10_loader(cfg) if objective == "triplet" else None
    data_iter = iter(loader) if loader is not None else None
    history_path = log_dir / "train.jsonl"
    start_time = time.time()
    last_log = {}
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
                    match_denom = _flat_mse_per_sample(x_s_ref, x_t).clamp_min(1.0e-6)
                    endpoint_denom = _flat_mse_per_sample(x_u_ref, x_t).clamp_min(1.0e-6)
                    match_loss = _safe_mean_normalized(match_per_sample, match_denom)
                    anchor_loss = anchor_per_sample.mean()
                    bridge_loss = bridge_per_sample.mean()
                    defect_loss = defect_raw.mean()
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

        denom = endpoint_denom.detach().float() if "endpoint_denom" in locals() else _flat_mse_per_sample(x_u_ref, x_t).clamp_min(1.0e-6).detach().float()
        if objective == "prior_fullstack" and str(train_cfg.get("warp_defect_signal", "raw")).lower() == "raw":
            norm_defect_per_sample = defect_raw.detach().float()
        else:
            norm_defect_per_sample = defect_raw.detach().float() / denom
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
            "perceptual_loss": float(perceptual_loss.detach().float().item()) if "perceptual_loss" in locals() else 0.0,
            "raw_match_mse": float(match_per_sample.detach().float().mean().item()),
            "raw_anchor_mse": float(anchor_per_sample.detach().float().mean().item()),
            "raw_bridge_mse": float(bridge_per_sample.detach().float().mean().item()) if "bridge_per_sample" in locals() else 0.0,
            "norm_defect": float(norm_defect.detach().float().item()),
            "warp_defect_signal": str(train_cfg.get("warp_defect_signal", "raw" if objective == "prior_fullstack" else "normalized")),
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
            if warp is not None:
                warp_payload = _warp_snapshot_payload(step=step, warp=warp, q_base=q_base, q_target=q_target, warp_stats=warp_stats)
                write_json(log_dir / "warp_latest.json", warp_payload)
                write_json(log_dir / f"warp_step{step}.json", warp_payload)
            if float(record["loss"]) <= best_loss:
                best_loss = float(record["loss"])
                ckpt["best_loss"] = best_loss
                _save_checkpoint(ckpt_dir / "best.pt", ckpt)

    write_json(run_root / "train_summary.json", {"last": last_log, "best_loss": best_loss})
    print(f"edm-first training completed: {run_root}")


if __name__ == "__main__":
    main()
