from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import time

import torch
import torch.distributed as dist
from torch import nn
import yaml

from dgfm.config import RunRoots
from dgfm.datasets import build_image_dataloaders
from dgfm.distributed import (
    DistributedContext,
    maybe_wrap_ddp,
    reduce_float_dict,
    set_dataloader_epoch,
    unwrap_model,
)
from dgfm.models import ModelEMA, build_map_model
from dgfm.schedulers import build_runtime_time_grid, build_timewarp_module, summarize_time_grid
from dgfm.utils import build_experiment_archive

from .cache import build_cache_dataloaders, interpolate_curvature, interpolate_state
from .defect import build_target_density, compute_dgtd_residual, compute_sample_defect, smooth_density, update_ema_bins
from .metrics import edm_weight, high_frequency_norm, laplacian_filter, metric_norm, min_snr_weight
from .teacher import (
    CONTINUATION_SOURCE_CACHED_AFFINE,
    CONTINUATION_SOURCE_CACHED_EXACT,
    CONTINUATION_SOURCE_BOOTSTRAP,
    CONTINUATION_SOURCE_ONLINE,
    build_teacher_adapter,
)


def _load_elapsed_history(history_path) -> tuple[int, float]:
    if not history_path.exists():
        return 0, 0.0
    count = 0
    total = 0.0
    with history_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            elapsed = payload.get("elapsed_sec")
            if elapsed is None:
                continue
            total += float(elapsed)
            count += 1
    return count, total


def _device_from_config(config: dict) -> torch.device:
    requested = config.get("runtime", {}).get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _autocast_context(device: torch.device, enabled: bool):
    if device.type != "cuda" or not enabled:
        return torch.autocast(device_type="cpu", enabled=False)
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)


def _move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    moved: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        moved[key] = value.to(device=device, non_blocking=True)
    return moved


def _extract_online_teacher_batch(batch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(batch, (list, tuple)):
        if not batch:
            raise ValueError("Empty batch received for online teacher mode")
        images = batch[0]
        cond = batch[1] if len(batch) > 1 and isinstance(batch[1], torch.Tensor) else None
    elif isinstance(batch, dict):
        image_key = next((key for key in ("images", "image", "x", "inputs") if key in batch), None)
        if image_key is None:
            raise KeyError("Online teacher batch dict must contain an image tensor")
        images = batch[image_key]
        cond = batch.get("label", batch.get("labels", batch.get("y")))
    else:
        raise TypeError(f"Unsupported batch type for online teacher mode: {type(batch).__name__}")
    images = images.to(device=device, non_blocking=True)
    if cond is not None:
        cond = cond.to(device=device, non_blocking=True)
    return images, cond


def _base_density(config: dict, *, num_bins: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    dgtd_cfg = config.get("dgtd", {})
    mode = str(dgtd_cfg.get("base_density", "logit_normal"))
    if bool(dgtd_cfg.get("uniform_time", False)):
        mode = "uniform"
    if mode == "uniform":
        return torch.full((num_bins,), 1.0 / num_bins, device=device, dtype=dtype)
    centers = torch.linspace(0.5 / num_bins, 1.0 - 0.5 / num_bins, steps=num_bins, device=device, dtype=dtype)
    p_mean = float(dgtd_cfg.get("base_p_mean", -1.2))
    p_std = float(dgtd_cfg.get("base_p_std", 1.2))
    sigma = torch.clamp((1.0 - centers) / torch.clamp(centers, min=1.0e-6), min=1.0e-6)
    logits = (torch.log(sigma) - p_mean) / max(p_std, 1.0e-6)
    density = torch.exp(-0.5 * logits.square()) + 1.0e-6
    density = density / density.sum()
    return density


def select_dgtd_dataloaders(config: dict, teacher_adapter) -> tuple[dict, bool]:
    dgtd_cfg = config.get("dgtd", {})
    online_requested = not bool(dgtd_cfg.get("disable_online_teacher", False))
    use_online_teacher_data = bool(dgtd_cfg.get("use_online_teacher_data", True))
    if online_requested and use_online_teacher_data and not teacher_adapter.has_online_teacher():
        raise RuntimeError(
            "Online teacher mode was requested, but the online teacher could not be built. "
            "Check teacher configuration, connectivity, and local model cache."
        )
    online_teacher_data = use_online_teacher_data and teacher_adapter.has_online_teacher()
    dataloaders = build_image_dataloaders(config) if online_teacher_data else build_cache_dataloaders(config)
    return dataloaders, online_teacher_data


def _bin_ids_from_time(time_value: torch.Tensor, *, num_bins: int) -> torch.Tensor:
    return torch.clamp((time_value * num_bins).floor().long(), min=0, max=num_bins - 1)


def _global_bin_means(
    bin_ids: torch.Tensor,
    values: torch.Tensor,
    *,
    num_bins: int,
    device: torch.device,
    ctx: DistributedContext,
) -> tuple[torch.Tensor, torch.Tensor]:
    sums = torch.zeros(num_bins, device=device, dtype=values.dtype)
    counts = torch.zeros(num_bins, device=device, dtype=values.dtype)
    sums.scatter_add_(0, bin_ids, values)
    counts.scatter_add_(0, bin_ids, torch.ones_like(values))
    if ctx.enabled:
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    valid = counts > 0
    if not valid.any():
        return torch.zeros(0, device=device, dtype=torch.long), torch.zeros(0, device=device, dtype=values.dtype)
    mean_values = sums[valid] / counts[valid]
    return torch.arange(num_bins, device=device, dtype=torch.long)[valid], mean_values


def _stage_values(config: dict, *, global_step: int, total_steps: int) -> dict[str, float | str]:
    dgtd_cfg = config.get("dgtd", {})
    warmup_frac = float(dgtd_cfg.get("warmup_frac", 0.12))
    flatten_frac = float(dgtd_cfg.get("flatten_frac", 0.2))
    warmup_end = max(1, int(total_steps * warmup_frac))
    flatten_start = max(warmup_end + 1, int(total_steps * (1.0 - flatten_frac)))
    beta_final = float(dgtd_cfg.get("beta_final", dgtd_cfg.get("defect_beta", 0.75)))
    eta_start = float(dgtd_cfg.get("eta_start", 0.95))
    eta_min = float(dgtd_cfg.get("eta_min", 0.4))
    if global_step < warmup_end:
        return {
            "stage": "warmup",
            "beta": 0.0,
            "eta": eta_start,
            "mu": float(dgtd_cfg.get("ema_mu_warmup", 0.95)),
            "flatten_mix": 0.0,
            "warmup_end": float(warmup_end),
            "flatten_start": float(flatten_start),
        }
    if global_step < flatten_start:
        alpha = (global_step - warmup_end) / max(1, flatten_start - warmup_end)
        return {
            "stage": "adaptive",
            "beta": beta_final * alpha,
            "eta": eta_start - alpha * (eta_start - eta_min),
            "mu": float(dgtd_cfg.get("ema_mu_main", 0.99)),
            "flatten_mix": 0.0,
            "warmup_end": float(warmup_end),
            "flatten_start": float(flatten_start),
        }
    alpha = (global_step - flatten_start) / max(1, total_steps - flatten_start)
    return {
        "stage": "flatten",
        "beta": beta_final * (1.0 - 0.5 * alpha),
        "eta": eta_min,
        "mu": float(dgtd_cfg.get("ema_mu_main", 0.99)),
        "flatten_mix": min(0.5, 0.5 * alpha),
        "warmup_end": float(warmup_end),
        "flatten_start": float(flatten_start),
    }


def _per_sample_mse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x - y).flatten(1).square().mean(dim=1)


def _region_masks(time_value: torch.Tensor) -> dict[str, torch.Tensor]:
    noisy = time_value < (1.0 / 3.0)
    mid = (time_value >= (1.0 / 3.0)) & (time_value < (2.0 / 3.0))
    clean = time_value >= (2.0 / 3.0)
    return {
        "noisy": noisy,
        "mid": mid,
        "clean": clean,
    }


def _mean_or_zero(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if bool(mask.any()):
        return values[mask].mean()
    return torch.zeros((), device=values.device, dtype=values.dtype)


@dataclass(slots=True)
class DGTDTrainer:
    config: dict
    roots: RunRoots
    dist_ctx: DistributedContext
    archive: object | None = field(init=False, default=None)
    stats: dict[str, dict[str, torch.Tensor]] = field(init=False, default_factory=dict)
    q_base: torch.Tensor | None = field(init=False, default=None)
    q_target: torch.Tensor | None = field(init=False, default=None)

    def prepare(self) -> None:
        self.roots.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.roots.sample_dir.mkdir(parents=True, exist_ok=True)
        self.roots.log_dir.mkdir(parents=True, exist_ok=True)
        self.archive = build_experiment_archive(self.roots)
        if self.dist_ctx.is_main_process:
            with (self.roots.log_dir / "config_resolved.yaml").open("w", encoding="utf-8") as handle:
                yaml.safe_dump(self.config, handle, sort_keys=False)
            self.archive.dump_yaml("config_resolved.yaml", self.config)

    def _init_stats(self, num_bins: int, device: torch.device) -> None:
        zeros = torch.zeros(num_bins, device=device, dtype=torch.float32)
        self.stats = {
            "D": {"bar": zeros.clone(), "count": zeros.clone()},
            "K": {"bar": zeros.clone(), "count": zeros.clone()},
            "HF": {"bar": zeros.clone(), "count": zeros.clone()},
        }
        self.q_base = _base_density(self.config, num_bins=num_bins, device=device, dtype=torch.float32)
        self.q_target = self.q_base.clone()

    def _run_epoch(
        self,
        *,
        model: nn.Module,
        raw_model: nn.Module,
        ema: ModelEMA,
        loader,
        optimizer,
        scaler: torch.amp.GradScaler,
        warp,
        warp_optimizer,
        teacher_adapter,
        device: torch.device,
        total_steps: int,
        train: bool,
        global_step_start: int,
        online_teacher_data: bool,
    ) -> tuple[dict[str, float], int]:
        model.train(train)
        dgtd_cfg = self.config.get("dgtd", {})
        runtime_cfg = self.config.get("runtime", {})
        use_amp = bool(runtime_cfg.get("amp", True))
        batch_limit_key = "max_train_batches" if train else "max_val_batches"
        batch_limit = int(self.config.get("train", {}).get(batch_limit_key, 0) or 0)
        num_bins = warp.num_bins
        totals: dict[str, float] = {
            "loss": 0.0,
            "defect": 0.0,
            "hf": 0.0,
            "curvature": 0.0,
            "low_sigma_hf": 0.0,
            "omega": 0.0,
            "direct_teacher_error": 0.0,
            "bridge_state_teacher_error": 0.0,
            "bridge_u_teacher_error": 0.0,
            "direct_bridge_gap": 0.0,
            "teacher_rel_error": 0.0,
            "target_build_sec": 0.0,
            "online_teacher_traj_sec": 0.0,
            "forward_sec": 0.0,
            "warp_sec": 0.0,
            "samples_seen": 0.0,
            "batches_seen": 0.0,
            "warp_loss": 0.0,
            "source_online": 0.0,
            "source_cached_affine": 0.0,
            "source_cached_exact": 0.0,
            "source_bootstrap": 0.0,
            "noisy_count": 0.0,
            "mid_count": 0.0,
            "clean_count": 0.0,
            "noisy_defect": 0.0,
            "mid_defect": 0.0,
            "clean_defect": 0.0,
            "noisy_hf": 0.0,
            "mid_hf": 0.0,
            "clean_hf": 0.0,
            "noisy_endpoint_error": 0.0,
            "mid_endpoint_error": 0.0,
            "clean_endpoint_error": 0.0,
            "online_anchor_used": 0.0,
            "online_continuation_used": 0.0,
            "cached_fallback_used": 0.0,
            "exact_mask_hits": 0.0,
            "alpha_online_sum": 0.0,
            "alpha_online_count": 0.0,
            "alpha_online_min": 0.0,
            "alpha_online_max": 0.0,
        }
        stage_name = "warmup"
        eta_value = 1.0
        beta_value = 0.0
        mu_value = float(dgtd_cfg.get("ema_mu_warmup", 0.95))
        flatten_mix_value = 0.0
        global_step = global_step_start
        ctx = torch.enable_grad if train else torch.no_grad
        with ctx():
            for batch_idx, batch in enumerate(loader):
                if batch_limit > 0 and batch_idx >= batch_limit:
                    break
                if online_teacher_data:
                    traj_t0 = time.perf_counter()
                    x_0, cond = _extract_online_teacher_batch(batch, device)
                    batch = teacher_adapter.online_trajectory_from_x0(x_0, cond=cond, device=device)
                    totals["online_teacher_traj_sec"] += time.perf_counter() - traj_t0
                else:
                    batch = _move_batch_to_device(batch, device)
                batch_size = int(batch["states"].shape[0])
                stage = _stage_values(self.config, global_step=global_step, total_steps=total_steps)
                stage_name = str(stage["stage"])
                eta_value = float(stage["eta"])
                beta_value = float(stage["beta"])
                mu_value = float(stage["mu"])
                flatten_mix_value = float(stage["flatten_mix"])
                t0 = time.perf_counter()
                r_t, r_s, r_u = warp.sample_triplets(batch_size=batch_size, device=device)
                if bool(dgtd_cfg.get("uniform_time", False)):
                    t, s, u = r_t, r_s, r_u
                else:
                    t = warp.r_to_t(r_t)
                    s = warp.r_to_t(r_s)
                    u = warp.r_to_t(r_u)
                if not torch.all(t < s) or not torch.all(s < u):
                    raise AssertionError("Current dgfm convention for DGTD requires t < s < u")
                x_t = interpolate_state(batch, t)
                x_s_teacher = interpolate_state(batch, s)
                x_u_teacher = interpolate_state(batch, u)
                curvature = interpolate_curvature(batch, u)
                totals["target_build_sec"] += time.perf_counter() - t0
                cond = teacher_adapter.get_condition(batch)
                model_extra = {"label": cond} if cond is not None else {}
                forward_t0 = time.perf_counter()
                with _autocast_context(device, use_amp):
                    residual_info = compute_dgtd_residual(
                        model,
                        teacher_adapter,
                        x_t,
                        t,
                        s,
                        u,
                        eta_value if not bool(dgtd_cfg.get("disable_teacher_anchor", False)) else 0.0,
                        trajectory=batch,
                        extra=model_extra,
                    )
                    residual = residual_info["residual"]
                    if bool(dgtd_cfg.get("symmetric_residual", True)):
                        direct_residual = residual_info["x_u_direct"] - residual_info["x_u_cont"].detach()
                        bridge_residual = residual_info["x_u_cont"] - residual_info["x_u_direct"].detach()
                        metric_direct = metric_norm(
                            direct_residual,
                            u,
                            sigma_fn=teacher_adapter.sigma,
                            lambda_hf_max=float(dgtd_cfg.get("lambda_hf_max", 0.1)),
                            sigma_detail=float(dgtd_cfg.get("sigma_detail", 0.2)),
                            disable_hf_metric=bool(dgtd_cfg.get("disable_hf_metric", False)),
                        )
                        metric_bridge = metric_norm(
                            bridge_residual,
                            u,
                            sigma_fn=teacher_adapter.sigma,
                            lambda_hf_max=float(dgtd_cfg.get("lambda_hf_max", 0.1)),
                            sigma_detail=float(dgtd_cfg.get("sigma_detail", 0.2)),
                            disable_hf_metric=bool(dgtd_cfg.get("disable_hf_metric", False)),
                        )
                        metric_value = 0.5 * (metric_direct + metric_bridge)
                    else:
                        metric_value = metric_norm(
                            residual,
                            u,
                            sigma_fn=teacher_adapter.sigma,
                            lambda_hf_max=float(dgtd_cfg.get("lambda_hf_max", 0.1)),
                            sigma_detail=float(dgtd_cfg.get("sigma_detail", 0.2)),
                            disable_hf_metric=bool(dgtd_cfg.get("disable_hf_metric", False)),
                        )
                    p_corr = warp.density_at(t) if not bool(dgtd_cfg.get("uniform_time", False)) else torch.ones_like(t)
                    omega = edm_weight(
                        u,
                        sigma_fn=teacher_adapter.sigma,
                        sigma_data=float(dgtd_cfg.get("sigma_data", self.config.get("model", {}).get("sigma_data", 0.5))),
                    )
                    omega = omega * min_snr_weight(
                        u,
                        sigma_fn=teacher_adapter.sigma,
                        gamma=float(dgtd_cfg.get("min_snr_gamma", 5.0)),
                    )
                    omega = omega * torch.pow(torch.clamp(p_corr, min=1.0e-6), -float(dgtd_cfg.get("kappa", 0.5)))
                    omega = omega / torch.clamp(omega.mean(), min=1.0e-6)
                    loss = torch.mean(omega.detach() * metric_value)
                totals["forward_sec"] += time.perf_counter() - forward_t0
                if train:
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    ema.update(unwrap_model(model))
                sample_defect = compute_sample_defect(
                    residual.detach(),
                    x_t,
                    x_u_teacher,
                    metric_value.detach(),
                )
                hf_values = high_frequency_norm(laplacian_filter(residual.detach()))
                direct_teacher_error = _per_sample_mse(residual_info["x_u_direct"].detach(), x_u_teacher)
                bridge_state_teacher_error = _per_sample_mse(residual_info["x_s_pred"].detach(), x_s_teacher)
                bridge_u_teacher_error = _per_sample_mse(residual_info["bridge_cont"].detach(), x_u_teacher)
                direct_bridge_gap = _per_sample_mse(residual_info["x_u_direct"].detach(), residual_info["x_u_cont"].detach())
                teacher_rel_error = residual_info["teacher_rel_error"]
                if teacher_rel_error is None:
                    teacher_rel_error = torch.zeros_like(direct_teacher_error)
                teacher_exact_mask = residual_info["teacher_exact_mask"]
                if teacher_exact_mask is None:
                    teacher_exact_mask = torch.zeros_like(direct_teacher_error, dtype=torch.bool)
                teacher_used_online_anchor = residual_info["teacher_used_online_anchor"]
                if teacher_used_online_anchor is None:
                    teacher_used_online_anchor = torch.zeros_like(direct_teacher_error, dtype=torch.bool)
                teacher_alpha = residual_info["teacher_alpha"]
                source_ids = residual_info["source_ids"]
                bin_ids = _bin_ids_from_time(u.detach(), num_bins=num_bins)
                ids, values = _global_bin_means(bin_ids, sample_defect, num_bins=num_bins, device=device, ctx=self.dist_ctx)
                if ids.numel() > 0:
                    update_ema_bins(self.stats["D"], ids, values, float(stage["mu"]))
                ids, values = _global_bin_means(bin_ids, curvature.detach(), num_bins=num_bins, device=device, ctx=self.dist_ctx)
                if ids.numel() > 0:
                    update_ema_bins(self.stats["K"], ids, values, float(stage["mu"]))
                ids, values = _global_bin_means(bin_ids, hf_values, num_bins=num_bins, device=device, ctx=self.dist_ctx)
                if ids.numel() > 0:
                    update_ema_bins(self.stats["HF"], ids, values, float(stage["mu"]))

                if train and global_step >= int(stage["warmup_end"]):
                    update_density_every = max(1, int(dgtd_cfg.get("update_density_every", 32)))
                    if global_step % update_density_every == 0:
                        q_target = build_target_density(
                            self.stats["D"]["bar"],
                            self.stats["K"]["bar"],
                            self.stats["HF"]["bar"],
                            self.q_base,
                            beta_value,
                            1.0e-6,
                            curvature_weight=float(
                                dgtd_cfg.get("qd_curvature_weight", dgtd_cfg.get("curvature_weight", 0.25))
                            ),
                            use_hf_bar=bool(dgtd_cfg.get("qd_use_hf_bar", True)),
                            hf_weight=float(dgtd_cfg.get("qd_hf_weight", dgtd_cfg.get("hf_weight", 0.5))),
                            ratio_cap=float(dgtd_cfg.get("qd_ratio_cap", 0.0)),
                        )
                        q_target = smooth_density(q_target)
                        if flatten_mix_value > 0.0:
                            q_target = (1.0 - flatten_mix_value) * q_target + flatten_mix_value * self.q_base
                            q_target = q_target / torch.clamp(q_target.sum(), min=1.0e-6)
                        self.q_target = q_target.detach()
                    if not bool(dgtd_cfg.get("disable_warp", False)):
                        update_warp_every = max(1, int(dgtd_cfg.get("update_warp_every", 32)))
                        if global_step % update_warp_every == 0 and warp_optimizer is not None:
                            warp_t0 = time.perf_counter()
                            warp_optimizer.zero_grad(set_to_none=True)
                            warp_loss = warp.kl_to_target_density(self.q_target)
                            warp_loss.backward()
                            warp_optimizer.step()
                            totals["warp_sec"] += time.perf_counter() - warp_t0
                            totals["warp_loss"] += float(warp_loss.detach().item())
                sigma_u = teacher_adapter.sigma(u.detach())
                low_sigma_mask = sigma_u <= float(dgtd_cfg.get("sigma_detail", 0.2))
                low_sigma_hf = hf_values[low_sigma_mask].mean() if bool(low_sigma_mask.any()) else torch.zeros((), device=device)

                totals["loss"] += float(loss.detach().item())
                totals["defect"] += float(sample_defect.mean().item())
                totals["hf"] += float(hf_values.mean().item())
                totals["curvature"] += float(curvature.mean().item())
                totals["low_sigma_hf"] += float(low_sigma_hf.item())
                totals["omega"] += float(omega.mean().item())
                totals["direct_teacher_error"] += float(direct_teacher_error.mean().item())
                totals["bridge_state_teacher_error"] += float(bridge_state_teacher_error.mean().item())
                totals["bridge_u_teacher_error"] += float(bridge_u_teacher_error.mean().item())
                totals["direct_bridge_gap"] += float(direct_bridge_gap.mean().item())
                totals["teacher_rel_error"] += float(teacher_rel_error.mean().item())
                totals["samples_seen"] += float(batch_size)
                totals["batches_seen"] += 1.0
                totals["source_online"] += float((source_ids == CONTINUATION_SOURCE_ONLINE).sum().item())
                totals["source_cached_affine"] += float((source_ids == CONTINUATION_SOURCE_CACHED_AFFINE).sum().item())
                totals["source_cached_exact"] += float((source_ids == CONTINUATION_SOURCE_CACHED_EXACT).sum().item())
                totals["source_bootstrap"] += float((source_ids == CONTINUATION_SOURCE_BOOTSTRAP).sum().item())
                totals["online_anchor_used"] += float(teacher_used_online_anchor.sum().item())
                totals["online_continuation_used"] += float((source_ids == CONTINUATION_SOURCE_ONLINE).sum().item())
                totals["cached_fallback_used"] += float(
                    ((source_ids == CONTINUATION_SOURCE_CACHED_AFFINE) | (source_ids == CONTINUATION_SOURCE_CACHED_EXACT)).sum().item()
                )
                totals["exact_mask_hits"] += float(teacher_exact_mask.sum().item())
                if teacher_alpha is not None:
                    alpha_online = teacher_alpha[teacher_used_online_anchor]
                    if alpha_online.numel() > 0:
                        totals["alpha_online_sum"] += float(alpha_online.sum().item())
                        totals["alpha_online_count"] += float(alpha_online.numel())
                        current_min = float(alpha_online.min().item())
                        current_max = float(alpha_online.max().item())
                        if totals["alpha_online_count"] == float(alpha_online.numel()):
                            totals["alpha_online_min"] = current_min
                            totals["alpha_online_max"] = current_max
                        else:
                            totals["alpha_online_min"] = min(totals["alpha_online_min"], current_min)
                            totals["alpha_online_max"] = max(totals["alpha_online_max"], current_max)
                region_masks = _region_masks(u.detach())
                for region_name, mask in region_masks.items():
                    count = float(mask.sum().item())
                    totals[f"{region_name}_count"] += count
                    if count <= 0.0:
                        continue
                    totals[f"{region_name}_defect"] += float(sample_defect[mask].sum().item())
                    totals[f"{region_name}_hf"] += float(hf_values[mask].sum().item())
                    totals[f"{region_name}_endpoint_error"] += float(direct_teacher_error[mask].sum().item())
                if train:
                    global_step += 1
        denom = max(1.0, totals["batches_seen"])
        payload = {
            "loss": totals["loss"] / denom,
            "defect": totals["defect"] / denom,
            "hf": totals["hf"] / denom,
            "curvature": totals["curvature"] / denom,
            "low_sigma_hf": totals["low_sigma_hf"] / denom,
            "omega": totals["omega"] / denom,
            "direct_teacher_error": totals["direct_teacher_error"] / denom,
            "bridge_state_teacher_error": totals["bridge_state_teacher_error"] / denom,
            "bridge_u_teacher_error": totals["bridge_u_teacher_error"] / denom,
            "direct_bridge_gap": totals["direct_bridge_gap"] / denom,
            "teacher_rel_error": totals["teacher_rel_error"] / denom,
            "teacher_rel_error_mean": totals["teacher_rel_error"] / denom,
            "target_build_sec": totals["target_build_sec"],
            "online_teacher_traj_sec": totals["online_teacher_traj_sec"],
            "forward_sec": totals["forward_sec"],
            "warp_sec": totals["warp_sec"],
            "warp_loss": totals["warp_loss"] / denom,
            "samples_seen": totals["samples_seen"],
            "batches_seen": totals["batches_seen"],
            "eta": eta_value,
            "beta": beta_value,
            "mu": mu_value,
            "flatten_mix": flatten_mix_value,
            "source_online": totals["source_online"],
            "source_cached_affine": totals["source_cached_affine"],
            "source_cached_exact": totals["source_cached_exact"],
            "source_bootstrap": totals["source_bootstrap"],
            "online_anchor_used": totals["online_anchor_used"],
            "online_continuation_used": totals["online_continuation_used"],
            "cached_fallback_used": totals["cached_fallback_used"],
            "exact_mask_hits": totals["exact_mask_hits"],
            "alpha_online_sum": totals["alpha_online_sum"],
            "alpha_online_count": totals["alpha_online_count"],
            "alpha_online_min": totals["alpha_online_min"],
            "alpha_online_max": totals["alpha_online_max"],
            "noisy_count": totals["noisy_count"],
            "mid_count": totals["mid_count"],
            "clean_count": totals["clean_count"],
            "noisy_defect": totals["noisy_defect"],
            "mid_defect": totals["mid_defect"],
            "clean_defect": totals["clean_defect"],
            "noisy_hf": totals["noisy_hf"],
            "mid_hf": totals["mid_hf"],
            "clean_hf": totals["clean_hf"],
            "noisy_endpoint_error": totals["noisy_endpoint_error"],
            "mid_endpoint_error": totals["mid_endpoint_error"],
            "clean_endpoint_error": totals["clean_endpoint_error"],
        }
        payload = reduce_float_dict(
            payload,
            device=device,
            ctx=self.dist_ctx,
            mean_keys=[
                "loss",
                "defect",
                "hf",
                "curvature",
                "low_sigma_hf",
                "omega",
                "warp_loss",
                "eta",
                "beta",
                "mu",
                "flatten_mix",
                "direct_teacher_error",
                "bridge_state_teacher_error",
                "bridge_u_teacher_error",
                "direct_bridge_gap",
                "teacher_rel_error",
                "teacher_rel_error_mean",
            ],
            sum_keys=[
                "samples_seen",
                "batches_seen",
                "target_build_sec",
                "online_teacher_traj_sec",
                "forward_sec",
                "warp_sec",
                "source_online",
                "source_cached_affine",
                "source_cached_exact",
                "source_bootstrap",
                "online_anchor_used",
                "online_continuation_used",
                "cached_fallback_used",
                "exact_mask_hits",
                "alpha_online_sum",
                "alpha_online_count",
                "noisy_count",
                "mid_count",
                "clean_count",
                "noisy_defect",
                "mid_defect",
                "clean_defect",
                "noisy_hf",
                "mid_hf",
                "clean_hf",
                "noisy_endpoint_error",
                "mid_endpoint_error",
                "clean_endpoint_error",
            ],
        )
        samples_seen = max(float(payload["samples_seen"]), 1.0)
        for source_name in ("online", "cached_affine", "cached_exact", "bootstrap"):
            payload[f"source_ratio_{source_name}"] = payload[f"source_{source_name}"] / samples_seen
        payload["online_anchor_used_rate"] = payload["online_anchor_used"] / samples_seen
        payload["online_continuation_rate"] = payload["online_continuation_used"] / samples_seen
        payload["cached_fallback_rate"] = payload["cached_fallback_used"] / samples_seen
        payload["exact_mask_hit_rate"] = payload["exact_mask_hits"] / samples_seen
        alpha_count = max(float(payload["alpha_online_count"]), 1.0)
        payload["alpha_online_mean"] = payload["alpha_online_sum"] / alpha_count if float(payload["alpha_online_count"]) > 0.0 else 0.0
        if float(payload["alpha_online_count"]) <= 0.0:
            payload["alpha_online_min"] = 0.0
            payload["alpha_online_max"] = 0.0
        for region_name in ("noisy", "mid", "clean"):
            count = max(float(payload[f"{region_name}_count"]), 1.0)
            payload[f"{region_name}_defect"] = payload[f"{region_name}_defect"] / count
            payload[f"{region_name}_hf"] = payload[f"{region_name}_hf"] / count
            payload[f"{region_name}_endpoint_error"] = payload[f"{region_name}_endpoint_error"] / count
        grid = build_runtime_time_grid(
            config=self.config,
            step_count=int(self.config.get("eval", {}).get("default_sample_steps", 16)),
            device=device,
            dtype=torch.float32,
            timewarp=warp,
        )
        q_phi = warp.density().detach()
        q_target = self.q_target.detach()
        q_base = self.q_base.detach()
        entropy_q_phi = float((-(q_phi * torch.log(torch.clamp(q_phi, min=1.0e-6))).sum()).item())
        kl_qD_qphi = float(warp.kl_to_target_density(q_target).detach().item())
        max_qphi_over_qbase = float(torch.max(q_phi / torch.clamp(q_base, min=1.0e-6)).item())
        argmax_qphi = int(torch.argmax(q_phi).item())
        payload.update(
            {
                "stage": stage_name,
                "q_phi": [float(item) for item in q_phi.cpu().tolist()],
                "q_D": [float(item) for item in q_target.cpu().tolist()],
                "D_bar": [float(item) for item in self.stats["D"]["bar"].detach().cpu().tolist()],
                "K_bar": [float(item) for item in self.stats["K"]["bar"].detach().cpu().tolist()],
                "HF_bar": [float(item) for item in self.stats["HF"]["bar"].detach().cpu().tolist()],
                "time_grid": summarize_time_grid(grid)["time_grid"],
                "entropy_q_phi": entropy_q_phi,
                "kl_qD_qphi": kl_qD_qphi,
                "max_qphi_over_qbase": max_qphi_over_qbase,
                "argmax_q_phi": argmax_qphi,
                "continuation_sources": {
                    source_name: float(payload[f"source_ratio_{source_name}"])
                    for source_name in ("online", "cached_affine", "cached_exact", "bootstrap")
                },
            }
        )
        return payload, global_step

    def run(self, resume: str | None = None, verbose: bool = False) -> None:
        self.prepare()
        device = self.dist_ctx.device if self.dist_ctx.enabled and self.dist_ctx.device is not None else _device_from_config(self.config)
        teacher_adapter = build_teacher_adapter(self.config)
        teacher_adapter.prepare(device)
        dataloaders, online_teacher_data = select_dgtd_dataloaders(self.config, teacher_adapter)
        raw_model = build_map_model(self.config).to(device)
        model = maybe_wrap_ddp(raw_model, self.dist_ctx, find_unused_parameters=False)
        ema = ModelEMA(raw_model, decay=float(self.config["train"].get("ema_decay", 0.9999)))
        optimizer = torch.optim.AdamW(
            raw_model.parameters(),
            lr=float(self.config["train"].get("lr", 2.0e-4)),
            weight_decay=float(self.config["train"].get("weight_decay", 0.0)),
            betas=tuple(self.config["train"].get("optimizer_betas", [0.9, 0.95])),
        )
        warp = build_timewarp_module(self.config, device=device, dtype=torch.float32)
        if warp is None:
            raise ValueError("DGTDTrainer requires scheduler.timewarp.enabled=true with a supported warp type")
        warp_optimizer = torch.optim.AdamW(
            warp.parameters(),
            lr=float(self.config.get("scheduler", {}).get("timewarp", {}).get("lr", 1.0e-3)),
            weight_decay=float(self.config.get("scheduler", {}).get("timewarp", {}).get("weight_decay", 0.0)),
            betas=tuple(self.config["train"].get("optimizer_betas", [0.9, 0.95])),
        )
        scaler = torch.amp.GradScaler(device="cuda", enabled=device.type == "cuda" and bool(self.config.get("runtime", {}).get("amp", True)))
        self._init_stats(num_bins=warp.num_bins, device=device)
        start_epoch = 0
        best_val = float("inf")
        global_step = 0
        if resume:
            ckpt = torch.load(resume, map_location=device)
            raw_model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            if "ema_model" in ckpt:
                ema.load_state_dict(ckpt["ema_model"])
            if ckpt.get("timewarp") is not None:
                warp.load_state_dict(ckpt["timewarp"])
            if ckpt.get("timewarp_optimizer") is not None:
                warp_optimizer.load_state_dict(ckpt["timewarp_optimizer"])
            if ckpt.get("dgtd_q_target") is not None:
                self.q_target = ckpt["dgtd_q_target"].to(device)
            if ckpt.get("dgtd_stats") is not None:
                saved_stats = ckpt["dgtd_stats"]
                for name in ("D", "K", "HF"):
                    self.stats[name]["bar"].copy_(saved_stats[name]["bar"].to(device))
                    self.stats[name]["count"].copy_(saved_stats[name]["count"].to(device))
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_val = float(ckpt.get("best_val", best_val))
            global_step = int(ckpt.get("global_step", 0))
            if ckpt.get("scaler") is not None and scaler.is_enabled():
                scaler.load_state_dict(ckpt["scaler"])

        epochs = int(self.config["train"].get("epochs", 1))
        history_path = self.roots.log_dir / "train.jsonl"
        completed_epochs, total_elapsed_sec = _load_elapsed_history(history_path)
        total_steps = epochs * max(1, len(dataloaders["train"]))
        for epoch in range(start_epoch, epochs):
            set_dataloader_epoch(dataloaders["train"], epoch)
            set_dataloader_epoch(dataloaders["val"], epoch)
            t0 = time.time()
            train_stats, global_step = self._run_epoch(
                model=model,
                raw_model=raw_model,
                ema=ema,
                loader=dataloaders["train"],
                optimizer=optimizer,
                scaler=scaler,
                warp=warp,
                warp_optimizer=warp_optimizer,
                teacher_adapter=teacher_adapter,
                device=device,
                total_steps=total_steps,
                train=True,
                global_step_start=global_step,
                online_teacher_data=online_teacher_data,
            )
            eval_model = ema.shadow if ema is not None else raw_model
            val_stats, _ = self._run_epoch(
                model=eval_model,
                raw_model=eval_model,
                ema=ema,
                loader=dataloaders["val"],
                optimizer=optimizer,
                scaler=scaler,
                warp=warp,
                warp_optimizer=warp_optimizer,
                teacher_adapter=teacher_adapter,
                device=device,
                total_steps=total_steps,
                train=False,
                global_step_start=global_step,
                online_teacher_data=online_teacher_data,
            )
            elapsed = time.time() - t0
            total_elapsed_sec += elapsed
            completed_epochs += 1
            avg_epoch_sec = total_elapsed_sec / max(completed_epochs, 1)
            payload = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "val_loss": val_stats["loss"],
                "train_defect": train_stats["defect"],
                "val_defect": val_stats["defect"],
                "train_low_sigma_hf": train_stats["low_sigma_hf"],
                "val_low_sigma_hf": val_stats["low_sigma_hf"],
                "train_direct_teacher_error": train_stats["direct_teacher_error"],
                "val_direct_teacher_error": val_stats["direct_teacher_error"],
                "train_bridge_state_teacher_error": train_stats["bridge_state_teacher_error"],
                "val_bridge_state_teacher_error": val_stats["bridge_state_teacher_error"],
                "train_bridge_u_teacher_error": train_stats["bridge_u_teacher_error"],
                "val_bridge_u_teacher_error": val_stats["bridge_u_teacher_error"],
                "train_direct_bridge_gap": train_stats["direct_bridge_gap"],
                "val_direct_bridge_gap": val_stats["direct_bridge_gap"],
                "train_teacher_rel_error_mean": train_stats["teacher_rel_error_mean"],
                "val_teacher_rel_error_mean": val_stats["teacher_rel_error_mean"],
                "train_target_build_sec": train_stats["target_build_sec"],
                "train_online_teacher_traj_sec": train_stats["online_teacher_traj_sec"],
                "train_forward_sec": train_stats["forward_sec"],
                "train_warp_sec": train_stats["warp_sec"],
                "train_warp_loss": train_stats["warp_loss"],
                "entropy_q_phi": train_stats["entropy_q_phi"],
                "kl_qD_qphi": train_stats["kl_qD_qphi"],
                "max_qphi_over_qbase": train_stats["max_qphi_over_qbase"],
                "argmax_q_phi": train_stats["argmax_q_phi"],
                "q_phi": train_stats["q_phi"],
                "q_D": train_stats["q_D"],
                "D_bar": train_stats["D_bar"],
                "K_bar": train_stats["K_bar"],
                "HF_bar": train_stats["HF_bar"],
                "time_grid": train_stats["time_grid"],
                "eta": train_stats["eta"],
                "beta": train_stats["beta"],
                "mu": train_stats["mu"],
                "flatten_mix": train_stats["flatten_mix"],
                "stage": train_stats["stage"],
                "continuation_sources": train_stats["continuation_sources"],
                "online_anchor_used_rate": train_stats["online_anchor_used_rate"],
                "online_continuation_rate": train_stats["online_continuation_rate"],
                "cached_fallback_rate": train_stats["cached_fallback_rate"],
                "exact_mask_hit_rate": train_stats["exact_mask_hit_rate"],
                "alpha_online_mean": train_stats["alpha_online_mean"],
                "alpha_online_min": train_stats["alpha_online_min"],
                "alpha_online_max": train_stats["alpha_online_max"],
                "train_noisy_defect": train_stats["noisy_defect"],
                "train_mid_defect": train_stats["mid_defect"],
                "train_clean_defect": train_stats["clean_defect"],
                "train_noisy_hf": train_stats["noisy_hf"],
                "train_mid_hf": train_stats["mid_hf"],
                "train_clean_hf": train_stats["clean_hf"],
                "train_noisy_endpoint_error": train_stats["noisy_endpoint_error"],
                "train_mid_endpoint_error": train_stats["mid_endpoint_error"],
                "train_clean_endpoint_error": train_stats["clean_endpoint_error"],
                "elapsed_sec": elapsed,
                "epoch_sec_avg": avg_epoch_sec,
                "global_step": global_step,
                "online_teacher_data": online_teacher_data,
            }
            if self.dist_ctx.is_main_process:
                with history_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload) + "\n")
                self.archive.append_jsonl("train.jsonl", payload)
            ckpt = {
                "epoch": epoch,
                "model": raw_model.state_dict(),
                "ema_model": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                "best_val": best_val,
                "global_step": global_step,
                "config": self.config,
                "timewarp": warp.state_dict(),
                "timewarp_optimizer": warp_optimizer.state_dict(),
                "dgtd_q_target": self.q_target.detach().cpu(),
                "dgtd_stats": {
                    name: {
                        "bar": state["bar"].detach().cpu(),
                        "count": state["count"].detach().cpu(),
                    }
                    for name, state in self.stats.items()
                },
            }
            if self.dist_ctx.is_main_process:
                torch.save(ckpt, self.roots.checkpoint_dir / "last.pt")
                self.archive.save_checkpoint("last.pt", ckpt)
                if val_stats["loss"] <= best_val:
                    best_val = val_stats["loss"]
                    ckpt["best_val"] = best_val
                    torch.save(ckpt, self.roots.checkpoint_dir / "best.pt")
                    self.archive.save_checkpoint("best.pt", ckpt)
                print(
                    f"epoch={epoch + 1}/{epochs} train_loss={train_stats['loss']:.6f} "
                    f"val_loss={val_stats['loss']:.6f} defect={train_stats['defect']:.6f} "
                    f"low_sigma_hf={train_stats['low_sigma_hf']:.6f} stage={train_stats['stage']} "
                    f"eta={train_stats['eta']:.4f} beta={train_stats['beta']:.4f} "
                    f"kl_qD_qphi={train_stats['kl_qD_qphi']:.6f} "
                    f"online_cont={train_stats['online_continuation_rate']:.3f} "
                    f"online_teacher_data={'yes' if online_teacher_data else 'no'} "
                    f"elapsed_sec={elapsed:.2f}",
                    flush=True,
                )
                if verbose:
                    print(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "q_phi": train_stats["q_phi"],
                                "q_D": train_stats["q_D"],
                                "D_bar": train_stats["D_bar"],
                                "K_bar": train_stats["K_bar"],
                                "HF_bar": train_stats["HF_bar"],
                                "time_grid": train_stats["time_grid"],
                                "continuation_sources": train_stats["continuation_sources"],
                                "online_anchor_used_rate": train_stats["online_anchor_used_rate"],
                                "online_continuation_rate": train_stats["online_continuation_rate"],
                                "cached_fallback_rate": train_stats["cached_fallback_rate"],
                                "exact_mask_hit_rate": train_stats["exact_mask_hit_rate"],
                                "alpha_online_mean": train_stats["alpha_online_mean"],
                                "alpha_online_min": train_stats["alpha_online_min"],
                                "alpha_online_max": train_stats["alpha_online_max"],
                                "entropy_q_phi": train_stats["entropy_q_phi"],
                                "kl_qD_qphi": train_stats["kl_qD_qphi"],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
