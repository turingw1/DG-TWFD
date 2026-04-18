from __future__ import annotations

import torch
from torch import Tensor

from .teacher import CONTINUATION_SOURCE_BOOTSTRAP


def compute_dgtd_residual(
    student,
    teacher_adapter,
    x_t: Tensor,
    t: Tensor,
    s: Tensor,
    u: Tensor,
    eta: float,
    *,
    trajectory=None,
    extra: dict | None = None,
) -> dict[str, Tensor | None]:
    if not torch.all(t < s):
        raise AssertionError("DGTD on the current dgfm line expects t < s")
    if not torch.all(s < u):
        raise AssertionError("DGTD on the current dgfm line expects s < u")
    model_extra = dict(extra or {})
    x_s_pred = student(x_t, t, s, extra=model_extra)
    x_u_direct = student(x_t, t, u, extra=model_extra)
    teacher_info = teacher_adapter.local_flow(
        trajectory,
        s,
        u,
        x_s_pred,
        extra=model_extra,
    )
    bridge_cont = student(x_s_pred, s, u, extra=model_extra)
    teacher_cont = teacher_info.continuation
    source_ids = teacher_info.source_ids
    if teacher_cont is None:
        x_u_cont = bridge_cont
        source_ids = torch.full_like(source_ids, CONTINUATION_SOURCE_BOOTSTRAP)
    else:
        x_u_cont = float(eta) * teacher_cont + (1.0 - float(eta)) * bridge_cont
    residual = x_u_direct - x_u_cont
    return {
        "x_s_pred": x_s_pred,
        "x_u_direct": x_u_direct,
        "x_u_cont": x_u_cont,
        "bridge_cont": bridge_cont,
        "teacher_cont": teacher_cont,
        "teacher_s": teacher_info.teacher_s,
        "teacher_u": teacher_info.teacher_u,
        "teacher_rel_error": teacher_info.rel_error,
        "teacher_alpha": teacher_info.alpha,
        "teacher_exact_mask": teacher_info.exact_mask,
        "teacher_used_online_anchor": teacher_info.used_online_anchor,
        "source_ids": source_ids,
        "residual": residual,
    }


def compute_sample_defect(
    R: Tensor,
    x_t: Tensor,
    x_u_teacher: Tensor,
    metric_value: Tensor,
    *,
    eps: float = 1.0e-6,
) -> Tensor:
    del R
    denom = (x_u_teacher - x_t).flatten(1).square().mean(dim=1)
    return metric_value / torch.clamp(denom + float(eps), min=float(eps))


def update_ema_bins(stats: dict[str, Tensor], bin_ids: Tensor, values: Tensor, mu: float) -> None:
    if "bar" not in stats or "count" not in stats:
        raise KeyError("stats must contain 'bar' and 'count'")
    bar = stats["bar"]
    count = stats["count"]
    for bin_id in torch.unique(bin_ids):
        mask = bin_ids == bin_id
        if not mask.any():
            continue
        idx = int(bin_id.item())
        mean_value = values[mask].mean().to(device=bar.device, dtype=bar.dtype)
        bar[idx] = float(mu) * bar[idx] + (1.0 - float(mu)) * mean_value
        count[idx] = count[idx] + mask.sum().to(device=count.device, dtype=count.dtype)


def _normalize_signal(x: Tensor, eps: float) -> Tensor:
    x = torch.clamp(x, min=0.0)
    total = x.sum()
    if float(total.item()) <= eps:
        return torch.full_like(x, 1.0 / x.numel())
    return x / torch.clamp(total, min=eps)


def _apply_ratio_cap(q: Tensor, q_base: Tensor, ratio_cap: float, eps: float) -> Tensor:
    q_base = _normalize_signal(q_base, eps)
    ratio_0 = q / torch.clamp(q_base, min=eps)
    fixed = torch.zeros_like(ratio_0, dtype=torch.bool)
    ratio = torch.zeros_like(ratio_0)
    for _ in range(ratio_0.numel() + 1):
        free = ~fixed
        fixed_mass = torch.sum(q_base[fixed] * float(ratio_cap))
        remaining_mass = max(0.0, 1.0 - float(fixed_mass.item()))
        if bool(free.any()):
            base_free = q_base[free]
            ratio_free = ratio_0[free]
            weighted = torch.sum(base_free * ratio_free)
            if float(weighted.item()) <= float(eps):
                scaled_free = torch.full_like(ratio_free, remaining_mass / max(float(base_free.sum().item()), float(eps)))
            else:
                scaled_free = ratio_free * (remaining_mass / float(weighted.item()))
            overflow = scaled_free > float(ratio_cap)
            if bool(overflow.any()):
                overflow_indices = torch.nonzero(free, as_tuple=False).view(-1)[overflow]
                fixed[overflow_indices] = True
                continue
            ratio[free] = scaled_free
        ratio[fixed] = float(ratio_cap)
        break
    capped = q_base * ratio
    return _normalize_signal(capped, eps)


def build_target_density(
    D_bar: Tensor,
    K_bar: Tensor,
    HF_bar: Tensor,
    q_base: Tensor,
    beta: float,
    eps: float,
    *,
    curvature_weight: float = 0.25,
    use_hf_bar: bool = True,
    hf_weight: float = 0.5,
    ratio_cap: float = 0.0,
) -> Tensor:
    q_base = _normalize_signal(q_base, eps)
    if beta <= 0.0:
        return q_base
    A = _normalize_signal(D_bar, eps) + float(curvature_weight) * _normalize_signal(K_bar, eps)
    if use_hf_bar and float(hf_weight) > 0.0:
        A = A + float(hf_weight) * _normalize_signal(HF_bar, eps)
    ratio = torch.pow(torch.clamp(A + float(eps), min=float(eps)), float(beta))
    q_D = ratio * q_base
    q_D = _normalize_signal(q_D, eps)
    if float(ratio_cap) > 0.0:
        q_D = _apply_ratio_cap(q_D, q_base, float(ratio_cap), float(eps))
    return q_D


def smooth_density(q: Tensor, kernel: Tensor | None = None) -> Tensor:
    q = q.view(1, 1, -1)
    if kernel is None:
        kernel = torch.tensor([0.25, 0.5, 0.25], device=q.device, dtype=q.dtype)
    kernel = kernel.view(1, 1, -1)
    pad = kernel.shape[-1] // 2
    padded = torch.nn.functional.pad(q, (pad, pad), mode="replicate")
    smoothed = torch.nn.functional.conv1d(padded, kernel)
    smoothed = smoothed.view(-1)
    return smoothed / torch.clamp(smoothed.sum(), min=1.0e-6)
