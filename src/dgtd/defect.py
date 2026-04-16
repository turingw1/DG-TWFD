from __future__ import annotations

import torch
from torch import Tensor


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
    teacher_cont = teacher_adapter.local_flow(
        trajectory,
        s,
        u,
        x_s_pred,
        extra=model_extra,
    )
    self_cont = student(x_s_pred.detach(), s, u, extra=model_extra).detach()
    if teacher_cont is None:
        target = self_cont
    else:
        target = float(eta) * teacher_cont.detach() + (1.0 - float(eta)) * self_cont
    residual = x_u_direct - target
    return {
        "x_s_pred": x_s_pred,
        "x_u_direct": x_u_direct,
        "teacher_cont": teacher_cont,
        "self_cont": self_cont,
        "target": target,
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


def build_target_density(
    D_bar: Tensor,
    K_bar: Tensor,
    HF_bar: Tensor,
    q_base: Tensor,
    beta: float,
    eps: float,
    *,
    curvature_weight: float = 0.25,
    hf_weight: float = 0.5,
) -> Tensor:
    q_base = _normalize_signal(q_base, eps)
    if beta <= 0.0:
        return q_base
    A = (
        _normalize_signal(D_bar, eps)
        + float(curvature_weight) * _normalize_signal(K_bar, eps)
        + float(hf_weight) * _normalize_signal(HF_bar, eps)
    )
    q_D = torch.pow(torch.clamp(A + float(eps), min=float(eps)), float(beta)) * q_base
    return _normalize_signal(q_D, eps)


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
