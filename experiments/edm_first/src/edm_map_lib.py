from __future__ import annotations

import copy
import json
import os
import pickle
import random
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
EDM_ROOT = REPO_ROOT / "refs" / "edm"
SRC_ROOT = REPO_ROOT / "src"


def prepare_imports() -> None:
    for path in (str(SRC_ROOT), str(EDM_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)


prepare_imports()

from dgtd.defect import build_target_density, smooth_density, update_ema_bins  # noqa: E402
from dgtd.warp import MonotoneDensityWarp, MonotoneRationalQuadraticSplineWarp  # noqa: E402


_ENV_DEFAULT_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):-([^}]*)\}")


def _expand_env_defaults(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name, fallback = match.group(1), match.group(2)
        return os.environ.get(name, fallback)

    return os.path.expandvars(_ENV_DEFAULT_RE.sub(replace, text))


def load_config(path: str | Path) -> dict:
    text = _expand_env_defaults(Path(path).read_text(encoding="utf-8"))
    cfg = yaml.safe_load(text) or {}
    cfg["_config_path"] = str(path)
    return cfg


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(value: str | Path, *, root: Path = REPO_ROOT) -> Path | str:
    raw = str(value)
    if raw.startswith("http://") or raw.startswith("https://"):
        return raw
    path = Path(raw)
    return path if path.is_absolute() else root / path


def load_edm_network(network: str | Path, *, device: torch.device, use_fp16: bool) -> torch.nn.Module:
    import dnnlib

    network_path = str(resolve_path(network))
    with dnnlib.util.open_url(network_path) as handle:
        net = pickle.load(handle)["ema"].to(device)
    if hasattr(net, "use_fp16") and device.type == "cuda":
        net.use_fp16 = bool(use_fp16)
    net.eval()
    return net


def clone_student_from_teacher(teacher: torch.nn.Module) -> torch.nn.Module:
    student = copy.deepcopy(teacher)
    student.train()
    student.requires_grad_(True)
    if hasattr(student, "use_fp16"):
        student.use_fp16 = False
    return student


def make_labels(label_ids: torch.Tensor, *, label_dim: int, device: torch.device) -> torch.Tensor | None:
    if label_dim <= 0:
        return None
    return F.one_hot(label_ids.to(device=device, dtype=torch.long), num_classes=label_dim).to(torch.float32)


def sample_labels(net: torch.nn.Module, batch_size: int, *, device: torch.device, generator=None) -> torch.Tensor | None:
    label_dim = int(getattr(net, "label_dim", 0) or 0)
    if label_dim <= 0:
        return None
    label_ids = torch.randint(label_dim, (batch_size,), device=device, generator=generator)
    return make_labels(label_ids, label_dim=label_dim, device=device)


def sigma_from_u(u: torch.Tensor, *, sigma_min: float, sigma_max: float, rho: float, net=None) -> torch.Tensor:
    u = u.clamp(0.0, 1.0)
    sigma = (sigma_max ** (1.0 / rho) + u * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))) ** rho
    if net is not None and hasattr(net, "round_sigma"):
        sigma = net.round_sigma(sigma)
    return sigma


def build_sigma_grid(
    *,
    step_count: int,
    warp: torch.nn.Module | None,
    device: torch.device,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    net=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r = torch.linspace(0.0, 1.0, steps=step_count + 1, device=device)
    u = warp.r_to_t(r) if warp is not None else r
    u[0] = 0.0
    u[-1] = 1.0
    sigma = sigma_from_u(u, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho, net=net)
    sigma[-1] = torch.zeros((), device=device, dtype=sigma.dtype)
    return u, sigma


def append_dims(value: torch.Tensor, ndim: int) -> torch.Tensor:
    return value.reshape(-1, *([1] * (ndim - 1)))


@torch.no_grad()
def teacher_transition(
    net: torch.nn.Module,
    x_t: torch.Tensor,
    sigma_t: torch.Tensor,
    sigma_s: torch.Tensor,
    labels: torch.Tensor | None,
) -> torch.Tensor:
    x_t = x_t.to(torch.float32)
    sigma_t = torch.as_tensor(sigma_t, device=x_t.device, dtype=x_t.dtype).view(-1)
    sigma_s = torch.as_tensor(sigma_s, device=x_t.device, dtype=x_t.dtype).view(-1)
    if sigma_t.numel() == 1:
        sigma_t = sigma_t.expand(x_t.shape[0])
    if sigma_s.numel() == 1:
        sigma_s = sigma_s.expand(x_t.shape[0])
    denoised = net(x_t, sigma_t, labels).to(x_t.dtype)
    d_cur = (x_t - denoised) / append_dims(torch.clamp(sigma_t, min=1.0e-6), x_t.ndim)
    x_euler = x_t + append_dims(sigma_s - sigma_t, x_t.ndim) * d_cur
    mask = sigma_s > 0.0
    if not bool(mask.any()):
        return x_euler
    x_next = x_euler.clone()
    labels_next = labels[mask] if labels is not None else None
    denoised_next = net(x_euler[mask], sigma_s[mask], labels_next).to(x_t.dtype)
    d_next = (x_euler[mask] - denoised_next) / append_dims(torch.clamp(sigma_s[mask], min=1.0e-6), x_t.ndim)
    x_next[mask] = x_t[mask] + append_dims(sigma_s[mask] - sigma_t[mask], x_t.ndim) * (0.5 * d_cur[mask] + 0.5 * d_next)
    return x_next


def student_transition(
    student: torch.nn.Module,
    x_t: torch.Tensor,
    sigma_t: torch.Tensor,
    sigma_s: torch.Tensor,
    labels: torch.Tensor | None,
) -> torch.Tensor:
    sigma_t = torch.as_tensor(sigma_t, device=x_t.device, dtype=x_t.dtype).view(-1)
    sigma_s = torch.as_tensor(sigma_s, device=x_t.device, dtype=x_t.dtype).view(-1)
    if sigma_t.numel() == 1:
        sigma_t = sigma_t.expand(x_t.shape[0])
    if sigma_s.numel() == 1:
        sigma_s = sigma_s.expand(x_t.shape[0])
    denoised = student(x_t, sigma_t, labels).to(x_t.dtype)
    d_cur = (x_t - denoised) / append_dims(torch.clamp(sigma_t, min=1.0e-6), x_t.ndim)
    return x_t + append_dims(sigma_s - sigma_t, x_t.ndim) * d_cur


@torch.no_grad()
def teacher_rollout_transition(
    net: torch.nn.Module,
    x_t: torch.Tensor,
    sigma_t: torch.Tensor | float,
    sigma_s: torch.Tensor | float,
    labels: torch.Tensor | None,
    *,
    num_steps: int,
    rho: float,
) -> torch.Tensor:
    """Roll a teacher transition with a Karras/EDM Heun grid.

    This is intentionally more expensive than `teacher_transition`: it provides
    a high-quality endpoint target for one-step consistency distillation.
    """
    if int(num_steps) < 1:
        raise ValueError("num_steps must be >= 1")
    device = x_t.device
    dtype = x_t.dtype
    sigma_start = torch.as_tensor(sigma_t, device=device, dtype=torch.float32).flatten()[0]
    sigma_end = torch.as_tensor(sigma_s, device=device, dtype=torch.float32).flatten()[0]
    if float(sigma_start.item()) <= float(sigma_end.item()):
        raise ValueError("teacher_rollout_transition expects sigma_t > sigma_s")
    if float(sigma_end.item()) <= 0.0:
        from generate import edm_sampler

        latents = x_t.to(torch.float64) / sigma_start.to(torch.float64)
        return edm_sampler(
            net,
            latents,
            labels,
            num_steps=int(num_steps),
            sigma_min=max(0.002, float(getattr(net, "sigma_min", 0.0))),
            sigma_max=float(sigma_start.item()),
            rho=float(rho),
        ).to(dtype)
    sigma_min = max(float(sigma_end.item()), float(getattr(net, "sigma_min", 0.0)))
    sigma_max = min(float(sigma_start.item()), float(getattr(net, "sigma_max", sigma_start.item())))
    if int(num_steps) == 1:
        t_steps = torch.as_tensor([sigma_max, 0.0], device=device, dtype=torch.float32)
    else:
        step_indices = torch.arange(int(num_steps), dtype=torch.float32, device=device)
        t_steps = (sigma_max ** (1.0 / rho) + step_indices / (int(num_steps) - 1) * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))) ** rho
        if hasattr(net, "round_sigma"):
            t_steps = net.round_sigma(t_steps)
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

    x_next = x_t.to(torch.float32)
    for step_index, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        sigma_cur = t_cur.expand(x_cur.shape[0]).to(device=device, dtype=x_cur.dtype)
        denoised = net(x_cur, sigma_cur, labels).to(x_cur.dtype)
        d_cur = (x_cur - denoised) / append_dims(torch.clamp(sigma_cur, min=1.0e-6), x_cur.ndim)
        x_euler = x_cur + append_dims((t_next - t_cur).to(x_cur.dtype), x_cur.ndim) * d_cur
        if step_index < int(num_steps) - 1 and float(t_next.item()) > 0.0:
            sigma_next = t_next.expand(x_cur.shape[0]).to(device=device, dtype=x_cur.dtype)
            denoised_next = net(x_euler, sigma_next, labels).to(x_cur.dtype)
            d_next = (x_euler - denoised_next) / append_dims(torch.clamp(sigma_next, min=1.0e-6), x_cur.ndim)
            x_next = x_cur + append_dims((t_next - t_cur).to(x_cur.dtype), x_cur.ndim) * (0.5 * d_cur + 0.5 * d_next)
        else:
            x_next = x_euler
    return x_next.to(dtype)


def build_cifar10_loader(cfg: dict):
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    train_cfg = cfg.get("train", {})
    data_root = Path(resolve_path(cfg["paths"]["data_root"]))
    transform = transforms.ToTensor()
    dataset = datasets.CIFAR10(root=str(data_root), train=True, download=False, transform=transform)
    return DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 64)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=True,
        drop_last=True,
        persistent_workers=int(train_cfg.get("num_workers", 4)) > 0,
    )


def init_warp(cfg: dict, *, device: torch.device) -> tuple[torch.nn.Module | None, torch.Tensor | None, torch.Tensor | None]:
    warp_cfg = cfg.get("timewarp", {})
    if not bool(warp_cfg.get("enabled", True)):
        return None, None, None
    warp_type = str(warp_cfg.get("type", "density")).lower()
    if warp_type in {"density", "piecewise_linear", "linear_cdf"}:
        warp = MonotoneDensityWarp(
            num_bins=int(warp_cfg.get("num_bins", 32)),
            init=str(warp_cfg.get("init", "logit_normal")),
        ).to(device=device)
    elif warp_type in {"rqs", "rational_quadratic", "rational_quadratic_spline"}:
        warp = MonotoneRationalQuadraticSplineWarp(
            num_bins=int(warp_cfg.get("num_bins", 96)),
            init=str(warp_cfg.get("init", "uniform")),
            min_bin_mass=float(warp_cfg.get("min_bin_mass", 1.0e-4)),
            min_derivative=float(warp_cfg.get("min_derivative", 1.0e-3)),
            inverse_iters=int(warp_cfg.get("inverse_iters", 36)),
            center_kl_weight=float(warp_cfg.get("center_kl_weight", 0.25)),
            derivative_smoothness_weight=float(warp_cfg.get("derivative_smoothness_weight", 0.0)),
        ).to(device=device)
    else:
        raise ValueError(f"Unsupported timewarp.type: {warp_type}")
    q_base = warp.density().detach().clone()
    q_target = q_base.clone()
    return warp, q_base, q_target


def sample_triplet_u(warp: torch.nn.Module | None, batch_size: int, *, device: torch.device):
    if warp is None:
        r_t, r_s, r_u = MonotoneDensityWarp(num_bins=32).sample_triplets(batch_size, device)
        return r_t, r_s, r_u
    r_t, r_s, r_u = warp.sample_triplets(batch_size, device)
    return warp.r_to_t(r_t), warp.r_to_t(r_s), warp.r_to_t(r_u)


def bin_ids_from_u(u: torch.Tensor, *, num_bins: int) -> torch.Tensor:
    return torch.clamp((u.detach() * num_bins).floor().long(), min=0, max=num_bins - 1)


def update_warp_from_defect(
    *,
    warp: torch.nn.Module,
    warp_optimizer: torch.optim.Optimizer,
    q_base: torch.Tensor,
    q_target: torch.Tensor,
    stats: dict[str, torch.Tensor],
    u: torch.Tensor,
    defect: torch.Tensor,
    cfg: dict,
) -> torch.Tensor:
    warp_cfg = cfg.get("timewarp", {})
    num_bins = warp.num_bins
    bin_ids = bin_ids_from_u(u, num_bins=num_bins)
    ids = torch.unique(bin_ids)
    if ids.numel() > 0:
        values = torch.stack([defect[bin_ids == idx].mean() for idx in ids])
        update_ema_bins({"bar": stats["D_bar"], "count": stats["D_count"]}, ids, values, float(warp_cfg.get("ema_mu", 0.995)))
    q_target_new = build_target_density(
        stats["D_bar"],
        torch.zeros_like(stats["D_bar"]),
        torch.zeros_like(stats["D_bar"]),
        q_base,
        float(warp_cfg.get("beta", 0.75)),
        1.0e-6,
        curvature_weight=float(warp_cfg.get("curvature_weight", 0.0)),
        use_hf_bar=False,
        hf_weight=float(warp_cfg.get("hf_weight", 0.0)),
        ratio_cap=float(warp_cfg.get("ratio_cap", 0.0)),
    )
    q_target_new = smooth_density(q_target_new)
    flatten_mix = float(warp_cfg.get("flatten_mix", 0.0))
    if flatten_mix > 0.0:
        flatten_mix = min(max(flatten_mix, 0.0), 1.0)
        q_target_new = (1.0 - flatten_mix) * q_target_new + flatten_mix * q_base
        q_target_new = q_target_new / torch.clamp(q_target_new.sum(), min=1.0e-6)
    q_target.copy_(q_target_new.detach())
    warp_loss = None
    for _ in range(max(1, int(warp_cfg.get("inner_steps", 1)))):
        warp_optimizer.zero_grad(set_to_none=True)
        warp_loss = warp.kl_to_target_density(q_target)
        warp_loss.backward()
        warp_optimizer.step()
    assert warp_loss is not None
    return warp_loss.detach()


def to_unit(images: torch.Tensor) -> torch.Tensor:
    return torch.clamp(images * 0.5 + 0.5, 0.0, 1.0)


@torch.no_grad()
def sample_with_student(
    *,
    student: torch.nn.Module,
    warp: torch.nn.Module | None,
    cfg: dict,
    step_count: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train_cfg = cfg.get("train", {})
    sigma_min = float(train_cfg.get("sigma_min", 0.002))
    sigma_max = float(train_cfg.get("sigma_max", 80.0))
    rho = float(train_cfg.get("rho", 7.0))
    generator = torch.Generator(device=device).manual_seed(int(seed))
    latents = torch.randn(
        batch_size,
        int(getattr(student, "img_channels", 3)),
        int(getattr(student, "img_resolution", 32)),
        int(getattr(student, "img_resolution", 32)),
        device=device,
        generator=generator,
    )
    labels = sample_labels(student, batch_size, device=device, generator=generator)
    u_grid, sigma_grid = build_sigma_grid(
        step_count=step_count,
        warp=warp,
        device=device,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
        net=student,
    )
    x = latents * sigma_grid[0].to(latents.dtype)
    for sigma_t, sigma_s in zip(sigma_grid[:-1], sigma_grid[1:]):
        x = student_transition(student, x, sigma_t.expand(batch_size), sigma_s.expand(batch_size), labels)
    return to_unit(x).detach(), u_grid.detach(), sigma_grid.detach()


def sample_stats(samples: torch.Tensor) -> dict:
    x = samples.detach().float().cpu()
    h_corr = torch.mean((x[..., :, :-1] - x.mean()) * (x[..., :, 1:] - x.mean())) / torch.clamp(x.var(), min=1.0e-8)
    v_corr = torch.mean((x[..., :-1, :] - x.mean()) * (x[..., 1:, :] - x.mean())) / torch.clamp(x.var(), min=1.0e-8)
    return {
        "shape": list(x.shape),
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "neighbor_corr_h": float(h_corr.item()),
        "neighbor_corr_v": float(v_corr.item()),
        "saturation_0_1": float(((x <= 0.0) | (x >= 1.0)).float().mean().item()),
    }


def save_grid(samples: torch.Tensor, path: str | Path, *, nrow: int = 8) -> None:
    from torchvision.utils import save_image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(samples.detach().cpu(), path, nrow=nrow)
