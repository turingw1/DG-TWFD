#!/usr/bin/env python3
"""Build the DG-TWFD time-warp mechanism figure from real checkpoints.

The figure is designed for the main paper mechanism claim:
DG-TWFD keeps the teacher path fixed and reallocates clock mass so that
low-step composed student waypoints spend more resolution in hard regions.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-dgtwfd")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
from matplotlib import gridspec, pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
EDM_FIRST_SRC = ROOT / "experiments" / "edm_first"
EDM_REF = ROOT / "refs" / "edm"

for path in [str(EDM_FIRST_SRC), str(EDM_REF)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from src.edm_map_lib import (  # noqa: E402
    append_dims,
    clone_student_from_teacher,
    init_warp,
    load_config,
    load_edm_network,
    make_labels,
    sigma_from_u,
    student_transition,
    teacher_transition,
)


DEFAULT_CONFIG = ROOT / "experiments" / "edm_first" / "configs" / "cifar10_edm_map_prior_fullstack_timewarp_v21_ctm_aligned.yaml"
DEFAULT_CHECKPOINT = (
    Path("/cache/Zhengwei/DG-TWFD-runtime/runs")
    / "edm_first_cifar10_prior_fullstack_timewarp_v21_ctm_aligned_from_v20_step3292"
    / "checkpoints"
    / "step2500.pt"
)
DEFAULT_OUTPUT = (
    ROOT
    / "docs"
    / "experiments"
    / "DG_TWFD_v3"
    / "figures"
    / "mechanism"
    / "timewarp_mechanism_20260502"
)

COLORS = {
    "teacher": "#696969",
    "identity": "#d95f02",
    "dg": "#1f77b4",
    "defect": "#f3b2a8",
    "variation": "#7f5aa2",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--num-trajectories", type=int, default=24)
    parser.add_argument("--dense-steps", type=int, default=32)
    parser.add_argument("--student-steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument("--selection-quantile", type=float, default=0.60)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _device(cfg: dict[str, Any], requested: str | None) -> torch.device:
    raw = requested or str(cfg.get("runtime", {}).get("device", "cuda"))
    if raw == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(raw)


def _latent_batch(
    *,
    seeds: list[int],
    channels: int,
    image_size: int,
    device: torch.device,
) -> torch.Tensor:
    values = []
    for seed in seeds:
        generator = torch.Generator(device=device).manual_seed(int(seed))
        values.append(torch.randn((channels, image_size, image_size), generator=generator, device=device))
    return torch.stack(values, dim=0)


def _load_models(
    *,
    cfg: dict[str, Any],
    checkpoint: Path,
    device: torch.device,
    use_ema: bool,
) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module | None, dict[str, Any]]:
    teacher = load_edm_network(cfg["paths"]["network"], device=device, use_fp16=False)
    teacher.eval().requires_grad_(False)
    student = clone_student_from_teacher(teacher, cfg=cfg).to(device)
    ckpt = torch.load(checkpoint, map_location="cpu")
    state_key = "student_ema" if use_ema and ckpt.get("student_ema") is not None else "student"
    student.load_state_dict(ckpt[state_key])
    student.eval().requires_grad_(False)
    warp, _q_base, _q_target = init_warp(cfg, device=device)
    if warp is not None and ckpt.get("warp") is not None:
        warp.load_state_dict(ckpt["warp"])
        warp.eval().requires_grad_(False)
    return teacher, student, warp, ckpt


def _sigma_grid(cfg: dict[str, Any], u: torch.Tensor, net: torch.nn.Module) -> torch.Tensor:
    train_cfg = cfg.get("train", {})
    sigma = sigma_from_u(
        u,
        sigma_min=float(train_cfg.get("sigma_min", 0.002)),
        sigma_max=float(train_cfg.get("sigma_max", 80.0)),
        rho=float(train_cfg.get("rho", 7.0)),
        net=net,
    )
    sigma = sigma.clone()
    sigma[-1] = torch.zeros((), device=u.device, dtype=u.dtype)
    return sigma


@torch.no_grad()
def _teacher_dense_paths(
    *,
    teacher: torch.nn.Module,
    cfg: dict[str, Any],
    latents: torch.Tensor,
    labels: torch.Tensor | None,
    dense_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = latents.device
    u_dense = torch.linspace(0.0, 1.0, steps=dense_steps + 1, device=device)
    sigma_dense = _sigma_grid(cfg, u_dense, teacher)
    x = latents.to(torch.float32) * sigma_dense[0].to(latents.dtype)
    states = [x.detach().clone()]
    for sigma_t, sigma_s in zip(sigma_dense[:-1], sigma_dense[1:]):
        x = teacher_transition(
            teacher,
            x,
            sigma_t.expand(x.shape[0]),
            sigma_s.expand(x.shape[0]),
            labels,
        )
        states.append(x.detach().clone())
    return torch.stack(states, dim=1), u_dense.detach(), sigma_dense.detach()


@torch.no_grad()
def _student_paths(
    *,
    student: torch.nn.Module,
    cfg: dict[str, Any],
    latents: torch.Tensor,
    labels: torch.Tensor | None,
    u_nodes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    sigma_nodes = _sigma_grid(cfg, u_nodes, student)
    x = latents.to(torch.float32) * sigma_nodes[0].to(latents.dtype)
    states = [x.detach().clone()]
    for sigma_t, sigma_s in zip(sigma_nodes[:-1], sigma_nodes[1:]):
        x = student_transition(
            student,
            x,
            sigma_t.expand(x.shape[0]),
            sigma_s.expand(x.shape[0]),
            labels,
        )
        states.append(x.detach().clone())
    return torch.stack(states, dim=1), sigma_nodes.detach()


def _interp_states_by_u(states: torch.Tensor, u_nodes: torch.Tensor, query_u: torch.Tensor) -> torch.Tensor:
    """Linearly interpolate a composed state trajectory on a common original-time grid."""
    query = query_u.detach().to(device=states.device, dtype=torch.float32).clamp(0.0, 1.0)
    nodes = u_nodes.to(device=states.device, dtype=torch.float32).clamp(0.0, 1.0)
    indices = torch.searchsorted(nodes.contiguous(), query.contiguous(), right=True) - 1
    lo = indices.long().clamp(min=0, max=nodes.numel() - 2)
    hi = lo + 1
    alpha = (query - nodes[lo]) / torch.clamp(nodes[hi] - nodes[lo], min=1.0e-8)
    alpha_view = alpha.view(1, -1, *([1] * (states.ndim - 2)))
    return states[:, lo] * (1.0 - alpha_view) + states[:, hi] * alpha_view


def _mse_by_node(student_states: torch.Tensor, teacher_at_nodes: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(2, student_states.ndim))
    return (student_states - teacher_at_nodes).square().mean(dim=dims)


@torch.no_grad()
def _semigroup_defect_by_clock(
    *,
    student: torch.nn.Module,
    cfg: dict[str, Any],
    teacher_states: torch.Tensor,
    u_dense: torch.Tensor,
    u_nodes: torch.Tensor,
    label_ids: list[int],
    label_dim: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure student composition error on held-out teacher states.

    For each segment [u_i, u_{i+1}], the direct transition is compared with two
    half transitions through the original-time midpoint. The input state is the
    dense teacher state at u_i, so the measured defect is tied to the same
    teacher trajectory without refitting any projection.
    """
    mids = 0.5 * (u_nodes[:-1] + u_nodes[1:])
    sigma_nodes = _sigma_grid(cfg, u_nodes.to(device), student)
    sigma_mid = _sigma_grid(cfg, mids.to(device), student)
    values: list[torch.Tensor] = []
    u_dense_device = u_dense.to(device)
    query_start = u_nodes[:-1].to(device)
    for start in range(0, teacher_states.shape[0], batch_size):
        stop = min(start + batch_size, teacher_states.shape[0])
        states = teacher_states[start:stop].to(device)
        x_start_all = _interp_states_by_u(states, u_dense_device, query_start)
        ids = torch.tensor(label_ids[start:stop], device=device, dtype=torch.long)
        labels = make_labels(ids, label_dim=label_dim, device=device) if label_dim > 0 else None
        batch_values = []
        for seg_idx in range(u_nodes.numel() - 1):
            x0 = x_start_all[:, seg_idx]
            s0 = sigma_nodes[seg_idx].expand(x0.shape[0])
            sm = sigma_mid[seg_idx].expand(x0.shape[0])
            s1 = sigma_nodes[seg_idx + 1].expand(x0.shape[0])
            direct = student_transition(student, x0, s0, s1, labels)
            first = student_transition(student, x0, s0, sm, labels)
            composed = student_transition(student, first, sm, s1, labels)
            batch_values.append((direct - composed).square().flatten(1).mean(dim=1).detach().cpu())
        values.append(torch.stack(batch_values, dim=1))
    return torch.cat(values, dim=0).numpy(), mids.detach().cpu().numpy()


def _project_teacher_basis(teacher_states: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = teacher_states.detach().cpu().float().flatten(1).numpy()
    origin = flat[0].copy()
    displacement = flat[-1] - flat[0]
    norm = float(np.linalg.norm(displacement))
    if norm < 1.0e-12:
        axis1 = np.zeros_like(displacement)
        axis1[0] = 1.0
    else:
        axis1 = displacement / norm
    centered = flat - origin
    residual = centered - centered @ axis1[:, None] * axis1[None, :]
    if np.linalg.norm(residual) < 1.0e-12:
        axis2 = np.zeros_like(axis1)
        axis2[1] = 1.0
    else:
        _u, _s, vh = np.linalg.svd(residual, full_matrices=False)
        axis2 = vh[0]
        axis2 = axis2 - float(axis2 @ axis1) * axis1
        axis2 = axis2 / max(float(np.linalg.norm(axis2)), 1.0e-12)
    return origin, axis1, axis2


def _project(states: torch.Tensor, origin: np.ndarray, axis1: np.ndarray, axis2: np.ndarray) -> np.ndarray:
    flat = states.detach().cpu().float().flatten(1).numpy() - origin[None, :]
    return np.stack([flat @ axis1, flat @ axis2], axis=1)


def _normalize_projection(
    teacher_proj: np.ndarray,
    identity_proj: np.ndarray,
    dg_proj: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_scale = max(abs(float(teacher_proj[-1, 0] - teacher_proj[0, 0])), 1.0e-12)
    combined = np.concatenate([teacher_proj, identity_proj, dg_proj], axis=0)
    y_scale = max(float(np.percentile(np.abs(combined[:, 1]), 95.0)), 1.0e-12)
    out = []
    for values in (teacher_proj, identity_proj, dg_proj):
        normalized = values.copy()
        normalized[:, 0] = (normalized[:, 0] - teacher_proj[0, 0]) / x_scale
        normalized[:, 1] = normalized[:, 1] / y_scale
        out.append(normalized)
    return out[0], out[1], out[2]


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not bool(finite.any()):
        return np.zeros_like(values)
    v_min = float(values[finite].min())
    v_max = float(values[finite].max())
    if v_max - v_min < 1.0e-12:
        return np.zeros_like(values)
    out = (values - v_min) / (v_max - v_min)
    out[~finite] = 0.0
    return out


def _bin_values_from_nodes(u_nodes: np.ndarray, values: np.ndarray, num_bins: int) -> np.ndarray:
    out = np.zeros(num_bins, dtype=np.float64)
    counts = np.zeros(num_bins, dtype=np.float64)
    for left, right, value in zip(u_nodes[:-1], u_nodes[1:], values):
        lo = int(np.clip(np.floor(left * num_bins), 0, num_bins - 1))
        hi = int(np.clip(np.ceil(right * num_bins), lo + 1, num_bins))
        out[lo:hi] += float(value)
        counts[lo:hi] += 1.0
    return out / np.maximum(counts, 1.0)


def _bootstrap_ci(values: np.ndarray, *, seed: int, samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean(axis=0)
    if values.shape[0] <= 1 or samples <= 0:
        se = values.std(axis=0, ddof=0) / np.sqrt(max(values.shape[0], 1))
        return mean, mean - 1.96 * se, mean + 1.96 * se
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.shape[0], size=(samples, values.shape[0]))
    boot = values[indices].mean(axis=1)
    lo = np.percentile(boot, 2.5, axis=0)
    hi = np.percentile(boot, 97.5, axis=0)
    return mean, lo, hi


def _make_colored_line(points: np.ndarray, values: np.ndarray, *, cmap: str = "Reds", lw: float = 2.2) -> LineCollection:
    segments = np.stack([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=plt.get_cmap(cmap), linewidth=lw, alpha=0.85)
    lc.set_array(np.asarray(values[:-1], dtype=np.float64))
    lc.set_clim(0.0, 1.0)
    return lc


def _format_panel(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=7.4, fontweight="bold", pad=3)
    ax.tick_params(axis="both", labelsize=6.5, width=0.65, length=2.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("#333333")


def _draw_panel_a(
    ax,
    *,
    warp: torch.nn.Module | None,
    ckpt: dict[str, Any],
    teacher_variation_bins: np.ndarray,
    student_steps: int,
    device: torch.device,
) -> dict[str, Any]:
    grid = torch.linspace(0.0, 1.0, steps=401, device=device)
    if warp is None:
        warped = grid
        density = torch.ones(96, device=device) / 96.0
    else:
        with torch.no_grad():
            warped = warp.t_to_r(grid)
            density = warp.density()
    p = grid.detach().cpu().numpy()
    q = warped.detach().cpu().numpy()
    density_np = density.detach().cpu().float().numpy()
    defect = ckpt.get("warp_stats", {}).get("D_bar")
    if torch.is_tensor(defect):
        defect_np = defect.detach().cpu().float().numpy()
    else:
        defect_np = density_np.copy()
    defect_norm = _normalize(defect_np)
    variation_norm = _normalize(teacher_variation_bins)
    num_bins = len(defect_norm)
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    for idx, value in enumerate(defect_norm):
        if value <= 0.0:
            continue
        ax.axvspan(
            bin_edges[idx],
            bin_edges[idx + 1],
            color=COLORS["defect"],
            alpha=0.08 + 0.24 * float(value),
            lw=0.0,
            zorder=0,
        )
    variation_y0 = 1.015
    for idx, value in enumerate(variation_norm):
        if value <= 0.03:
            continue
        center = 0.5 * (bin_edges[idx] + bin_edges[idx + 1])
        ax.plot(
            [center, center],
            [variation_y0, variation_y0 + 0.055 * float(value)],
            color=COLORS["variation"],
            lw=0.45,
            alpha=0.72,
            clip_on=False,
            solid_capstyle="butt",
        )

    ax.plot([0.0, 1.0], [0.0, 1.0], color="#b7b7b7", lw=1.05, ls="--", label="identity")
    ax.plot(p, q, color=COLORS["dg"], lw=2.1, label="learned warp")

    r_nodes = torch.linspace(0.0, 1.0, steps=student_steps + 1, device=device)
    if warp is None:
        u_nodes = r_nodes
    else:
        with torch.no_grad():
            u_nodes = warp.r_to_t(r_nodes)
    u_nodes = u_nodes.detach().clone()
    u_nodes[0] = 0.0
    u_nodes[-1] = 1.0
    u_np = u_nodes.cpu().numpy()
    r_np = r_nodes.cpu().numpy()
    ax.scatter(u_np, r_np, s=16, color=COLORS["dg"], edgecolors="white", linewidths=0.5, zorder=5, label="8-step nodes")
    for x_pos, y_pos in zip(u_np[1:-1], r_np[1:-1]):
        ax.plot([x_pos, x_pos], [0.0, y_pos], color=COLORS["dg"], lw=0.55, alpha=0.23, zorder=1)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.08)
    ax.set_xlabel("original progress $p=1-t$", fontsize=7.2)
    ax.set_ylabel("warped progress $q=1-g(t)$", fontsize=7.2)
    _format_panel(ax, "A. Learned clock")
    ax.text(
        0.02,
        0.965,
        "rug: teacher variation",
        fontsize=5.8,
        color=COLORS["variation"],
        transform=ax.transAxes,
        va="top",
    )
    return {
        "u_nodes": u_np.tolist(),
        "r_nodes": r_np.tolist(),
        "defect_bins": defect_norm.tolist(),
        "variation_bins": variation_norm.tolist(),
        "density_mass": density_np.tolist(),
    }


def _draw_panel_b(
    ax,
    *,
    teacher_proj: np.ndarray,
    identity_proj: np.ndarray,
    dg_proj: np.ndarray,
    u_dense: np.ndarray,
    identity_values: np.ndarray,
    dg_values: np.ndarray,
) -> None:
    defect_proxy = _normalize(np.linalg.norm(np.diff(teacher_proj, axis=0), axis=1))
    ax.add_collection(_make_colored_line(teacher_proj, defect_proxy, lw=2.15))
    ax.plot(teacher_proj[:, 0], teacher_proj[:, 1], color=COLORS["teacher"], lw=1.2, alpha=0.42, label="teacher path")
    ax.plot(
        identity_proj[:, 0],
        identity_proj[:, 1],
        color=COLORS["identity"],
        lw=1.6,
        marker="o",
        ms=3.4,
        label="identity clock",
    )
    ax.plot(
        dg_proj[:, 0],
        dg_proj[:, 1],
        color=COLORS["dg"],
        lw=1.8,
        marker="o",
        ms=3.4,
        label="DG-TWFD clock",
    )
    ax.scatter(teacher_proj[0, 0], teacher_proj[0, 1], s=16, color="#222222", marker="x", zorder=6)
    ax.scatter(teacher_proj[-1, 0], teacher_proj[-1, 1], s=18, color="#222222", marker="s", zorder=6)
    ax.set_xlabel("endpoint axis (normalized)", fontsize=7.2)
    ax.set_ylabel("residual PC1", fontsize=7.0)
    _format_panel(ax, "B. Same teacher path")
    ax.text(
        0.02,
        0.03,
        f"60th pct. defect: {identity_values.mean():.2g} -> {dg_values.mean():.2g}",
        transform=ax.transAxes,
        fontsize=5.8,
        color="#333333",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
    )
    del u_dense


def _draw_panel_c(
    ax,
    *,
    identity_u: np.ndarray,
    dg_u: np.ndarray,
    identity_error: np.ndarray,
    dg_error: np.ndarray,
    density_mass: np.ndarray,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    id_mean, id_lo, id_hi = _bootstrap_ci(identity_error, seed=seed, samples=bootstrap_samples)
    dg_mean, dg_lo, dg_hi = _bootstrap_ci(dg_error, seed=seed + 1, samples=bootstrap_samples)
    ax.plot(identity_u, id_mean, color=COLORS["identity"], lw=1.9, label="identity clock")
    ax.fill_between(identity_u, id_lo, id_hi, color=COLORS["identity"], alpha=0.16, lw=0)
    ax.plot(dg_u, dg_mean, color=COLORS["dg"], lw=2.0, label="DG-TWFD clock")
    ax.fill_between(dg_u, dg_lo, dg_hi, color=COLORS["dg"], alpha=0.16, lw=0)
    y_max = max(float(np.nanmax(id_hi)), float(np.nanmax(dg_hi)), 1.0e-8)
    y_min = min(float(np.nanmin(id_lo)), float(np.nanmin(dg_lo)), 0.0)
    mass = np.asarray(density_mass, dtype=np.float64)
    mass_norm = mass / max(float(mass.max()), 1.0e-12)
    x = (np.arange(len(mass_norm)) + 0.5) / float(len(mass_norm))
    bar_h = 0.12 * y_max
    ax.bar(
        x,
        mass_norm * bar_h,
        bottom=y_min,
        width=0.9 / len(mass_norm),
        color=COLORS["dg"],
        alpha=0.18,
        edgecolor="none",
        label="warped mass",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(y_min, y_max * 1.08)
    ax.set_xlabel("original progress $p$", fontsize=7.2)
    ax.set_ylabel("defect MSE", fontsize=7.0)
    _format_panel(ax, "C. Defect reduction")
    return {
        "identity_mean": id_mean.tolist(),
        "identity_lo": id_lo.tolist(),
        "identity_hi": id_hi.tolist(),
        "dg_mean": dg_mean.tolist(),
        "dg_lo": dg_lo.tolist(),
        "dg_hi": dg_hi.tolist(),
    }


def _save_caption(output_dir: Path) -> None:
    caption = r"""\textbf{Effect of defect-guided time warping on real teacher trajectories.}
The learned warp expands high-defect regions in warped time, so uniformly spaced warped steps induce denser original-time waypoints where composition is hardest.
Projected held-out trajectories show that DG-TWFD preserves the same teacher path while changing the lower-step waypoint traversal.
Aggregate defect curves report held-out means with bootstrap confidence intervals.
"""
    (output_dir / "caption.tex").write_text(caption, encoding="utf-8")


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _display_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    args = _parse_args()
    config_path = _repo_path(args.config)
    checkpoint_path = _repo_path(args.checkpoint)
    output_dir = _repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(config_path)
    device = _device(cfg, args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    teacher, student, warp, ckpt = _load_models(cfg=cfg, checkpoint=checkpoint_path, device=device, use_ema=args.use_ema)
    label_dim = int(getattr(teacher, "label_dim", 0) or 0)
    channels = int(getattr(teacher, "img_channels", 3))
    image_size = int(getattr(teacher, "img_resolution", 32))

    all_teacher = []
    all_identity = []
    all_dg = []
    labels_all = []
    seeds_all = []
    label_ids_all = []

    identity_u = torch.linspace(0.0, 1.0, steps=args.student_steps + 1, device=device)
    r_nodes = torch.linspace(0.0, 1.0, steps=args.student_steps + 1, device=device)
    if warp is None:
        dg_u = r_nodes
    else:
        with torch.no_grad():
            dg_u = warp.r_to_t(r_nodes)
    dg_u = dg_u.detach().clone()
    dg_u[0] = 0.0
    dg_u[-1] = 1.0

    produced = 0
    while produced < args.num_trajectories:
        current = min(args.batch_size, args.num_trajectories - produced)
        seeds = [args.seed + produced + idx for idx in range(current)]
        label_ids = torch.tensor(
            [(args.seed + 17 * (produced + idx)) % max(label_dim, 1) for idx in range(current)],
            device=device,
            dtype=torch.long,
        )
        labels = make_labels(label_ids, label_dim=label_dim, device=device) if label_dim > 0 else None
        latents = _latent_batch(seeds=seeds, channels=channels, image_size=image_size, device=device)
        teacher_states, u_dense, _sigma_dense = _teacher_dense_paths(
            teacher=teacher,
            cfg=cfg,
            latents=latents,
            labels=labels,
            dense_steps=args.dense_steps,
        )
        identity_states, _ = _student_paths(student=student, cfg=cfg, latents=latents, labels=labels, u_nodes=identity_u)
        dg_states, _ = _student_paths(student=student, cfg=cfg, latents=latents, labels=labels, u_nodes=dg_u)
        all_teacher.append(teacher_states.detach().cpu())
        all_identity.append(identity_states.detach().cpu())
        all_dg.append(dg_states.detach().cpu())
        if labels is not None:
            labels_all.append(labels.detach().cpu())
        seeds_all.extend(seeds)
        label_ids_all.extend([int(item) for item in label_ids.detach().cpu().tolist()])
        produced += current

    teacher_states = torch.cat(all_teacher, dim=0)
    identity_states = torch.cat(all_identity, dim=0)
    dg_states = torch.cat(all_dg, dim=0)
    u_dense_cpu = u_dense.detach().cpu()
    identity_u_cpu = identity_u.detach().cpu()
    dg_u_cpu = dg_u.detach().cpu()

    common_u_cpu = u_dense_cpu
    identity_common = _interp_states_by_u(identity_states, identity_u_cpu, common_u_cpu)
    dg_common = _interp_states_by_u(dg_states, dg_u_cpu, common_u_cpu)
    identity_path_error = _mse_by_node(identity_common, teacher_states)[:, 1:].numpy()
    dg_path_error = _mse_by_node(dg_common, teacher_states)[:, 1:].numpy()
    teacher_identity_nodes = _interp_states_by_u(teacher_states, u_dense_cpu, identity_u_cpu)
    teacher_dg_nodes = _interp_states_by_u(teacher_states, u_dense_cpu, dg_u_cpu)
    identity_node_error = _mse_by_node(identity_states, teacher_identity_nodes)[:, 1:].numpy()
    dg_node_error = _mse_by_node(dg_states, teacher_dg_nodes)[:, 1:].numpy()
    identity_defect, identity_mid_u = _semigroup_defect_by_clock(
        student=student,
        cfg=cfg,
        teacher_states=teacher_states,
        u_dense=u_dense_cpu,
        u_nodes=identity_u_cpu,
        label_ids=label_ids_all,
        label_dim=label_dim,
        batch_size=args.batch_size,
        device=device,
    )
    dg_defect, dg_mid_u = _semigroup_defect_by_clock(
        student=student,
        cfg=cfg,
        teacher_states=teacher_states,
        u_dense=u_dense_cpu,
        u_nodes=dg_u_cpu,
        label_ids=label_ids_all,
        label_dim=label_dim,
        batch_size=args.batch_size,
        device=device,
    )
    valid_defect_segments = np.isfinite(identity_defect).all(axis=0) & np.isfinite(dg_defect).all(axis=0)
    identity_defect_plot = identity_defect[:, valid_defect_segments]
    dg_defect_plot = dg_defect[:, valid_defect_segments]
    identity_mid_u_plot = identity_mid_u[valid_defect_segments]
    dg_mid_u_plot = dg_mid_u[valid_defect_segments]
    if identity_defect_plot.shape[1] == 0:
        raise RuntimeError("No finite semigroup-defect segments were available for the mechanism figure.")
    improvement = identity_defect_plot.mean(axis=1) - dg_defect_plot.mean(axis=1)
    order = np.argsort(improvement)
    q_index = int(np.clip(round(args.selection_quantile * (len(order) - 1)), 0, len(order) - 1))
    selected_index = int(order[q_index])

    selected_teacher = teacher_states[selected_index]
    selected_identity = identity_states[selected_index]
    selected_dg = dg_states[selected_index]
    origin, axis1, axis2 = _project_teacher_basis(selected_teacher)
    teacher_proj = _project(selected_teacher, origin, axis1, axis2)
    identity_proj = _project(selected_identity, origin, axis1, axis2)
    dg_proj = _project(selected_dg, origin, axis1, axis2)
    teacher_proj, identity_proj, dg_proj = _normalize_projection(teacher_proj, identity_proj, dg_proj)

    teacher_step_mse = (teacher_states[:, 1:] - teacher_states[:, :-1]).square().flatten(2).mean(dim=2).mean(dim=0).numpy()
    teacher_variation_bins = _bin_values_from_nodes(
        u_dense_cpu.numpy(),
        teacher_step_mse,
        int(cfg.get("timewarp", {}).get("num_bins", 96)),
    )

    fig = plt.figure(figsize=(7.1, 2.82))
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.02, 1.02, 1.18], wspace=0.42)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    panel_a = _draw_panel_a(
        ax_a,
        warp=warp,
        ckpt=ckpt,
        teacher_variation_bins=teacher_variation_bins,
        student_steps=args.student_steps,
        device=device,
    )
    _draw_panel_b(
        ax_b,
        teacher_proj=teacher_proj,
        identity_proj=identity_proj,
        dg_proj=dg_proj,
        u_dense=u_dense_cpu.numpy(),
        identity_values=identity_defect_plot[selected_index],
        dg_values=dg_defect_plot[selected_index],
    )
    panel_c = _draw_panel_c(
        ax_c,
        identity_u=identity_mid_u_plot,
        dg_u=dg_mid_u_plot,
        identity_error=identity_defect_plot,
        dg_error=dg_defect_plot,
        density_mass=np.asarray(panel_a["density_mass"], dtype=np.float64),
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
    )

    handles = [
        Line2D([0], [0], color=COLORS["teacher"], lw=1.6, label="teacher path"),
        Line2D([0], [0], color=COLORS["identity"], lw=1.8, marker="o", ms=3.3, label="identity clock"),
        Line2D([0], [0], color=COLORS["dg"], lw=1.9, marker="o", ms=3.3, label="DG-TWFD clock"),
        Line2D([0], [0], color=COLORS["defect"], lw=6.0, alpha=0.55, label="defect / variation"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.99),
        ncol=4,
        frameon=False,
        fontsize=6.2,
        handlelength=1.45,
        columnspacing=0.95,
    )
    fig.subplots_adjust(left=0.062, right=0.995, bottom=0.22, top=0.82)

    pdf_path = output_dir / "timewarp_mechanism_main.pdf"
    png_path = output_dir / "timewarp_mechanism_main.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=450)
    plt.close(fig)

    np.savez_compressed(
        output_dir / "timewarp_mechanism_data.npz",
        teacher_proj=teacher_proj,
        identity_proj=identity_proj,
        dg_proj=dg_proj,
        identity_path_error=identity_path_error,
        dg_path_error=dg_path_error,
        identity_node_error=identity_node_error,
        dg_node_error=dg_node_error,
        identity_defect=identity_defect,
        dg_defect=dg_defect,
        identity_defect_plot=identity_defect_plot,
        dg_defect_plot=dg_defect_plot,
        improvement=improvement,
        identity_u=identity_u_cpu.numpy(),
        dg_u=dg_u_cpu.numpy(),
        common_u=common_u_cpu.numpy(),
        identity_mid_u=identity_mid_u,
        dg_mid_u=dg_mid_u,
        identity_mid_u_plot=identity_mid_u_plot,
        dg_mid_u_plot=dg_mid_u_plot,
        valid_defect_segments=valid_defect_segments,
        u_dense=u_dense_cpu.numpy(),
        density_mass=np.asarray(panel_a["density_mass"], dtype=np.float64),
        defect_bins=np.asarray(panel_a["defect_bins"], dtype=np.float64),
        variation_bins=np.asarray(panel_a["variation_bins"], dtype=np.float64),
    )
    _save_caption(output_dir)

    manifest = {
        "figure": {
            "pdf": _display_path(pdf_path),
            "png": _display_path(png_path),
            "data": _display_path(output_dir / "timewarp_mechanism_data.npz"),
            "caption": _display_path(output_dir / "caption.tex"),
        },
        "config": _display_path(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(ckpt.get("step", -1)),
        "use_ema": bool(args.use_ema),
        "device": str(device),
        "num_trajectories": int(args.num_trajectories),
        "dense_steps": int(args.dense_steps),
        "student_steps": int(args.student_steps),
        "seed": int(args.seed),
        "selection_quantile": float(args.selection_quantile),
        "selected_index": int(selected_index),
        "selected_seed": int(seeds_all[selected_index]),
        "selected_label_id": int(label_ids_all[selected_index]),
        "selected_improvement": float(improvement[selected_index]),
        "aggregate": {
            "identity_mean_defect": float(identity_defect_plot.mean()),
            "dg_mean_defect": float(dg_defect_plot.mean()),
            "mean_improvement": float(improvement.mean()),
            "median_improvement": float(np.median(improvement)),
            "fraction_dg_better": float(np.mean(improvement > 0.0)),
            "common_path_identity_mean_error": float(identity_path_error.mean()),
            "common_path_dg_mean_error": float(dg_path_error.mean()),
            "node_identity_mean_error": float(identity_node_error.mean()),
            "node_dg_mean_error": float(dg_node_error.mean()),
            **panel_c,
        },
        "projection": {
            "basis_fit": "selected dense teacher trajectory only",
            "axis1": "teacher endpoint displacement",
            "axis2": "PC1 of teacher residual orthogonal to endpoint displacement",
        },
        "panel_a": panel_a,
        "notes": [
            "Student trajectories are projected onto a basis fit only from the dense teacher path.",
            "The representative trajectory is selected by the requested improvement quantile, not by visual inspection.",
            "Teacher variation is the mean dense-teacher state MSE between adjacent dense PF-ODE nodes.",
            "Panel C uses finite semigroup-defect segments only; the final sigma=0 endpoint segment can be numerically singular for this adapter diagnostic.",
        ],
    }
    (output_dir / "timewarp_mechanism_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["figure"], indent=2))
    print(
        "aggregate_defect",
        json.dumps(
            {
                "identity": manifest["aggregate"]["identity_mean_defect"],
                "dg": manifest["aggregate"]["dg_mean_defect"],
                "mean_improvement": manifest["aggregate"]["mean_improvement"],
                "fraction_dg_better": manifest["aggregate"]["fraction_dg_better"],
                "common_path_identity": manifest["aggregate"]["common_path_identity_mean_error"],
                "common_path_dg": manifest["aggregate"]["common_path_dg_mean_error"],
            },
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()
