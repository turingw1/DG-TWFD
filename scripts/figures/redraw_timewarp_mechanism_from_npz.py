#!/usr/bin/env python3
"""Redraw the DG-TWFD mechanism figure from saved real-trajectory evidence.

This script intentionally does not recompute trajectories. It reads the
checkpoint-derived NPZ produced by build_timewarp_mechanism_figure.py and
redraws the main figure around the supported claim:

DG-TWFD reallocates original-time waypoints toward high-defect regions and
reduces paired semigroup defect, without claiming lower pointwise path error.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-dgtwfd")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
from matplotlib import gridspec, pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "docs"
    / "experiments"
    / "DG_TWFD_v3"
    / "figures"
    / "mechanism"
    / "timewarp_mechanism_20260502"
)

COLORS = {
    "teacher": "#6e6e6e",
    "identity": "#d95f02",
    "dg": "#1f77b4",
    "defect": "#ef8a76",
    "variation": "#7b5aa6",
    "black": "#333333",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def _normalize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if not bool(finite.any()):
        return np.zeros_like(arr)
    lo = float(arr[finite].min())
    hi = float(arr[finite].max())
    if hi - lo < 1.0e-12:
        out = np.zeros_like(arr, dtype=np.float64)
    else:
        out = (arr - lo) / (hi - lo)
    out[~finite] = 0.0
    return out


def _finite_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def _bootstrap_ci(values: np.ndarray, *, seed: int, samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float64)
    mean = arr.mean(axis=0)
    if arr.shape[0] <= 1 or samples <= 0:
        se = arr.std(axis=0, ddof=0) / np.sqrt(max(arr.shape[0], 1))
        return mean, mean - 1.96 * se, mean + 1.96 * se
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, arr.shape[0], size=(samples, arr.shape[0]))
    boot = arr[indices].mean(axis=1)
    lo = np.percentile(boot, 2.5, axis=0)
    hi = np.percentile(boot, 97.5, axis=0)
    return mean, lo, hi


def _teacher_waypoints(teacher_proj: np.ndarray, u_dense: np.ndarray, u_nodes: np.ndarray) -> np.ndarray:
    nodes = np.asarray(u_nodes, dtype=np.float64)
    x = np.interp(nodes, u_dense, teacher_proj[:, 0])
    y = np.interp(nodes, u_dense, teacher_proj[:, 1])
    return np.stack([x, y], axis=1)


def _format_panel(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=7.15, fontweight="bold", pad=3)
    ax.tick_params(axis="both", labelsize=6.4, width=0.65, length=2.4)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("#333333")


def _draw_panel_a(ax, data: dict[str, np.ndarray]) -> dict[str, Any]:
    density = np.asarray(data["density_mass"], dtype=np.float64)
    density = np.clip(density, 0.0, None)
    density = density / max(float(density.sum()), 1.0e-12)
    defect = _normalize(data["defect_bins"])
    variation = _normalize(data["variation_bins"])
    n_bins = density.shape[0]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    q_edges = np.concatenate([[0.0], np.cumsum(density)])
    q_edges[-1] = 1.0

    for idx, value in enumerate(defect):
        ax.axvspan(
            edges[idx],
            edges[idx + 1],
            ymin=0.11,
            ymax=0.98,
            color=COLORS["defect"],
            alpha=0.035 + 0.23 * float(value),
            lw=0.0,
            zorder=0,
        )

    ax.plot([0.0, 1.0], [0.0, 1.0], color="#4a4a4a", lw=1.0, ls=(0, (3, 2)), label="identity")
    ax.plot(edges, q_edges, color=COLORS["dg"], lw=2.0, label="DG-TWFD warp")

    r_nodes = np.linspace(0.0, 1.0, len(data["dg_u"]))
    dg_u = np.asarray(data["dg_u"], dtype=np.float64)
    identity_u = np.asarray(data["identity_u"], dtype=np.float64)
    ax.scatter(dg_u, r_nodes, s=17, color=COLORS["dg"], edgecolors="white", linewidths=0.45, zorder=4)
    for x_pos in dg_u:
        ax.plot([x_pos, x_pos], [-0.055, 0.0], color=COLORS["dg"], lw=0.8, clip_on=False, zorder=3)
    for x_pos in identity_u:
        ax.plot([x_pos, x_pos], [1.015, 1.055], color="#777777", lw=0.55, clip_on=False, zorder=3)

    rug_y0 = -0.105
    active = np.flatnonzero(variation > 0.04)
    for idx in active:
        center = 0.5 * (edges[idx] + edges[idx + 1])
        ax.plot(
            [center, center],
            [rug_y0, rug_y0 + 0.055 * float(variation[idx])],
            color=COLORS["variation"],
            lw=0.45,
            alpha=0.72,
            clip_on=False,
            solid_capstyle="butt",
        )

    ax.text(
        0.975,
        0.12,
        "uniform warped grid\n-> nonuniform original waypoints",
        transform=ax.transAxes,
        fontsize=5.45,
        ha="right",
        va="bottom",
        color=COLORS["dg"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.0},
    )
    ax.text(0.02, 0.93, "red: empirical defect", transform=ax.transAxes, fontsize=5.7, color="#8a3d34")
    ax.text(
        0.02,
        0.065,
        "purple rug: variation",
        transform=ax.transAxes,
        fontsize=5.6,
        color=COLORS["variation"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.8},
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.12, 1.08)
    ax.set_xlabel("original progress $p=1-t$", fontsize=7.0)
    ax.set_ylabel("warped progress $q=1-g(t)$", fontsize=7.0)
    _format_panel(ax, "A. Defect-guided time allocation")
    ax.legend(loc="lower right", fontsize=5.7, frameon=False, handlelength=2.1)
    return {
        "density_mass_sum": float(density.sum()),
        "corr_density_defect": _finite_corr(density, defect),
        "corr_density_variation": _finite_corr(density, variation),
        "dg_original_waypoints": dg_u.tolist(),
    }


def _teacher_defect_for_segments(u_dense: np.ndarray, defect_bins: np.ndarray) -> np.ndarray:
    mids = 0.5 * (u_dense[:-1] + u_dense[1:])
    n_bins = len(defect_bins)
    indices = np.clip(np.floor(mids * n_bins).astype(int), 0, n_bins - 1)
    return _normalize(defect_bins)[indices]


def _draw_direction_arrow(ax, points: np.ndarray, idx: int, color: str) -> None:
    if idx + 1 >= len(points):
        return
    ax.annotate(
        "",
        xy=points[idx + 1],
        xytext=points[idx],
        arrowprops={
            "arrowstyle": "-|>",
            "mutation_scale": 6.5,
            "lw": 0.8,
            "color": color,
            "shrinkA": 2.5,
            "shrinkB": 2.5,
            "alpha": 0.85,
        },
        zorder=6,
    )


def _draw_panel_b(ax, data: dict[str, np.ndarray]) -> dict[str, Any]:
    teacher = np.asarray(data["teacher_proj"], dtype=np.float64)
    u_dense = np.asarray(data["u_dense"], dtype=np.float64)
    identity_nodes = _teacher_waypoints(teacher, u_dense, data["identity_u"])
    dg_nodes = _teacher_waypoints(teacher, u_dense, data["dg_u"])

    ax.plot(teacher[:, 0], teacher[:, 1], color=COLORS["teacher"], lw=1.55, alpha=0.55, zorder=1)

    seg_values = _teacher_defect_for_segments(u_dense, data["defect_bins"])
    threshold = float(np.percentile(seg_values, 75.0))
    segments = np.stack([teacher[:-1], teacher[1:]], axis=1)
    high_segments = segments[seg_values >= threshold]
    if len(high_segments) > 0:
        lc = LineCollection(high_segments, colors=COLORS["defect"], linewidths=3.3, alpha=0.48, zorder=2)
        ax.add_collection(lc)
        target = high_segments[len(high_segments) // 2].mean(axis=0)
        ax.annotate(
            "high-defect arc",
            xy=target,
            xytext=(target[0] - 0.28, target[1] + 0.42),
            fontsize=5.7,
            color="#8a3d34",
            arrowprops={"arrowstyle": "->", "lw": 0.65, "color": "#8a3d34", "shrinkA": 1, "shrinkB": 1},
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.0},
            zorder=7,
        )

    ax.plot(
        identity_nodes[:, 0],
        identity_nodes[:, 1],
        color=COLORS["identity"],
        lw=1.05,
        marker="o",
        ms=3.1,
        mec="white",
        mew=0.35,
        alpha=0.92,
        zorder=4,
        label="identity waypoints",
    )
    ax.plot(
        dg_nodes[:, 0],
        dg_nodes[:, 1],
        color=COLORS["dg"],
        lw=1.25,
        marker="o",
        ms=3.4,
        mec="white",
        mew=0.38,
        alpha=0.96,
        zorder=5,
        label="DG-TWFD waypoints",
    )
    _draw_direction_arrow(ax, identity_nodes, 2, COLORS["identity"])
    _draw_direction_arrow(ax, dg_nodes, 5, COLORS["dg"])

    label_offsets = {
        0: (0.015, -0.030),
        2: (0.015, -0.020),
        4: (0.015, 0.035),
        6: (0.015, 0.035),
        8: (0.015, -0.030),
    }
    for k in [0, 2, 4, 6, 8]:
        point = dg_nodes[k]
        dx, dy = label_offsets[k]
        ax.text(
            point[0] + dx,
            point[1] + dy,
            f"k={k}",
            fontsize=5.4,
            color=COLORS["black"],
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.74, "pad": 0.8},
            zorder=8,
        )

    ax.scatter(teacher[0, 0], teacher[0, 1], s=18, color="#202020", marker="x", zorder=8)
    ax.scatter(teacher[-1, 0], teacher[-1, 1], s=18, color="#202020", marker="s", zorder=8)
    ax.set_xlabel("endpoint axis", fontsize=7.0)
    ax.set_ylabel("orthogonal residual PC1", fontsize=7.0)
    _format_panel(ax, "B. Same path, redistributed waypoints")
    handles = [
        Line2D([0], [0], color=COLORS["teacher"], lw=1.5, label="teacher path"),
        Line2D([0], [0], color=COLORS["identity"], lw=1.1, marker="o", ms=3.0, label="identity"),
        Line2D([0], [0], color=COLORS["dg"], lw=1.2, marker="o", ms=3.0, label="DG-TWFD"),
        Line2D([0], [0], color=COLORS["defect"], lw=3.0, alpha=0.5, label="high defect"),
    ]
    ax.legend(handles=handles, loc="best", fontsize=5.5, frameon=False, handlelength=1.8)
    return {
        "waypoint_source": "teacher path interpolation; student composed states intentionally omitted",
        "high_defect_percentile": 75.0,
        "labelled_waypoints": [0, 2, 4, 6, 8],
    }


def _draw_panel_c(ax_top, ax_bar, data: dict[str, np.ndarray], *, seed: int, bootstrap_samples: int) -> dict[str, Any]:
    identity = np.asarray(data["identity_defect_plot"], dtype=np.float64)
    dg = np.asarray(data["dg_defect_plot"], dtype=np.float64)
    segment_x = np.arange(1, identity.shape[1] + 1)
    id_mean, id_lo, id_hi = _bootstrap_ci(identity, seed=seed, samples=bootstrap_samples)
    dg_mean, dg_lo, dg_hi = _bootstrap_ci(dg, seed=seed + 1, samples=bootstrap_samples)
    paired = identity - dg
    paired_mean = paired.mean(axis=0)

    ax_top.plot(segment_x, id_mean, color=COLORS["identity"], lw=1.7, marker="o", ms=3.0, label="identity")
    ax_top.fill_between(segment_x, id_lo, id_hi, color=COLORS["identity"], alpha=0.15, lw=0)
    ax_top.plot(segment_x, dg_mean, color=COLORS["dg"], lw=1.85, marker="o", ms=3.0, label="DG-TWFD")
    ax_top.fill_between(segment_x, dg_lo, dg_hi, color=COLORS["dg"], alpha=0.15, lw=0)
    ax_top.set_xlim(0.75, identity.shape[1] + 0.25)
    ax_top.set_xticklabels([])
    ax_top.set_ylabel("defect MSE ($10^{-4}$)", fontsize=7.0)
    ax_top.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value * 1.0e4:.1f}"))
    _format_panel(ax_top, "C. Paired semigroup defect")
    ax_top.legend(loc="upper right", fontsize=5.6, frameon=False)

    n = identity.shape[0]
    mean_identity = float(identity.mean())
    mean_dg = float(dg.mean())
    reduction = (mean_identity - mean_dg) / max(mean_identity, 1.0e-12)
    frac_better = float(np.mean(identity.mean(axis=1) > dg.mean(axis=1)))
    ax_top.text(
        0.03,
        0.92,
        f"n={n}, mean drop {100.0 * reduction:.1f}%",
        transform=ax_top.transAxes,
        fontsize=5.8,
        va="top",
        color=COLORS["black"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 1.0},
    )

    colors = [COLORS["dg"] if value >= 0 else COLORS["identity"] for value in paired_mean]
    ax_bar.axhline(0.0, color="#555555", lw=0.65)
    ax_bar.bar(segment_x, paired_mean, width=0.62, color=colors, alpha=0.78, edgecolor="none")
    ax_bar.set_xlim(0.75, identity.shape[1] + 0.25)
    ax_bar.set_xlabel("segment index $k$", fontsize=7.0)
    ax_bar.set_ylabel("id-DG\n($10^{-5}$)", fontsize=6.2)
    ax_bar.tick_params(axis="both", labelsize=6.1, width=0.65, length=2.2)
    ax_bar.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value * 1.0e5:.0f}"))
    for spine in ax_bar.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("#333333")

    return {
        "n": int(n),
        "segment_count": int(identity.shape[1]),
        "identity_mean_defect": mean_identity,
        "dg_mean_defect": mean_dg,
        "relative_reduction": float(reduction),
        "fraction_dg_better": frac_better,
        "paired_segment_mean_identity_minus_dg": paired_mean.tolist(),
        "identity_ci_low": id_lo.tolist(),
        "identity_ci_high": id_hi.tolist(),
        "dg_ci_low": dg_lo.tolist(),
        "dg_ci_high": dg_hi.tolist(),
    }


def _save_caption(output_dir: Path) -> None:
    caption = r"""\textbf{Real-trajectory mechanism of defect-guided time warping.}
The learned clock allocates larger warped-time mass to original-time regions with high empirical semigroup defect, so uniform warped-time sampling induces non-uniform original-time waypoints.
On a held-out CIFAR-10 trajectory, the teacher path is unchanged while the low-step waypoint traversal is redistributed toward the high-defect segment.
Across held-out trajectories, DG-TWFD reduces paired semigroup defect under the same step budget.
"""
    (output_dir / "caption.tex").write_text(caption, encoding="utf-8")


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_manifest(output_dir: Path, manifest: dict[str, Any], redraw: dict[str, Any]) -> None:
    manifest = dict(manifest)
    manifest["figure"] = {
        "pdf": str((output_dir / "timewarp_mechanism_main.pdf").relative_to(ROOT)),
        "png": str((output_dir / "timewarp_mechanism_main.png").relative_to(ROOT)),
        "data": str((output_dir / "timewarp_mechanism_data.npz").relative_to(ROOT)),
        "caption": str((output_dir / "caption.tex").relative_to(ROOT)),
    }
    manifest["redraw"] = redraw
    (output_dir / "timewarp_mechanism_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir = output_dir if output_dir.is_absolute() else ROOT / output_dir
    data_path = output_dir / "timewarp_mechanism_data.npz"
    manifest_path = output_dir / "timewarp_mechanism_manifest.json"
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    loaded = np.load(data_path)
    data = {key: loaded[key] for key in loaded.files}
    manifest = _load_manifest(manifest_path)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
        }
    )

    fig = plt.figure(figsize=(7.15, 2.78))
    outer = gridspec.GridSpec(
        1,
        3,
        figure=fig,
        width_ratios=[1.03, 1.03, 1.08],
        wspace=0.42,
        left=0.055,
        right=0.992,
        top=0.91,
        bottom=0.20,
    )
    ax_a = fig.add_subplot(outer[0, 0])
    ax_b = fig.add_subplot(outer[0, 1])
    c_grid = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, 2], height_ratios=[3.0, 1.02], hspace=0.06)
    ax_c = fig.add_subplot(c_grid[0, 0])
    ax_c_bar = fig.add_subplot(c_grid[1, 0], sharex=ax_c)

    panel_a = _draw_panel_a(ax_a, data)
    panel_b = _draw_panel_b(ax_b, data)
    panel_c = _draw_panel_c(ax_c, ax_c_bar, data, seed=args.seed, bootstrap_samples=args.bootstrap_samples)

    pdf_path = output_dir / "timewarp_mechanism_main.pdf"
    png_path = output_dir / "timewarp_mechanism_main.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    _save_caption(output_dir)
    redraw = {
        "version": "v2_neurips_mechanism_redraw",
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "claim": "time allocation and paired semigroup-defect reduction; no pointwise teacher-path closeness claim",
        "data_source": str(data_path.relative_to(ROOT)),
        "panels": {
            "A": panel_a,
            "B": panel_b,
            "C": panel_c,
        },
        "path_error_caution": {
            "common_path_identity_mean_error": manifest.get("aggregate", {}).get("common_path_identity_mean_error"),
            "common_path_dg_mean_error": manifest.get("aggregate", {}).get("common_path_dg_mean_error"),
            "node_identity_mean_error": manifest.get("aggregate", {}).get("node_identity_mean_error"),
            "node_dg_mean_error": manifest.get("aggregate", {}).get("node_dg_mean_error"),
            "interpretation": "These pointwise path metrics do not support claiming DG is closer to the teacher path.",
        },
    }
    _write_manifest(output_dir, manifest, redraw)
    print(json.dumps({"pdf": str(pdf_path), "png": str(png_path), "redraw": redraw["version"]}, indent=2))


if __name__ == "__main__":
    main()
