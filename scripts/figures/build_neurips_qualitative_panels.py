#!/usr/bin/env python3
"""Build NeurIPS main-text qualitative panels with vector labels.

The main figure uses class-conditional rows only. CIFAR-10 JAX consistency
checkpoints are unconditional, so they are rendered as a separate seed-only
reference panel and are not mixed into the class-preservation figure.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-dgtwfd")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
QUAL_DIR = ROOT / "docs" / "experiments" / "DG_TWFD_v3" / "figures" / "qualitative"
OUTDIR = QUAL_DIR / "neurips_main_20260502_identity_4_10_18_30"
CIFAR_ROOT = QUAL_DIR / "class_locked_samples" / "cifar10_neurips_main_20260502"
IMAGENET_ROOT = QUAL_DIR / "class_locked_samples" / "imagenet64_neurips_main_20260502_identity_4_10_18_30"
CIFAR_SEED_ONLY_ROOT = QUAL_DIR / "class_locked_samples" / "cifar10_20260502_paper"

STEPS = [1, 2, 4, 8]
MAIN_CIFAR_INDICES = [0, 1, 2, 5]
MAIN_IMAGENET_INDICES = [0, 1, 3, 5]

CIFAR_SAMPLES = [
    {"class_id": 0, "class_name": "airplane", "short_name": "plane", "seed": 1000},
    {"class_id": 1, "class_name": "automobile", "short_name": "car", "seed": 1001},
    {"class_id": 3, "class_name": "cat", "short_name": "cat", "seed": 1003},
    {"class_id": 7, "class_name": "horse", "short_name": "horse", "seed": 1007},
    {"class_id": 8, "class_name": "ship", "short_name": "ship", "seed": 1008},
    {"class_id": 9, "class_name": "truck", "short_name": "truck", "seed": 1009},
    {"class_id": 2, "class_name": "bird", "short_name": "bird", "seed": 1002},
    {"class_id": 4, "class_name": "deer", "short_name": "deer", "seed": 1004},
]

IMAGENET_SAMPLES = [
    {"class_id": 22, "class_name": "bird", "short_name": "bird", "seed": 31},
    {"class_id": 207, "class_name": "dog", "short_name": "dog", "seed": 32},
    {"class_id": 281, "class_name": "cat-like", "short_name": "cat", "seed": 33},
    {"class_id": 407, "class_name": "vehicle", "short_name": "vehicle", "seed": 34},
    {"class_id": 817, "class_name": "sports car", "short_name": "car", "seed": 35},
    {"class_id": 701, "class_name": "clutter", "short_name": "clutter", "seed": 36},
    {"class_id": 444, "class_name": "object", "short_name": "object", "seed": 37},
    {"class_id": 8, "class_name": "bird detail", "short_name": "bird2", "seed": 38},
]

CIFAR_ROWS_MAIN = [
    {"label": "DG-TWFD", "row": "dg_twfd_full_learned_clock", "id_mode": "seed"},
    {"label": "Identity", "row": "dg_twfd_identity_same_checkpoint", "id_mode": "seed"},
    {"label": "CTM", "row": "ctm_official_cond", "id_mode": "seed"},
]

CIFAR_ROWS_APPENDIX = [
    *CIFAR_ROWS_MAIN,
    {"label": "CTM no-GAN", "row": "ctm_nogan_dsm_10k", "id_mode": "seed"},
]

CIFAR_ROWS_SEED_ONLY = [
    {"label": "CD-LPIPS", "row": "cd_lpips_cifar10_jax", "id_mode": "seed"},
    {"label": "CD-L2", "row": "cd_l2_cifar10_jax", "id_mode": "seed"},
    {"label": "CT-LPIPS", "row": "ct_lpips_cifar10_jax", "id_mode": "seed"},
]

IMAGENET_ROWS_MAIN = [
    {"label": "EDM ref.", "row": "edm_imagenet64_cond_adm_32_48_64_128", "id_mode": "index"},
    {"label": "EDM clock", "row": "edm_imagenet64_identity_proxy_4_10_18_30", "id_mode": "index"},
    {"label": "CD-LPIPS", "row": "cd_lpips_imagenet64", "id_mode": "index"},
    {"label": "CD-L2", "row": "cd_l2_imagenet64", "id_mode": "index"},
    {"label": "CT", "row": "ct_imagenet64", "id_mode": "index"},
    {"label": "CTM", "row": "ctm_imagenet64_official", "id_mode": "index"},
]


def _load_image(root: Path, row: str, step: int, sample: dict, sample_index: int, id_mode: str) -> np.ndarray:
    sample_id = int(sample["seed"]) if id_mode == "seed" else int(sample_index)
    path = root / row / f"steps{step}" / f"{sample_id:06d}.png"
    if not path.exists():
        raise FileNotFoundError(path)
    image = Image.open(path).convert("RGB")
    scale = 5 if min(image.size) <= 32 else 3
    image = image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)
    return np.asarray(image)


def _render_block(
    fig,
    *,
    root: Path,
    rows: list[dict],
    samples: list[dict],
    sample_indices: list[int],
    dataset_label: str,
    subtitle: str,
    fig_w: float,
    fig_h: float,
    top: float,
    left: float,
    tile: float,
    sample_gap: float,
    step_gap: float,
    row_gap: float,
) -> float:
    selected = [samples[idx] for idx in sample_indices]
    n_samples = len(selected)
    group_w = n_samples * tile + (n_samples - 1) * sample_gap

    fig.text(left / fig_w, top / fig_h, dataset_label, ha="left", va="top", fontsize=7.4, weight="bold")
    if subtitle:
        fig.text((left + 0.78) / fig_w, top / fig_h, subtitle, ha="left", va="top", fontsize=5.2)
    y = top - 0.13

    for step_idx, step in enumerate(STEPS):
        x0 = left + 0.62 + step_idx * (group_w + step_gap)
        for local_idx, sample in enumerate(selected):
            label = sample.get("short_name", sample["class_name"])
            fig.text(
                (x0 + local_idx * (tile + sample_gap) + tile / 2) / fig_w,
                y / fig_h,
                label,
                ha="center",
                va="bottom",
                fontsize=4.9,
            )

    y -= 0.055
    for row_idx, row in enumerate(rows):
        yy = y - row_idx * (tile + row_gap) - tile
        fig.text((left + 0.55) / fig_w, (yy + tile / 2) / fig_h, row["label"], ha="right", va="center", fontsize=6.0)
        for step_idx, step in enumerate(STEPS):
            x0 = left + 0.62 + step_idx * (group_w + step_gap)
            for local_idx, sample_idx in enumerate(sample_indices):
                sample = samples[sample_idx]
                image = _load_image(root, row["row"], step, sample, sample_idx, row["id_mode"])
                ax = fig.add_axes(
                    [
                        (x0 + local_idx * (tile + sample_gap)) / fig_w,
                        yy / fig_h,
                        tile / fig_w,
                        tile / fig_h,
                    ]
                )
                ax.imshow(image, interpolation="nearest")
                ax.set_axis_off()

    bottom = y - len(rows) * tile - (len(rows) - 1) * row_gap
    for step_idx, step in enumerate(STEPS):
        x0 = left + 0.62 + step_idx * (group_w + step_gap)
        fig.text((x0 + group_w / 2) / fig_w, (bottom - 0.05) / fig_h, f"{step} step", ha="center", va="top", fontsize=5.9)
    return bottom - 0.18


def _save_figure(fig, path_base: Path) -> dict[str, str]:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = path_base.with_suffix(".pdf")
    png_path = path_base.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(png_path, dpi=450, bbox_inches="tight", pad_inches=0.01)
    return {
        "pdf": str(pdf_path.relative_to(ROOT)),
        "png": str(png_path.relative_to(ROOT)),
    }


def _build_main() -> dict[str, str]:
    fig_w, fig_h = 6.48, 3.78
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    top = fig_h - 0.08
    top = _render_block(
        fig,
        root=CIFAR_ROOT,
        rows=CIFAR_ROWS_MAIN,
        samples=CIFAR_SAMPLES,
        sample_indices=MAIN_CIFAR_INDICES,
        dataset_label="CIFAR-10",
        subtitle="",
        fig_w=fig_w,
        fig_h=fig_h,
        top=top,
        left=0.08,
        tile=0.292,
        sample_gap=0.006,
        step_gap=0.047,
        row_gap=0.014,
    )
    _render_block(
        fig,
        root=IMAGENET_ROOT,
        rows=IMAGENET_ROWS_MAIN,
        samples=IMAGENET_SAMPLES,
        sample_indices=MAIN_IMAGENET_INDICES,
        dataset_label="ImageNet64",
        subtitle="",
        fig_w=fig_w,
        fig_h=fig_h,
        top=top,
        left=0.08,
        tile=0.292,
        sample_gap=0.006,
        step_gap=0.047,
        row_gap=0.014,
    )
    return _save_figure(fig, OUTDIR / "main_text" / "qualitative_neurips_main")


def _build_appendix(dataset: str, *, root: Path, rows: list[dict], samples: list[dict], title: str) -> dict[str, str]:
    fig_w, fig_h = 7.6, 4.9
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    _render_block(
        fig,
        root=root,
        rows=rows,
        samples=samples,
        sample_indices=list(range(len(samples))),
        dataset_label=title,
        subtitle="all preselected samples",
        fig_w=fig_w,
        fig_h=fig_h,
        top=fig_h - 0.08,
        left=0.08,
        tile=0.205,
        sample_gap=0.006,
        step_gap=0.045,
        row_gap=0.016,
    )
    return _save_figure(fig, OUTDIR / "appendix" / f"{dataset}_appendix_all_samples")


def _build_seed_only_reference() -> dict[str, str]:
    fig_w, fig_h = 7.6, 2.3
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    _render_block(
        fig,
        root=CIFAR_SEED_ONLY_ROOT,
        rows=CIFAR_ROWS_SEED_ONLY,
        samples=CIFAR_SAMPLES,
        sample_indices=list(range(len(CIFAR_SAMPLES))),
        dataset_label="CIFAR-10 seed-only reference",
        subtitle="OpenAI JAX checkpoints are unconditional; not evidence for class preservation",
        fig_w=fig_w,
        fig_h=fig_h,
        top=fig_h - 0.08,
        left=0.08,
        tile=0.205,
        sample_gap=0.006,
        step_gap=0.045,
        row_gap=0.016,
    )
    return _save_figure(fig, OUTDIR / "appendix" / "cifar10_seed_only_reference")


def main() -> None:
    manifest = {
        "purpose": "NeurIPS main-text qualitative panels for low-step class preservation and visual coherence.",
        "selection_rule": (
            "Class/seed pairs were preselected from semantic categories before inspecting this generation output. "
            "Main text uses four pairs per dataset; appendix contains all eight preselected pairs."
        ),
        "display_steps": STEPS,
        "imagenet_identity_proxy_actual_edm_steps": {"1": 4, "2": 10, "4": 18, "8": 30},
        "cifar_samples": CIFAR_SAMPLES,
        "imagenet_samples": IMAGENET_SAMPLES,
        "main_indices": {
            "cifar10": MAIN_CIFAR_INDICES,
            "imagenet64": MAIN_IMAGENET_INDICES,
        },
        "rows": {
            "cifar10_main": [row["label"] for row in CIFAR_ROWS_MAIN],
            "imagenet64_main": [row["label"] for row in IMAGENET_ROWS_MAIN],
            "cifar10_seed_only_reference": [row["label"] for row in CIFAR_ROWS_SEED_ONLY],
        },
        "caveats": [
            "CIFAR-10 OpenAI JAX consistency checkpoints are unconditional and are rendered only in a separate seed-only reference panel.",
            "No ImageNet64 DG-TWFD checkpoint is available in the current workspace; ImageNet identity clock is an EDM proxy and is labeled as such.",
        ],
        "outputs": {
            "main_text": _build_main(),
            "cifar10_appendix": _build_appendix("cifar10", root=CIFAR_ROOT, rows=CIFAR_ROWS_APPENDIX, samples=CIFAR_SAMPLES, title="CIFAR-10 appendix"),
            "imagenet64_appendix": _build_appendix("imagenet64", root=IMAGENET_ROOT, rows=IMAGENET_ROWS_MAIN, samples=IMAGENET_SAMPLES, title="ImageNet64 appendix"),
            "cifar10_seed_only_reference": _build_seed_only_reference(),
        },
        "caption_suggestion": (
            "Qualitative samples under 1, 2, 4, and 8 sampling steps test low-step class preservation and visual coherence. "
            "Rows use matched class labels and initial noise seeds for class-conditional methods; unconditional CIFAR-10 consistency checkpoints are shown separately as seed-only references."
        ),
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTDIR / "caption.txt").write_text(manifest["caption_suggestion"] + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
