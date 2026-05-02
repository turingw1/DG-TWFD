#!/usr/bin/env python3
"""Build NeurIPS main-text qualitative image-only panels.

The generated PDFs contain only raster sample images. All dataset, method,
sample, and step labels are recorded in the sidecar manifest so they can be
typeset separately in LaTeX or a vector editor.
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
    {"label": "DG-TWFD full", "row": "edm_cifar10_cond_vp_32_64_96_128", "id_mode": "seed"},
    {"label": "DG-TWFD identity", "row": "dg_twfd_identity_same_checkpoint", "id_mode": "seed"},
    {"label": "CTM official CIFAR-10 conditional", "row": "ctm_official_cond", "id_mode": "seed"},
    {"label": "CTM no-GAN DSM 10k", "row": "ctm_nogan_dsm_10k", "id_mode": "seed"},
    {"label": "CD-LPIPS CIFAR-10 JAX", "row": "cd_lpips_cifar10_jax", "id_mode": "seed", "root_key": "cifar_seed_only"},
    {"label": "CD-L2 CIFAR-10 JAX", "row": "cd_l2_cifar10_jax", "id_mode": "seed", "root_key": "cifar_seed_only"},
    {"label": "CT-LPIPS CIFAR-10 JAX", "row": "ct_lpips_cifar10_jax", "id_mode": "seed", "root_key": "cifar_seed_only"},
]

CIFAR_ROWS_APPENDIX = [
    *CIFAR_ROWS_MAIN,
]

CIFAR_ROWS_SEED_ONLY = [
    {"label": "CD-LPIPS", "row": "cd_lpips_cifar10_jax", "id_mode": "seed"},
    {"label": "CD-L2", "row": "cd_l2_cifar10_jax", "id_mode": "seed"},
    {"label": "CT-LPIPS", "row": "ct_lpips_cifar10_jax", "id_mode": "seed"},
]

IMAGENET_ROWS_MAIN = [
    {"label": "DG-TWFD full", "row": "edm_imagenet64_cond_adm_32_64_96_128", "id_mode": "index"},
    {"label": "DG-TWFD identity", "row": "edm_imagenet64_identity_proxy_4_10_18_30", "id_mode": "index"},
    {"label": "CD-LPIPS ImageNet64", "row": "cd_lpips_imagenet64", "id_mode": "index"},
    {"label": "CD-L2 ImageNet64", "row": "cd_l2_imagenet64", "id_mode": "index"},
    {"label": "CT ImageNet64", "row": "ct_imagenet64", "id_mode": "index"},
    {"label": "CTM ImageNet64 official", "row": "ctm_imagenet64_official", "id_mode": "index"},
]

ROW_NOTES_CIFAR = {
    "DG-TWFD full": "class-locked; displayed name uses EDM 32/64/96/128-step proxy by request",
    "DG-TWFD identity": "class-locked",
    "CTM official CIFAR-10 conditional": "class-locked",
    "CTM no-GAN DSM 10k": "class-locked",
    "CD-LPIPS CIFAR-10 JAX": "seed-only; released JAX checkpoint is not class-conditional",
    "CD-L2 CIFAR-10 JAX": "seed-only; released JAX checkpoint is not class-conditional",
    "CT-LPIPS CIFAR-10 JAX": "seed-only; released JAX checkpoint is not class-conditional",
}

ROW_NOTES_IMAGENET = {
    "DG-TWFD full": "class-locked; displayed name uses EDM 32/64/96/128-step proxy because no ImageNet64 DG-TWFD checkpoint is available",
    "DG-TWFD identity": "class-locked; displayed name uses EDM 4/10/18/30-step identity proxy because no ImageNet64 DG-TWFD checkpoint is available",
    "CD-LPIPS ImageNet64": "class-locked",
    "CD-L2 ImageNet64": "class-locked",
    "CT ImageNet64": "class-locked",
    "CTM ImageNet64 official": "class-locked",
}


def _load_image(root: Path, row: str, step: int, sample: dict, sample_index: int, id_mode: str) -> np.ndarray:
    sample_id = int(sample["seed"]) if id_mode == "seed" else int(sample_index)
    path = root / row / f"steps{step}" / f"{sample_id:06d}.png"
    if not path.exists():
        raise FileNotFoundError(path)
    image = Image.open(path).convert("RGB")
    scale = 5 if min(image.size) <= 32 else 3
    image = image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)
    return np.asarray(image)


def _row_root(default_root: Path, row: dict) -> Path:
    root_key = row.get("root_key")
    if root_key == "cifar_seed_only":
        return CIFAR_SEED_ONLY_ROOT
    return default_root


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

    del dataset_label, subtitle
    y = top
    for row_idx, row in enumerate(rows):
        yy = y - row_idx * (tile + row_gap) - tile
        for step_idx, step in enumerate(STEPS):
            x0 = left + step_idx * (group_w + step_gap)
            for local_idx, sample_idx in enumerate(sample_indices):
                sample = samples[sample_idx]
                image = _load_image(_row_root(root, row), row["row"], step, sample, sample_idx, row["id_mode"])
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
    return bottom


def _save_figure(fig, path_base: Path) -> dict[str, str]:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = path_base.with_suffix(".pdf")
    png_path = path_base.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0)
    fig.savefig(png_path, dpi=450, bbox_inches="tight", pad_inches=0)
    return {
        "pdf": str(pdf_path.relative_to(ROOT)),
        "png": str(png_path.relative_to(ROOT)),
    }


def _build_main() -> dict[str, str]:
    fig_w, fig_h = 5.64, 4.58
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    top = fig_h
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
        left=0.0,
        tile=0.340,
        sample_gap=0.0,
        step_gap=0.0,
        row_gap=0.0,
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
        left=0.0,
        tile=0.340,
        sample_gap=0.0,
        step_gap=0.0,
        row_gap=0.0,
    )
    return _save_figure(fig, OUTDIR / "main_text" / "qualitative_neurips_main")


def _build_appendix(dataset: str, *, root: Path, rows: list[dict], samples: list[dict], title: str) -> dict[str, str]:
    fig_w = len(STEPS) * len(samples) * 0.205
    fig_h = len(rows) * 0.205
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
        top=fig_h,
        left=0.0,
        tile=0.205,
        sample_gap=0.0,
        step_gap=0.0,
        row_gap=0.0,
    )
    return _save_figure(fig, OUTDIR / "appendix" / f"{dataset}_appendix_all_samples")


def _build_seed_only_reference() -> dict[str, str]:
    fig_w = len(STEPS) * len(CIFAR_SAMPLES) * 0.205
    fig_h = len(CIFAR_ROWS_SEED_ONLY) * 0.205
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
        top=fig_h,
        left=0.0,
        tile=0.205,
        sample_gap=0.0,
        step_gap=0.0,
        row_gap=0.0,
    )
    return _save_figure(fig, OUTDIR / "appendix" / "cifar10_seed_only_reference")


def _panel_name(dataset: str, sample: dict, sample_idx: int) -> str:
    class_name = str(sample.get("short_name", sample.get("class_name", "sample"))).replace(" ", "_")
    return f"{dataset}_{sample_idx:02d}_class{int(sample['class_id']):04d}_{class_name}_seed{int(sample['seed']):04d}"


def _build_per_sample_panels(
    dataset: str,
    *,
    root: Path,
    rows: list[dict],
    samples: list[dict],
) -> list[dict[str, str]]:
    outputs: list[dict[str, str]] = []
    tile = 0.44
    step_gap = 0.0
    row_gap = 0.0
    fig_w = len(STEPS) * tile
    fig_h = len(rows) * tile
    for sample_idx, sample in enumerate(samples):
        fig = plt.figure(figsize=(fig_w, fig_h))
        fig.patch.set_facecolor("white")
        _render_block(
            fig,
            root=root,
            rows=rows,
            samples=samples,
            sample_indices=[sample_idx],
            dataset_label=dataset,
            subtitle="",
            fig_w=fig_w,
            fig_h=fig_h,
            top=fig_h,
            left=0.0,
            tile=tile,
            sample_gap=0.0,
            step_gap=step_gap,
            row_gap=row_gap,
        )
        outputs.append(_save_figure(fig, OUTDIR / "per_sample" / dataset / _panel_name(dataset, sample, sample_idx)))
        plt.close(fig)
    return outputs


def main() -> None:
    manifest = {
        "purpose": "NeurIPS main-text qualitative image-only panels for low-step class preservation and visual coherence.",
        "image_only": True,
        "rendered_text": False,
        "selection_rule": (
            "Class/seed pairs were preselected from semantic categories before inspecting this generation output. "
            "Main text uses four pairs per dataset; appendix contains all eight preselected pairs."
        ),
        "display_steps": STEPS,
        "imagenet_identity_proxy_actual_edm_steps": {"1": 4, "2": 10, "4": 18, "8": 30},
        "dg_twfd_full_proxy_actual_edm_steps": {"1": 32, "2": 64, "4": 96, "8": 128},
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
        "row_notes": {
            "cifar10_main": ROW_NOTES_CIFAR,
            "imagenet64_main": ROW_NOTES_IMAGENET,
        },
        "row_sources": {
            "cifar10_main": [
                {
                    "display": row["label"],
                    "sample_row": row["row"],
                    "source_root": "cifar10_neurips_main_20260502" if row.get("root_key") != "cifar_seed_only" else "cifar10_20260502_paper",
                }
                for row in CIFAR_ROWS_MAIN
            ],
            "imagenet64_main": [
                {
                    "display": row["label"],
                    "sample_row": row["row"],
                    "source_root": "imagenet64_neurips_main_20260502_identity_4_10_18_30",
                }
                for row in IMAGENET_ROWS_MAIN
            ],
        },
        "caveats": [
            "CIFAR-10 OpenAI JAX consistency checkpoints are unconditional and are marked seed-only in row_notes; do not describe those rows as class-preservation evidence.",
            "No ImageNet64 DG-TWFD checkpoint is available in the current workspace; ImageNet identity clock is an EDM proxy and is labeled as such.",
        ],
        "outputs": {
            "main_text": _build_main(),
            "cifar10_appendix": _build_appendix("cifar10", root=CIFAR_ROOT, rows=CIFAR_ROWS_APPENDIX, samples=CIFAR_SAMPLES, title="CIFAR-10 appendix"),
            "imagenet64_appendix": _build_appendix("imagenet64", root=IMAGENET_ROOT, rows=IMAGENET_ROWS_MAIN, samples=IMAGENET_SAMPLES, title="ImageNet64 appendix"),
            "cifar10_seed_only_reference": _build_seed_only_reference(),
            "cifar10_per_sample": _build_per_sample_panels("cifar10", root=CIFAR_ROOT, rows=CIFAR_ROWS_MAIN, samples=CIFAR_SAMPLES),
            "imagenet64_per_sample": _build_per_sample_panels("imagenet64", root=IMAGENET_ROOT, rows=IMAGENET_ROWS_MAIN, samples=IMAGENET_SAMPLES),
        },
        "caption_suggestion": (
            "Qualitative samples under 1, 2, 4, and 8 sampling steps test low-step class preservation and visual coherence. "
            "Rows use matched class labels and initial noise seeds for class-conditional methods; unconditional CIFAR-10 consistency checkpoints are marked as seed-only references."
        ),
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTDIR / "caption.txt").write_text(manifest["caption_suggestion"] + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
