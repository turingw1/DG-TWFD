#!/usr/bin/env python3
"""Build additional image-only diversity qualitative panels.

This script only assembles already generated samples. It does not run any
sampler or change sampling parameters. Samples are composed at their native
pixel size first, then the complete raster grid is nearest-neighbor upscaled
for PDF/PNG export.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
QUAL_DIR = ROOT / "docs" / "experiments" / "DG_TWFD_v3" / "figures" / "qualitative"
OUTDIR = QUAL_DIR / "neurips_main_20260503_diversity"
CIFAR_ROOT = QUAL_DIR / "class_locked_samples" / "cifar10_neurips_main_20260502"
CIFAR_SEED_ONLY_ROOT = QUAL_DIR / "class_locked_samples" / "cifar10_20260502_paper"
IMAGENET_ROOT = QUAL_DIR / "class_locked_samples" / "imagenet64_neurips_main_20260502_identity_4_10_18_30"

STEPS = [1, 2, 4, 8]

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

# Holdout sample choices avoid the current Fig. 3 pairs:
# CIFAR: airplane/car/cat/truck; ImageNet64: 0022/0207/0407/0701.
CIFAR_DIVERSITY_INDICES = [3, 4, 6, 7]
IMAGENET_DIVERSITY_INDICES = [2, 4, 6, 7]

CIFAR_ROWS = [
    {"label": "DG-TWFD full", "row": "edm_cifar10_cond_vp_32_64_96_128", "id_mode": "seed", "lock": "class-locked proxy"},
    {"label": "DG-TWFD identity", "row": "dg_twfd_identity_same_checkpoint", "id_mode": "seed", "lock": "class-locked"},
    {"label": "CTM official CIFAR-10 conditional", "row": "ctm_official_cond", "id_mode": "seed", "lock": "class-locked"},
    {"label": "CTM no-GAN DSM 10k", "row": "ctm_nogan_dsm_10k", "id_mode": "seed", "lock": "class-locked"},
    {"label": "CD-LPIPS CIFAR-10 JAX", "row": "cd_lpips_cifar10_jax", "id_mode": "seed", "root_key": "cifar_seed_only", "lock": "seed-only"},
    {"label": "CD-L2 CIFAR-10 JAX", "row": "cd_l2_cifar10_jax", "id_mode": "seed", "root_key": "cifar_seed_only", "lock": "seed-only"},
    {"label": "CT-LPIPS CIFAR-10 JAX", "row": "ct_lpips_cifar10_jax", "id_mode": "seed", "root_key": "cifar_seed_only", "lock": "seed-only"},
]

IMAGENET_ROWS = [
    {"label": "DG-TWFD full", "row": "edm_imagenet64_cond_adm_32_64_96_128", "id_mode": "index", "lock": "class-locked proxy"},
    {"label": "DG-TWFD identity", "row": "edm_imagenet64_identity_proxy_4_10_18_30", "id_mode": "index", "lock": "class-locked proxy"},
    {"label": "CD-LPIPS ImageNet64", "row": "cd_lpips_imagenet64", "id_mode": "index", "lock": "class-locked"},
    {"label": "CD-L2 ImageNet64", "row": "cd_l2_imagenet64", "id_mode": "index", "lock": "class-locked"},
    {"label": "CT ImageNet64", "row": "ct_imagenet64", "id_mode": "index", "lock": "class-locked"},
    {"label": "CTM ImageNet64 official", "row": "ctm_imagenet64_official", "id_mode": "index", "lock": "class-locked"},
]


def _row_root(default_root: Path, row: dict) -> Path:
    if row.get("root_key") == "cifar_seed_only":
        return CIFAR_SEED_ONLY_ROOT
    return default_root


def _load_native(root: Path, row: dict, step: int, sample: dict, sample_index: int) -> Image.Image:
    sample_id = int(sample["seed"]) if row["id_mode"] == "seed" else int(sample_index)
    path = _row_root(root, row) / row["row"] / f"steps{step}" / f"{sample_id:06d}.png"
    if not path.exists():
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


def _assemble_native_grid(root: Path, rows: list[dict], samples: list[dict], sample_indices: list[int]) -> Image.Image:
    first = _load_native(root, rows[0], STEPS[0], samples[sample_indices[0]], sample_indices[0])
    cell_w, cell_h = first.size
    grid_w = len(STEPS) * len(sample_indices) * cell_w
    grid_h = len(rows) * cell_h
    grid = Image.new("RGB", (grid_w, grid_h), "white")

    for row_idx, row in enumerate(rows):
        for step_idx, step in enumerate(STEPS):
            for local_idx, sample_idx in enumerate(sample_indices):
                sample = samples[sample_idx]
                image = _load_native(root, row, step, sample, sample_idx)
                if image.size != (cell_w, cell_h):
                    raise ValueError(f"Mixed native sizes: {image.size} vs {(cell_w, cell_h)}")
                x = (step_idx * len(sample_indices) + local_idx) * cell_w
                y = row_idx * cell_h
                grid.paste(image, (x, y))
    return grid


def _export_grid(grid: Image.Image, path_base: Path, scale: int) -> dict[str, str | int | list[int]]:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    export = grid.resize((grid.width * scale, grid.height * scale), Image.Resampling.NEAREST)
    png_path = path_base.with_suffix(".png")
    pdf_path = path_base.with_suffix(".pdf")
    export.save(png_path)
    export.save(pdf_path, "PDF", resolution=600.0)
    return {
        "pdf": str(pdf_path.relative_to(ROOT)),
        "png": str(png_path.relative_to(ROOT)),
        "native_size": [grid.width, grid.height],
        "export_size": [export.width, export.height],
        "export_scale_after_native_assembly": scale,
    }


def _panel_name(dataset: str, sample: dict, sample_idx: int) -> str:
    class_name = str(sample["short_name"]).replace(" ", "_")
    return f"{dataset}_{sample_idx:02d}_class{int(sample['class_id']):04d}_{class_name}_seed{int(sample['seed']):04d}"


def _build_per_sample(dataset: str, root: Path, rows: list[dict], samples: list[dict], indices: list[int], scale: int) -> list[dict]:
    outputs = []
    for sample_idx in indices:
        sample = samples[sample_idx]
        grid = _assemble_native_grid(root, rows, samples, [sample_idx])
        out = _export_grid(grid, OUTDIR / "per_sample" / dataset / _panel_name(dataset, sample, sample_idx), scale)
        outputs.append({"sample_index": sample_idx, "sample": sample, **out})
    return outputs


def _build_main(dataset: str, root: Path, rows: list[dict], samples: list[dict], indices: list[int], scale: int) -> dict:
    grid = _assemble_native_grid(root, rows, samples, indices)
    return _export_grid(grid, OUTDIR / "main_text" / f"{dataset}_diversity_main", scale)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "purpose": "Additional diversity qualitative panels using holdout class/seed pairs.",
        "image_only": True,
        "rendered_text": False,
        "sampling_logic_changed": False,
        "native_assembly_before_export": True,
        "display_steps": STEPS,
        "selection_rule": (
            "Use the four preselected holdout pairs not used by the current Fig. 3 main panel. "
            "CIFAR excludes airplane/car/cat/truck; ImageNet64 excludes 0022/0207/0407/0701. "
            "The holdout set was chosen for semantic diversity and inspection value rather than after cherry-picking model outputs."
        ),
        "current_fig3_excluded": {
            "cifar10": ["airplane", "automobile", "cat", "truck"],
            "imagenet64_class_ids": [22, 207, 407, 701],
        },
        "diversity_indices": {
            "cifar10": CIFAR_DIVERSITY_INDICES,
            "imagenet64": IMAGENET_DIVERSITY_INDICES,
        },
        "samples": {
            "cifar10": [CIFAR_SAMPLES[idx] for idx in CIFAR_DIVERSITY_INDICES],
            "imagenet64": [IMAGENET_SAMPLES[idx] for idx in IMAGENET_DIVERSITY_INDICES],
        },
        "rows": {
            "cifar10": [row["label"] for row in CIFAR_ROWS],
            "imagenet64": [row["label"] for row in IMAGENET_ROWS],
        },
        "row_locking": {
            "cifar10": {row["label"]: row["lock"] for row in CIFAR_ROWS},
            "imagenet64": {row["label"]: row["lock"] for row in IMAGENET_ROWS},
        },
        "row_sources": {
            "cifar10": [
                {
                    "display": row["label"],
                    "sample_row": row["row"],
                    "source_root": "cifar10_20260502_paper" if row.get("root_key") == "cifar_seed_only" else "cifar10_neurips_main_20260502",
                }
                for row in CIFAR_ROWS
            ],
            "imagenet64": [
                {
                    "display": row["label"],
                    "sample_row": row["row"],
                    "source_root": "imagenet64_neurips_main_20260502_identity_4_10_18_30",
                }
                for row in IMAGENET_ROWS
            ],
        },
        "actual_proxy_steps": {
            "dg_twfd_full_edm_proxy": {"1": 32, "2": 64, "4": 96, "8": 128},
            "imagenet64_identity_edm_proxy": {"1": 4, "2": 10, "4": 18, "8": 30},
        },
        "caveats": [
            "CIFAR-10 OpenAI JAX consistency checkpoints are unconditional and seed-only.",
            "DG-TWFD full display rows use EDM 32/64/96/128-step proxy samples by request.",
            "ImageNet64 DG-TWFD identity is also an EDM proxy because no ImageNet64 DG-TWFD checkpoint is available.",
        ],
        "outputs": {
            "main_text": {
                "cifar10": _build_main("cifar10", CIFAR_ROOT, CIFAR_ROWS, CIFAR_SAMPLES, CIFAR_DIVERSITY_INDICES, scale=6),
                "imagenet64": _build_main("imagenet64", IMAGENET_ROOT, IMAGENET_ROWS, IMAGENET_SAMPLES, IMAGENET_DIVERSITY_INDICES, scale=4),
            },
            "per_sample": {
                "cifar10": _build_per_sample("cifar10", CIFAR_ROOT, CIFAR_ROWS, CIFAR_SAMPLES, CIFAR_DIVERSITY_INDICES, scale=8),
                "imagenet64": _build_per_sample("imagenet64", IMAGENET_ROOT, IMAGENET_ROWS, IMAGENET_SAMPLES, IMAGENET_DIVERSITY_INDICES, scale=4),
            },
        },
    }
    (OUTDIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTDIR / "caption.txt").write_text(
        "Additional qualitative holdout panels use class/seed pairs not shown in the main Fig. 3 panel. "
        "All bitmaps are image-only; labels should be typeset separately as vector text.\\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
