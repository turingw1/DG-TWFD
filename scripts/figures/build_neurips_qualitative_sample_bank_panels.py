#!/usr/bin/env python3
"""Build per-sample qualitative panels from the large sample bank.

The sampler outputs are produced separately by the dataset-specific generation
scripts. This script only assembles existing PNGs. It keeps each cell at its
native sample resolution while composing the grid, then upscales the completed
grid with nearest-neighbor for paper-friendly PDF/PNG export.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
QUAL_DIR = ROOT / "docs" / "experiments" / "DG_TWFD_v3" / "figures" / "qualitative"
OUTDIR = QUAL_DIR / "neurips_sample_bank_20260503"
CIFAR_ROOT = QUAL_DIR / "class_locked_samples" / "cifar10_sample_bank_20260503"
IMAGENET_ROOT = QUAL_DIR / "class_locked_samples" / "imagenet64_sample_bank_20260503"

STEPS = [1, 2, 4, 8]

CIFAR_ROWS = [
    {"label": "DG-TWFD full", "row": "edm_cifar10_cond_vp_32_64_96_128", "id_mode": "seed", "lock": "class-locked proxy"},
    {"label": "DG-TWFD identity", "row": "dg_twfd_identity_same_checkpoint", "id_mode": "seed", "lock": "class-locked"},
    {"label": "CTM official CIFAR-10 conditional", "row": "ctm_official_cond", "id_mode": "seed", "lock": "class-locked"},
    {"label": "CTM no-GAN DSM 10k", "row": "ctm_nogan_dsm_10k", "id_mode": "seed", "lock": "class-locked"},
    {"label": "CD-LPIPS CIFAR-10 JAX", "row": "cd_lpips_cifar10_jax", "id_mode": "seed", "lock": "seed-only"},
    {"label": "CD-L2 CIFAR-10 JAX", "row": "cd_l2_cifar10_jax", "id_mode": "seed", "lock": "seed-only"},
    {"label": "CT-LPIPS CIFAR-10 JAX", "row": "ct_lpips_cifar10_jax", "id_mode": "seed", "lock": "seed-only"},
]

IMAGENET_ROWS = [
    {"label": "DG-TWFD full", "row": "edm_imagenet64_cond_adm_32_64_96_128", "id_mode": "index", "lock": "class-locked proxy"},
    {"label": "DG-TWFD identity", "row": "edm_imagenet64_identity_proxy_4_10_18_30", "id_mode": "index", "lock": "class-locked proxy"},
    {"label": "CD-LPIPS ImageNet64", "row": "cd_lpips_imagenet64", "id_mode": "index", "lock": "class-locked"},
    {"label": "CD-L2 ImageNet64", "row": "cd_l2_imagenet64", "id_mode": "index", "lock": "class-locked"},
    {"label": "CT ImageNet64", "row": "ct_imagenet64", "id_mode": "index", "lock": "class-locked"},
    {"label": "CTM ImageNet64 official", "row": "ctm_imagenet64_official", "id_mode": "index", "lock": "class-locked"},
]


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _cifar_samples() -> list[dict]:
    manifest = _load_json(CIFAR_ROOT / "manifest.json")
    return [
        {
            "index": int(col["index"]),
            "class_id": int(col["class_id"]),
            "class_name": str(col["class_name"]),
            "short_name": str(col["class_name"]).replace(" ", "_"),
            "seed": int(col["latent_seed"]),
        }
        for col in manifest["columns"]
    ]


def _imagenet_samples() -> list[dict]:
    manifest = _load_json(IMAGENET_ROOT / "manifest.json")
    return [
        {
            "index": int(col["index"]),
            "class_id": int(col["class_id"]),
            "class_name": f"imagenet_class_{int(col['class_id']):04d}",
            "short_name": f"id{int(col['class_id']):04d}",
            "seed": int(col["seed"]),
        }
        for col in manifest["columns"]
    ]


def _sample_file(root: Path, row: dict, step: int, sample: dict) -> Path:
    sample_id = int(sample["seed"]) if row["id_mode"] == "seed" else int(sample["index"])
    return root / row["row"] / f"steps{step}" / f"{sample_id:06d}.png"


def _load_native(root: Path, row: dict, step: int, sample: dict) -> Image.Image:
    path = _sample_file(root, row, step, sample)
    if not path.exists():
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


def _assemble_native_grid(root: Path, rows: list[dict], sample: dict) -> Image.Image:
    first = _load_native(root, rows[0], STEPS[0], sample)
    cell_w, cell_h = first.size
    grid = Image.new("RGB", (len(STEPS) * cell_w, len(rows) * cell_h), "white")
    for row_idx, row in enumerate(rows):
        for step_idx, step in enumerate(STEPS):
            image = _load_native(root, row, step, sample)
            if image.size != (cell_w, cell_h):
                raise ValueError(f"Mixed native sizes for {sample}: {image.size} vs {(cell_w, cell_h)}")
            grid.paste(image, (step_idx * cell_w, row_idx * cell_h))
    return grid


def _export_grid(grid: Image.Image, path_base: Path, scale: int) -> dict:
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


def _panel_name(dataset: str, sample: dict) -> str:
    return (
        f"{dataset}_{int(sample['index']):03d}_class{int(sample['class_id']):04d}_"
        f"{sample['short_name']}_seed{int(sample['seed']):04d}"
    )


def _build_dataset(dataset: str, root: Path, rows: list[dict], samples: list[dict], scale: int) -> list[dict]:
    outputs = []
    for sample in samples:
        grid = _assemble_native_grid(root, rows, sample)
        out = _export_grid(grid, OUTDIR / "per_sample" / dataset / _panel_name(dataset, sample), scale)
        outputs.append({"sample": sample, **out})
    return outputs


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cifar_samples = _cifar_samples()
    imagenet_samples = _imagenet_samples()
    manifest = {
        "purpose": "Large qualitative sample bank for browsing many per-sample model comparisons.",
        "image_only": True,
        "rendered_text": False,
        "sampling_logic_changed_by_panel_builder": False,
        "native_assembly_before_export": True,
        "display_steps": STEPS,
        "source_roots": {
            "cifar10": str(CIFAR_ROOT.relative_to(ROOT)),
            "imagenet64": str(IMAGENET_ROOT.relative_to(ROOT)),
        },
        "sample_counts": {
            "cifar10": len(cifar_samples),
            "imagenet64": len(imagenet_samples),
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
            "cifar10": [{"display": row["label"], "sample_row": row["row"], "source_root": "cifar10_sample_bank_20260503"} for row in CIFAR_ROWS],
            "imagenet64": [{"display": row["label"], "sample_row": row["row"], "source_root": "imagenet64_sample_bank_20260503"} for row in IMAGENET_ROWS],
        },
        "actual_proxy_steps": {
            "dg_twfd_full_edm_proxy": {"1": 32, "2": 64, "4": 96, "8": 128},
            "imagenet64_identity_edm_proxy": {"1": 4, "2": 10, "4": 18, "8": 30},
        },
        "caveats": [
            "CIFAR-10 JAX consistency checkpoints are unconditional and seed-only.",
            "DG-TWFD full display rows use EDM 32/64/96/128-step proxy samples by request.",
            "ImageNet64 DG-TWFD identity uses EDM 4/10/18/30-step proxy samples.",
        ],
        "outputs": {
            "per_sample": {
                "cifar10": _build_dataset("cifar10", CIFAR_ROOT, CIFAR_ROWS, cifar_samples, scale=8),
                "imagenet64": _build_dataset("imagenet64", IMAGENET_ROOT, IMAGENET_ROWS, imagenet_samples, scale=4),
            },
        },
    }
    (OUTDIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTDIR / "caption.txt").write_text(
        "Large image-only qualitative sample bank. Labels should be typeset separately as vector text; "
        "CIFAR consistency rows are seed-only references.\\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
