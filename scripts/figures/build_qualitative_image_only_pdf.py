#!/usr/bin/env python3
"""Build image-only qualitative PDF panels.

The output PDF contains no text. Rows and columns are documented in the
sidecar manifest so the PDF can be labeled later in a vector editor.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "docs" / "experiments" / "DG_TWFD_v3" / "figures" / "qualitative"
STEPS = [1, 2, 4, 8]
SEEDS = [0, 1, 2, 3]
SAMPLE_GAP = 2
CELL_GAP = 8
ROW_GAP = 8
BACKGROUND = (255, 255, 255)


def _tensor_to_image(x: torch.Tensor) -> Image.Image:
    if x.ndim != 3:
        raise ValueError(f"expected CHW tensor, got shape {tuple(x.shape)}")
    if x.shape[0] not in (1, 3):
        raise ValueError(f"expected 1 or 3 channels, got shape {tuple(x.shape)}")
    x = x.detach().cpu().float()
    if float(x.min()) < -0.01:
        x = (x + 1.0) / 2.0
    x = x.clamp(0, 1)
    arr = (x.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype("uint8")
    if arr.shape[2] == 1:
        arr = arr[:, :, 0]
        return Image.fromarray(arr, mode="L").convert("RGB")
    return Image.fromarray(arr, mode="RGB")


def _load_tensor_samples(pattern: str, step: int) -> list[Image.Image]:
    path = ROOT / pattern.format(step=step)
    tensor = torch.load(path, map_location="cpu")
    return [_tensor_to_image(tensor[i]) for i in SEEDS]


def _sample_path(image_root: Path, seed: int) -> Path:
    nested = image_root / f"{seed - seed % 1000:06d}" / f"{seed:06d}.png"
    if nested.exists():
        return nested
    flat = image_root / f"{seed:06d}.png"
    if flat.exists():
        return flat
    raise FileNotFoundError(f"missing sample for seed {seed}: {image_root}")


def _load_png_samples(pattern: str, step: int) -> list[Image.Image]:
    image_root = ROOT / pattern.format(step=step)
    return [Image.open(_sample_path(image_root, seed)).convert("RGB") for seed in SEEDS]


def _make_cell(images: list[Image.Image]) -> Image.Image:
    widths = [im.width for im in images]
    heights = [im.height for im in images]
    width = sum(widths) + SAMPLE_GAP * (len(images) - 1)
    height = max(heights)
    cell = Image.new("RGB", (width, height), BACKGROUND)
    x = 0
    for im in images:
        cell.paste(im, (x, 0))
        x += im.width + SAMPLE_GAP
    return cell


def _compose_panel(rows: list[dict], output_stem: str) -> dict:
    cells_by_row: list[list[Image.Image]] = []
    for row in rows:
        row_cells = []
        for step in STEPS:
            if row["source_type"] == "tensor":
                samples = _load_tensor_samples(row["pattern"], step)
            elif row["source_type"] == "pngdir":
                samples = _load_png_samples(row["pattern"], step)
            else:
                raise ValueError(f"unknown source_type: {row['source_type']}")
            row_cells.append(_make_cell(samples))
        cells_by_row.append(row_cells)

    col_widths = [max(row[col].width for row in cells_by_row) for col in range(len(STEPS))]
    row_heights = [max(cell.height for cell in row) for row in cells_by_row]
    width = sum(col_widths) + CELL_GAP * (len(STEPS) - 1)
    height = sum(row_heights) + ROW_GAP * (len(rows) - 1)
    panel = Image.new("RGB", (width, height), BACKGROUND)

    y = 0
    for row_idx, row in enumerate(cells_by_row):
        x = 0
        for col_idx, cell in enumerate(row):
            panel.paste(cell, (x, y))
            x += col_widths[col_idx] + CELL_GAP
        y += row_heights[row_idx] + ROW_GAP

    OUTDIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTDIR / f"{output_stem}.png"
    pdf_path = OUTDIR / f"{output_stem}.pdf"
    panel.save(png_path)
    panel.save(pdf_path, "PDF", resolution=72.0)
    return {
        "output_png": str(png_path.relative_to(ROOT)),
        "output_pdf": str(pdf_path.relative_to(ROOT)),
        "pixel_size": [width, height],
        "rows": [row["label"] for row in rows],
        "columns": [f"NFE {step}" for step in STEPS],
        "seeds": SEEDS,
    }


def main() -> None:
    cifar_rows = [
        {
            "label": "DG-TWFD best / v17 auto warp step7750",
            "source_type": "tensor",
            "pattern": "eval/edm_first_cifar10_prior_fullstack_timewarp_v17_rqs_fastwarp_from_step11855_step7750/steps{step}/fixed_seed_samples.pt",
        },
        {
            "label": "DG-TWFD identity / v17 step7750 identity",
            "source_type": "tensor",
            "pattern": "eval/edm_first_cifar10_prior_fullstack_timewarp_v17_rqs_fastwarp_from_step11855_step7750_identity/steps{step}/fixed_seed_samples.pt",
        },
        {
            "label": "CTM no-GAN DSM 10k EMA0.999",
            "source_type": "pngdir",
            "pattern": "runs/ctm_nogan_20260429/cifar10_ema010000_50k/samples/steps{step}/images",
        },
    ]
    imagenet_rows = [
        {
            "label": "EDM official ImageNet64",
            "source_type": "pngdir",
            "pattern": "runs/edm_imagenet64_public_eval_full/samples/steps{step}/images",
        },
        {
            "label": "CD-LPIPS official ImageNet64",
            "source_type": "pngdir",
            "pattern": "runs/cd_imagenet64_lpips_5k/samples/steps{step}/images",
        },
        {
            "label": "CD-L2 official ImageNet64",
            "source_type": "pngdir",
            "pattern": "runs/cd_imagenet64_l2_5k/samples/steps{step}/images",
        },
        {
            "label": "CT official ImageNet64",
            "source_type": "pngdir",
            "pattern": "runs/ct_imagenet64_5k/samples/steps{step}/images",
        },
    ]

    manifest = {
        "note": "PDF panels contain images only; labels are intentionally absent.",
        "omitted": [
            "OpenAI CIFAR-10 CM CD/CT rows are pending local sample generation from JAX checkpoints.",
            "CIFAR-10 EDM official row is pending image generation under the target qualitative protocol.",
        ],
        "panels": [
            _compose_panel(cifar_rows, "qualitative_cifar10_images_only"),
            _compose_panel(imagenet_rows, "qualitative_imagenet64_images_only"),
        ],
    }
    manifest_path = OUTDIR / "qualitative_images_only_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
